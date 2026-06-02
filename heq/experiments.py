"""Scaling experiments comparing diagnostic probing and DAS.

The quantity of interest is each method's **train-set fit** as a function of the
number of training examples ``n``, measured on both the **correct** labels (signal)
and a **fixed random** labelling (spurious-fitting capacity, i.e. an empirical
Rademacher-complexity estimate). At small ``n`` a method can memorise anything, so
both fits sit near 1.0; as ``n`` grows the random-label fit must fall toward chance
while the correct-label fit persists. The ``n`` at which the random fit collapses is
the capacity signature we want to compare across methods.
"""

import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from heq.config import CACHE_DIR, DAS_SCALE_CONFIGS, SCALE_N
from heq.das import build_das_intervenable, eval_das, reset_das_rotation, train_das
from heq.data import load_or_generate_counterfactual
from heq.probing import collect_activations, probe_labels

# Roughly constant optimiser budget per DAS fit (with accumulation=1), so that the
# achieved train fit reflects capacity rather than under-training.
DAS_TARGET_UPDATES = 5000


def run_probe_scaling(model, train_ds, test_ds, embedding_dim,
                      n_values=SCALE_N, layer_idx=0, device="cpu"):
    """Probe train-fit (capacity) and test accuracy vs n, on correct and random labels."""
    train_acts = collect_activations(model, train_ds, layer_idx, device=device)
    test_acts = collect_activations(model, test_ds, layer_idx, device=device)
    train_labels = probe_labels(train_ds, embedding_dim)
    test_labels = probe_labels(test_ds, embedding_dim)

    results = {concept: {"correct": {"train": [], "test": []},
                         "random":  {"train": [], "test": []}} for concept in ["WX"]}
    for n in n_values:
        idx = np.random.choice(len(train_acts), min(n, len(train_acts)), replace=False)
        acts = train_acts[idx]
        for concept in ["WX"]:
            true_labels = train_labels[concept][idx]
            rand_labels = np.random.permutation(true_labels)  # fixed random labelling for this n
            for key, lbls in [("correct", true_labels), ("random", rand_labels)]:
                probe = LogisticRegression(C=1e6, max_iter=5000)  # ~unregularised: measure true capacity
                probe.fit(acts, lbls)
                # train fit on the labels it was fit to == spurious capacity (for `random`)
                results[concept][key]["train"].append(probe.score(acts, lbls))
                results[concept][key]["test"].append(probe.score(test_acts, test_labels[concept]))

    return results


def _shuffle_labels(dataset, seed=0):
    """Return a copy of the counterfactual dataset with its labels permuted once.

    A *fixed* random labelling makes "fit to random targets" a well-defined,
    memorisable objective; this is the causal-probing analogue of a probe's
    control-task labels.
    """
    labels = np.array(dataset["labels"])
    perm = np.random.RandomState(seed).permutation(len(labels))
    return (dataset.remove_columns("labels")
                   .add_column("labels", labels[perm].tolist())
                   .with_format("torch"))


def _rotation_state(intervenable):
    """A CPU copy of the (shared) rotation's state, for checkpointing."""
    for v in intervenable.interventions.values():
        return {k: t.detach().cpu().clone() for k, t in v.rotate_layer.state_dict().items()}


def run_das_scaling(trained, train_causal_model, test_dataset, embedding_dim,
                    sampler, cache_params, das_configs=DAS_SCALE_CONFIGS,
                    eval_batch_size=640, lr=0.01, warmup_steps=10, seed=0, device="cpu"):
    """DAS train-fit (capacity) vs n, on correct and on fixed-random labels.

    accumulation_steps=1 and a roughly constant optimiser budget ensure each fit is
    saturated, so train accuracy reflects capacity rather than under-training.
    """
    intervenable = build_das_intervenable(trained, device)
    results = {"correct": {"train": [], "test": []}, "random": {"train": []}}
    rotations = {}  # trained rotation per (n, condition), for checkpointing

    for n, b in tqdm(das_configs, desc="DAS scale"):
        epochs = max(30, DAS_TARGET_UPDATES * b // n)  # ~DAS_TARGET_UPDATES optimiser steps (accum=1)
        cf_params = {**cache_params, "bs": b}
        dataset = load_or_generate_counterfactual(
            "scale", train_causal_model, n, b, sampler, **cf_params
        )
        rand_dataset = _shuffle_labels(dataset, seed=seed)

        for key, ds in [("correct", dataset), ("random", rand_dataset)]:
            reset_das_rotation(intervenable, embedding_dim * 4)
            train_das(
                intervenable, ds, embedding_dim,
                batch_size=b, epochs=epochs, accumulation_steps=1,
                lr=lr, warmup_steps=warmup_steps, randomize_labels=False,
                verbose=False, device=device,
            )
            rotations[f"n{n}_{key}"] = _rotation_state(intervenable)
            # train fit: evaluate on the very set it was fit to (batch=b keeps one
            # intervention type per batch, as _run_intervenable_batch requires).
            results[key]["train"].append(
                eval_das(intervenable, ds, embedding_dim, batch_size=b, verbose=False, device=device))
            if key == "correct":
                results["correct"]["test"].append(
                    eval_das(intervenable, test_dataset, embedding_dim,
                             batch_size=eval_batch_size, verbose=False, device=device))

    return results, rotations


def _aggregate_scaling(probe_runs, das_runs):
    """Stack per-seed curves into mean and std (across seeds) for each plotted curve."""
    def agg(curves):
        a = np.array(curves, dtype=float)  # [n_seeds, n_sizes]
        return {"mean": a.mean(0), "std": a.std(0)}
    return {
        "probe_correct": agg([r["WX"]["correct"]["train"] for r in probe_runs]),
        "probe_random":  agg([r["WX"]["random"]["train"]  for r in probe_runs]),
        "das_correct":   agg([r["correct"]["train"] for r in das_runs]),
        "das_random":    agg([r["random"]["train"]  for r in das_runs]),
    }


def run_scaling_seeds(trained, train_ds, test_ds, train_causal_model, test_dataset,
                      embedding_dim, sampler, cache_params, seeds=(0, 1, 2), device="cpu"):
    """Run the probe and DAS scaling sweeps over several seeds and aggregate them.

    Each seed re-randomises the *fitting* (the probe's subsample and random labelling;
    DAS's rotation init and random labelling) on the same cached representations, so the
    bands reflect run-to-run variance of the methods, not of the data. Returns
    ``(aggregate, probe_runs, das_runs, das_rotations)``.
    """
    probe_runs, das_runs, das_rotations = [], [], {}
    for s in seeds:
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)
        print(f"\n--- scaling seed {s} ---")
        probe_runs.append(
            run_probe_scaling(trained, train_ds, test_ds, embedding_dim, device=device))
        das_res, rots = run_das_scaling(
            trained, train_causal_model, test_dataset, embedding_dim,
            sampler, cache_params, seed=s, device=device)
        das_runs.append(das_res)
        das_rotations[f"seed{s}"] = rots
    return _aggregate_scaling(probe_runs, das_runs), probe_runs, das_runs, das_rotations


def save_scaling_checkpoint(aggregate, probe_runs, das_runs, das_rotations, path=None):
    """Persist aggregated + per-seed scaling results and the trained DAS rotations, so
    the (slow) sweep need not be re-run to re-plot or re-evaluate. Load with torch.load.
    """
    path = path or (CACHE_DIR / "scaling_checkpoint.pt")
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save({"aggregate": aggregate,
                "probe_runs": probe_runs,
                "das_runs": das_runs,
                "das_rotations": das_rotations}, str(path))
    print(f"Saved scaling checkpoint to {path}")


def plot_scaling_seeds(aggregate, save_path="scaling.png"):
    """Train-set fit vs n: mean over seeds with shaded ±1 std bands. Solid = correct
    labels (signal), dashed = random labels (spurious capacity); blue = probe, green =
    DAS. The method whose random (dashed) band drops to chance at smaller n has lower
    spurious-fitting capacity.
    """
    import matplotlib.pyplot as plt

    n = SCALE_N[:len(aggregate["probe_correct"]["mean"])]
    c = {"probe": "#1f77b4", "das": "#2ca02c"}
    curves = [
        ("probe_correct", c["probe"], "-",  "o", "Probe – correct"),
        ("probe_random",  c["probe"], "--", "o", "Probe – random"),
        ("das_correct",   c["das"],   "-",  "s", "DAS – correct"),
        ("das_random",    c["das"],   "--", "s", "DAS – random"),
    ]
    _, ax = plt.subplots(figsize=(7, 5))
    for key, color, ls, marker, label in curves:
        m, s = np.array(aggregate[key]["mean"]), np.array(aggregate[key]["std"])
        ax.plot(n, m, linestyle=ls, marker=marker, color=color, label=label)
        ax.fill_between(n, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_xscale("log")
    ax.set_xlabel("Training examples (n)")
    ax.set_ylabel("Train-set fit (accuracy)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path} (mean +/- 1 std over seeds)")
    plt.show()
