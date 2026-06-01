"""Scaling experiments comparing diagnostic probing and DAS.

The quantity of interest is each method's **train-set fit** as a function of the
number of training examples ``n``, measured on both the **correct** labels (signal)
and a **fixed random** labelling (spurious-fitting capacity, i.e. an empirical
Rademacher-complexity estimate). At small ``n`` a method can memorise anything, so
both fits sit near 1.0; as ``n`` grows the random-label fit must fall toward chance
while the correct-label fit persists. The ``n`` at which the random fit collapses is
the capacity signature we want to compare across methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from heq.config import DAS_SCALE_CONFIGS, PROBE_SCALE_N
from heq.das import build_das_intervenable, eval_das, reset_das_rotation, train_das
from heq.data import load_or_generate_counterfactual
from heq.probing import collect_activations, probe_labels

# Roughly constant optimiser budget per DAS fit (with accumulation=1), so that the
# achieved train fit reflects capacity rather than under-training.
DAS_TARGET_UPDATES = 1000


def run_probe_scaling(model, train_ds, test_ds, embedding_dim,
                      n_values=PROBE_SCALE_N, layer_idx=0, device="cpu"):
    """Probe train-fit (capacity) and test accuracy vs n, on correct and random labels."""
    train_acts = collect_activations(model, train_ds, layer_idx, device=device)
    test_acts = collect_activations(model, test_ds, layer_idx, device=device)
    train_labels = probe_labels(train_ds, embedding_dim)
    test_labels = probe_labels(test_ds, embedding_dim)

    results = {concept: {"correct": {"train": [], "test": []},
                         "random":  {"train": [], "test": []}} for concept in ["WX", "YZ"]}
    for n in n_values:
        idx = np.random.choice(len(train_acts), min(n, len(train_acts)), replace=False)
        acts = train_acts[idx]
        for concept in ["WX", "YZ"]:
            true_labels = train_labels[concept][idx]
            rand_labels = np.random.permutation(true_labels)  # fixed random labelling for this n
            for key, lbls in [("correct", true_labels), ("random", rand_labels)]:
                probe = LogisticRegression(max_iter=1000)
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


def run_das_scaling(trained, train_causal_model, test_dataset, embedding_dim,
                    sampler, cache_params, das_configs=DAS_SCALE_CONFIGS,
                    eval_batch_size=640, lr=0.01, warmup_steps=10, device="cpu"):
    """DAS train-fit (capacity) vs n, on correct and on fixed-random labels.

    accumulation_steps=1 and a roughly constant optimiser budget ensure each fit is
    saturated, so train accuracy reflects capacity rather than under-training.
    """
    intervenable = build_das_intervenable(trained, device)
    results = {"correct": {"train": [], "test": []}, "random": {"train": []}}

    for n, b in tqdm(das_configs, desc="DAS scale"):
        epochs = max(30, DAS_TARGET_UPDATES * b // n)  # ~DAS_TARGET_UPDATES optimiser steps (accum=1)
        cf_params = {**cache_params, "bs": b}
        dataset = load_or_generate_counterfactual(
            "scale", train_causal_model, n, b, sampler, **cf_params
        )
        rand_dataset = _shuffle_labels(dataset)

        for key, ds in [("correct", dataset), ("random", rand_dataset)]:
            reset_das_rotation(intervenable, embedding_dim * 4)
            train_das(
                intervenable, ds, embedding_dim,
                batch_size=b, epochs=epochs, accumulation_steps=1,
                lr=lr, warmup_steps=warmup_steps, randomize_labels=False,
                verbose=False, device=device,
            )
            # train fit: evaluate on the very set it was fit to (batch=b keeps one
            # intervention type per batch, as _run_intervenable_batch requires).
            results[key]["train"].append(
                eval_das(intervenable, ds, embedding_dim, batch_size=b, verbose=False, device=device))
            if key == "correct":
                results["correct"]["test"].append(
                    eval_das(intervenable, test_dataset, embedding_dim,
                             batch_size=eval_batch_size, verbose=False, device=device))

    return results


def plot_scaling_experiment(probe_results, das_results, save_path="scaling.png"):
    import matplotlib.pyplot as plt

    n_probe = PROBE_SCALE_N[:len(probe_results["WX"]["correct"]["train"])]
    n_das = [cfg[0] for cfg in DAS_SCALE_CONFIGS[:len(das_results["correct"]["train"])]]

    colors = {"WX": "#1f77b4", "YZ": "#ff7f0e", "DAS": "#2ca02c"}
    _, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    probe_items = [
        ("WX – correct (train)", colors["WX"], "-",  probe_results["WX"]["correct"]["train"]),
        ("WX – random (train)",  colors["WX"], "--", probe_results["WX"]["random"]["train"]),
        ("YZ – correct (train)", colors["YZ"], "-",  probe_results["YZ"]["correct"]["train"]),
        ("YZ – random (train)",  colors["YZ"], "--", probe_results["YZ"]["random"]["train"]),
    ]
    das_items = [
        ("DAS – correct (train)", colors["DAS"], "-",  das_results["correct"]["train"]),
        ("DAS – random (train)",  colors["DAS"], "--", das_results["random"]["train"]),
        ("DAS – correct (test)",  colors["DAS"], ":",  das_results["correct"]["test"]),
    ]

    for ax, (title, n_vals, items) in zip(axes, [
        ("Linear Probe (Layer 0)", n_probe, probe_items),
        ("DAS (linear)", n_das, das_items),
    ]):
        for label, color, ls, vals in items:
            ax.plot(n_vals, vals, marker="o", color=color, linestyle=ls, label=label)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
        ax.set_xscale("log")
        ax.set_xlabel("Training examples (n)")
        ax.set_ylabel("Train-set fit (accuracy)")
        ax.set_title(title)
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()
