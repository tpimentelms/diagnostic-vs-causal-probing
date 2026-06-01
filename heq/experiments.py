"""Scaling experiments comparing diagnostic probing and DAS, each evaluated on
both correct and randomised (control) labels, plus the plotting helper.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from heq.config import DAS_SCALE_CONFIGS, PROBE_SCALE_N
from heq.das import build_das_intervenable, eval_das, reset_das_rotation, train_das
from heq.data import load_or_generate_counterfactual
from heq.probing import collect_activations, probe_labels


def run_probe_scaling(model, train_ds, test_ds, embedding_dim,
                      n_values=PROBE_SCALE_N, layer_idx=0, device="cpu"):
    """Probe accuracy vs n_training_examples, with correct and random labels."""
    train_acts = collect_activations(model, train_ds, layer_idx, device=device)
    test_acts = collect_activations(model, test_ds, layer_idx, device=device)
    train_labels = probe_labels(train_ds, embedding_dim)
    test_labels = probe_labels(test_ds, embedding_dim)

    results = {concept: {"correct": [], "random": []} for concept in ["WX", "YZ"]}
    for n in n_values:
        idx = np.random.choice(len(train_acts), min(n, len(train_acts)), replace=False)
        acts = train_acts[idx]
        for concept in ["WX", "YZ"]:
            true_labels = train_labels[concept][idx]
            rand_labels = np.random.permutation(true_labels)
            for key, lbls in [("correct", true_labels), ("random", rand_labels)]:
                probe = LogisticRegression(max_iter=1000)
                probe.fit(acts, lbls)
                results[concept][key].append(probe.score(test_acts, test_labels[concept]))

    return results


def run_das_scaling(trained, train_causal_model, test_dataset, embedding_dim,
                    sampler, cache_params, das_configs=DAS_SCALE_CONFIGS,
                    eval_batch_size=640, device="cpu"):
    """DAS accuracy vs n_training_examples, with correct and random labels."""
    intervenable = build_das_intervenable(trained, device)
    results = {"correct": [], "random": []}

    for n, b in tqdm(das_configs, desc="DAS scale"):
        # ~200 optimizer steps regardless of n: epochs * (n/b) ≈ 200
        epochs = max(5, 200 * b // n)
        cf_params = {**cache_params, "bs": b}
        dataset = load_or_generate_counterfactual(
            "scale", train_causal_model, n, b, sampler, **cf_params
        )
        for randomize in [False, True]:
            reset_das_rotation(intervenable, embedding_dim * 4)
            train_das(
                intervenable, dataset, embedding_dim,
                batch_size=b, epochs=epochs, accumulation_steps=1,
                lr=0.001, warmup_steps=10, randomize_labels=randomize,
                verbose=False, device=device,
            )
            acc = eval_das(
                intervenable, test_dataset, embedding_dim,
                batch_size=eval_batch_size, verbose=False, device=device,
            )
            results["random" if randomize else "correct"].append(acc)

    return results


def plot_scaling_experiment(probe_results, das_results, save_path="scaling.png"):
    import matplotlib.pyplot as plt

    n_probe = PROBE_SCALE_N[:len(probe_results["WX"]["correct"])]
    n_das = [cfg[0] for cfg in DAS_SCALE_CONFIGS[:len(das_results["correct"])]]

    colors = {"WX": "#1f77b4", "YZ": "#ff7f0e", "DAS": "#2ca02c"}
    _, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (title, n_vals, data_items) in zip(axes, [
        ("Linear Probe (Layer 0)", n_probe,
         [("WX – correct", colors["WX"], "-", probe_results["WX"]["correct"]),
          ("WX – random",  colors["WX"], "--", probe_results["WX"]["random"]),
          ("YZ – correct", colors["YZ"], "-", probe_results["YZ"]["correct"]),
          ("YZ – random",  colors["YZ"], "--", probe_results["YZ"]["random"])]),
        ("DAS", n_das,
         [("DAS – correct", colors["DAS"], "-",  das_results["correct"]),
          ("DAS – random",  colors["DAS"], "--", das_results["random"])]),
    ]):
        for label, color, ls, vals in data_items:
            ax.plot(n_vals, vals, marker="o", color=color, linestyle=ls, label=label)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
        ax.set_xscale("log")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Test accuracy")
        ax.set_title(title)
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()
