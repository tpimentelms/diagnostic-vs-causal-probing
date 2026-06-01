"""Entry point for the hierarchical-equality probing/DAS experiments.

Prepares data and the MLP, then runs any subset of the analysis steps selected
with ``--steps`` (probe, das, scaling). Heavy artifacts are cached under
``cache/``, so re-runs are cheap.

Examples
--------
    python main.py                      # all steps
    python main.py --steps probe        # diagnostic probe only
    python main.py --steps das,scaling  # DAS + the scaling sweep
"""

import argparse
import os
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import wandb
from sklearn.metrics import classification_report

from heq.data import (
    build_causal_model,
    load_or_generate_counterfactual,
    load_or_generate_factual,
    make_input_sampler,
)
from heq.das import build_das_intervenable, eval_das, train_das
from heq.experiments import plot_scaling_experiment, run_das_scaling, run_probe_scaling
from heq.models import load_or_train_mlp
from heq.probing import run_probe_experiment

ALL_STEPS = ("probe", "das", "scaling")


def fix_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_arg=None):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_steps(value):
    if value.strip().lower() == "all":
        return list(ALL_STEPS)
    steps = [s.strip().lower() for s in value.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step(s): {unknown}; choose from {ALL_STEPS} or 'all'")
    return steps


def get_args():
    parser = argparse.ArgumentParser(description="Hierarchical equality: probing & DAS")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=parse_steps, default="all",
                        help="comma-separated subset of {probe,das,scaling}, or 'all'")
    # Data / model sizes (lifted out of the body so runs are reproducible & tunable).
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--n-entities", type=int, default=100)
    parser.add_argument("--n-mlp-examples", type=int, default=2 ** 20)
    parser.add_argument("--n-test-examples", type=int, default=10000)
    parser.add_argument("--das-batch-size", type=int, default=640)
    parser.add_argument("--n-das-train-examples", type=int, default=200 * 6400)  # ~200 opt. steps/epoch
    parser.add_argument("--n-das-test-examples", type=int, default=3 * 6400)     # one block per intervention type
    parser.add_argument("--mlp-epochs", type=int, default=10)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.01,
                        help="weight decay for MLP training; higher => less over-confident logits")
    parser.add_argument("--das-epochs", type=int, default=20)
    parser.add_argument("--das-lr", type=float, default=0.001)
    parser.add_argument("--das-warmup", type=int, default=100, help="DAS warmup in optimizer steps")
    # Logging.
    parser.add_argument("--wandb-project", type=str, default="das-hierarchical-equality")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main(args):
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    device = get_device(args.device)
    print(f"Using device: {device}")
    dim = args.embedding_dim
    steps = args.steps
    needs_cf = ("das" in steps) or ("scaling" in steps)

    print("\n=== Generating or Loading Data for Hierarchical Equality ===")
    sampler = make_input_sampler(dim)
    train_causal_model = build_causal_model(dim, n_entities=args.n_entities)
    test_causal_model = build_causal_model(dim, n_entities=args.n_entities)

    cache_params = dict(dim=dim, seed=args.seed, nentities=args.n_entities)
    train_ds = load_or_generate_factual("train", train_causal_model, args.n_mlp_examples, sampler, **cache_params)
    test_ds = load_or_generate_factual("test", test_causal_model, args.n_test_examples, sampler, **cache_params)

    test_dataset = None
    train_dataset = None
    if needs_cf:
        cf_cache_params = dict(**cache_params, bs=args.das_batch_size)
        test_dataset = load_or_generate_counterfactual(
            "test", test_causal_model, args.n_das_test_examples, args.das_batch_size, sampler, **cf_cache_params)
    if "das" in steps:
        cf_cache_params = dict(**cache_params, bs=args.das_batch_size)
        train_dataset = load_or_generate_counterfactual(
            "train", train_causal_model, args.n_das_train_examples, args.das_batch_size, sampler, **cf_cache_params)

    print("\n=== Training and Evaluating MLP on Hierarchical Equality ===")
    trained = load_or_train_mlp(train_ds, test_ds, dim, args.n_mlp_examples,
                                epochs=args.mlp_epochs, weight_decay=args.mlp_weight_decay,
                                device=device, **cache_params)
    trained.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(np.array(test_ds["inputs_embeds"]), dtype=torch.float32).to(device)
        test_logits = trained(inputs_embeds=test_inputs.unsqueeze(1))[0]
    y_test = np.array(test_ds["labels"]).argmax(1)
    print("Trained MLP factual accuracy:")
    print(classification_report(y_test, test_logits.argmax(1).cpu().numpy()))

    if "probe" in steps:
        print("\n=== Diagnostic Probe Experiment ===")
        run_probe_experiment(trained, train_ds, test_ds, dim, n_layers=3, device=device)

    if "das" in steps:
        print("\n=== Distributed Alignment Search ===")
        intervenable = build_das_intervenable(trained, device)
        accumulation_steps = 6400 // args.das_batch_size
        train_das(intervenable, train_dataset, dim, batch_size=args.das_batch_size,
                  epochs=args.das_epochs, accumulation_steps=accumulation_steps,
                  lr=args.das_lr, warmup_steps=args.das_warmup, device=device)
        print("DAS counterfactual accuracy:")
        eval_das(intervenable, test_dataset, dim, batch_size=args.das_batch_size, device=device)

    if "scaling" in steps:
        print("\n=== Scaling Experiment: Probes vs DAS (correct vs random labels) ===")
        probe_results = run_probe_scaling(trained, train_ds, test_ds, dim, device=device)
        das_results = run_das_scaling(
            trained, train_causal_model, test_dataset, dim,
            sampler, cache_params, device=device,
        )
        plot_scaling_experiment(probe_results, das_results)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    fix_seed(args.seed)
    main(args)
