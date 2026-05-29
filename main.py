import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import wandb
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import Trainer, TrainingArguments

from pyvene import CausalModel, IntervenableModel, create_mlp_classifier
from pyvene import (
    IntervenableConfig,
    RepresentationConfig,
    RotatedSpaceIntervention,
)
from pyvene.models.mlp.modelings_mlp import MLPConfig

CACHE_DIR = Path("cache")


def fix_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="DAS tutorial: hierarchical equality task")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="das-hierarchical-equality")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def randvec(n, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for _ in range(n)])


def build_causal_model(embedding_dim, n_entities=20):
    variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]
    reps = [randvec(embedding_dim) for _ in range(n_entities)]

    values = {var: reps for var in ["W", "X", "Y", "Z"]}
    values["WX"] = [True, False]
    values["YZ"] = [True, False]
    values["O"] = [True, False]

    parents = {
        "W": [], "X": [], "Y": [], "Z": [],
        "WX": ["W", "X"],
        "YZ": ["Y", "Z"],
        "O": ["WX", "YZ"],
    }

    def filler():
        return reps[0]

    functions = {
        "W": filler, "X": filler, "Y": filler, "Z": filler,
        "WX": lambda x, y: np.array_equal(x, y),
        "YZ": lambda x, y: np.array_equal(x, y),
        "O": lambda x, y: x == y,
    }

    return CausalModel(variables, values, parents, functions)


def make_input_sampler(embedding_dim):
    def sampler(output_var=None, output_var_value=None):
        A, B, C, D = [randvec(embedding_dim) for _ in range(4)]

        if output_var is None:
            choices = [
                {"W": A, "X": B, "Y": C, "Z": D},
                {"W": A, "X": A, "Y": C, "Z": D},
                {"W": A, "X": B, "Y": C, "Z": C},
                {"W": A, "X": A, "Y": C, "Z": C},
            ]
        elif output_var == "WX":
            if output_var_value:
                choices = [{"W": A, "X": A, "Y": C, "Z": D}, 
                           {"W": A, "X": A, "Y": C, "Z": C}]
            else:
                choices = [{"W": A, "X": B, "Y": C, "Z": D}, 
                           {"W": A, "X": B, "Y": C, "Z": C}]
        elif output_var == "YZ":
            if output_var_value:
                choices = [{"W": A, "X": B, "Y": C, "Z": C}, 
                           {"W": A, "X": A, "Y": C, "Z": C}]
            else:
                choices = [{"W": A, "X": B, "Y": C, "Z": D}, 
                           {"W": A, "X": A, "Y": C, "Z": D}]
        else:
            raise ValueError(f"Unknown output_var: {output_var!r}")

        return random.choice(choices)

    return sampler


def intervention_id(intervention):
    if "WX" in intervention and "YZ" in intervention:
        return 2
    if "WX" in intervention:
        return 0
    return 1


# ── Handcrafted MLP ────────────────────────────────────────────────────────────

def build_handcrafted_mlp(embedding_dim=2):
    assert embedding_dim == 2, "Handcrafted weights are hardcoded for embedding_dim=2"
    config = MLPConfig(
        h_dim=embedding_dim * 4,
        activation_function="relu",
        n_layer=2,
        num_classes=2,
        pdrop=0.0,
    )
    _, _, model = create_mlp_classifier(config)

    # Layer 1: compute per-pair absolute differences
    W1 = [
        [1, 0, -1, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 0, 0],
        [-1, 0, 1, 0, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, -1, 0, 1],
    ]
    model.mlp.h[0].ff1.weight = torch.nn.Parameter(torch.FloatTensor(W1))
    model.mlp.h[0].ff1.bias = torch.nn.Parameter(torch.zeros(8))

    # Layer 2: detect equality from differences
    W2 = [
        [1, -1, 0, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0, 0, 0],
    ]
    model.mlp.h[1].ff1.weight = torch.nn.Parameter(torch.FloatTensor(W2).T)
    model.mlp.h[1].ff1.bias = torch.nn.Parameter(torch.zeros(8))

    W3 = [[1, 0], [1, 0], [-0.999999, 0], [-0.999999, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    model.score.weight = torch.nn.Parameter(torch.FloatTensor(W3).T)
    model.score.bias = torch.nn.Parameter(torch.FloatTensor([0, 1e-14]))

    return model


def eval_factual(model, causal_model, sampler, prefix="handcrafted", n_examples=100000):
    examples = causal_model.generate_factual_dataset(n_examples, sampler)
    X = torch.stack([ex["input_ids"] for ex in examples])
    y = torch.stack([ex["labels"] for ex in examples])

    model.eval()
    with torch.no_grad():
        preds = model(inputs_embeds=X)

    print(classification_report(y, preds[0].argmax(1)))
    if wandb.run is not None:
        report = classification_report(y, preds[0].argmax(1), output_dict=True)
        wandb.log({f"{prefix}/accuracy": report["accuracy"]})


# ── Caching ───────────────────────────────────────────────────────────────────


def _cache_path(name, n_examples, **cache_params):
    param_str = "_".join(f"{k}{v}" for k, v in sorted(cache_params.items()))
    return CACHE_DIR / f"{name}_n{n_examples}_{param_str}"


def load_or_generate_factual(name, causal_model, n_examples, sampler, **cache_params):
    path = _cache_path(f"{name}_factual", n_examples, **cache_params)
    if path.exists():
        print(f"Loading {name} factual dataset from {path}")
        return Dataset.load_from_disk(str(path))
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Generating {name} factual dataset ({n_examples} examples)...")
    examples = causal_model.generate_factual_dataset(n_examples, sampler)
    ds = make_hf_dataset(examples)
    ds.save_to_disk(str(path))
    return ds


def _pyvene_cf_to_hf_dataset(dataset):
    chunks = {}
    for batch in DataLoader(dataset, batch_size=65536):
        for k, v in batch.items():
            chunks.setdefault(k, []).append(v.numpy())
    return Dataset.from_dict({k: np.concatenate(v) for k, v in chunks.items()})


def load_or_generate_counterfactual(name, causal_model, n_examples, batch_size, sampler, **cache_params):
    path = _cache_path(f"{name}_counterfactual", n_examples, **cache_params)
    if path.exists():
        print(f"Loading {name} counterfactual dataset from {path}")
        return Dataset.load_from_disk(str(path)).with_format("torch")
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Generating {name} counterfactual dataset ({n_examples} examples)...")
    dataset = causal_model.generate_counterfactual_dataset(n_examples, intervention_id, batch_size, sampler=sampler)
    hf_ds = _pyvene_cf_to_hf_dataset(dataset)
    hf_ds.save_to_disk(str(path))
    return hf_ds.with_format("torch")


# ── Train MLP ─────────────────────────────────────────────────────────────────

def make_hf_dataset(examples):
    return Dataset.from_dict({
        "labels": [
            torch.FloatTensor([0, 1]) if ex["labels"].item() == 1 else torch.FloatTensor([1, 0])
            for ex in examples
        ],
        "inputs_embeds": torch.stack([ex["input_ids"] for ex in examples]),
    })


def compute_metrics(x):
    return {
        "accuracy": classification_report(
            x.predictions.argmax(1), x.label_ids.argmax(1), output_dict=True
        )["accuracy"]
    }


def train_mlp(train_ds, test_ds, embedding_dim, batch_size=1024, epochs=3):
    config = MLPConfig(
        h_dim=embedding_dim * 4,
        activation_function="relu",
        n_layer=3,
        num_classes=2,
        pdrop=0.0,
    )
    _, _, model = create_mlp_classifier(config)

    print("Configuring training")
    training_args = TrainingArguments(
        output_dir="test_trainer",
        eval_strategy="epoch",
        learning_rate=0.001,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to="wandb" if wandb.run is not None else "none",
        disable_tqdm=False,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    print("Starting MLP model training")
    trainer.train()

    return model, trainer


# ── DAS ───────────────────────────────────────────────────────────────────────

def build_das_intervenable(model, device):
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(0, "block_output", "pos", 1, subspace_partition=None, intervention_link_key=0),
            RepresentationConfig(0, "block_output", "pos", 1, subspace_partition=None, intervention_link_key=0),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    intervenable = IntervenableModel(config, model, use_fast=True)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable


def _run_intervenable_batch(intervenable, batch, embedding_dim):
    assert batch["intervention_id"].unique().numel() == 1, \
        "All examples in a batch must share the same intervention_id"
    batch_size = batch["input_ids"].shape[0]
    wx_subspace = [[i for i in range(0, embedding_dim * 2)]] * batch_size
    yz_subspace = [[i for i in range(embedding_dim * 2, embedding_dim * 4)]] * batch_size
    pos = [[[0]] * batch_size]

    if batch["intervention_id"][0] == 2:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [{"inputs_embeds": batch["source_input_ids"][:, 0]},
             {"inputs_embeds": batch["source_input_ids"][:, 1]}],
            {"sources->base": (pos + pos, pos + pos)},
            subspaces=[wx_subspace, yz_subspace],
        )
    elif batch["intervention_id"][0] == 0:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [{"inputs_embeds": batch["source_input_ids"][:, 0]}, None],
            {"sources->base": (pos + [None], pos + [None])},
            subspaces=[wx_subspace, None],
        )
    else:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [None, {"inputs_embeds": batch["source_input_ids"][:, 0]}],
            {"sources->base": ([None] + pos, [None] + pos)},
            subspaces=[None, yz_subspace],
        )


def train_das(intervenable, dataset, embedding_dim, batch_size=6400, epochs=10, device="cpu"):
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v.rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
    ce_loss = torch.nn.CrossEntropyLoss()

    intervenable.model.train()
    print("Trainable intervention parameters:", intervenable.count_parameters())

    step = 0
    for epoch in trange(epochs, desc="Epoch"):
        epoch_iter = tqdm(DataLoader(dataset, batch_size=batch_size), desc=f"Epoch {epoch}", leave=True)
        for batch in epoch_iter:
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            _, outputs = _run_intervenable_batch(intervenable, batch, embedding_dim)
            labels = batch["labels"].squeeze().to(torch.long)
            loss = ce_loss(outputs[0], labels)
            acc = (outputs[0].argmax(1) == labels).float().mean().item()
            epoch_iter.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
            if wandb.run is not None:
                wandb.log({"das/loss": loss.item(), "das/acc": acc, "das/step": step})

            loss.backward()
            optimizer.step()
            intervenable.set_zero_grad()
            step += 1


def eval_das(intervenable, test_dataset, embedding_dim, batch_size=6400, device="cpu"):
    eval_labels, eval_preds = [], []
    intervenable.model.eval()

    with torch.no_grad():
        for batch in tqdm(DataLoader(test_dataset, batch_size=batch_size), desc="Eval"):
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            _, outputs = _run_intervenable_batch(intervenable, batch, embedding_dim)
            eval_labels.append(batch["labels"].squeeze().cpu())
            eval_preds.append(outputs[0].argmax(1).cpu())

    y_true = torch.cat(eval_labels).numpy()
    y_pred = torch.cat(eval_preds).numpy()
    print(classification_report(y_true, y_pred))
    if wandb.run is not None:
        report = classification_report(y_true, y_pred, output_dict=True)
        wandb.log({"das/eval_accuracy": report["accuracy"]})


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = get_device(args.device)
    print(f"Using device: {device}")

    # print("\n=== Handcrafted MLP ===")
    # hc_causal_model = build_causal_model(embedding_dim=2)
    # handcrafted = build_handcrafted_mlp(embedding_dim=2)
    # print("Factual accuracy:")
    # eval_factual(handcrafted, hc_causal_model, hc_causal_model.sample_input_tree_balanced, prefix="handcrafted")

    print("\n=== Generating or Loading Data for Hierarchical Equality ===")
    embedding_dim, n_train_examples, n_test_examples, das_batch_size = 4, 2**20, 10000, 6400
    n_train_entities, n_test_entities = 100, 100
    sampler = make_input_sampler(embedding_dim)
    train_causal_model = build_causal_model(embedding_dim, n_entities=n_train_entities)
    test_causal_model = build_causal_model(embedding_dim, n_entities=n_test_entities)

    cache_params = dict(dim=embedding_dim, seed=args.seed, nentities=n_train_entities)
    train_ds = load_or_generate_factual("train", train_causal_model, n_train_examples, sampler, **cache_params)
    test_ds = load_or_generate_factual("test", test_causal_model, n_test_examples, sampler, **cache_params)
    train_dataset = load_or_generate_counterfactual("train", train_causal_model, n_train_examples, das_batch_size, sampler, **cache_params)
    test_dataset = load_or_generate_counterfactual("test", test_causal_model, n_test_examples, das_batch_size, sampler, **cache_params)

    print("\n=== Training and Evaluating MLP on Hierarchical Equality ===")
    trained, trainer = train_mlp(train_ds, test_ds, embedding_dim, epochs=3)
    test_preds = trainer.predict(test_ds)
    y_test = np.array(test_ds["labels"]).argmax(1)
    print("Trained MLP factual accuracy:")
    print(classification_report(y_test, test_preds[0].argmax(1)))

    print("\n=== Distributed Alignment Search ===")
    intervenable = build_das_intervenable(trained, device)
    train_das(intervenable, train_dataset, embedding_dim, batch_size=das_batch_size, device=device)
    print("DAS counterfactual accuracy:")
    eval_das(intervenable, test_dataset, embedding_dim, batch_size=das_batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    fix_seed(args.seed)
    main(args)
