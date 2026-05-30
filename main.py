import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import wandb
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Sampler
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


def load_or_train_mlp(train_ds, test_ds, embedding_dim, n_examples, epochs=5, device='cpu', **cache_params):
    mlp_params = {**cache_params, "epochs": epochs}
    path = Path(str(_cache_path("mlp", n_examples, **mlp_params)) + ".pt")
    config = MLPConfig(h_dim=embedding_dim * 4, activation_function="relu", n_layer=3, num_classes=2, pdrop=0.0)
    if path.exists():
        print(f"Loading cached MLP from {path}")
        _, _, model = create_mlp_classifier(config)
        model.to(device)
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        return model
    CACHE_DIR.mkdir(exist_ok=True)
    trained, _ = train_mlp(train_ds, test_ds, embedding_dim, epochs=epochs)
    torch.save(trained.state_dict(), str(path))
    return trained


# ── DAS ───────────────────────────────────────────────────────────────────────

class BatchedRandomSampler(Sampler):
    """Shuffles batch order without mixing examples across batch boundaries.

    pyvene stores counterfactual examples in contiguous blocks of batch_size,
    each block sharing the same intervention_id. Shuffling individual examples
    would mix types within a DataLoader batch; this sampler shuffles block order
    instead so each epoch sees a different interleaving of intervention types.
    """
    def __init__(self, n_examples, batch_size):
        self.n_batches = n_examples // batch_size
        self.batch_size = batch_size

    def __iter__(self):
        for b in torch.randperm(self.n_batches).tolist():
            yield from range(b * self.batch_size, (b + 1) * self.batch_size)

    def __len__(self):
        return self.n_batches * self.batch_size


def get_batched_sampler(batch_size):
    def batched_random_sampler(data):
        batch_indices = [_ for _ in range(int(len(data) / batch_size))]
        random.shuffle(batch_indices)
        for b_i in batch_indices:
            for i in range(b_i * batch_size, (b_i + 1) * batch_size):
                yield i
    return batched_random_sampler


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


def handcraft_das(intervenable, embedding_dim):
    n = embedding_dim * 4  # = 8
    with torch.no_grad():
        for k, v in intervenable.interventions.items():
            v.rotate_layer.parametrizations.weight[0].base.copy_(torch.eye(n))
            v.rotate_layer.parametrizations.weight.original.data.copy_(-torch.eye(n))
            break  # shared rotation, only set once

def _run_intervenable_batch(intervenable, batch, embedding_dim):
    assert batch["intervention_id"].unique().numel() == 1, \
        "All examples in a batch must share the same intervention_id"
    batch_size = batch["input_ids"].shape[0]
    wx_subspace = [[_ for _ in range(0, embedding_dim * 2)]] * batch_size
    yz_subspace = [[_ for _ in range(embedding_dim * 2, embedding_dim * 4)]] * batch_size
    pos = [[[[0]] * batch_size] for _ in range(4)]

    if batch["intervention_id"][0] == 2:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [{"inputs_embeds": batch["source_input_ids"][:, 0]},
             {"inputs_embeds": batch["source_input_ids"][:, 1]}],
            {"sources->base": (pos[0] + pos[1], pos[2] + pos[3])},
            subspaces=[wx_subspace, yz_subspace],
        )
    elif batch["intervention_id"][0] == 0:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [{"inputs_embeds": batch["source_input_ids"][:, 0]}, None],
            {"sources->base": (pos[0] + [None], pos[1] + [None])},
            subspaces=[wx_subspace, None],
        )
    elif batch["intervention_id"][0] == 1:
        return intervenable(
            {"inputs_embeds": batch["input_ids"]},
            [None, {"inputs_embeds": batch["source_input_ids"][:, 0]}],
            {"sources->base": ([None] + pos[0], [None] + pos[1])},
            subspaces=[None, yz_subspace],
        )


def train_das(intervenable, dataset, embedding_dim, batch_size=6400, epochs=10, accumulation_steps=1, lr=0.001, warmup_steps=100, device="cpu"):
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v.rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )
    ce_loss = torch.nn.CrossEntropyLoss()

    intervenable.model.train()
    print("Trainable intervention parameters:", intervenable.count_parameters())
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    total_steps = 1
    optimizer_steps = 0
    for epoch in trange(epochs, desc="Epoch"):
        sampler = get_batched_sampler(batch_size)
        epoch_iter = tqdm(DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset)), desc=f"Epoch {epoch}", leave=True)
        total_loss, total_acc = 0, 0
        for i, batch in enumerate(epoch_iter):
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            _, outputs = _run_intervenable_batch(intervenable, batch, embedding_dim)
            labels = batch["labels"].squeeze().to(torch.long)
            loss = ce_loss(outputs[0], labels)
            acc = (outputs[0].argmax(1) == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

            epoch_iter.set_postfix({"loss": f"{(total_loss/(i+1)):.4f}", "acc": f"{(total_acc/(i+1)):.4f}"})
            if wandb.run is not None:
                wandb.log({"das/loss": loss.item(), "das/acc": acc, "das/step": total_steps})

            (loss / accumulation_steps).backward()

            if total_steps % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                optimizer_steps += 1
            total_steps += 1


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


# ── Diagnostic Probe ─────────────────────────────────────────────────────────

def _collect_activations(model, dataset, layer_idx, batch_size=1024, device="cpu"):
    all_acts = []

    def hook_fn(module, input, output):
        all_acts.append(output.detach().squeeze(1).cpu().numpy())

    hook = model.mlp.h[layer_idx].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset.with_format("torch"), batch_size=batch_size), desc="Collecting", leave=False):
            inputs = batch["inputs_embeds"].to(device).unsqueeze(1)
            model(inputs_embeds=inputs)
    hook.remove()
    return np.concatenate(all_acts)


def _probe_labels(ds, embedding_dim):
    inputs = np.array(ds["inputs_embeds"])
    d = embedding_dim
    WX = (np.abs(inputs[:, :d] - inputs[:, d:2*d]).sum(1) < 1e-6).astype(int)
    YZ = (np.abs(inputs[:, 2*d:3*d] - inputs[:, 3*d:]).sum(1) < 1e-6).astype(int)
    O = np.array(ds["labels"]).argmax(1)
    return {"WX": WX, "YZ": YZ, "O": O}


def run_probe_experiment(model, train_ds, test_ds, embedding_dim, n_layers=3, batch_size=1024, device="cpu"):
    train_labels = _probe_labels(train_ds, embedding_dim)
    test_labels = _probe_labels(test_ds, embedding_dim)

    log_data = {}
    for layer_idx in range(n_layers):
        train_acts = _collect_activations(model, train_ds, layer_idx, batch_size, device)
        test_acts = _collect_activations(model, test_ds, layer_idx, batch_size, device)
        for concept in ["WX", "YZ", "O"]:
            probe = LogisticRegression(max_iter=1000)
            probe.fit(train_acts, train_labels[concept])
            acc = probe.score(test_acts, test_labels[concept])
            print(f"  Layer {layer_idx} -> {concept}: {acc:.4f}")
            log_data[f"probe/layer{layer_idx}/{concept}"] = acc

    if wandb.run is not None:
        wandb.log(log_data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = get_device(args.device)
    # device = 'cpu'
    print(f"Using device: {device}")

    # print("\n=== Handcrafted MLP ===")
    # hc_causal_model = build_causal_model(embedding_dim=2)
    # n_layers = 2
    # trained = build_handcrafted_mlp(embedding_dim=2).to(device)
    # print("Factual accuracy:")
    # eval_factual(handcrafted, hc_causal_model, hc_causal_model.sample_input_tree_balanced, prefix="handcrafted")

    print("\n=== Generating or Loading Data for Hierarchical Equality ===")
    embedding_dim, n_mlp_examples, n_test_examples, das_batch_size = 4, 2**20, 10000, 640
    n_das_train_examples = 100 * 6400   # 1,280,000 — ~200 effective optimizer steps/epoch regardless of das_batch_size
    n_das_test_examples = 3 * 6400       # 19,200 — one block of each intervention type
    n_train_entities, n_test_entities = 100, 100
    sampler = make_input_sampler(embedding_dim)
    train_causal_model = build_causal_model(embedding_dim, n_entities=n_train_entities)
    test_causal_model = build_causal_model(embedding_dim, n_entities=n_test_entities)

    cache_params = dict(dim=embedding_dim, seed=args.seed, nentities=n_train_entities)
    train_ds = load_or_generate_factual("train", train_causal_model, n_mlp_examples, sampler, **cache_params)
    test_ds = load_or_generate_factual("test", test_causal_model, n_test_examples, sampler, **cache_params)
    cf_cache_params = dict(dim=embedding_dim, seed=args.seed, nentities=n_train_entities, bs=das_batch_size)
    train_dataset = load_or_generate_counterfactual("train", train_causal_model, n_das_train_examples, das_batch_size, sampler, **cf_cache_params)
    test_dataset = load_or_generate_counterfactual("test", test_causal_model, n_das_test_examples, das_batch_size, sampler, **cf_cache_params)

    print("\n=== Training and Evaluating MLP on Hierarchical Equality ===")
    n_layers = 3
    trained = load_or_train_mlp(train_ds, test_ds, embedding_dim, n_mlp_examples, epochs=10, device=device, **cache_params)
    trained.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(np.array(test_ds["inputs_embeds"]), dtype=torch.float32).to(device)
        test_logits = trained(inputs_embeds=test_inputs.unsqueeze(1))[0]
    y_test = np.array(test_ds["labels"]).argmax(1)
    print("Trained MLP factual accuracy:")
    print(classification_report(y_test, test_logits.argmax(1).cpu().numpy()))

    print("\n=== Diagnostic Probe Experiment ===")
    run_probe_experiment(trained, train_ds, test_ds, embedding_dim, n_layers=n_layers, device=device)

    print("\n=== Distributed Alignment Search ===")
    intervenable = build_das_intervenable(trained, device)
    accumulation_steps = 6400 // das_batch_size
    # accumulation_steps = 5
    # handcraft_das(intervenable, embedding_dim)
    train_das(intervenable, train_dataset, embedding_dim, batch_size=das_batch_size, epochs=20, accumulation_steps=accumulation_steps, device=device)
    print("DAS counterfactual accuracy:")
    eval_das(intervenable, test_dataset, embedding_dim, batch_size=das_batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    fix_seed(args.seed)
    main(args)
