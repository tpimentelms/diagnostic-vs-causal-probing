"""The hierarchical-equality task: causal model, input sampler, and the
(cached) factual / counterfactual datasets used to train the MLP and DAS.

The task asks whether two pairs of vectors are *both* equal or *both* unequal:
``O = (W == X) == (Y == Z)``. The intermediate variables ``WX`` and ``YZ`` are
the ground-truth concepts a probe or alignment map ought to recover.
"""

import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from pyvene import CausalModel

from heq.config import CACHE_DIR


def randvec(n, lower=-1, upper=1):
    """A random vector with 2-decimal entries (so exact equality is meaningful)."""
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


# ── Caching ─────────────────────────────────────────────────────────────────

def cache_path(name, n_examples, **cache_params):
    param_str = "_".join(f"{k}{v}" for k, v in sorted(cache_params.items()))
    return CACHE_DIR / f"{name}_n{n_examples}_{param_str}"


def make_hf_dataset(examples):
    return Dataset.from_dict({
        "labels": [
            torch.FloatTensor([0, 1]) if ex["labels"].item() == 1 else torch.FloatTensor([1, 0])
            for ex in examples
        ],
        "inputs_embeds": torch.stack([ex["input_ids"] for ex in examples]),
    })


def load_or_generate_factual(name, causal_model, n_examples, sampler, **cache_params):
    path = cache_path(f"{name}_factual", n_examples, **cache_params)
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
    path = cache_path(f"{name}_counterfactual", n_examples, **cache_params)
    if path.exists():
        print(f"Loading {name} counterfactual dataset from {path}")
        return Dataset.load_from_disk(str(path)).with_format("torch")
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Generating {name} counterfactual dataset ({n_examples} examples)...")
    dataset = causal_model.generate_counterfactual_dataset(n_examples, intervention_id, batch_size, sampler=sampler)
    hf_ds = _pyvene_cf_to_hf_dataset(dataset)
    hf_ds.save_to_disk(str(path))
    return hf_ds.with_format("torch")


def filter_intervention_type(dataset, tid):
    """Subset a counterfactual dataset to a single intervention type (0/1/2)."""
    # Read the column straight from Arrow. Indexing a column on a torch-formatted
    # dataset runs the formatter row-by-row (~6s per 200k rows); to_numpy() is ~300x
    # faster. Call sites pass a freshly loaded dataset (no indices mapping), so the
    # Arrow column order matches the dataset order.
    ids = dataset.data.column("intervention_id").to_numpy()
    idx = np.where(ids == tid)[0]
    return dataset.select(idx).with_format("torch")
