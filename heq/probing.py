"""Diagnostic probing: train linear probes on the MLP's hidden activations to
decode the ground-truth concepts (WX, YZ) and the output (O).
"""

import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm


def collect_activations(model, dataset, layer_idx, batch_size=1024, device="cpu"):
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


def probe_labels(ds, embedding_dim):
    inputs = np.array(ds["inputs_embeds"])
    d = embedding_dim
    WX = (np.abs(inputs[:, :d] - inputs[:, d:2*d]).sum(1) < 1e-6).astype(int)
    YZ = (np.abs(inputs[:, 2*d:3*d] - inputs[:, 3*d:]).sum(1) < 1e-6).astype(int)
    O = np.array(ds["labels"]).argmax(1)
    return {"WX": WX, "YZ": YZ, "O": O}


def run_probe_experiment(model, train_ds, test_ds, embedding_dim, n_layers=3, batch_size=1024, device="cpu"):
    train_labels = probe_labels(train_ds, embedding_dim)
    test_labels = probe_labels(test_ds, embedding_dim)

    log_data = {}
    for layer_idx in range(n_layers):
        train_acts = collect_activations(model, train_ds, layer_idx, batch_size, device)
        test_acts = collect_activations(model, test_ds, layer_idx, batch_size, device)
        for concept in ["WX", "YZ", "O"]:
            probe = LogisticRegression(max_iter=1000)
            probe.fit(train_acts, train_labels[concept])
            acc = probe.score(test_acts, test_labels[concept])
            print(f"  Layer {layer_idx} -> {concept}: {acc:.4f}")
            log_data[f"probe/layer{layer_idx}/{concept}"] = acc

    if wandb.run is not None:
        wandb.log(log_data)
