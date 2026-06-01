"""Distributed Alignment Search (DAS) with a linear (rotation) alignment map.

We learn an orthogonal map onto the MLP's hidden space, intervene on the WX and
YZ subspaces in that rotated basis, and measure the interchange-intervention
accuracy with which the (frozen) MLP reproduces the target algorithm's
counterfactual outputs.
"""

import math
import random

import torch
import wandb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    RotatedSpaceIntervention,
)


def build_das_intervenable(model, device):
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(0, "block_output", "pos", 1, subspace_partition=None, intervention_link_key=0),
            RepresentationConfig(0, "block_output", "pos", 1, subspace_partition=None, intervention_link_key=0),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    # use_fast=False: the fast path keeps only the *first* location tag, which
    # silently drops the second source in type-2 (both-subspace) interventions.
    intervenable = IntervenableModel(config, model, use_fast=False)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable


def get_batched_sampler(batch_size):
    """Shuffle batch order without mixing examples across batch boundaries.

    pyvene stores counterfactual examples in contiguous blocks of ``batch_size``,
    each block sharing the same ``intervention_id``. Shuffling individual examples
    would mix intervention types within a DataLoader batch; we shuffle block order
    instead so each epoch sees a different interleaving of the types.
    """
    def batched_random_sampler(data):
        batch_indices = list(range(len(data) // batch_size))
        random.shuffle(batch_indices)
        for b_i in batch_indices:
            for i in range(b_i * batch_size, (b_i + 1) * batch_size):
                yield i
    return batched_random_sampler


def reset_das_rotation(intervenable, n):
    """Re-initialise the shared rotation to a random orthogonal matrix."""
    for v in intervenable.interventions.values():
        with torch.no_grad():
            new_base = torch.empty(n, n)
            torch.nn.init.orthogonal_(new_base)
            v.rotate_layer.parametrizations.weight[0].base.copy_(new_base)
            v.rotate_layer.parametrizations.weight.original.data.copy_(-torch.eye(n))
        break


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


def train_das(intervenable, dataset, embedding_dim, batch_size=6400, epochs=10,
              accumulation_steps=1, lr=0.001, warmup_steps=100, grad_clip=1.0,
              randomize_labels=False, eval_dataset=None, eval_batch_size=640,
              eval_every=1, verbose=True, device="cpu"):
    rotation_params = []
    for k, v in intervenable.interventions.items():
        rotation_params = list(v.rotate_layer.parameters())
        break
    optimizer = torch.optim.Adam(rotation_params, lr=lr)

    # Warmup then cosine decay. Holding lr flat after warmup made the rotation
    # overshoot once it reached a good (and, with a confident MLP, sharp) alignment,
    # sending the loss back up; decaying to ~0 lets it settle into the solution.
    minibatches_per_epoch = max(1, len(dataset) // batch_size)
    total_optim_steps = max(1, (epochs * minibatches_per_epoch) // accumulation_steps)

    def lr_schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    ce_loss = torch.nn.CrossEntropyLoss()

    intervenable.model.train()
    if verbose:
        print("Trainable intervention parameters:", intervenable.count_parameters())
        print(f"Effective batch size: {batch_size * accumulation_steps}")

    total_steps = 1
    optimizer_steps = 0
    epoch_iter_outer = trange(epochs, desc="Epoch") if verbose else range(epochs)
    for epoch in epoch_iter_outer:
        sampler = get_batched_sampler(batch_size)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
        epoch_iter = tqdm(loader, desc=f"Epoch {epoch}", leave=True) if verbose else loader
        # Track loss/acc per intervention type. The three types differ in difficulty
        # (type 2 swaps both subspaces and is hardest), so an aggregate running mean is
        # misleading: it drifts purely with the order types happen to arrive in.
        type_loss = {0: 0.0, 1: 0.0, 2: 0.0}
        type_acc = {0: 0.0, 1: 0.0, 2: 0.0}
        type_cnt = {0: 0, 1: 0, 2: 0}
        for i, batch in enumerate(epoch_iter):
            tid = int(batch["intervention_id"][0])
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            _, outputs = _run_intervenable_batch(intervenable, batch, embedding_dim)
            labels = batch["labels"].squeeze().to(torch.long)
            if randomize_labels:
                labels = labels[torch.randperm(len(labels))]
            loss = ce_loss(outputs[0], labels)
            acc = (outputs[0].argmax(1) == labels).float().mean().item()

            type_loss[tid] += loss.item()
            type_acc[tid] += acc
            type_cnt[tid] += 1

            if verbose:
                epoch_iter.set_postfix(
                    {f"acc{t}": f"{type_acc[t] / type_cnt[t]:.3f}" for t in (0, 1, 2) if type_cnt[t]}
                )
            if wandb.run is not None:
                wandb.log({"das/loss": loss.item(), "das/acc": acc,
                           f"das/acc_type{tid}": acc, "das/step": total_steps})

            (loss / accumulation_steps).backward()

            if total_steps % accumulation_steps == 0:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(rotation_params, grad_clip)
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                optimizer_steps += 1
            total_steps += 1

        # Periodic clean evaluation: order-independent, averaged over all types — the
        # number to actually trust (unlike the within-epoch running means above).
        if eval_dataset is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            eval_acc = eval_das(intervenable, eval_dataset, embedding_dim,
                                batch_size=eval_batch_size, verbose=False, device=device)
            intervenable.model.train()  # eval_das switched to eval(); resume training
            if verbose:
                per_type = " ".join(f"acc{t}={type_acc[t] / type_cnt[t]:.3f}"
                                    for t in (0, 1, 2) if type_cnt[t])
                print(f"[epoch {epoch}] eval IIA={eval_acc:.4f} | train {per_type}")
            if wandb.run is not None:
                wandb.log({"das/eval_acc_epoch": eval_acc, "das/epoch": epoch})


def eval_das(intervenable, test_dataset, embedding_dim, batch_size=6400, verbose=True, device="cpu"):
    eval_labels, eval_preds = [], []
    intervenable.model.eval()

    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=batch_size)
        loader = tqdm(loader, desc="Eval") if verbose else loader
        for batch in loader:
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
    report = classification_report(y_true, y_pred, output_dict=True)
    if verbose:
        print(classification_report(y_true, y_pred))
    if wandb.run is not None:
        wandb.log({"das/eval_accuracy": report["accuracy"]})
    return report["accuracy"]
