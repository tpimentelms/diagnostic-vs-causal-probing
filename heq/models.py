"""The MLP classifier for the hierarchical-equality task: training, caching,
and metrics. The architecture itself lives in :func:`heq.config.mlp_config`.
"""

from pathlib import Path

import torch
import wandb
from sklearn.metrics import classification_report

from pyvene import create_mlp_classifier
from transformers import Trainer, TrainingArguments

from heq.config import CACHE_DIR, mlp_config
from heq.data import cache_path


def compute_metrics(eval_pred):
    return {
        "accuracy": classification_report(
            eval_pred.label_ids.argmax(1), eval_pred.predictions.argmax(1), output_dict=True
        )["accuracy"]
    }


def train_mlp(train_ds, test_ds, embedding_dim, batch_size=1024, epochs=3):
    _, _, model = create_mlp_classifier(mlp_config(embedding_dim))

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


def load_or_train_mlp(train_ds, test_ds, embedding_dim, n_examples, epochs=5, device="cpu", **cache_params):
    mlp_params = {**cache_params, "epochs": epochs}
    path = Path(str(cache_path("mlp", n_examples, **mlp_params)) + ".pt")
    if path.exists():
        print(f"Loading cached MLP from {path}")
        _, _, model = create_mlp_classifier(mlp_config(embedding_dim))
        model.to(device)
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        return model
    CACHE_DIR.mkdir(exist_ok=True)
    trained, _ = train_mlp(train_ds, test_ds, embedding_dim, epochs=epochs)
    # HF Trainer trains on its own auto-selected device; pin the result to the
    # device the rest of the pipeline (probing, DAS) expects before returning.
    trained.to(device)
    torch.save(trained.state_dict(), str(path))
    return trained
