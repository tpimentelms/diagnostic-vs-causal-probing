"""Shared configuration: paths, the MLP architecture, and experiment sweeps.

This module is the single source of truth for the MLP architecture so that the
training, loading, and probing code paths can never silently disagree (a
mismatch would otherwise corrupt the cache keys in :mod:`heq.data`).
"""

from pathlib import Path

from pyvene.models.mlp.modelings_mlp import MLPConfig

CACHE_DIR = Path("cache")


def mlp_config(embedding_dim, n_layer=3):
    """Return the MLP architecture used throughout the project.

    The hidden size is ``4 * embedding_dim`` (one slot per input vector W, X, Y,
    Z), with a binary output head for the hierarchical-equality decision.
    """
    return MLPConfig(
        h_dim=embedding_dim * 4,
        activation_function="relu",
        n_layer=n_layer,
        num_classes=2,
        pdrop=0.0,
    )


# ── Scaling-experiment sweeps ───────────────────────────────────────────────

# (n_total_examples, das_batch_size) pairs for the DAS scaling sweep.
# Each n must be divisible by batch_size * 3 (one block per intervention type).
DAS_SCALE_CONFIGS = [
    (30, 10), (90, 30), (300, 100), (960, 320),
    (1920, 640), (6400, 640), (19200, 640),
]

# Training-set sizes for the diagnostic-probe scaling sweep.
PROBE_SCALE_N = [10, 32, 100, 320, 1000, 3200, 10000]
