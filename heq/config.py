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

# Shared dataset sizes (n) for BOTH the probing and DAS scaling sweeps, so the two
# methods are compared on the same x-axis.
SCALE_N = [30, 90, 300, 960, 1920, 6400, 19200]

# DAS additionally needs a batch size per n: the counterfactual data is stored in
# contiguous blocks of `batch_size`, each holding a single intervention type, so n
# must be a multiple of batch_size (and n // batch_size a multiple of 3 to balance
# the three types).
DAS_SCALE_CONFIGS = [(30, 10), (90, 30), (300, 100), (960, 320),
                     (1920, 640), (6400, 640), (19200, 640)]
assert [n for n, _ in DAS_SCALE_CONFIGS] == SCALE_N, "DAS sizes must match SCALE_N"
