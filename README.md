# AISI-test

Diagnostic probing vs. distributed alignment search (DAS) on the hierarchical
equality task.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). Dependencies
(including the correct PyTorch build per platform) are pinned in `uv.lock`.

```bash
uv sync                      # create .venv from the lockfile
```

PyTorch is selected automatically by platform: the CUDA 12.8 build (required for
Blackwell / `sm_120` GPUs) on Linux, and the default MPS-capable wheel on macOS.
On an NVIDIA GPU, confirm the build can target your card with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

## Usage

Run the full pipeline, or any subset of steps:

```bash
uv run python main.py                      # all steps
uv run python main.py --steps probe        # diagnostic probe only
uv run python main.py --steps das,scaling  # DAS + the scaling sweep
uv run python main.py --help               # all options
```

Heavy artifacts (datasets, trained MLP) are cached under `cache/`, so re-runs are
cheap. The accompanying technical report is in `report/`.
