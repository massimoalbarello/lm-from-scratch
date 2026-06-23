# lm-from-scratch

Experiment space for building a language model from scratch, following [Stanford CS336: Language Modeling from Scratch](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 (`.python-version`).

```bash
uv sync          # create .venv and install dependencies
```

## Running the notebook

```bash
uv run jupyter lab
```

Open `notebooks/scratchpad.ipynb` and select the **Python 3 (lm-from-scratch)** kernel.

## Adding packages

Add dependencies only as you need them, e.g.:

```bash
uv add torch numpy regex      # then restart the notebook kernel
```