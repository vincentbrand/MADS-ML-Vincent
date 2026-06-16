# MADS-ML-Vincent

**Student**: Vincent Brand
**Course**: Machine Learning — Master Applied Data Science (MADS), HAN University
**Instructor**: raoulg
**Date**: April 2026

## Description

This repository contains all exercises and experiments for the MADS Machine Learning course. Each chapter builds on the previous one, progressing from dense networks through CNNs, RNNs, attention mechanisms, and hyperparameter tuning.

## Structure

```
notebooks/
├── 0_baseline/          Ch0: NumPy, pandas, OOP baseline exercises
├── 1_pytorch_intro/     Ch1: Dense network hyperparameter experiments (Fashion MNIST)
├── 2_convolutions/      Ch2: CNNs, dropout, batchnorm, MLflow logging
├── 3_recurrent_networks/ Ch3: GRU, LSTM, Conv1D+GRU on gesture data
├── 4_hypertuning/       Ch4: Ray Tune hyperparameter pipeline
├── 5_attention/         Ch5: EEG eye state classification with attention
└── 6_unsupervised/      Ch6: Autoencoders

reports/                 1-page reports per chapter (PDF)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/MADS-ML-Vincent.git
cd MADS-ML-Vincent

# Create environment with uv
uv sync

# Run a notebook
uv run jupyter notebook
```

## Reproducing experiments

Each notebook is self-contained. Open a notebook and run all cells top-to-bottom. Experiment logs are written to `modellogs/` (TensorBoard + TOML) and `mlflow.db` (MLflow).

```bash
# View TensorBoard
uv run tensorboard --logdir notebooks/1_pytorch_intro/modellogs

# View MLflow dashboard
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Ch4 — hypertuning pipeline (Flowers)

Ch4 is a standalone Ray Tune pipeline (not a notebook), implemented under `src/`:

```
src/data.py          # Flowers datastreamers (downscaled for hardware)
src/models/cnn.py    # configurable CNN: conv/residual blocks, dropout, batchnorm
src/models/config.py # Pydantic configs (architecture + tuning settings)
src/tune.py          # Ray Tune entrypoint (grid / search modes)
src/analysis.py      # build heatmap / parallel-coordinate figures from results
```

Run from the project root:

```bash
# Architecture grid for the heatmap (fixed epochs, NO hyperband)
uv run python -m src.tune --mode grid --epochs 10

# Broader random search with ASHA for exploration
uv run python -m src.tune --mode search --samples 30 --epochs 15

# Build the report figures from a results CSV (no re-training)
uv run python -m src.analysis --results-csv ray_results/ch4_flowers_grid_results.csv
```

Results are written to `ray_results/` and figures to `reports/figures/`.

## Linting

```bash
make format   # auto-format with ruff
make lint     # check with ruff
```
