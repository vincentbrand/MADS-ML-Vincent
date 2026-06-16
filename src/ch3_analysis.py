"""Build the Ch3 architecture-comparison figure from the gestures MLflow runs.

The Ch3 exercise notebook only logs to MLflow / TensorBoard / TOML -- it draws no
figures itself. This script reads the final accuracy of each logged run and turns
it into the bar chart the report references, *without* re-training anything:

- ``ch3-architecture-comparison.png`` : final accuracy per architecture (GRU vs
  LSTM vs Conv1D+GRU), with the 90% target marked.

Run from the project root, e.g.::

    uv run python -m src.ch3_analysis \
        --mlflow-db notebooks/3_recurrent_networks/mlflow.db \
        --experiment gestures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient

ACC_METRIC = "metric/Accuracy"
TARGET = 0.90


def family(tag: str) -> str:
    """Group a run tag into its architecture family for colouring."""
    if tag.startswith("Conv1D"):
        return "Conv1D+GRU"
    if tag.startswith("LSTM"):
        return "LSTM"
    return "GRU"


def load_runs(mlflow_db: Path, experiment: str) -> pd.DataFrame:
    """Read each run's model tag and final accuracy from the MLflow store."""
    client = MlflowClient(tracking_uri=f"sqlite:///{mlflow_db}")
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"experiment {experiment!r} not found in {mlflow_db}")
    rows = []
    for run in client.search_runs(exp.experiment_id, max_results=1000):
        tag = run.data.tags.get("model")
        acc = run.data.metrics.get(ACC_METRIC)
        if tag is None or acc is None:
            continue
        rows.append({"model": tag, "accuracy": acc, "family": family(tag)})
    df = pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} runs from experiment {experiment!r}")
    return df


def comparison_plot(df: pd.DataFrame, out_path: Path) -> Path:
    """Horizontal bar chart of final accuracy per architecture, coloured by family."""
    palette = {"GRU": "#4C72B0", "LSTM": "#DD8452", "Conv1D+GRU": "#55A868"}
    colours = [palette[f] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(df["model"], df["accuracy"], color=colours)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(TARGET, 1.005)
    ax.set_xlabel("Final accuracy (higher is better)")
    ax.set_title("Gestures: final accuracy by architecture (target > 90%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(
        handles,
        palette.keys(),
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.success(f"Wrote comparison plot to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Ch3 architecture-comparison figure")
    parser.add_argument(
        "--mlflow-db",
        type=Path,
        default=Path("notebooks/3_recurrent_networks/mlflow.db").resolve(),
    )
    parser.add_argument("--experiment", type=str, default="gestures")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/figures/ch3-architecture-comparison.png").resolve(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_runs(args.mlflow_db, args.experiment)
    comparison_plot(df, args.out)


if __name__ == "__main__":
    main()
