"""Turn Ray Tune results into the figures the Ch4 report references.

Reads a results CSV written by ``src.tune.run`` and regenerates plots *without*
re-running any training, so the report can be rebuilt offline:

- ``heatmap``               : architecture grid (num_blocks x filters) -> test_loss
- ``parallel_coordinates``  : search-mode hyperparameters -> test_loss

Run from the project root, e.g.::

    uv run python -m src.analysis --results-csv ray_results/ch4_flowers_grid_results.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

CONFIG_PREFIX = "config/"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load a tune results CSV and strip the ``config/`` prefix from hyperparameter columns."""
    df = pd.read_csv(csv_path)
    renamed = {c: c[len(CONFIG_PREFIX) :] for c in df.columns if c.startswith(CONFIG_PREFIX)}
    df = df.rename(columns=renamed)
    logger.info(f"Loaded {len(df)} trials from {csv_path}")
    return df


COLUMN_LABELS = {
    "filters": "filters (base channels)",
    "use_skip": "skip connections",
}


def heatmap(
    df: pd.DataFrame,
    out_path: Path,
    metric: str = "test_loss",
    columns: str = "filters",
) -> Path:
    """Heatmap of an architecture grid: rows=num_blocks, cols=``columns``, value=metric."""
    pivot = df.pivot_table(index="num_blocks", columns=columns, values=metric, aggfunc="min")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"Flowers CNN grid: {metric} (lower is better)")
    ax.set_xlabel(COLUMN_LABELS.get(columns, columns))
    ax.set_ylabel("num_blocks (depth)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.success(f"Wrote heatmap to {out_path}")
    return out_path


def parallel_coordinates(df: pd.DataFrame, out_path: Path, metric: str = "test_loss") -> Path:
    """Parallel-coordinates plot of numeric hyperparameters, coloured by the metric."""
    candidates = ["num_blocks", "filters", "units", "dropout", "lr"]
    cols = [c for c in candidates if c in df.columns and df[c].nunique() > 1]
    if not cols:
        logger.warning("No varying numeric hyperparameters found; skipping parallel-coordinates plot")
        return out_path

    data = df[cols + [metric]].dropna()
    norm = (data[cols] - data[cols].min()) / (data[cols].max() - data[cols].min() + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis_r")
    mvals = data[metric]
    lo, hi = mvals.min(), mvals.max()
    for i in range(len(data)):
        colour = cmap((mvals.iloc[i] - lo) / (hi - lo + 1e-12))
        ax.plot(range(len(cols)), norm.iloc[i].values, color=colour, alpha=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_ylabel("normalised value [0, 1]")
    ax.set_title(f"Hyperparameter search: lines coloured by {metric} (yellow = lower, better)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo, vmax=hi))
    fig.colorbar(sm, ax=ax, label=metric)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.success(f"Wrote parallel-coordinates plot to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ch4 figures from a tune results CSV")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--figdir", type=Path, default=Path("reports/figures").resolve())
    parser.add_argument("--metric", type=str, default="test_loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_results(args.results_csv)
    # hyphenate so the filenames are LaTeX-safe (underscores break text mode)
    stem = args.results_csv.stem.replace("_", "-")
    # heatmap rows are always depth; the column is whichever architecture axis varies
    # (filters for the capacity grid, skip connections for the skipgrid).
    for columns in ("filters", "use_skip"):
        if "num_blocks" in df.columns and columns in df.columns and df[columns].nunique() > 1:
            heatmap(df, args.figdir / f"{stem}-heatmap.png", metric=args.metric, columns=columns)
            break
    parallel_coordinates(df, args.figdir / f"{stem}-parallel.png", metric=args.metric)


if __name__ == "__main__":
    main()
