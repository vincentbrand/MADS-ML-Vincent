"""Ray Tune hypertuning pipeline for the Flowers dataset (Ch4).

Two run modes, matching the iterative method described in the report:

- ``grid``   : deterministic grid over architecture (num_blocks x filters) at a
               *fixed* epoch budget and **no scheduler** — the fair comparison used
               for heatmaps. Never use hyperband here (unequal epochs distort it).
- ``search`` : broader random search over architecture + regularisation + lr, with
               an ASHA (hyperband) scheduler for cheap exploration.

Run from the project root, e.g.::

    uv run python -m src.tune --mode grid --epochs 10
    uv run python -m src.tune --mode search --samples 30 --epochs 15
"""

import argparse
import os
from pathlib import Path

import ray
import torch
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import optim

from src.data import get_flower_streamers
from src.models.cnn import FlowerCNN
from src.models.config import CNNConfig, TuneSettings

MODEL_FIELDS = ("num_blocks", "filters", "use_skip", "use_batchnorm", "dropout", "units")


def select_device() -> str:
    """cuda -> cpu. The mps (Metal/GPU) branch is intentionally omitted.

    On this M5 the mps backend crashes the Python kernel during training, so we
    run on CPU here. cuda is kept so the pipeline still uses a GPU on other machines.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def train_flowers(config: dict) -> None:
    """Ray Tune trainable: build data + model from ``config`` and train one trial.

    Reports ``train_loss`` / ``test_loss`` / ``Accuracy`` to Ray every epoch via
    mltrainer's ``ReportTypes.RAY``.
    """
    streamers = get_flower_streamers(batchsize=config["batchsize"], img_size=config["img_size"])
    model_config = CNNConfig(**{k: config[k] for k in MODEL_FIELDS})
    model = FlowerCNN(model_config)

    settings = TrainerSettings(
        epochs=config["epochs"],
        metrics=[metrics.Accuracy()],
        logdir=config["logdir"],
        train_steps=streamers.train_steps,
        valid_steps=streamers.valid_steps,
        reporttypes=[ReportTypes.RAY],
        optimizer_kwargs={"lr": config["lr"], "weight_decay": 1e-5},
        scheduler_kwargs={"factor": 0.5, "patience": 3},
        earlystop_kwargs={"save": False, "verbose": False, "patience": config["epochs"] + 1},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        traindataloader=streamers.train.stream(),
        validdataloader=streamers.valid.stream(),
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=select_device(),
    )
    trainer.loop()


def build_search_space(mode: str, settings: TuneSettings) -> dict:
    """Construct the Ray Tune config for the requested mode.

    Runtime values (img_size, batchsize, epochs, lr, logdir) are injected so each
    trial process is self-contained.
    """
    runtime = {
        "img_size": settings.img_size,
        "batchsize": settings.batchsize,
        "epochs": settings.epochs,
        "lr": settings.learning_rate,
        "logdir": str(settings.tune_dir),
    }

    if mode == "grid":
        space = {
            "num_blocks": tune.grid_search([2, 3, 4]),
            "filters": tune.grid_search([16, 32, 64]),
            "use_skip": False,
            "use_batchnorm": True,
            "dropout": 0.0,
            "units": 128,
        }
    elif mode == "skipgrid":
        # equal-epoch depth x skip grid (no scheduler) — the fair test for H1.
        space = {
            "num_blocks": tune.grid_search([2, 3, 4]),
            "use_skip": tune.grid_search([False, True]),
            "filters": 32,
            "use_batchnorm": True,
            "dropout": 0.0,
            "units": 128,
        }
    elif mode == "search":
        space = {
            "num_blocks": tune.randint(2, 5),
            "filters": tune.choice([16, 32, 64, 128]),
            "use_skip": tune.choice([True, False]),
            "use_batchnorm": tune.choice([True, False]),
            "dropout": tune.uniform(0.0, 0.5),
            "units": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-2),
        }
        # lr is tuned in search mode; drop the fixed runtime lr
        runtime.pop("lr")
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'grid', 'skipgrid' or 'search')")

    space.update(runtime)
    return space


def run(settings: TuneSettings, mode: str) -> Path:
    """Run the tune experiment and persist the results dataframe. Returns the CSV path."""
    device = select_device()
    logger.info(f"Starting Ray Tune: mode={mode}, device={device}, settings={settings.model_dump()}")

    # Download/extract the dataset once in the driver so parallel trials don't race
    # on the first download (which would read a half-written cache directory).
    get_flower_streamers(batchsize=settings.batchsize, img_size=settings.img_size)

    # mltrainer's ReportTypes.RAY calls ray.train.report, which Ray's Train-v2 API
    # (default in ray>=2.x) raises on inside a Tune function. Force the v1 behaviour
    # in every worker process by setting the env var via the runtime environment.
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_TRAIN_V2_ENABLED": "0"}})

    config = build_search_space(mode, settings)
    cpus = os.cpu_count() or 2
    cpu_per_trial = max(1, (cpus - 1) // settings.max_concurrent)

    # ASHA (hyperband) only for exploration; never for the grid used in heatmaps.
    scheduler = ASHAScheduler(max_t=settings.epochs, grace_period=1, reduction_factor=2) if mode == "search" else None

    settings.tune_dir.mkdir(parents=True, exist_ok=True)
    analysis = tune.run(
        train_flowers,
        name=f"{settings.experiment_name}_{mode}",
        config=config,
        metric=settings.metric,
        mode=settings.mode,
        num_samples=settings.num_samples if mode == "search" else 1,
        scheduler=scheduler,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": settings.gpus_per_trial},
        storage_path=str(settings.tune_dir),
        verbose=1,
    )

    out_csv = settings.tune_dir / f"{settings.experiment_name}_{mode}_results.csv"
    analysis.results_df.to_csv(out_csv, index=False)
    best = analysis.get_best_config(metric=settings.metric, mode=settings.mode)
    logger.success(f"Best config ({settings.metric}={settings.mode}): {best}")
    logger.success(f"Results written to {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray Tune hypertuning on the Flowers dataset")
    parser.add_argument(
        "--mode",
        choices=["grid", "skipgrid", "search"],
        default="grid",
        help="grid (depth x filters heatmap), skipgrid (depth x skip heatmap) or search (explore)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20, help="num_samples (search mode only)")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--experiment-name", type=str, default="ch4_flowers")
    parser.add_argument("--tune-dir", type=Path, default=Path("ray_results").resolve())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TuneSettings(
        img_size=args.img_size,
        batchsize=args.batchsize,
        epochs=args.epochs,
        num_samples=args.samples,
        experiment_name=args.experiment_name,
        max_concurrent=args.max_concurrent,
        tune_dir=args.tune_dir.resolve(),
    )
    run(settings, args.mode)


if __name__ == "__main__":
    # Delegate to the importable module path so Ray pickles the trainable by
    # reference. A function defined in the __main__ module is pickled by value,
    # which makes tune.run() fail with a cloudpickle error.
    from src.tune import main as _main

    _main()
