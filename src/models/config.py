"""Configuration objects for the Ch4 hypertuning pipeline.

All settings live in Pydantic models (no plain dicts, no hardcoded paths) so the
model architecture, the data pipeline and the Ray Tune run can each be configured
and validated independently.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class CNNConfig(BaseModel):
    """Architecture of the configurable CNN.

    The hyperparameter hierarchy the report reasons about maps onto these fields:
    architecture (``num_blocks``, ``use_skip``) > capacity (``filters``, ``units``)
    > regularisation (``use_batchnorm``, ``dropout``).
    """

    input_channels: int = 3
    num_classes: int = 5
    num_blocks: int = 2
    filters: int = 32
    use_batchnorm: bool = True
    use_skip: bool = False
    dropout: float = 0.0
    units: int = 128

    @field_validator("num_blocks")
    @classmethod
    def at_least_one_block(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_blocks must be >= 1")
        return v

    @field_validator("dropout")
    @classmethod
    def dropout_in_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return v


class TuneSettings(BaseModel):
    """Settings for the data pipeline and the Ray Tune run."""

    # data
    img_size: int = 96  # downscaled from the native 224 for hardware
    batchsize: int = 32
    # training
    epochs: int = 10
    learning_rate: float = 1e-3
    # tuning
    experiment_name: str = "ch4_flowers"
    num_samples: int = 20
    metric: str = "test_loss"
    mode: Literal["min", "max"] = "min"
    max_concurrent: int = 2
    gpus_per_trial: float = 0.0
    tune_dir: Path = Path("ray_results").resolve()

    @field_validator("img_size", "batchsize", "epochs", "num_samples")
    @classmethod
    def positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("value must be >= 1")
        return v
