"""Flowers data pipeline for the Ch4 hypertuning experiments.

Wraps ``mads_datasets`` so a trial just asks for train/valid streamers at a given
batch size and (downscaled) image size. The dataset is downloaded once and cached
under ``~/.cache/mads_datasets``; subsequent trials read from that cache.
"""

from dataclasses import dataclass

from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor


@dataclass
class FlowerStreamers:
    """Train/valid streamers plus the step counts the Trainer needs."""

    train: object
    valid: object
    train_steps: int
    valid_steps: int


def get_flower_streamers(batchsize: int = 32, img_size: int = 96) -> FlowerStreamers:
    """Create train/valid datastreamers for the Flowers dataset.

    Args:
        batchsize: images per batch.
        img_size: square size to resize images to (native is 224; smaller is faster).
    """
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    # downscale from the native (224, 224) to keep CNN-from-scratch tuning tractable
    factory.settings.img_size = (img_size, img_size)

    streamers = factory.create_datastreamer(batchsize=batchsize, preprocessor=BasePreprocessor())
    train = streamers["train"]
    valid = streamers["valid"]
    logger.info(f"Flowers loaded: img_size={img_size}, batchsize={batchsize}, train={len(train)}, valid={len(valid)}")
    return FlowerStreamers(
        train=train,
        valid=valid,
        train_steps=len(train),
        valid_steps=len(valid),
    )
