"""
pmd.py
"Poor Man's Dataloader" Datasets
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.strategy import Dataset


class MemmapDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.cache_train = None
        self.cache_val = None
        self.block_size = block_size

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else f"{name}_{suffix}"  # type: ignore
        self.path = path

        self.has_val = (path / "val.bin").exists()

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get batches based on the "poor man's dataloader" strategy"""

        if not self.has_val and split == "val":
            split = "train"

        # args is the run configuration + config is the GPT config
        data_dir = self.path
        block_size = self.block_size

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            if self.cache_train is not None:
                data = self.cache_train
            else:
                data = np.memmap(
                    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
                )
                self.cache_train = data  # type: ignore
        else:
            if self.cache_val is not None:
                data = self.cache_val
            else:
                data = np.memmap(
                    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
                )
                self.cache_val = data  # type: ignore

        if deterministic_key:
            portion = batch_size * block_size
            ix = np.arange(
                deterministic_key * portion,
                (deterministic_key + 1) * portion,
                block_size,
            )
        else:
            ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))

        x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # For now, all tokens are real (no padding)
        padding_mask = np.ones_like(x, dtype=np.bool_)

        return x, y, padding_mask
