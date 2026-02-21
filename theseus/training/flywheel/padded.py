"""
padded.py
Padded Dataset - datasets with pre-padded sequences and masks.
"""

import json
import os
from pathlib import Path
from typing import Optional

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.strategy import Dataset


class PaddedDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.cache_train = None
        self.cache_val = None
        self.cache_train_mask = None
        self.cache_val_mask = None
        self.block_size = block_size

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        with open(path / "shape.json", "r") as f:
            self.shape = json.load(f)

        self.has_val = (path / "val.bin").exists()

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """get batches from padded dataset with masks"""

        if not self.has_val and split == "val":
            split = "train"

        shape = tuple(self.shape[split])
        data_dir = self.path
        block_size = self.block_size

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            if self.cache_train is not None:
                data = self.cache_train
                mask = self.cache_train_mask
            else:
                data = np.memmap(
                    os.path.join(data_dir, "train.bin"),
                    dtype=np.uint32,
                    mode="r",
                    shape=shape,
                )
                self.cache_train = data  # type: ignore
                mask = np.memmap(
                    os.path.join(data_dir, "train.bin.mask"),
                    dtype=np.bool_,
                    mode="r",
                    shape=shape,
                )
                self.cache_train_mask = mask  # type: ignore
        else:
            if self.cache_val is not None:
                data = self.cache_val
                mask = self.cache_val_mask
            else:
                data = np.memmap(
                    os.path.join(data_dir, "val.bin"),
                    dtype=np.uint32,
                    mode="r",
                    shape=shape,
                )
                self.cache_val = data  # type: ignore
                mask = np.memmap(
                    os.path.join(data_dir, "val.bin.mask"),
                    dtype=np.bool_,
                    mode="r",
                    shape=shape,
                )
                self.cache_val_mask = mask  # type: ignore

        # check that the dataset is at least as long as the block size
        assert data.shape[1] >= block_size, "Dataset is smaller than block size."
        data = data[:, -(block_size + 1) :]  # type: ignore
        mask = mask[:, -(block_size + 1) :]  # type: ignore

        if deterministic_key is not None:
            # Deterministic sampling: use modulo to wrap around dataset
            start_idx = (deterministic_key * batch_size) % (data.shape[0] - batch_size)
            end_idx = start_idx + batch_size
            if end_idx <= data.shape[0]:
                ix = np.arange(start_idx, end_idx)
            else:
                # Wrap around to beginning if we overflow
                ix = np.concatenate(
                    [
                        np.arange(start_idx, data.shape[0]),
                        np.arange(0, end_idx - data.shape[0]),
                    ]
                )
        else:
            ix = np.random.randint(0, len(data), size=(batch_size,))

        x = data[ix][:, :-1].astype(np.int64)
        y = data[ix][:, 1:].astype(np.int64)
        padding_mask = mask[ix][:, :-1].astype(np.bool_)

        # set padding tokens to -1
        y[~(mask[ix][:, 1:])] = -1

        # pad up to block size if needed
        if x.shape[-1] < block_size:
            pad_len = block_size - x.shape[-1]
            x = np.concatenate(
                (np.zeros((x.shape[0], pad_len), dtype=np.int64), x), axis=-1
            )
            y = np.concatenate(
                (np.full((y.shape[0], pad_len), -1, dtype=np.int64), y), axis=-1
            )
            padding_mask = np.concatenate(
                (
                    np.full((padding_mask.shape[0], pad_len), False, dtype=np.bool_),
                    padding_mask,
                ),
                axis=-1,
            )

        return {"x": x, "y": y, "padding_mask": padding_mask}
