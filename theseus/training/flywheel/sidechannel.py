"""
SideChannelPaddedDataset — loads side-channel format memmap files.

Returns batches with think_x, think_y, sidechannel, sidechannel_mask, padding_mask.
"""

import json
import os
from pathlib import Path
from typing import Optional

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.strategy import Dataset


class SideChannelPaddedDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.block_size = block_size

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        with open(path / "shape.json", "r") as f:
            self.shape = json.load(f)

        self.has_val = (path / "val.think.bin").exists()

        # Caches
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_split(self, split: str) -> dict[str, np.ndarray]:
        if split in self._cache:
            return self._cache[split]

        shape_info = self.shape[split]
        think_shape = tuple(shape_info["think"])
        sc_shape = tuple(shape_info["sidechannel"])
        mask_shape = tuple(shape_info["sidechannel_mask"])

        data: dict[str, np.ndarray] = {
            "think": np.memmap(
                os.path.join(self.path, f"{split}.think.bin"),
                dtype=np.uint32,
                mode="r",
                shape=think_shape,
            ),
            "think_mask": np.memmap(
                os.path.join(self.path, f"{split}.think.bin.mask"),
                dtype=np.bool_,
                mode="r",
                shape=think_shape,
            ),
            "sidechannel": np.memmap(
                os.path.join(self.path, f"{split}.sidechannel.bin"),
                dtype=np.uint32,
                mode="r",
                shape=sc_shape,
            ),
            "sidechannel_mask": np.memmap(
                os.path.join(self.path, f"{split}.sidechannel_mask.bin"),
                dtype=np.int32,
                mode="r",
                shape=mask_shape,
            ),
        }

        self._cache[split] = data
        return data

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if not self.has_val and split == "val":
            split = "train"

        data = self._load_split(split)
        n_samples = data["think"].shape[0]

        if deterministic_key is not None:
            start_idx = (deterministic_key * batch_size) % max(
                1, n_samples - batch_size
            )
            ix = np.arange(start_idx, start_idx + batch_size) % n_samples
        else:
            ix = np.random.randint(0, n_samples, size=(batch_size,))

        think = data["think"][ix]  # (B, L)
        think_mask = data["think_mask"][ix]  # (B, L)
        sidechannel = data["sidechannel"][ix]  # (B, N, L)
        sidechannel_mask = data["sidechannel_mask"][ix]  # (B, L)

        block_size = self.block_size

        # Create x/y shift (next-token prediction)
        think_x = think[:, :-1].astype(np.int64)
        think_y = think[:, 1:].astype(np.int64)
        padding_mask = think_mask[:, :-1].astype(np.bool_)

        # Mask out padding in targets
        think_y[~think_mask[:, 1:]] = -1

        # Truncate sidechannel_mask to match think_x length
        sc_mask = sidechannel_mask[:, :-1].astype(np.int32)

        # Pad to block_size if needed
        if think_x.shape[-1] < block_size:
            pad_len = block_size - think_x.shape[-1]
            think_x = np.concatenate(
                (np.zeros((think_x.shape[0], pad_len), dtype=np.int64), think_x),
                axis=-1,
            )
            think_y = np.concatenate(
                (np.full((think_y.shape[0], pad_len), -1, dtype=np.int64), think_y),
                axis=-1,
            )
            padding_mask = np.concatenate(
                (
                    np.full(
                        (padding_mask.shape[0], pad_len), False, dtype=np.bool_
                    ),
                    padding_mask,
                ),
                axis=-1,
            )
            sc_mask = np.concatenate(
                (np.zeros((sc_mask.shape[0], pad_len), dtype=np.int32), sc_mask),
                axis=-1,
            )

        return {
            "think_x": think_x,
            "think_y": think_y,
            "sidechannel": sidechannel.astype(np.int64),
            "sidechannel_mask": sc_mask,
            "padding_mask": padding_mask,
        }
