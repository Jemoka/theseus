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

BYTES_PER_BLOCK = 4 * 1024 * 1024  # 4MB block size
BUFFER_BLOCKS = 64  # Number of blocks to buffer (256MB total, ~64M tokens)


class MemmapDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.cache_train: Optional[np.memmap] = None
        self.cache_val: Optional[np.memmap] = None
        self.block_size = block_size

        data_dir: Path = spec.hardware.hosts[jax.process_index()].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        self.has_val = (path / "val.bin").exists()

        # Buffer state for cache-optimal loading (per split)
        self._tokens_per_block = BYTES_PER_BLOCK // 4  # uint32 = 4 bytes
        self._train_buffer: Optional[np.ndarray] = None
        self._train_sample_indices: Optional[np.ndarray] = None
        self._train_sample_ptr = 0
        self._train_next_block = 0
        self._val_buffer: Optional[np.ndarray] = None
        self._val_sample_indices: Optional[np.ndarray] = None
        self._val_sample_ptr = 0
        self._val_next_block = 0

    def _get_memmap(self, split: str) -> np.memmap:
        """Get or create memmap for the given split."""
        if split == "train":
            if self.cache_train is None:
                self.cache_train = np.memmap(
                    os.path.join(self.path, "train.bin"), dtype=np.uint32, mode="r"
                )
            result = self.cache_train
        else:
            if self.cache_val is None:
                self.cache_val = np.memmap(
                    os.path.join(self.path, "val.bin"), dtype=np.uint32, mode="r"
                )
            result = self.cache_val
        assert result is not None
        return result

    def _refill_buffer(self, data: np.memmap, split: str) -> None:
        """Read next BUFFER_BLOCKS sequentially into buffer, generate shuffled indices."""
        file_tokens = len(data)
        total_blocks = max(1, file_tokens // self._tokens_per_block)

        # Get current state for this split
        if split == "train":
            next_block = self._train_next_block
        else:
            next_block = self._val_next_block

        # Wrap around and determine read range
        start_block = next_block % total_blocks
        blocks_to_read = min(BUFFER_BLOCKS, total_blocks)

        # Calculate token range (sequential read - cache friendly!)
        start_token = start_block * self._tokens_per_block
        end_token = min(
            start_token + blocks_to_read * self._tokens_per_block, file_tokens
        )

        # Handle wrap-around at end of file - check if we don't have enough tokens for a full buffer
        if end_token - start_token < self.block_size + 1:
            # Wrap to beginning when we can't get even one valid sample
            start_token = 0
            end_token = min(blocks_to_read * self._tokens_per_block, file_tokens)
            start_block = 0

        # Single sequential read into buffer
        buffer = np.array(data[start_token:end_token])

        # Debug: Check for empty or suspiciously small buffer
        if len(buffer) < self.block_size + 1:
            from loguru import logger

            logger.debug(
                "BUFFER | WARNING: Buffer too small in {} split: {} tokens. "
                "start_token={}, end_token={}, file_tokens={}",
                split,
                len(buffer),
                start_token,
                end_token,
                file_tokens,
            )

        # Generate shuffled sample indices within buffer
        # To avoid overlap, space samples by at least block_size
        num_samples = max(1, (len(buffer) - self.block_size - 1) // self.block_size)
        # Generate non-overlapping positions: 0, block_size, 2*block_size, ...
        sample_indices = np.arange(0, num_samples * self.block_size, self.block_size)
        # Shuffle them to randomize order
        np.random.shuffle(sample_indices)

        # Update state for this split
        if split == "train":
            self._train_buffer = buffer
            self._train_sample_indices = sample_indices
            self._train_sample_ptr = 0
            self._train_next_block = (start_block + blocks_to_read) % total_blocks

            # Log buffer refresh for debugging
            from loguru import logger

            logger.debug(
                "BUFFER | REFILL train: block {}->{}, tokens {}->{} ({} tokens), next_block {}",
                start_block,
                start_block + blocks_to_read,
                start_token,
                end_token,
                end_token - start_token,
                self._train_next_block,
            )
        else:
            self._val_buffer = buffer
            self._val_sample_indices = sample_indices
            self._val_sample_ptr = 0
            self._val_next_block = (start_block + blocks_to_read) % total_blocks

            # Log buffer refresh for debugging
            from loguru import logger

            logger.debug(
                "BUFFER | REFILL val: block {}->{}, tokens {}->{} ({} tokens), next_block {}",
                start_block,
                start_block + blocks_to_read,
                start_token,
                end_token,
                end_token - start_token,
                self._val_next_block,
            )

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get batches using cache-optimal sequential block reads."""
        if not self.has_val and split == "val":
            split = "train"

        data = self._get_memmap(split)
        block_size = self.block_size

        # Deterministic access for validation. Use wrapped sample indices so we
        # never read past EOF and accidentally return empty sequences.
        if deterministic_key is not None:
            max_start = len(data) - block_size - 1
            if max_start < 0:
                raise ValueError(
                    f"Dataset too small for block_size={block_size}: len={len(data)}"
                )

            valid_starts = np.arange(0, max_start + 1, block_size, dtype=np.int64)
            if valid_starts.size == 0:
                valid_starts = np.array([0], dtype=np.int64)

            start = deterministic_key * batch_size
            take = (start + np.arange(batch_size, dtype=np.int64)) % valid_starts.size
            ix = valid_starts[take]

            x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
            y = np.stack(
                [data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix]
            )
            padding_mask = np.ones_like(x, dtype=np.bool_)
            return x, y, padding_mask

        # Random access: use buffer for cache-optimal loading
        if split == "train":
            buffer = self._train_buffer
            sample_indices = self._train_sample_indices
            sample_ptr = self._train_sample_ptr
        else:
            buffer = self._val_buffer
            sample_indices = self._val_sample_indices
            sample_ptr = self._val_sample_ptr

        # Refill buffer if needed
        if (
            buffer is None
            or sample_indices is None
            or sample_ptr + batch_size > len(sample_indices)
        ):
            if buffer is not None and sample_indices is not None:
                from loguru import logger

                logger.debug(
                    "BUFFER | EXHAUSTED {}: sampled {}/{} positions, need {} more",
                    split,
                    sample_ptr,
                    len(sample_indices),
                    batch_size,
                )
            self._refill_buffer(data, split)
            if split == "train":
                buffer = self._train_buffer
                sample_indices = self._train_sample_indices
                sample_ptr = 0
            else:
                buffer = self._val_buffer
                sample_indices = self._val_sample_indices
                sample_ptr = 0

        assert buffer is not None and sample_indices is not None

        # Sample from buffer using pre-shuffled indices
        ix = sample_indices[sample_ptr : sample_ptr + batch_size]

        # Update pointer
        if split == "train":
            self._train_sample_ptr = sample_ptr + batch_size
        else:
            self._val_sample_ptr = sample_ptr + batch_size

        # Extract sequences from buffer (all in-memory, no file access)
        x_list = []
        y_list = []
        for i in ix:
            x_seq = buffer[i : i + block_size]
            y_seq = buffer[i + 1 : i + 1 + block_size]

            # Check if we got full sequences
            if len(x_seq) < block_size or len(y_seq) < block_size:
                from loguru import logger

                logger.debug(
                    "BUFFER | WARNING: Incomplete sequence in {}: x_len={}, y_len={}, "
                    "block_size={}, buffer_len={}, index={}",
                    split,
                    len(x_seq),
                    len(y_seq),
                    block_size,
                    len(buffer),
                    i,
                )
                # Pad with zeros if necessary
                x_seq = np.pad(x_seq, (0, block_size - len(x_seq)), constant_values=0)
                y_seq = np.pad(y_seq, (0, block_size - len(y_seq)), constant_values=0)

            x_list.append(x_seq.astype(np.int64))
            y_list.append(y_seq.astype(np.int64))

        x = np.stack(x_list)
        y = np.stack(y_list)
        padding_mask = np.ones_like(x, dtype=np.bool_)

        return x, y, padding_mask
