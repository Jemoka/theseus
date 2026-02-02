"""
strategy.py
Data Loading Strategy
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np
from enum import Enum

from theseus.base.job import ExecutionSpec


class DatasetStyle(Enum):
    PADDED = "padded"
    PMD = "pmd"


class Dataset(ABC):
    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of data.

        Returns:
            Tuple of (x, y, padding_mask) where:
                x: input tokens
                y: target tokens
                padding_mask: bool array, True for real tokens, False for padding
        """
        pass


@dataclass
class Sampling:
    name: str
    rate: float
    style: DatasetStyle = DatasetStyle.PADDED
    suffix: str = ""


class AsyncStrategy:
    def __init__(self, strategy: "Strategy", kwargs: Dict[str, Any]):
        """Asynchronous data loading strategy using threading.

        Args:
            strategy: The underlying strategy to fetch batches from
            kwargs: Arguments to pass to strategy.get_batch()
        """
        self.strategy = strategy
        self.kwargs = kwargs
        self.queue: Queue[Tuple[np.ndarray, np.ndarray, np.ndarray] | Exception] = (
            Queue(maxsize=512)
        )
        self.stop_flag = False
        self.error: Optional[Exception] = None

        self.thread = Thread(target=self._fetch_worker, daemon=True)
        self.thread.start()

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the next batch. Blocks until a batch is available.

        Returns:
            Tuple of numpy arrays (x, y, padding_mask)

        Raises:
            Exception if the worker thread encountered an error
        """
        if self.error is not None:
            raise self.error

        item = self.queue.get()

        if isinstance(item, Exception):
            self.error = item
            raise item

        return item

    def _fetch_worker(self) -> None:
        """Worker thread that continuously fetches batches."""
        try:
            while not self.stop_flag:
                batch = self.strategy.get_batch(**self.kwargs)
                self.queue.put(batch)
        except Exception as e:
            self.queue.put(e)

    def close(self) -> None:
        """Stop the worker thread and clean up."""
        self.stop_flag = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.thread.join(timeout=1.0)

    def __del__(self) -> None:
        """Cleanup when object is garbage collected."""
        self.close()


class Strategy:
    def __init__(self, spec: ExecutionSpec, block_size: int, mixture: list[Sampling]):
        self.mixture = mixture
        self.spec = spec
        self.block_size = block_size

        # make samplings if you got dicts
        nmixture = []
        for i in mixture:
            if isinstance(i, dict):
                nmixture.append(Sampling(**i))
            else:
                nmixture.append(i)
        mixture = nmixture

        # validate that rates sum to 1
        total_rate = sum(sampling.rate for sampling in mixture)
        if not abs(total_rate - 1.0) < 1e-6:
            raise ValueError(f"Sampling rates must sum to 1, got {total_rate}")

        # Create dataset objects
        self.datasets: list[Dataset] = []
        for sampling in mixture:
            ds: Dataset
            if sampling.style == DatasetStyle.PADDED:
                from theseus.training.flywheel.padded import PaddedDataset

                ds = PaddedDataset(spec, block_size, sampling.name, sampling.suffix)
            elif sampling.style == DatasetStyle.PMD:
                from theseus.training.flywheel.pmd import MemmapDataset

                ds = MemmapDataset(spec, block_size, sampling.name, sampling.suffix)
            else:
                raise ValueError(f"Unknown dataset style: {sampling.style}")
            self.datasets.append(ds)

    def get_async_batches(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> AsyncStrategy:
        return AsyncStrategy(
            self,
            {
                "batch_size": batch_size,
                "split": split,
                "deterministic_key": deterministic_key,
            },
        )

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = random if deterministic_key is None else random.Random(deterministic_key)

        # Sample batch distribution: how many samples from each dataset
        counts = self._sample_batch_distribution(batch_size, r)  # type: ignore

        # Gather samples from each dataset
        x_parts, y_parts, mask_parts = [], [], []
        key_offset = 0
        for dataset, count in zip(self.datasets, counts):
            if count > 0:
                det_key = (
                    deterministic_key + key_offset
                    if deterministic_key is not None
                    else None
                )
                bx, by, bm = dataset.get_batch(count, split, det_key)
                x_parts.append(bx)
                y_parts.append(by)
                mask_parts.append(bm)
                key_offset += 1

        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        padding_mask = np.concatenate(mask_parts, axis=0)

        # Shuffle to mix samples from different datasets
        perm = r.sample(range(batch_size), batch_size)
        x = x[perm]
        y = y[perm]
        padding_mask = padding_mask[perm]

        # Validate batch: resample any all-zero rows
        valid_rows = ~(x == 0).all(axis=-1)
        cut_batch_x = x[valid_rows]
        cut_batch_y = y[valid_rows]
        cut_batch_mask = padding_mask[valid_rows]

        if cut_batch_x.shape[0] < batch_size:
            x_addn, y_addn, mask_addn = self.get_batch(
                batch_size - cut_batch_x.shape[0],
                split,
                deterministic_key + 1000 if deterministic_key is not None else None,
            )
            x = np.concatenate([cut_batch_x, x_addn], axis=0)
            y = np.concatenate([cut_batch_y, y_addn], axis=0)
            padding_mask = np.concatenate([cut_batch_mask, mask_addn], axis=0)

        return (x, y, padding_mask)

    def _sample_batch_distribution(
        self, batch_size: int, r: random.Random
    ) -> list[int]:
        """Sample how many items to take from each dataset.

        Uses multinomial-style sampling to distribute batch_size across datasets
        according to their rates.
        """
        rates = [s.rate for s in self.mixture]
        counts = [0] * len(self.mixture)

        # Assign each sample to a dataset based on rates
        for _ in range(batch_size):
            rand_val = r.random()
            cumulative = 0.0
            for i, rate in enumerate(rates):
                cumulative += rate
                if rand_val < cumulative:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1

        return counts
