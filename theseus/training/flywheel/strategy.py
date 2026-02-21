"""
strategy.py
Data Loading Strategy
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional, List

import numpy as np
from enum import Enum

from theseus.base.job import ExecutionSpec


class DatasetStyle(Enum):
    PADDED = "padded"
    PMD = "pmd"
    CONTRASTIVE = "contrastive"


class Dataset(ABC):
    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        deterministic_key: Optional[int] = None,
    ) -> Dict[str, np.ndarray]: ...


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
        self.queue: Queue[Any] = Queue(maxsize=512)
        self.stop_flag = False
        self.error: Optional[Exception] = None

        self.thread = Thread(target=self._fetch_worker, daemon=True)
        self.thread.start()

    def get_batch(self) -> Any:
        """Get the next batch. Blocks until a batch is available.

        Returns:
            PyTree containing batch data

        Raises:
            Exception: If the worker thread encountered an error.
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
        self.mixture = mixture

        # validate that rates sum to 1
        total_rate = sum(sampling.rate for sampling in mixture)
        if not abs(total_rate - 1.0) < 1e-6:
            raise ValueError(f"Sampling rates must sum to 1, got {total_rate}")

        # Create dataset objects
        self.datasets: list[Dataset] = []
        styles_lower: list[str] = []
        for sampling in mixture:
            ds: Dataset
            style_val = sampling.style
            style_str = (
                style_val.value
                if isinstance(style_val, DatasetStyle)
                else str(style_val)
            )
            style_lower = style_str.lower()
            styles_lower.append(style_lower)

            if style_lower == DatasetStyle.PADDED.value:
                from theseus.training.flywheel.padded import PaddedDataset

                ds = PaddedDataset(spec, block_size, sampling.name, sampling.suffix)
            elif style_lower == DatasetStyle.CONTRASTIVE.value:
                from theseus.training.flywheel.contrastive import (
                    ContrastivePaddedDataset,
                )

                ds = ContrastivePaddedDataset(
                    spec, block_size, sampling.name, sampling.suffix
                )
            elif style_lower == DatasetStyle.PMD.value:
                from theseus.training.flywheel.pmd import MemmapDataset

                ds = MemmapDataset(spec, block_size, sampling.name, sampling.suffix)
            else:
                raise ValueError(f"Unknown dataset style: {sampling.style}")
            self.datasets.append(ds)

        self.is_contrastive = all(
            s == DatasetStyle.CONTRASTIVE.value for s in styles_lower
        )

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
        _recursion_depth: int = 0,
    ) -> Dict[str, np.ndarray]:
        MAX_RECURSION_DEPTH = 10

        r = random if deterministic_key is None else random.Random(deterministic_key)

        # Sample batch distribution: how many samples from each dataset
        counts = self._sample_batch_distribution(batch_size, r)  # type: ignore

        # Gather samples from each dataset
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        mask_parts: List[np.ndarray] = []
        key_offset = 0
        for dataset, count in zip(self.datasets, counts):
            if count > 0:
                det_key = (
                    deterministic_key + key_offset
                    if deterministic_key is not None
                    else None
                )
                batch = dataset.get_batch(count, split, det_key)
                # dict batches only
                if "pos" in batch:
                    x_parts.append(np.asarray(batch["pos"]))
                    y_parts.append(np.asarray(batch["neg"]))
                    mask_parts.append(
                        np.stack(
                            [
                                np.asarray(batch["padding_mask_pos"]),
                                np.asarray(batch["padding_mask_neg"]),
                            ],
                            axis=1,
                        )
                    )
                else:
                    x_parts.append(np.asarray(batch["x"]))
                    y_parts.append(np.asarray(batch["y"]))
                    mask_parts.append(np.asarray(batch["padding_mask"]))
                key_offset += 1

        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        padding_mask = np.concatenate(mask_parts, axis=0)

        # Shuffle to mix samples from different datasets
        perm = r.sample(range(min(batch_size, len(x))), min(batch_size, len(x)))
        x = x[perm]
        y = y[perm]
        padding_mask = padding_mask[perm]

        # Validate batch: resample any all-zero rows (consider pos and neg if present)
        reduce_axes = tuple(range(1, x.ndim))
        valid_rows = ~((x == 0).all(axis=reduce_axes) & (y == 0).all(axis=reduce_axes))
        cut_batch_x = x[valid_rows]
        cut_batch_y = y[valid_rows]
        cut_batch_mask = padding_mask[valid_rows]

        # Try to fill remaining batch, but stop after max depth
        if cut_batch_x.shape[0] < batch_size and _recursion_depth < MAX_RECURSION_DEPTH:
            extra = self.get_batch(
                batch_size - cut_batch_x.shape[0],
                split,
                deterministic_key + 1000 if deterministic_key is not None else None,
                _recursion_depth=_recursion_depth + 1,
            )
            x_addn = extra.get("x", extra.get("pos"))
            y_addn = extra.get("y", extra.get("neg"))
            mask_addn = extra.get("padding_mask")
            if mask_addn is None and "padding_mask_pos" in extra:
                mask_addn = np.stack(
                    [extra["padding_mask_pos"], extra["padding_mask_neg"]], axis=1
                )
            x = np.concatenate([cut_batch_x, x_addn], axis=0)
            y = np.concatenate([cut_batch_y, y_addn], axis=0)
            padding_mask = np.concatenate([cut_batch_mask, mask_addn], axis=0)
        else:
            # Hit max depth or have enough samples, return what we have
            x = cut_batch_x
            y = cut_batch_y
            padding_mask = cut_batch_mask

        if self.is_contrastive:
            padding_mask_pos = padding_mask[:, 0]
            padding_mask_neg = padding_mask[:, 1]
            return {
                "pos": x,
                "neg": y,
                "padding_mask_pos": padding_mask_pos,
                "padding_mask_neg": padding_mask_neg,
            }

        return {"x": x, "y": y, "padding_mask": padding_mask}

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
