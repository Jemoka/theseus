import os
import json
import random
import time
from typing import Optional, Union, cast, Any
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger

from theseus.job import BasicJob
from theseus.config import field
from theseus.data.datasets import (
    DATASETS,
    ChatTemplate,
    ChatTemplateDataset,
    StringDataset,
    StreamingStringDataset,
    StreamingChatTemplateDataset,
)
from theseus.data.tokenizer import get_chatml_encoder, encode_chat_template


# ========== Dataclass Configs ==========


@dataclass
class TokenizeDatasetConfigBase:
    """Base config for dataset tokenization"""

    name: str = field("name")
    suffix: Optional[str] = field("suffix", default=None)
    val_pct: float = field("tokenization/val_pct", default=0.05)
    seed: int = field("tokenization/seed", default=2357)

    def __post_init__(self) -> None:
        """Validate dataset name"""
        if self.name not in DATASETS:
            available = ", ".join(sorted(DATASETS.keys()))
            raise ValueError(
                f"Dataset '{self.name}' not found in registry. Available datasets: {available}"
            )


@dataclass
class TokenizeDatasetConfig(TokenizeDatasetConfigBase):
    """Config for tokenizing non-pretraining datasets with fixed block size"""

    split: str = field("split", default="train")
    block_size: int = field("architecture/block_size", default=512)
    pad_token: int = field("tokenization/pad_token", default=0)
    num_proc: int = field("num_proc", default=8)
    system_prompt: Optional[str] = field("tokenization/system_prompt", default=None)


@dataclass
class TokenizePretrainingDatasetConfig(TokenizeDatasetConfigBase):
    """Config for tokenizing pretraining datasets with streaming"""

    max_samples: Optional[int] = field("max_samples", default=None)


# ========== Dataset Preparation Jobs ==========


class TokenizeBlockwiseDatasetJob(BasicJob[TokenizeDatasetConfig]):
    """
    Prepare non-pretraining datasets with fixed block size.
    Creates .bin and .bin.mask files for train/val splits.
    """

    config = TokenizeDatasetConfig

    @property
    def done(self) -> bool:
        """Check if dataset preparation is complete with the same config"""
        args = self.args

        # Construct output path
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix is None else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name

        # Check if config.json exists
        config_path = output_path / "config.json"
        if not config_path.exists():
            return False

        try:
            # Load and compare config
            with open(config_path) as f:
                saved_config = json.load(f)

            current_config = asdict(args)
            if saved_config != current_config:
                return False

            # Check if shape.json exists
            shape_json_path = output_path / "shape.json"
            if not shape_json_path.exists():
                return False

            with open(shape_json_path) as f:
                existing_shapes = json.load(f)

            # Check if all files exist and have correct shapes
            for split_name, shape in existing_shapes.items():
                tokens_file = output_path / f"{split_name}.bin"
                mask_file = output_path / f"{split_name}.bin.mask"

                if not (tokens_file.exists() and mask_file.exists()):
                    return False

                # Verify file sizes match expected shape
                expected_size = shape[0] * shape[1] * np.uint32().itemsize
                if tokens_file.stat().st_size != expected_size:
                    return False

            return True
        except Exception:
            return False

    def run(self) -> None:
        args = self.args

        # Only main process does the work for data preparation
        if not self.main_process():
            return

        # Construct output path
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix is None else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if already complete
        if self.done:
            logger.info(f"Dataset already prepared at {output_path}")
            shape_json_path = output_path / "shape.json"
            with open(shape_json_path) as f:
                existing_shapes = json.load(f)
            logger.info(f"Shapes: {existing_shapes}")
            return

        # Get dataset from registry
        dataset_cls: Any = DATASETS[args.name]
        dataset: Union[ChatTemplateDataset, StringDataset] = dataset_cls(
            split=args.split
        )

        # Get encoder
        encoder = get_chatml_encoder()

        # Determine if it's a chat dataset by checking the first item
        first_item = dataset[0]
        is_chat = isinstance(first_item, list)

        # Create train/val split
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.seed(args.seed)
        random.shuffle(indices)

        val_size = int(dataset_size * args.val_pct)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        splits = {"train": train_indices, "val": val_indices}

        # Track shapes for shape.json
        shapes = {}

        # Process each split
        for split_name, split_indices in splits.items():
            num_samples = len(split_indices)

            # Statistics tracking
            num_truncated = 0
            num_padded = 0
            total_length = 0
            max_length = 0
            min_length = float("inf")

            start_time = time.time()
            log_interval = max(1, num_samples // 20)  # Log ~20 times per split

            # Create memmap files
            tokens_filename = os.path.join(output_path, f"{split_name}.bin")
            mask_filename = os.path.join(output_path, f"{split_name}.bin.mask")

            dtype = np.uint32  # Use uint32 for cl100k (vocab size ~100k)
            tokens_arr = np.memmap(
                tokens_filename,
                dtype=dtype,
                mode="w+",
                shape=(num_samples, args.block_size),
            )
            mask_arr = np.memmap(
                mask_filename,
                dtype=np.bool_,
                mode="w+",
                shape=(num_samples, args.block_size),
            )

            logger.info(f"Processing {split_name} split: {num_samples} samples")

            # Process each example
            for arr_idx, dataset_idx in enumerate(split_indices):
                item = dataset[dataset_idx]

                # Encode based on type
                if is_chat:
                    chat_item = cast(ChatTemplate, item)
                    ids = encode_chat_template(chat_item, encoder, args.system_prompt)
                else:
                    # String dataset
                    string_item = cast(str, item)
                    ids = encoder.encode(string_item)

                seq_len = len(ids)

                # Update statistics
                total_length += seq_len
                max_length = max(max_length, seq_len)
                min_length = min(min_length, seq_len)

                if seq_len > args.block_size:
                    # Left truncate: keep rightmost block_size tokens
                    num_truncated += 1
                    ids = ids[-args.block_size :]
                    tokens_arr[arr_idx] = np.array(ids, dtype=dtype)
                    mask_arr[arr_idx] = True
                else:
                    # Left pad with pad_token
                    if seq_len < args.block_size:
                        num_padded += 1
                    padding_len = args.block_size - seq_len
                    padded = [args.pad_token] * padding_len + ids
                    tokens_arr[arr_idx] = np.array(padded, dtype=dtype)

                    # Mask: False for padding, True for real tokens
                    mask = [False] * padding_len + [True] * seq_len
                    mask_arr[arr_idx] = np.array(mask, dtype=np.bool_)

                # Periodic logging
                if (arr_idx + 1) % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (arr_idx + 1) / elapsed
                    avg_len = total_length / (arr_idx + 1)
                    total_tokens = (arr_idx + 1) * args.block_size
                    logger.info(
                        f"[{split_name}] {arr_idx + 1}/{num_samples} samples "
                        f"({100 * (arr_idx + 1) / num_samples:.1f}%) | "
                        f"{rate:.1f} samples/s | {total_tokens:,} tokens | "
                        f"avg_len={avg_len:.1f}"
                    )

            # Flush to disk
            tokens_arr.flush()
            mask_arr.flush()

            # Track shape
            shapes[split_name] = [num_samples, args.block_size]

            # Log statistics
            avg_length = total_length / num_samples if num_samples > 0 else 0
            logger.info(f"{split_name.upper()} STATISTICS:")
            logger.info(f"  Total samples: {num_samples}")
            logger.info(f"  Shape: ({num_samples}, {args.block_size})")
            logger.info(
                f"  Truncated: {num_truncated} ({100 * num_truncated / num_samples:.2f}%)"
            )
            logger.info(
                f"  Padded: {num_padded} ({100 * num_padded / num_samples:.2f}%)"
            )
            logger.info(f"  Exact fit: {num_samples - num_truncated - num_padded}")
            logger.info(
                f"  Sequence lengths - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}"
            )

        # Write shape.json
        with open(os.path.join(output_path, "shape.json"), "w") as f:
            json.dump(shapes, f, indent=4)
        logger.info(f"Wrote shape.json to {output_path}: {shapes}")

        # Write config.json for idempotency checking
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(asdict(args), f, indent=4)
        logger.info(f"Wrote config.json to {output_path}")


class TokenizeVariableDatasetJob(BasicJob[TokenizePretrainingDatasetConfig]):
    """
    Prepare pretraining datasets with streaming support.
    Creates train.bin and val.bin files with variable-length sequences.
    """

    config = TokenizePretrainingDatasetConfig

    @property
    def done(self) -> bool:
        """
        Check if dataset preparation is complete with the same config.
        For streaming datasets, we can only verify completion if max_samples is set.
        """
        args = self.args

        # If max_samples is not set, streaming job is never truly "done"
        if args.max_samples is None:
            return False

        # Construct output path
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix is None else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name

        # Check if config.json exists and matches
        config_path = output_path / "config.json"
        if not config_path.exists():
            return False

        try:
            # Load and compare config
            with open(config_path) as f:
                saved_config = json.load(f)

            current_config = asdict(args)
            if saved_config != current_config:
                return False

            train_filename = output_path / "train.bin"
            val_filename = output_path / "val.bin"

            # Check if files exist
            if not (train_filename.exists() and val_filename.exists()):
                return False

            # For streaming datasets with max_samples, check if files are non-empty
            # We can't easily verify exact completion without processing, so we check for non-zero size
            train_size = train_filename.stat().st_size
            val_size = val_filename.stat().st_size

            # Files should have data (at least some tokens)
            return train_size > 0 and val_size > 0
        except Exception:
            return False

    def run(self) -> None:
        args = self.args

        # Only main process does the work for data preparation
        if not self.main_process():
            return

        # Construct output path
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix is None else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Set random seed
        random.seed(args.seed)

        # Get dataset from registry
        dataset_cls: Any = DATASETS[args.name]
        dataset: Union[StreamingChatTemplateDataset, StreamingStringDataset] = (
            dataset_cls()
        )

        # Get encoder
        encoder = get_chatml_encoder()

        # Determine if it's a chat dataset by checking the first item
        iterator = iter(dataset)
        first_item = next(iterator)
        is_chat = isinstance(first_item, list)

        # Recreate iterator
        dataset = dataset_cls()

        # Create memmaps for train and val
        train_filename = os.path.join(output_path, "train.bin")
        val_filename = os.path.join(output_path, "val.bin")
        dtype = np.uint32  # Use uint32 for cl100k

        # Check for existing data and resume if needed
        train_idx = 0
        val_idx = 0
        samples_to_skip = 0

        if os.path.exists(train_filename) and os.path.exists(val_filename):
            logger.info("Found existing data files, checking for resume point...")

            # Find last non-zero position in train file using binary search
            train_resume_idx = self._find_last_nonzero(train_filename, dtype)
            val_resume_idx = self._find_last_nonzero(val_filename, dtype)

            if train_resume_idx > 0 or val_resume_idx > 0:
                train_idx = train_resume_idx
                val_idx = val_resume_idx

                # Estimate samples processed based on ratio
                # This is approximate - we use the val_pct to estimate
                total_tokens = train_idx + val_idx
                samples_to_skip = int(total_tokens * 0.1)  # Conservative estimate

                logger.info(
                    f"Resuming from train_idx={train_idx:,}, val_idx={val_idx:,}"
                )
                logger.info(f"Skipping approximately {samples_to_skip:,} samples")

                # Truncate files to resume point
                with open(train_filename, "r+b") as f:
                    f.truncate(train_idx * dtype().itemsize)
                with open(val_filename, "r+b") as f:
                    f.truncate(val_idx * dtype().itemsize)

        # Start with initial size and grow as needed
        initial_size = max(1_000_000, train_idx + 1_000_000)
        train_arr = np.memmap(
            train_filename,
            dtype=dtype,
            mode="r+" if train_idx > 0 else "w+",
            shape=(initial_size,),
        )
        val_initial_size = max(1_000_000, val_idx + 1_000_000)
        val_arr = np.memmap(
            val_filename,
            dtype=dtype,
            mode="r+" if val_idx > 0 else "w+",
            shape=(val_initial_size,),
        )

        # Logging state
        start_time = time.time()
        last_log_time = start_time
        log_interval = 1000  # Log every 1000 samples
        last_log_sample = 0
        last_log_train_idx = train_idx
        last_log_val_idx = val_idx

        logger.info("Processing streaming dataset...")
        if samples_to_skip > 0:
            logger.info(f"Resuming from {samples_to_skip:,} samples")

        # Iterate through dataset
        sample_count = 0
        for item in dataset:
            # Skip already processed samples
            if sample_count < samples_to_skip:
                sample_count += 1
                continue

            # Limit samples if specified
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            sample_count += 1

            # Encode based on type
            if is_chat:
                chat_item = cast(ChatTemplate, item)
                ids = encode_chat_template(chat_item, encoder)
            else:
                # String dataset - for pretraining, add EOT token
                string_item = cast(str, item)
                ids = encoder.encode(string_item)
                # Add end of text token (this is special token index in cl100k_base)
                ids.append(encoder.eot_token)

            sample_ids = np.array(ids, dtype=dtype)
            sample_len = len(sample_ids)

            # Randomly assign to train or val
            is_val = random.random() < args.val_pct

            if is_val:
                # Resize val memmap if needed
                if val_idx + sample_len > val_arr.shape[0]:
                    val_arr.flush()
                    new_size = max(val_arr.shape[0] * 2, val_idx + sample_len)
                    val_arr = np.memmap(
                        val_filename, dtype=dtype, mode="r+", shape=(new_size,)
                    )

                # Write sample
                val_arr[val_idx : val_idx + sample_len] = sample_ids
                val_idx += sample_len

                if val_idx % 100_000 == 0:
                    val_arr.flush()
            else:
                # Resize train memmap if needed
                if train_idx + sample_len > train_arr.shape[0]:
                    train_arr.flush()
                    new_size = max(train_arr.shape[0] * 2, train_idx + sample_len)
                    train_arr = np.memmap(
                        train_filename, dtype=dtype, mode="r+", shape=(new_size,)
                    )

                # Write sample
                train_arr[train_idx : train_idx + sample_len] = sample_ids
                train_idx += sample_len

                if train_idx % 100_000 == 0:
                    train_arr.flush()

            # Periodic logging
            if sample_count % log_interval == 0:
                current_time = time.time()
                elapsed_total = current_time - start_time
                elapsed_since_log = current_time - last_log_time

                samples_since_log = sample_count - last_log_sample
                train_tokens_since_log = train_idx - last_log_train_idx
                val_tokens_since_log = val_idx - last_log_val_idx
                tokens_since_log = train_tokens_since_log + val_tokens_since_log

                sample_rate = (
                    samples_since_log / elapsed_since_log
                    if elapsed_since_log > 0
                    else 0
                )
                token_rate = (
                    tokens_since_log / elapsed_since_log if elapsed_since_log > 0 else 0
                )

                logger.info(
                    f"[{sample_count:,} samples] "
                    f"{sample_rate:.1f} samples/s | {token_rate:,.0f} tokens/s | "
                    f"train: {train_idx:,} tokens | val: {val_idx:,} tokens | "
                    f"elapsed: {elapsed_total:.1f}s"
                )

                last_log_time = current_time
                last_log_sample = sample_count
                last_log_train_idx = train_idx
                last_log_val_idx = val_idx

        # Trim train file to actual size
        train_arr.flush()
        train_arr._mmap.close()
        del train_arr

        with open(train_filename, "r+b") as f:
            f.truncate(train_idx * dtype().itemsize)

        # Trim val file to actual size
        val_arr.flush()
        val_arr._mmap.close()
        del val_arr

        with open(val_filename, "r+b") as f:
            f.truncate(val_idx * dtype().itemsize)

        logger.info(f"Done! Created in {output_path}:")
        logger.info(f"  train.bin: {train_idx:,} tokens")
        logger.info(f"  val.bin: {val_idx:,} tokens")

        # Write config.json for idempotency checking
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(asdict(args), f, indent=4)
        logger.info(f"Wrote config.json to {output_path}")

    def _find_last_nonzero(self, filepath: str, dtype: type) -> int:
        """
        Find the last non-zero position in a memmap file using binary search.
        Returns the index of the last non-zero element + 1 (i.e., the write position).
        """
        if not os.path.exists(filepath):
            return 0

        filesize = os.path.getsize(filepath)
        if filesize == 0:
            return 0

        data = np.memmap(filepath, dtype=dtype, mode="r")

        # If file is very small or first element is zero, start from beginning
        if len(data) == 0 or data[0] == 0:
            del data
            return 0

        # Binary search for last non-zero position
        lo = 0
        hi = len(data) - 1

        while lo < hi:
            mid = ((hi - lo) // 2) + lo

            # Check if we're in a run of zeros
            if mid > 0 and data[mid] == 0 and data[mid - 1] == 0:
                hi = mid
            elif data[mid] != 0:
                lo = mid + 1
            else:
                # Linear search backwards to find exact boundary
                while mid > 0 and data[mid] == 0:
                    mid -= 1
                lo = mid + 1
                break

        result = lo
        del data
        return result
