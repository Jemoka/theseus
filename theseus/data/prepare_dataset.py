import os
import json
import random
from typing import Optional, Union, cast, Any
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field

from theseus.jobs import BasicJob
from theseus.data.datasets import (
    DATASETS,
    ChatTemplate,
    ChatTemplateDataset,
    StringDataset,
    StreamingStringDataset,
    StreamingChatTemplateDataset,
)
from theseus.data.tokenizer import get_chatml_encoder, encode_chat_template


# ========== Pydantic Configs ==========


class PrepareDatasetConfigBase(BaseModel):
    """Base config for dataset preparation"""

    name: str = Field(description="Name of dataset in DATASETS registry")
    suffix: Optional[str] = Field(
        default=None, description="Optional suffix for output directory name"
    )
    val_pct: float = Field(default=0.05, description="Validation split percentage")
    seed: int = Field(default=2357, description="Random seed for splitting")


class PrepareDatasetConfig(PrepareDatasetConfigBase):
    """Config for preparing non-pretraining datasets with fixed block size"""

    split: str = Field(default="train", description="Dataset split to use")
    block_size: int = Field(default=512, description="Fixed sequence length")
    pad_token: int = Field(default=0, description="Token ID for padding")
    num_proc: int = Field(default=8, description="Number of processes for mapping")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for chat datasets"
    )


class PreparePretrainingDatasetConfig(PrepareDatasetConfigBase):
    """Config for preparing pretraining datasets with streaming"""

    max_samples: Optional[int] = Field(
        default=None, description="Max samples to process (None for all)"
    )


# ========== Dataset Preparation Jobs ==========


class PrepareDatasetJob(BasicJob[PrepareDatasetConfig]):
    """
    Prepare non-pretraining datasets with fixed block size.
    Creates .bin and .bin.mask files for train/val splits.
    """

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

            # Process each example
            for arr_idx, dataset_idx in enumerate(
                tqdm(split_indices, desc=f"writing {split_name}.bin")
            ):
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

            # Flush to disk
            tokens_arr.flush()
            mask_arr.flush()

            # Track shape
            shapes[split_name] = [num_samples, args.block_size]

            # Print statistics
            avg_length = total_length / num_samples if num_samples > 0 else 0
            print(f"\n{split_name.upper()} STATISTICS:")
            print(f"  Total samples: {num_samples}")
            print(f"  Shape: ({num_samples}, {args.block_size})")
            print(
                f"  Truncated: {num_truncated} ({100 * num_truncated / num_samples:.2f}%)"
            )
            print(f"  Padded: {num_padded} ({100 * num_padded / num_samples:.2f}%)")
            print(f"  Exact fit: {num_samples - num_truncated - num_padded}")
            print(
                f"  Sequence lengths - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}"
            )

        # Write shape.json
        with open(os.path.join(output_path, "shape.json"), "w") as f:
            json.dump(shapes, f, indent=4)
        print(f"\nWrote shape.json to {output_path}: {shapes}")


class PreparePretrainingDatasetJob(BasicJob[PreparePretrainingDatasetConfig]):
    """
    Prepare pretraining datasets with streaming support.
    Creates train.bin and val.bin files with variable-length sequences.
    """

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

        # Start with initial size and grow as needed
        initial_size = 1_000_000
        train_arr = np.memmap(
            train_filename, dtype=dtype, mode="w+", shape=(initial_size,)
        )
        val_arr = np.memmap(val_filename, dtype=dtype, mode="w+", shape=(initial_size,))

        train_idx = 0
        val_idx = 0

        # Iterate through dataset
        sample_count = 0
        for item in tqdm(dataset, desc="processing dataset"):
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

        print(f"\nDone! Created in {output_path}:")
        print(f"  train.bin: {train_idx:,} tokens")
        print(f"  val.bin: {val_idx:,} tokens")
