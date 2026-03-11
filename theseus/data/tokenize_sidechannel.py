"""
Tokenization job for side-channel format data.

Converts multi-turn chat data into the DMA-CoT side-channel format:
- Assistant turns → think_tokens (main reasoning stream)
- User/system turns → sidechannel tokens (one channel per turn, up to N)
- sidechannel_mask → maps each position in think_tokens to the active channel
"""

import os
import json
import random
import time
from typing import Any, Type, cast
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger

from theseus.job import BasicJob
from theseus.registry import job, DATASETS
from theseus.config import field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.data.tokenizer import (
    TokenizerConfig,
    encode_chat_template,
    get_tokenizer,
)


@dataclass
class TokenizeSideChannelConfig:
    """Config for side-channel tokenization."""

    name: str = field("data/dataset")
    config: str = field("data/config", default="")
    suffix: str = field("data/suffix", default="")
    val_pct: float = field("data/val_pct", default=0.05)
    seed: int = field("data/seed", default=2357)
    split: str = field("data/split", default="train")
    block_size: int = field("architecture/block_size", default=512)
    n_channels: int = field("architecture/sidechannel/n_channels", default=4)
    pad_token: int = field("data/pad_token", default=0)
    num_proc: int = field("system/num_proc", default=8)
    system_prompt: str = field("data/system_prompt", default="")

    def __post_init__(self) -> None:
        from theseus.registry import DATASETS

        if self.name not in DATASETS:
            available = ", ".join(sorted(DATASETS.keys()))
            raise ValueError(
                f"Dataset '{self.name}' not found in registry. "
                f"Available datasets: {available}"
            )


def _split_conversation(
    template: ChatTemplate,
) -> tuple[list[ChatTurn], list[ChatTurn]]:
    """Split a conversation into channel turns (system/user) and think turns (assistant).

    Returns:
        (channel_turns, think_turns)
    """
    channel_turns: list[ChatTurn] = []
    think_turns: list[ChatTurn] = []

    for turn in template:
        if turn.role == "assistant":
            think_turns.append(turn)
        else:
            channel_turns.append(turn)

    return channel_turns, think_turns


def _encode_sidechannel_item(
    template: ChatTemplate,
    tokenizer: Any,
    block_size: int,
    n_channels: int,
    pad_token: int,
    system_prompt: str,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]
    | None
):
    """Encode a single conversation into side-channel format.

    Returns:
        (think_tokens, think_mask, sidechannel, sidechannel_mask, truncated) or None if empty.
        - think_tokens: (block_size,) uint32
        - think_mask: (block_size,) bool
        - sidechannel: (n_channels, block_size) uint32
        - sidechannel_mask: (block_size,) int32
        - truncated: bool
    """
    channel_turns, think_turns = _split_conversation(template)

    if not think_turns:
        return None

    # Encode think stream (all assistant turns concatenated)
    think_template: ChatTemplate = []
    for turn in think_turns:
        think_template.append(ChatTurn(role="assistant", message=turn.message))

    think_ids = encode_chat_template(think_template, tokenizer, system_prompt="", tokenize=True)

    if not think_ids:
        return None

    # Truncate/pad think tokens
    seq_len = len(think_ids)
    truncated = seq_len > block_size

    if truncated:
        think_ids = think_ids[-block_size:]

    padding_len = block_size - len(think_ids)
    padded_think = [pad_token] * padding_len + think_ids
    think_mask_list = [False] * padding_len + [True] * len(think_ids)

    think_tokens = np.array(padded_think, dtype=np.uint32)
    think_mask = np.array(think_mask_list, dtype=np.bool_)

    # Encode each channel turn
    # If more than n_channels turns, compress older ones into channel 0
    if len(channel_turns) > n_channels:
        # Merge oldest turns into the first channel
        overflow = channel_turns[: len(channel_turns) - n_channels + 1]
        merged_text = " ".join(t.message for t in overflow)
        merged_turn = ChatTurn(role=overflow[0].role, message=merged_text)
        channel_turns = [merged_turn] + channel_turns[len(channel_turns) - n_channels + 1 :]

    sidechannel = np.full((n_channels, block_size), pad_token, dtype=np.uint32)

    for ch_idx, turn in enumerate(channel_turns):
        ch_template: ChatTemplate = [ChatTurn(role=turn.role, message=turn.message)]
        ch_ids = encode_chat_template(ch_template, tokenizer, system_prompt="", tokenize=True)

        if len(ch_ids) > block_size:
            ch_ids = ch_ids[-block_size:]

        ch_padding = block_size - len(ch_ids)
        padded_ch = [pad_token] * ch_padding + ch_ids
        sidechannel[ch_idx] = np.array(padded_ch, dtype=np.uint32)

    # Build sidechannel_mask: maps each think position to its active channel
    # Strategy: positions after the i-th user turn point to channel i
    # For simplicity, divide think tokens proportionally among channels
    n_active_channels = min(len(channel_turns), n_channels)
    if n_active_channels == 0:
        n_active_channels = 1

    sidechannel_mask = np.zeros(block_size, dtype=np.int32)

    if n_active_channels > 1:
        # Distribute think positions across channels
        # Each channel gets a proportional segment
        positions_per_channel = block_size // n_active_channels
        for ch_idx in range(n_active_channels):
            start = ch_idx * positions_per_channel
            end = (
                (ch_idx + 1) * positions_per_channel
                if ch_idx < n_active_channels - 1
                else block_size
            )
            sidechannel_mask[start:end] = ch_idx

    return think_tokens, think_mask, sidechannel, sidechannel_mask, truncated


@job("data/tokenize_sidechannel_dataset")
class TokenizeSideChannelDatasetJob(BasicJob[TokenizeSideChannelConfig]):
    """Tokenize multi-turn chat data into side-channel format.

    Produces:
    - {split}.think.bin: (num_samples, block_size) uint32
    - {split}.think.bin.mask: (num_samples, block_size) bool
    - {split}.sidechannel.bin: (num_samples, n_channels, block_size) uint32
    - {split}.sidechannel_mask.bin: (num_samples, block_size) int32
    - shape.json, config.json
    """

    @classmethod
    def config(cls) -> Type[TokenizeSideChannelConfig] | list[Type[Any]]:
        return [TokenizeSideChannelConfig, TokenizerConfig]

    @property
    def done(self) -> bool:
        args = self.args
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix == "" else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name

        config_path = output_path / "config.json"
        if not config_path.exists():
            return False

        try:
            with open(config_path) as f:
                saved_config = json.load(f)
            if saved_config != asdict(args):
                return False

            shape_path = output_path / "shape.json"
            if not shape_path.exists():
                return False

            with open(shape_path) as f:
                shapes = json.load(f)

            for split_name in shapes:
                for suffix in (
                    ".think.bin",
                    ".think.bin.mask",
                    ".sidechannel.bin",
                    ".sidechannel_mask.bin",
                ):
                    fpath = output_path / f"{split_name}{suffix}"
                    if not fpath.exists():
                        return False
            return True
        except Exception:
            return False

    def run(self) -> None:
        args = self.args

        if not self.main_process():
            return

        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        output_name = args.name if args.suffix == "" else f"{args.name}_{args.suffix}"
        output_path = data_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        if self.done:
            logger.info(f"Dataset already prepared at {output_path}")
            return

        dataset_cls: Any = DATASETS[args.name]
        dataset: ChatTemplateDataset = dataset_cls(
            split=args.split, config=args.config
        )

        tokenizer = get_tokenizer()

        # Create train/val split
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.seed(args.seed)
        random.shuffle(indices)

        val_size = int(dataset_size * args.val_pct)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        splits = {"train": train_indices, "val": val_indices}
        shapes: dict[str, dict[str, list[int]]] = {}

        for split_name, split_indices in splits.items():
            num_samples = len(split_indices)
            logger.info(f"Processing {split_name} split: {num_samples} samples")

            # Pre-allocate arrays
            think_arr = np.memmap(
                os.path.join(output_path, f"{split_name}.think.bin"),
                dtype=np.uint32,
                mode="w+",
                shape=(num_samples, args.block_size),
            )
            think_mask_arr = np.memmap(
                os.path.join(output_path, f"{split_name}.think.bin.mask"),
                dtype=np.bool_,
                mode="w+",
                shape=(num_samples, args.block_size),
            )
            sc_arr = np.memmap(
                os.path.join(output_path, f"{split_name}.sidechannel.bin"),
                dtype=np.uint32,
                mode="w+",
                shape=(num_samples, args.n_channels, args.block_size),
            )
            sc_mask_arr = np.memmap(
                os.path.join(output_path, f"{split_name}.sidechannel_mask.bin"),
                dtype=np.int32,
                mode="w+",
                shape=(num_samples, args.block_size),
            )

            num_truncated = 0
            num_skipped = 0
            start_time = time.time()
            log_interval = max(1, num_samples // 20)
            write_idx = 0

            for arr_idx, dataset_idx in enumerate(split_indices):
                item = dataset[dataset_idx]

                result = _encode_sidechannel_item(
                    cast(ChatTemplate, item),
                    tokenizer,
                    args.block_size,
                    args.n_channels,
                    args.pad_token,
                    args.system_prompt,
                )

                if result is None:
                    num_skipped += 1
                    continue

                think_tokens, think_mask, sidechannel, sidechannel_mask, truncated = (
                    result
                )

                think_arr[write_idx] = think_tokens
                think_mask_arr[write_idx] = think_mask
                sc_arr[write_idx] = sidechannel
                sc_mask_arr[write_idx] = sidechannel_mask
                write_idx += 1

                if truncated:
                    num_truncated += 1

                if (arr_idx + 1) % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (arr_idx + 1) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[{split_name}] {arr_idx + 1}/{num_samples} "
                        f"({100 * (arr_idx + 1) / num_samples:.1f}%) | "
                        f"{rate:.1f} samples/s"
                    )

            # Truncate to actual written samples
            actual_samples = write_idx

            think_arr.flush()
            think_mask_arr.flush()
            sc_arr.flush()
            sc_mask_arr.flush()

            shapes[split_name] = {
                "think": [actual_samples, args.block_size],
                "sidechannel": [actual_samples, args.n_channels, args.block_size],
                "sidechannel_mask": [actual_samples, args.block_size],
            }

            logger.info(f"{split_name.upper()}: {actual_samples} samples written")
            logger.info(f"  Truncated: {num_truncated}, Skipped: {num_skipped}")

        with open(os.path.join(output_path, "shape.json"), "w") as f:
            json.dump(shapes, f, indent=4)

        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(asdict(args), f, indent=4)

        logger.info(f"Wrote side-channel data to {output_path}")
