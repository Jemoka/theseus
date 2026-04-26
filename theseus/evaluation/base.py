"""
Evaluation framework for theseus trainers.

Provides abstract base classes for different evaluation types:
- RolloutEvaluation: Autoregressive generation tasks
- EncodingEvaluation: Next-token prediction accuracy
- PerplexityEvaluation: Dataset perplexity (returns ppl, lower is better)
- PerplexityComparisonEvaluation: Multiple-choice via perplexity comparison

Also provides:
- Evaluator: InferenceJob subclass that runs multiple evaluations
"""

import json
import random
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional, Union, Generic, TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from loguru import logger

from theseus.base import Axis, ExecutionSpec

from theseus.config import field, configure, configuration
from theseus.inference import InferenceJob, M
from theseus.model.module import Module
from theseus.data.tokenizer import Tokenizer, TokenizerConfig, get_tokenizer
from theseus.data.datasets.dataset import ChatTemplate

if TYPE_CHECKING:
    from theseus.training.base import BaseTrainer


def _select_indices(inference: Any, n: int) -> list[int]:
    """Pick which examples to evaluate from an n-sample dataset.

    Reads ``inference.length`` (set by ``Evaluator``); when 0 < length < n,
    samples that many indices via the evaluator's per-call-deterministic
    ``random.Random``. Otherwise returns ``range(n)``. Every host computes
    the same indices because every Evaluator's RNG is seeded identically.
    """
    length = getattr(inference, "length", -1)
    rng = getattr(inference, "random", None)
    if rng is None or length <= 0 or length >= n:
        return list(range(n))
    sampled: list[int] = rng.sample(range(n), length)
    return sampled


def _pad_eval_inputs(
    batch_unit: int, *sequences: list[Any]
) -> tuple[int, tuple[list[Any], ...]]:
    """Pad example lists to a multiple of the global batch size by repeating the last item."""
    if not sequences:
        raise ValueError("Expected at least one sequence to pad.")

    original_size = len(sequences[0])
    if original_size == 0:
        raise ValueError("Evaluation dataset is empty.")

    if any(len(seq) != original_size for seq in sequences[1:]):
        raise ValueError("All evaluation sequences must have the same length.")

    padded_size = ((original_size + batch_unit - 1) // batch_unit) * batch_unit
    pad_count = padded_size - original_size
    if pad_count == 0:
        return original_size, tuple(sequences)

    padded = tuple(seq + [seq[-1]] * pad_count for seq in sequences)
    return original_size, padded


class Evaluation(ABC):
    """Abstract base class for all evaluations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this evaluation."""
        ...

    def prefix(self) -> str:
        """Prefix for metrics from this evaluation."""
        return self.name + "_"

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in this evaluation."""
        ...

    @abstractmethod
    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the evaluation and return a score (and optionally intermediates).

        When ``return_intermediates=True``, also returns a list of
        ``(x, padding_mask)`` numpy arrays — one per sample — available on every
        host so that an RL trainer can use them as a training batch.
        """
        ...

    def _score(self, *args: Any, reduce: str = "mean") -> Any:
        """Reduce per-sample scores from ``self.score(...)`` into a final value.

        ``reduce="mean"`` and ``"sum"`` return a Python float;
        ``reduce="none"`` returns the per-sample np.ndarray.
        """
        per_sample = np.asarray(list(self.score(*args)), dtype=np.float32)
        if reduce == "mean":
            return float(per_sample.mean()) if per_sample.size > 0 else 0.0
        if reduce == "sum":
            return float(per_sample.sum())
        if reduce == "none":
            return per_sample
        raise ValueError(f"unknown reduce mode: {reduce!r}")

    def score(self, *args: Any) -> List[float]:
        """Return one float per evaluation sample. Subclasses override."""
        raise NotImplementedError("Override score() to return one float per sample.")

    @staticmethod
    def find_accumulation_steps(
        dataset_size: int, max_batch_size: int, dp_replicate: int
    ) -> Tuple[int, int] | Tuple[None, None]:
        """Find batch size and accumulation steps that evenly divide dataset.

        Args:
            dataset_size: Total number of samples
            max_batch_size: Maximum per-device batch size
            dp_replicate: Data parallel replication factor

        Returns:
            (batch_size, accumulation_steps) or (None, None) if no valid size found
        """
        for batch_size in reversed(range(1, max_batch_size + 1)):
            if dataset_size % (batch_size * dp_replicate) == 0:
                return batch_size, dataset_size // (batch_size * dp_replicate)

        logger.warning(
            f"Couldn't find a good size for dataset_size {dataset_size} "
            f"and max_batch_size {max_batch_size} and dp_replicate {dp_replicate}. "
            "Will chop off the end of this evaluation."
        )
        return None, None


class RolloutEvaluation(Evaluation):
    """Evaluation using autoregressive generation."""

    def score(self, ys: list[str], y_hats: list[str]) -> List[float]:
        """Per-sample scores. Default: cast each ``check()`` to float."""
        return [float(self.check(y, y_hat)) for y, y_hat in zip(ys, y_hats)]

    def check(self, y: str, y_hat: str) -> bool:
        """Check if y_hat matches y.

        Args:
            y: Ground truth
            y_hat: Generated result

        Returns:
            Whether y_hat matches y
        """
        raise NotImplementedError(
            "Please override this method or self.score, not neither!"
        )

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean generated result before checking.

        Args:
            y_hat: Generated result, which *can include* the prompt

        Returns:
            Cleaned/normalized result available for comparison
        """
        ...

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, str]:
        """Get sample at index.

        Returns:
            (input_string, expected_output_string)
        """
        ...

    def max_new_tokens(self, inference: "InferenceJob[Any, M]") -> int:
        """Maximum tokens to generate. Subclasses MUST override.

        Drives the prompt/generation split (``prompt_max = block_size -
        max_new_tokens``) so the JIT shapes are constant across refills —
        defaulting to ``block_size`` would leave zero room for prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override max_new_tokens()."
        )

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode_batch/decode_batch methods
            reduce: how to reduce per-sample scores ("mean" | "sum" | "none")
            return_intermediates: also return per-sample (rollout, mask) numpy
                arrays on every host (for RL consumers).
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling threshold
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            Evaluation score, or (score, intermediates) when return_intermediates.
        """
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        original_size = len(indices)

        # Pin prompt + total lengths so the JIT shapes are constant across
        # refills (varying-length prompts otherwise force XLA recompiles).
        max_new_tokens = eval_data.max_new_tokens(inference)
        prompt_max = inference.block_size - max_new_tokens
        if prompt_max <= 0:
            raise ValueError(
                f"{eval_data.name}: max_new_tokens={max_new_tokens} leaves no "
                f"room under inference.block_size={inference.block_size}."
            )
        total_tokens = inference.block_size

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x_raw, y_raw = zip(*[eval_data.get(i) for i in indices])
            x = list(x_raw)
            original_y = list(y_raw)
            _, (x, _) = _pad_eval_inputs(batch_unit, x, original_y)
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[:prompt_max] for seq in encoded]
            prompt_lengths = [len(seq) for seq in encoded]
            xs, masks = inference.pad(encoded, pad_to=prompt_max)
        else:
            original_y = None
            prompt_lengths = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Keep a global view of the prompt mask for intermediates (every host).
        masks_full = masks

        # Divide across processes
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )

        # Create subkey
        inference.key, key = jax.random.split(inference.key)

        def evaluate_chunk(
            state: Any, xs_chunk: Any, masks_chunk: Any, key: Any
        ) -> Any:
            """Evaluate a chunk of batches."""

            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch = batch
                results = inference._autoregress(
                    state,
                    key,
                    x_batch,
                    mask_batch,
                    total_tokens,
                    temperature,
                    top_p,
                    **kwargs,
                )
                return None, results

            _, rollouts = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
            results = jnp.reshape(rollouts, (-1, rollouts.shape[-1]))
            return results

        # JIT compile chunk function once
        data_sharding = NamedSharding(inference.mesh, data_pspec)
        wrapped_evaluate_chunk = jax.jit(
            evaluate_chunk,
            in_shardings=(inference.state_sharding, data_sharding, data_sharding, None),
            out_shardings=None,
        )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_results = []

        if jax.process_index() == 0:
            logger.info(
                "EVAL | {} | samples={} prompt={} gen={} batches={}",
                eval_data.name,
                original_size,
                prompt_max,
                max_new_tokens,
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.info(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_results = wrapped_evaluate_chunk(
                inference.state, xs_chunk, masks_chunk, key
            )
            all_results.append(chunk_results)

        # Concatenate all chunk results
        results = jnp.concatenate(all_results, axis=0)

        # Collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten and strip left-pad tokens before decoding
        results = jnp.reshape(results, (-1, results.shape[-1]))
        results_list = results.tolist()

        if jax.process_index() == 0:
            assert prompt_lengths is not None
            max_prompt_len = max(prompt_lengths[:original_size])
            stripped = [
                row[max_prompt_len - prompt_lengths[i] :]
                for i, row in enumerate(results_list[:original_size])
            ]
        else:
            stripped = results_list[:original_size]
        decoded_results = encoding.decode_batch(stripped)

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            assert original_y is not None
            score = eval_data._score(
                original_y,
                [eval_data.clean(i) for i in decoded_results],
                reduce=reduce,
            )
        else:
            score = (
                np.zeros(original_size, dtype=np.float32) if reduce == "none" else 0.0
            )
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        # Per-sample (x, action_mask, padding_mask) on every host. The eval
        # already knows where the prompt ended and where generation began, so
        # it returns both masks rather than making the consumer re-derive.
        #   action_mask : True only over generated tokens (where to tune).
        #   padding_mask: True over prompt + generated (the standard attention
        #                 mask the model expects).
        T_in_max = int(masks_full.shape[-1])
        T_total = int(results.shape[-1])
        gen_len = max(T_total - T_in_max, 0)
        action_mask = jnp.concatenate(
            [
                jnp.zeros_like(masks_full, dtype=jnp.bool_),
                jnp.ones((masks_full.shape[0], gen_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        padding_mask = jnp.concatenate(
            [
                masks_full.astype(jnp.bool_),
                jnp.ones((masks_full.shape[0], gen_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        results_np = np.asarray(results)
        action_mask_np = np.asarray(action_mask)
        padding_mask_np = np.asarray(padding_mask)
        intermediates = [
            (results_np[i], action_mask_np[i], padding_mask_np[i])
            for i in range(original_size)
        ]
        return score, intermediates


class EncodingEvaluation(Evaluation):
    """Evaluation using next-token prediction accuracy."""

    def score(self, xs: list[str], y_hats: list[str]) -> List[float]:
        """Per-sample scores. Default: cast each ``check()`` to float."""
        return [float(self.check(x, y_hat)) for x, y_hat in zip(xs, y_hats)]

    def check(self, x: str, y_hat: str) -> bool:
        """Check if prediction is correct given input.

        Args:
            x: Input string
            y_hat: Model prediction (cleaned, decoded argmax)

        Returns:
            Whether prediction is correct
        """
        raise NotImplementedError("Please override this method or self.score!")

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean model prediction before checking.

        Args:
            y_hat: Raw decoded model prediction

        Returns:
            Cleaned/normalized result available for comparison
        """
        ...

    @abstractmethod
    def get(self, indx: int) -> str:
        """Get input string at index."""
        ...

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        original_size = len(indices)

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            original_x = [eval_data.get(i) for i in indices]
            _, (x,) = _pad_eval_inputs(batch_unit, original_x)
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[: inference.block_size] for seq in encoded]
            xs, masks = inference.pad(encoded)
        else:
            original_x = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Keep a global view for intermediates.
        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )

        def evaluate_chunk(state: Any, xs_chunk: Any, masks_chunk: Any) -> Any:
            """Evaluate a chunk of batches."""

            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch = batch
                # Use inference's forward method - returns (logits, loss, meta)
                logits, _, _ = inference.forward(
                    state,
                    state.params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )
                # Take argmax to get predicted tokens
                predictions = jnp.argmax(logits[:, :-1, :], axis=-1)
                return None, predictions

            _, results = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
            results = jnp.reshape(results, (-1, results.shape[-1]))
            return results

        # JIT compile chunk function once
        data_sharding = NamedSharding(inference.mesh, data_pspec)
        wrapped_evaluate_chunk = jax.jit(
            evaluate_chunk,
            in_shardings=(inference.state_sharding, data_sharding, data_sharding),
            out_shardings=None,
        )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_results = []

        if jax.process_index() == 0:
            logger.info(
                "EVAL | {} | samples={} seq={} batches={}",
                eval_data.name,
                original_size,
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.info(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_results = wrapped_evaluate_chunk(
                inference.state, xs_chunk, masks_chunk
            )
            all_results.append(chunk_results)

        # Concatenate all chunk results
        results = jnp.concatenate(all_results, axis=0)

        # Collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten and decode
        results = jnp.reshape(results, (-1, results.shape[-1]))
        decoded_outputs = encoding.decode_batch(results.tolist()[:original_size])

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            assert original_x is not None
            score = eval_data._score(
                original_x,
                [eval_data.clean(i) for i in decoded_outputs],
                reduce=reduce,
            )
        else:
            score = (
                np.zeros(original_size, dtype=np.float32) if reduce == "none" else 0.0
            )
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        # No "action" notion for encoding evals — every real token is fair game.
        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [
            (xs_np[i], masks_np[i], masks_np[i]) for i in range(original_size)
        ]
        return score, intermediates


class PerplexityEvaluation(Evaluation):
    """Evaluation that computes dataset perplexity and returns ppl (lower is better).

    Runs a blockwise forward pass like EncodingEvaluation, computes the mean
    negative log-likelihood over all non-padding tokens, and returns perplexity.
    """

    @abstractmethod
    def get(self, indx: int) -> str:
        """Get input string at index."""
        ...

    def score(
        self, per_sample_nll: np.ndarray, per_sample_count: np.ndarray
    ) -> List[float]:
        """Per-sample perplexity (= exp(nll / max(count, 1))."""
        nll = np.asarray(per_sample_nll, dtype=np.float64)
        count = np.maximum(np.asarray(per_sample_count, dtype=np.float64), 1.0)
        return list(np.exp(nll / count).astype(np.float32))

    def _score(  # type: ignore[override]
        self,
        per_sample_nll: np.ndarray,
        per_sample_count: np.ndarray,
        reduce: str = "mean",
    ) -> Any:
        """Token-weighted aggregate ppl for mean/sum; per-sample ppl for none."""
        if reduce == "none":
            return np.asarray(
                self.score(per_sample_nll, per_sample_count), dtype=np.float32
            )
        nll = float(np.asarray(per_sample_nll, dtype=np.float64).sum())
        count = float(np.asarray(per_sample_count, dtype=np.float64).sum())
        return float(np.exp(nll / max(count, 1.0)))

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        original_size = len(indices)

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x = [eval_data.get(i) for i in indices]
            _, (x,) = _pad_eval_inputs(batch_unit, x)
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[: inference.block_size] for seq in encoded]
            xs, masks = inference.pad(encoded)
        else:
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )

        def evaluate_chunk(state: Any, xs_chunk: Any, masks_chunk: Any) -> Any:
            """Compute total NLL and token count for a chunk of batches."""

            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch = batch

                # Compute next-token targets: shift x left by 1
                y_batch = jnp.roll(x_batch, -1, axis=-1)
                y_batch = y_batch.at[:, -1].set(-1)  # last position has no target
                y_batch = jnp.where(mask_batch == 0, -1, y_batch)  # mask padding

                logits, _, _ = inference.forward(
                    state,
                    state.params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )

                logits_f32 = logits.astype(jnp.float32)
                log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

                token_mask = y_batch != -1
                y_safe = jnp.where(token_mask, y_batch, 0)

                token_nll = -jnp.take_along_axis(
                    log_probs, y_safe[..., None], axis=-1
                ).squeeze(-1)
                token_nll = jnp.where(token_mask, token_nll, 0.0)

                # Keep per-sample stats so padded examples can be dropped after gather.
                return None, jnp.stack(
                    [
                        jnp.sum(token_nll, axis=-1),
                        jnp.sum(token_mask, axis=-1).astype(jnp.float32),
                    ],
                    axis=-1,
                )

            _, stats = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
            # stats shape: (chunk_steps, batch_size, 2)
            return jnp.reshape(stats, (-1, 2))

        # JIT compile chunk function once
        data_sharding = NamedSharding(inference.mesh, data_pspec)
        wrapped_evaluate_chunk = jax.jit(
            evaluate_chunk,
            in_shardings=(inference.state_sharding, data_sharding, data_sharding),
            out_shardings=None,
        )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_stats = []

        if jax.process_index() == 0:
            logger.info(
                "EVAL | {} | samples={} seq={} batches={}",
                eval_data.name,
                original_size,
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.info(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_stats = wrapped_evaluate_chunk(inference.state, xs_chunk, masks_chunk)
            all_stats.append(chunk_stats)

        # Concatenate all chunk stats: shape (total_local_examples, 2)
        stats = jnp.concatenate(all_stats, axis=0)

        # Gather across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        stats = multihost_utils.process_allgather(stats)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten process dimension and drop repeated pad examples.
        stats = jnp.reshape(stats, (-1, 2))
        stats = stats[:original_size]
        stats_np = np.asarray(stats)
        per_sample_nll = stats_np[:, 0]
        per_sample_count = stats_np[:, 1]

        # _score is deterministic from broadcast inputs; identical on every host.
        score = eval_data._score(per_sample_nll, per_sample_count, reduce=reduce)
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [
            (xs_np[i], masks_np[i], masks_np[i]) for i in range(original_size)
        ]
        return score, intermediates


class PerplexityComparisonEvaluation(Evaluation):
    """Evaluation using perplexity comparison for multiple-choice tasks."""

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Get sample at index.

        Returns:
            (prefix, list_of_continuations, correct_index)
        """
        ...

    def score(self, correct_flags: List[float]) -> List[float]:
        """Per-sample correctness (1.0 / 0.0)."""
        return [float(c) for c in correct_flags]

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        n_samples = len(indices)

        # Gather all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            all_data = [eval_data.get(i) for i in indices]

            # Flatten: create (prefix+continuation, sample_idx, continuation_idx, prefix_len)
            flattened_inputs = []
            prefix_lengths = []
            metadata = []  # (sample_idx, continuation_idx, num_continuations)

            for sample_idx, (prefix, continuations, correct_idx) in enumerate(all_data):
                prefix_encoded = encoding.encode(prefix)
                prefix_len = len(prefix_encoded)

                for cont_idx, continuation in enumerate(continuations):
                    full_text = prefix + continuation
                    flattened_inputs.append(full_text)
                    prefix_lengths.append(prefix_len)
                    metadata.append((sample_idx, cont_idx, len(continuations)))

            original_flat_size = len(flattened_inputs)
            _, (flattened_inputs, prefix_lengths, metadata) = _pad_eval_inputs(
                batch_unit, flattened_inputs, prefix_lengths, metadata
            )

            # Encode all inputs
            encoded_inputs = encoding.encode_batch(
                flattened_inputs, allowed_special="all"
            )
            encoded_inputs = [seq[: inference.block_size] for seq in encoded_inputs]
            xs, masks = inference.pad(encoded_inputs)
            prefix_lengths_array = jnp.array(prefix_lengths, dtype=jnp.int32)
            metadata_array = jnp.array(metadata, dtype=jnp.int32)
            correct_indices_array = jnp.array([d[2] for d in all_data], dtype=jnp.int32)
            original_flat_size_array = jnp.array(original_flat_size, dtype=jnp.int32)
        else:
            xs, masks = None, None
            (
                prefix_lengths_array,
                metadata_array,
                correct_indices_array,
                original_flat_size_array,
            ) = (
                None,
                None,
                None,
                None,
            )

        # Broadcast to all hosts
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        prefix_lengths_array = multihost_utils.broadcast_one_to_all(
            prefix_lengths_array
        )
        metadata_array = multihost_utils.broadcast_one_to_all(metadata_array)
        correct_indices_array = multihost_utils.broadcast_one_to_all(
            correct_indices_array
        )
        original_flat_size_array = multihost_utils.broadcast_one_to_all(
            original_flat_size_array
        )
        multihost_utils.sync_global_devices("eval_gather_all:post")

        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        pieces_prefix_lens = jnp.array_split(
            prefix_lengths_array, jax.process_count(), axis=0
        )
        pieces_metadata = jnp.array_split(metadata_array, jax.process_count(), axis=0)

        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]
        prefix_lens_local = pieces_prefix_lens[jax.process_index()]
        metadata_local = pieces_metadata[jax.process_index()]

        # Reshape into (accumulate_steps, batch_size, T)
        batch_size = inference.local_replicas * inference.per_device_batch_size
        xs = xs.reshape(-1, batch_size, xs.shape[-1])
        masks = masks.reshape(-1, batch_size, masks.shape[-1])
        prefix_lens_local = prefix_lens_local.reshape(-1, batch_size)

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        prefix_lens_pspec = P(None, Axis.BATCH)  # type: ignore[no-untyped-call]

        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )
        prefix_lens_local = multihost_utils.host_local_array_to_global_array(
            prefix_lens_local, inference.mesh, prefix_lens_pspec
        )

        def evaluate_chunk(
            state: Any, xs_chunk: Any, masks_chunk: Any, prefix_lens_chunk: Any
        ) -> Any:
            """Compute per-sample loss only on continuation tokens for a chunk."""

            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch, prefix_len_batch = batch

                # Shift x to create y for next token prediction
                y_batch = jnp.roll(x_batch, -1, axis=-1)
                y_batch = y_batch.at[:, -1].set(0)

                # With LEFT padding, content starts after padding
                seq_len = x_batch.shape[-1]
                num_real_tokens = jnp.sum(
                    mask_batch.astype(jnp.int32), axis=-1, keepdims=True
                )
                content_start = seq_len - num_real_tokens

                seq_positions = jnp.arange(seq_len)[None, :]
                prefix_end = content_start + prefix_len_batch[:, None]
                is_padding_or_prefix = seq_positions < prefix_end
                y_batch = jnp.where(is_padding_or_prefix, -1, y_batch)

                # Use inference's forward method - returns (logits, loss, meta)
                logits, _, _ = inference.forward(
                    state,
                    state.params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )

                # Compute cross-entropy loss per token
                logits_f32 = logits.astype(jnp.float32)
                logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
                targets_flat = y_batch.reshape(-1)

                mask = targets_flat != -1
                targets_masked = jnp.where(mask, targets_flat, 0)

                log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
                token_losses = -jnp.take_along_axis(
                    log_probs, targets_masked[:, None], axis=-1
                ).squeeze(-1)
                token_losses = jnp.where(mask, token_losses, 0.0)

                # Average per sample
                token_losses = token_losses.reshape(x_batch.shape[0], x_batch.shape[1])
                mask = mask.reshape(x_batch.shape[0], x_batch.shape[1])
                per_sample_loss = jnp.sum(token_losses, axis=-1) / jnp.maximum(
                    jnp.sum(mask, axis=-1), 1.0
                )

                return None, per_sample_loss

            _, losses = jax.lax.scan(
                reduce, None, (xs_chunk, masks_chunk, prefix_lens_chunk)
            )
            losses = jnp.reshape(losses, (-1,))
            return losses

        # JIT compile chunk function once
        data_sharding = NamedSharding(inference.mesh, data_pspec)
        prefix_lens_sharding = NamedSharding(inference.mesh, prefix_lens_pspec)
        wrapped_evaluate_chunk = jax.jit(
            evaluate_chunk,
            in_shardings=(
                inference.state_sharding,
                data_sharding,
                data_sharding,
                prefix_lens_sharding,
            ),
            out_shardings=None,
        )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_losses = []

        if jax.process_index() == 0:
            logger.info(
                "EVAL | {} | samples={} flat={} seq={} batches={}",
                eval_data.name,
                n_samples,
                int(original_flat_size_array),
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]
            prefix_lens_chunk = prefix_lens_local[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.info(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_losses = wrapped_evaluate_chunk(
                inference.state, xs_chunk, masks_chunk, prefix_lens_chunk
            )
            all_losses.append(chunk_losses)

        # Concatenate all chunk results
        losses = jnp.concatenate(all_losses, axis=0)

        # Gather results across all hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        losses = multihost_utils.process_allgather(losses)
        metadata_gathered = multihost_utils.process_allgather(metadata_local)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten
        losses = jnp.reshape(losses, (-1,))
        metadata_gathered = jnp.reshape(metadata_gathered, (-1, 3))
        original_flat_size = int(jax.device_get(original_flat_size_array))
        losses = losses[:original_flat_size]
        metadata_gathered = metadata_gathered[:original_flat_size]

        # Convert to numpy
        losses = jax.device_get(losses)
        metadata_gathered = jax.device_get(metadata_gathered)
        correct_indices_array = jax.device_get(correct_indices_array)

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            # Group losses by sample_idx
            sample_losses = defaultdict(list)
            sample_num_continuations = {}

            for i, (sample_idx, cont_idx, num_conts) in enumerate(metadata_gathered):
                sample_idx = int(sample_idx)
                sample_losses[sample_idx].append(losses[i])
                sample_num_continuations[sample_idx] = int(num_conts)

            # Per-sample correctness
            correct_flags: List[float] = []
            for sample_idx in sorted(sample_losses.keys()):
                expected_conts = sample_num_continuations[sample_idx]
                actual_conts = len(sample_losses[sample_idx])

                if actual_conts != expected_conts:
                    continue

                losses_for_sample = jnp.array(sample_losses[sample_idx])
                pred = int(jnp.argmin(losses_for_sample))
                correct_idx = int(correct_indices_array[sample_idx])

                correct_flags.append(1.0 if pred == correct_idx else 0.0)

            score = eval_data._score(correct_flags, reduce=reduce)
        else:
            score = np.zeros(n_samples, dtype=np.float32) if reduce == "none" else 0.0
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [(xs_np[i], masks_np[i]) for i in range(original_flat_size)]
        return score, intermediates


@dataclass
class EvaluatorConfig:
    """Configuration for Evaluator."""

    components: List[str] = field("eval/evaluations")
    length: int = field("eval/length", default=-1)


@dataclass
class RLEvaluatorConfig:
    """Configuration for RL trainers — list of evaluation components used as
    rollout sources for on-policy learning."""

    components: List[str] = field("training/rl/components", default_factory=list)
    batch_size: float = field("training/batch_size", default=512)


class Evaluator(InferenceJob[EvaluatorConfig, M], Generic[M]):
    """InferenceJob that runs evaluations and saves results.

    Created from a trainer or checkpoint, holds a list of evaluations,
    and runs them when run() is called.

    Example:
        evaluator = Evaluator.from_trainer(trainer, evaluations, encoding)
        evaluator()  # Runs evaluations and saves results
    """

    MODEL: type[M] = Module  # type: ignore[assignment]
    evaluations: List[Evaluation]
    encoding: Tokenizer
    length: int
    random: random.Random

    @classmethod
    def config(cls) -> List[Any]:
        return [EvaluatorConfig, TokenizerConfig]

    def _get_results_path(self) -> Path:
        """Get path for saving evaluation results."""
        return self.spec.result_path("results.json")

    @property
    def done(self) -> bool:
        """Check if evaluation results already exist."""
        return self._get_results_path().exists()

    @classmethod
    def from_trainer(
        cls,
        trainer: "BaseTrainer[Any, Any]",
        config: Optional[Any] = None,
    ) -> "Evaluator[M]":
        """Create Evaluator from trainer.

        Args:
            trainer: BaseTrainer instance to get inference state from
            config: Optional config object whose ``.components`` field names
                the evaluations to run. If None, hydrates ``EvaluatorConfig``
                from the global config. Pass an ``RLEvaluatorConfig`` to
                build a separate evaluator for RL rollouts.

        Returns:
            Evaluator instance ready to run evaluations
        """
        from theseus.registry import EVALUATIONS

        evaluator = super().from_trainer(trainer)
        evaluator.encoding = get_tokenizer()
        evaluator.random = random.Random(0xC0FFEE)

        if config is None:
            config = configure(EvaluatorConfig)

        if isinstance(config, RLEvaluatorConfig):
            evaluator.length = int(config.batch_size)
        else:
            evaluator.length = config.length

        try:
            evaluator.evaluations = [EVALUATIONS[name]() for name in config.components]
        except KeyError as e:
            raise ValueError(f"Unknown evaluation dataset: {e.args[0]}") from e

        return evaluator

    @classmethod
    def from_checkpoint(
        cls,
        suffix: str | Path,
        spec: ExecutionSpec,
        runtime_cfg: Any | None = None,
    ) -> Tuple["Evaluator[M]", Any]:
        """Create Evaluator from checkpoint.

        Args:
            suffix: Checkpoint suffix
            spec: ExecutionSpec with topology
            runtime_cfg: Optional runtime config overlay

        Returns:
            (evaluator, config) tuple
        """
        evaluator, cfg = super().from_checkpoint(suffix, spec, runtime_cfg=runtime_cfg)
        with configuration(cfg):
            evaluator.encoding = get_tokenizer()
            evaluator.length = configure(EvaluatorConfig).length
        evaluator.random = random.Random(0xC0FFEE)
        return evaluator, cfg

    def rollout(
        self,
        inputs: List[Union[str, ChatTemplate]],
        encoding: Optional[Tokenizer] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
    ) -> List[Union[str, ChatTemplate]]:
        return super().rollout(
            inputs,
            encoding if encoding is not None else self.encoding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            chunk_size=chunk_size,
        )

    def evaluate(
        self,
        reduce: str = "mean",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run all evaluations.

        Args:
            reduce: passed through to each evaluation. "mean"/"sum" → float per
                evaluation; "none" → np.ndarray of per-sample scores.
            return_intermediates: when True, also return the per-evaluation list
                of (x, mask) rollouts (one inner list per evaluation).
            **kwargs: forwarded to each evaluation's __call__ (e.g. temperature,
                top_p, chunk_size).
        """
        results: dict[str, Any] = {}
        all_intermediates: List[List[Tuple[np.ndarray, np.ndarray]]] = []

        for evaluation in self.evaluations:
            logger.info("EVAL | Running {}", evaluation.name)
            if return_intermediates:
                score, intermediates = evaluation(
                    self,
                    self.encoding,
                    reduce=reduce,
                    return_intermediates=True,
                    **kwargs,
                )
                all_intermediates.append(intermediates)
            else:
                score = evaluation(
                    self,
                    self.encoding,
                    reduce=reduce,
                    return_intermediates=False,
                    **kwargs,
                )
            results[evaluation.name] = score
            logger.info("EVAL | {} done", evaluation.name)

        if return_intermediates:
            return results, all_intermediates
        return results

    def run(self) -> None:
        """Run all evaluations and save results to disk."""

        logger.info("EVAL | Starting evaluations for job {}", self.spec.name)
        results = self.evaluate()

        # Log summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info("  {}: {:.4f}", key, value)
        logger.info("=" * 60)

        # Save results to JSON (only on main process)
        with self.spec.result("results.json", main_process_only=True) as f:
            if f is not None:
                json.dump(
                    {
                        "job": self.spec.name,
                        "project": self.spec.project,
                        "group": self.spec.group,
                        "results": {k: float(v) for k, v in results.items()},
                    },
                    f,
                    indent=2,
                )
                logger.info("EVAL | Results saved to {}", Path(f.name))
