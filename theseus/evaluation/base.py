"""
Evaluation framework for theseus trainers.

Provides abstract base classes for different evaluation types:
- RolloutEvaluation: Autoregressive generation tasks
- EncodingEvaluation: Next-token prediction accuracy
- PerplexityEvaluation: Dataset perplexity (returns 1/ppl, higher is better)
- PerplexityComparisonEvaluation: Multiple-choice via perplexity comparison

Also provides:
- Evaluator: InferenceJob subclass that runs multiple evaluations
"""

import json
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional, Union, Generic, TYPE_CHECKING

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
        self, inference: "InferenceJob[Any, M]", encoding: Any, **kwargs: Any
    ) -> float:
        """Run the evaluation and return a score."""
        ...

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

    def score(self, ys: list[str], y_hats: list[str]) -> float:
        """Compute score from generated results.

        Args:
            ys: Ground truth strings
            y_hats: Generated results

        Returns:
            Score (higher is better)
        """
        results = [self.check(y, y_hat) for y, y_hat in zip(ys, y_hats)]
        return sum(results) / len(results)

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
        """Maximum tokens to generate. Override in subclasses for shorter rollouts.

        Default is full block_size, but most evaluations only need ~10-100 tokens.
        """
        return inference.block_size

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> float:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode_batch/decode_batch methods
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling threshold
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            Evaluation score
        """
        eval_data = self

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x, y = zip(*[eval_data.get(i) for i in range(len(eval_data))])
            xs, masks = inference.pad(encoding.encode_batch(x, allowed_special="all"))
        else:
            x, y = None, None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Truncate to valid batch size
        valid_size = (
            xs.shape[0]
            // (inference.replicas * inference.per_device_batch_size)
            * (inference.replicas * inference.per_device_batch_size)
        )
        xs = xs[:valid_size]
        masks = masks[:valid_size]

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

        # Calculate total tokens needed: max prompt length + max_new_tokens
        max_new_tokens = eval_data.max_new_tokens(inference)
        max_prompt_length = int(jnp.max(jnp.sum(masks, axis=-1)))
        total_tokens = min(max_prompt_length + max_new_tokens, inference.block_size)

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

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            # Log progress (only on main process)
            if jax.process_index() == 0:
                progress = (chunk_end / num_batches) * 100
                logger.info(
                    f"EVAL | {eval_data.name} - Processing batches {chunk_start + 1}-{chunk_end}/{num_batches} ({progress:.1f}%)"
                )

            # Run chunk
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

        # Flatten and decode
        results = jnp.reshape(results, (-1, results.shape[-1]))
        decoded_results = encoding.decode_batch(results.tolist())

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            assert y is not None
            score = eval_data.score(
                list(y), [eval_data.clean(i) for i in decoded_results]
            )
        else:
            score = None
        score = multihost_utils.broadcast_one_to_all(jnp.array(score))
        return float(score)


class EncodingEvaluation(Evaluation):
    """Evaluation using next-token prediction accuracy."""

    def score(self, xs: list[str], y_hats: list[str]) -> float:
        """Compute score from input and model predictions.

        Args:
            xs: Input strings
            y_hats: Model predictions (argmax of logits, shifted by 1)

        Returns:
            Score (higher is better)
        """
        results = [self.check(x, y_hat) for x, y_hat in zip(xs, y_hats)]
        return sum(results) / len(results)

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
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> float:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode_batch/decode_batch methods
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            Evaluation score
        """
        eval_data = self

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x = [eval_data.get(i) for i in range(len(eval_data))]
            xs, masks = inference.pad(encoding.encode_batch(x, allowed_special="all"))
        else:
            x = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Truncate to valid batch size
        valid_size = (
            xs.shape[0]
            // (inference.replicas * inference.per_device_batch_size)
            * (inference.replicas * inference.per_device_batch_size)
        )
        xs = xs[:valid_size]
        masks = masks[:valid_size]

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

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            # Log progress (only on main process)
            if jax.process_index() == 0:
                progress = (chunk_end / num_batches) * 100
                logger.info(
                    f"EVAL | {eval_data.name} - Processing batches {chunk_start + 1}-{chunk_end}/{num_batches} ({progress:.1f}%)"
                )

            # Run chunk
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
        decoded_outputs = encoding.decode_batch(results.tolist())

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            assert x is not None
            score = eval_data.score(x, [eval_data.clean(i) for i in decoded_outputs])
        else:
            score = None
        score = multihost_utils.broadcast_one_to_all(jnp.array(score))
        return float(score)


class PerplexityEvaluation(Evaluation):
    """Evaluation that computes dataset perplexity and returns 1/ppl (higher is better).

    Runs a blockwise forward pass like EncodingEvaluation, computes the mean
    negative log-likelihood over all non-padding tokens, and returns 1/perplexity.
    """

    @abstractmethod
    def get(self, indx: int) -> str:
        """Get input string at index."""
        ...

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> float:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode_batch methods
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            1/perplexity (higher is better)
        """
        eval_data = self

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x = [eval_data.get(i) for i in range(len(eval_data))]
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[: inference.block_size] for seq in encoded]
            xs, masks = inference.pad(encoded)
        else:
            x = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Truncate to valid batch size
        valid_size = (
            xs.shape[0]
            // (inference.replicas * inference.per_device_batch_size)
            * (inference.replicas * inference.per_device_batch_size)
        )
        xs = xs[:valid_size]
        masks = masks[:valid_size]

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

                # Return (sum_nll, token_count) as a 2-element array
                return None, jnp.array(
                    [jnp.sum(token_nll), jnp.sum(token_mask).astype(jnp.float32)]
                )

            _, stats = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
            # stats shape: (chunk_steps, 2)
            return stats

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

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if jax.process_index() == 0:
                progress = (chunk_end / num_batches) * 100
                logger.info(
                    f"EVAL | {eval_data.name} - Processing batches {chunk_start + 1}-{chunk_end}/{num_batches} ({progress:.1f}%)"
                )

            chunk_stats = wrapped_evaluate_chunk(inference.state, xs_chunk, masks_chunk)
            all_stats.append(chunk_stats)

        # Concatenate all chunk stats: shape (total_local_batches, 2)
        stats = jnp.concatenate(all_stats, axis=0)

        # Gather across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        stats = multihost_utils.process_allgather(stats)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten process dimension (process_allgather prepends a process axis)
        stats = jnp.reshape(stats, (-1, 2))

        # Compute global perplexity and return 1/ppl
        total_nll = jnp.sum(stats[:, 0])
        total_count = jnp.sum(stats[:, 1])
        mean_nll = total_nll / jnp.maximum(total_count, 1.0)
        ppl = jnp.exp(mean_nll)
        score = multihost_utils.broadcast_one_to_all(1.0 / ppl)
        return float(score)


class PerplexityComparisonEvaluation(Evaluation):
    """Evaluation using perplexity comparison for multiple-choice tasks."""

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Get sample at index.

        Returns:
            (prefix, list_of_continuations, correct_index)
        """
        ...

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> float:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode/encode_batch methods
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            Accuracy score
        """
        eval_data = self

        # Gather all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            all_data = [eval_data.get(i) for i in range(len(eval_data))]

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

            # Encode all inputs
            encoded_inputs = encoding.encode_batch(
                flattened_inputs, allowed_special="all"
            )
            xs, masks = inference.pad(encoded_inputs)
            prefix_lengths_array = jnp.array(prefix_lengths, dtype=jnp.int32)
            metadata_array = jnp.array(metadata, dtype=jnp.int32)
            correct_indices_array = jnp.array([d[2] for d in all_data], dtype=jnp.int32)
        else:
            xs, masks = None, None
            prefix_lengths_array, metadata_array, correct_indices_array = (
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
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Truncate to valid batch size
        valid_size = (
            xs.shape[0]
            // (inference.replicas * inference.per_device_batch_size)
            * (inference.replicas * inference.per_device_batch_size)
        )
        xs = xs[:valid_size]
        masks = masks[:valid_size]
        prefix_lengths_array = prefix_lengths_array[:valid_size]
        metadata_array = metadata_array[:valid_size]

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

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]
            prefix_lens_chunk = prefix_lens_local[chunk_start:chunk_end]

            # Log progress (only on main process)
            if jax.process_index() == 0:
                progress = (chunk_end / num_batches) * 100
                logger.info(
                    f"EVAL | {eval_data.name} - Processing batches {chunk_start + 1}-{chunk_end}/{num_batches} ({progress:.1f}%)"
                )

            # Run chunk
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

            # Evaluate complete samples only
            correct = 0
            total = 0

            for sample_idx in sorted(sample_losses.keys()):
                expected_conts = sample_num_continuations[sample_idx]
                actual_conts = len(sample_losses[sample_idx])

                if actual_conts != expected_conts:
                    continue

                losses_for_sample = jnp.array(sample_losses[sample_idx])
                pred = int(jnp.argmin(losses_for_sample))
                correct_idx = int(correct_indices_array[sample_idx])

                correct += int(pred == correct_idx)
                total += 1

            accuracy = correct / total if total > 0 else 0.0
        else:
            accuracy = None

        accuracy = multihost_utils.broadcast_one_to_all(jnp.array(accuracy))
        return float(accuracy)


@dataclass
class EvaluatorConfig:
    """Configuration for Evaluator."""

    evaluations: List[str] = field("eval/evaluations")


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

    @classmethod
    def config(cls) -> List[Any]:
        return [EvaluatorConfig, TokenizerConfig]

    def __init__(self, spec: ExecutionSpec):
        """Direct __init__ not supported - use from_trainer() or from_checkpoint()."""
        raise NotImplementedError(
            f"Cannot instantiate {self.__class__.__name__} directly. "
            "Use from_trainer() or from_checkpoint() instead."
        )

    def _get_results_path(self) -> Path:
        """Get path for saving evaluation results."""
        results_dir = self.spec.hardware.hosts[0].cluster.results_dir
        project = self.spec.project or "general"
        group = self.spec.group if self.spec.group else "default"
        return Path(results_dir) / project / group / self.spec.name / "results.json"

    @property
    def done(self) -> bool:
        """Check if evaluation results already exist."""
        return self._get_results_path().exists()

    @classmethod
    def from_trainer(cls, trainer: "BaseTrainer[Any, Any]") -> "Evaluator[M]":
        """Create Evaluator from trainer.

        Args:
            trainer: BaseTrainer instance to get inference state from
            evaluations: List of Evaluation instances to run
            encoding: Tokenizer with encode/decode methods

        Returns:
            Evaluator instance ready to run evaluations
        """
        from theseus.evaluation.datasets.registry import DATASETS

        evaluator = super().from_trainer(trainer)
        evaluator.encoding = get_tokenizer()

        cfg = configure(EvaluatorConfig)
        try:
            evaluator.evaluations = [DATASETS[name]() for name in cfg.evaluations]
        except KeyError as e:
            raise ValueError(f"Unknown evaluation dataset: {e.args[0]}") from e

        return evaluator

    @classmethod
    def from_checkpoint(
        cls, suffix: str | Path, spec: ExecutionSpec
    ) -> Tuple["Evaluator[M]", Any]:
        """Create Evaluator from checkpoint.

        Args:
            suffix: Checkpoint suffix
            spec: ExecutionSpec with topology
            evaluations: List of Evaluation instances to run
            encoding: Tokenizer with encode/decode methods

        Returns:
            (evaluator, config) tuple
        """
        evaluator, cfg = super().from_checkpoint(suffix, spec)
        with configuration(cfg):
            evaluator.encoding = get_tokenizer()
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

    def evaluate(self) -> dict[str, float]:
        results: dict[str, float] = {}

        for evaluation in self.evaluations:
            logger.info("EVAL | Running {}", evaluation.name)
            score = evaluation(self, self.encoding)
            results[evaluation.name] = score
            logger.info("EVAL | {} = {:.4f}", evaluation.name, score)

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
        if jax.process_index() == 0:
            output_path = self._get_results_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
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
            logger.info("EVAL | Results saved to {}", output_path)
