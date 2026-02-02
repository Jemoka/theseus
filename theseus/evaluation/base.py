"""
Evaluation framework for theseus trainers.

Provides abstract base classes for different evaluation types:
- RolloutEvaluation: Autoregressive generation tasks
- EncodingEvaluation: Next-token prediction accuracy
- PerplexityEvaluation: Multiple-choice via perplexity comparison
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from loguru import logger

from theseus.base import Axis


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

    def __call__(
        self,
        trainer: Any,
        encoding: Any,
        truncate: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> float:
        """Run evaluation.

        Args:
            trainer: BaseTrainer instance (or subclass)
            encoding: Tokenizer with encode_batch/decode_batch methods
            truncate: Whether to truncate dataset if batch size doesn't divide evenly
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling threshold

        Returns:
            Evaluation score
        """
        eval_data = self

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x, y = zip(*[eval_data.get(i) for i in range(len(eval_data))])
            xs, masks = trainer.pad(encoding.encode_batch(x))
        else:
            x, y = None, None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Find batch sizes
        if not truncate:
            per_device_batch_size, accumulate_steps = self.find_accumulation_steps(
                xs.shape[0], trainer.per_device_batch_size, trainer.replicas
            )
            if per_device_batch_size is None:
                truncate = True

        if truncate:
            valid_size = (
                xs.shape[0]
                // (trainer.replicas * trainer.per_device_batch_size)
                * (trainer.replicas * trainer.per_device_batch_size)
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
            -1, trainer.local_replicas * trainer.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, trainer.local_replicas * trainer.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, trainer.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, trainer.mesh, data_pspec
        )

        # Create subkey
        trainer.key, key = jax.random.split(trainer.key)

        def evaluate(state: Any, xs: Any, masks: Any, key: Any) -> Any:
            def reduce(_: Any, batch: Any) -> Any:
                results = trainer._autoregress(
                    state,
                    key,
                    batch[0],
                    batch[1],
                    trainer.args.block_size,
                    temperature,
                    top_p,
                    **kwargs,
                )
                return None, results

            _, rollouts = jax.lax.scan(reduce, None, (xs, masks))
            results = jnp.reshape(rollouts, (-1, rollouts.shape[-1]))
            return results

        # JIT compile
        data_sharding = NamedSharding(trainer.mesh, data_pspec)
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(trainer.state_sharding, data_sharding, data_sharding, None),
            out_shardings=None,
        )

        results = wrapped_evaluate(trainer.state, xs, masks, key)

        # Collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten and decode
        results = jnp.reshape(results, (-1, results.shape[-1]))
        decoded_results = encoding.decode_batch(results.tolist())

        # Score (y was gathered on process 0, broadcast to all)
        assert y is not None
        score = eval_data.score(list(y), [eval_data.clean(i) for i in decoded_results])
        return score


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

    def __call__(self, trainer: Any, encoding: Any, truncate: bool = False) -> float:
        """Run evaluation.

        Args:
            trainer: BaseTrainer instance (or subclass)
            encoding: Tokenizer with encode_batch/decode_batch methods
            truncate: Whether to truncate dataset if batch size doesn't divide evenly

        Returns:
            Evaluation score
        """
        eval_data = self

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x = [eval_data.get(i) for i in range(len(eval_data))]
            xs, masks = trainer.pad(encoding.encode_batch(x))
        else:
            x = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Find batch sizes
        if not truncate:
            per_device_batch_size, accumulate_steps = self.find_accumulation_steps(
                xs.shape[0], trainer.per_device_batch_size, trainer.replicas
            )
            if per_device_batch_size is None:
                truncate = True

        if truncate:
            valid_size = (
                xs.shape[0]
                // (trainer.replicas * trainer.per_device_batch_size)
                * (trainer.replicas * trainer.per_device_batch_size)
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
            -1, trainer.local_replicas * trainer.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, trainer.local_replicas * trainer.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, trainer.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, trainer.mesh, data_pspec
        )

        def evaluate(state: Any, xs: Any, masks: Any) -> Any:
            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch = batch
                # Use trainer's forward method - returns (logits, loss)
                logits, _ = trainer.forward(
                    state,
                    state.params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )
                # Take argmax to get predicted tokens
                predictions = jnp.argmax(logits[:, :-1, :], axis=-1)
                return None, predictions

            _, results = jax.lax.scan(reduce, None, (xs, masks))
            results = jnp.reshape(results, (-1, results.shape[-1]))
            return results

        # JIT compile
        data_sharding = NamedSharding(trainer.mesh, data_pspec)
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(trainer.state_sharding, data_sharding, data_sharding),
            out_shardings=None,
        )

        results = wrapped_evaluate(trainer.state, xs, masks)

        # Collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten and decode
        results = jnp.reshape(results, (-1, results.shape[-1]))
        decoded_outputs = encoding.decode_batch(results.tolist())

        # Score
        assert x is not None
        score = eval_data.score(x, [eval_data.clean(i) for i in decoded_outputs])
        return score


class PerplexityEvaluation(Evaluation):
    """Evaluation using perplexity comparison for multiple-choice tasks."""

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Get sample at index.

        Returns:
            (prefix, list_of_continuations, correct_index)
        """
        ...

    def __call__(self, trainer: Any, encoding: Any, truncate: bool = False) -> float:
        """Run evaluation.

        Args:
            trainer: BaseTrainer instance (or subclass)
            encoding: Tokenizer with encode/encode_batch methods
            truncate: Whether to truncate dataset if batch size doesn't divide evenly

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
            encoded_inputs = encoding.encode_batch(flattened_inputs)
            xs, masks = trainer.pad(encoded_inputs)
            prefix_lengths_array = jnp.array(prefix_lengths, dtype=jnp.int32)
            metadata_array = jnp.array(metadata, dtype=jnp.int32)
            correct_indices_array = jnp.array([d[2] for d in all_data], dtype=jnp.int32)
            # Fix: correct_indices should be the third element of each tuple
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

        # Find batch sizes
        if not truncate:
            per_device_batch_size, accumulate_steps = self.find_accumulation_steps(
                xs.shape[0], trainer.per_device_batch_size, trainer.replicas
            )
            if per_device_batch_size is None:
                truncate = True

        if truncate:
            valid_size = (
                xs.shape[0]
                // (trainer.replicas * trainer.per_device_batch_size)
                * (trainer.replicas * trainer.per_device_batch_size)
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
        batch_size = trainer.local_replicas * trainer.per_device_batch_size
        xs = xs.reshape(-1, batch_size, xs.shape[-1])
        masks = masks.reshape(-1, batch_size, masks.shape[-1])
        prefix_lens_local = prefix_lens_local.reshape(-1, batch_size)

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        prefix_lens_pspec = P(None, Axis.BATCH)  # type: ignore[no-untyped-call]

        xs = multihost_utils.host_local_array_to_global_array(
            xs, trainer.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, trainer.mesh, data_pspec
        )
        prefix_lens_local = multihost_utils.host_local_array_to_global_array(
            prefix_lens_local, trainer.mesh, prefix_lens_pspec
        )

        def evaluate(state: Any, xs: Any, masks: Any, prefix_lens: Any) -> Any:
            """Compute per-sample loss only on continuation tokens."""

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

                # Use trainer's forward method - returns (logits, loss)
                logits, _ = trainer.forward(
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

            _, losses = jax.lax.scan(reduce, None, (xs, masks, prefix_lens))
            losses = jnp.reshape(losses, (-1,))
            return losses

        # JIT compile
        data_sharding = NamedSharding(trainer.mesh, data_pspec)
        prefix_lens_sharding = NamedSharding(trainer.mesh, prefix_lens_pspec)
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(
                trainer.state_sharding,
                data_sharding,
                data_sharding,
                prefix_lens_sharding,
            ),
            out_shardings=None,
        )

        losses = wrapped_evaluate(trainer.state, xs, masks, prefix_lens_local)

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
        return float(accuracy)
