"""
Inference job for running inference on trained models.

Provides abstract base class InferenceJob that can be created from a trainer
or loaded from a checkpoint.
"""

from pathlib import Path
from typing import (
    Any,
    Tuple,
    Generic,
    TypeVar,
    Optional,
    List,
    Union,
    TYPE_CHECKING,
)
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from flax.training import train_state
import flax


from theseus.model.module import Module
from theseus.job import RestoreableJob
from theseus.base import Axis
from theseus.config import configure, current_config
from theseus.data.datasets.dataset import ChatTemplate
from theseus.data.tokenizer import Tokenizer, encode_chat_template, decode_chat_template

if TYPE_CHECKING:
    from theseus.training.base import BaseTrainer

C = TypeVar("C")
M = TypeVar("M", bound=Module)


class InferenceJob(RestoreableJob[C], Generic[C, M]):
    """Abstract base for inference jobs. Must be subclassed with custom forward().

    Subclasses define MODEL class attribute and forward() method.
    Users create instances via from_trainer() or from_checkpoint(), not __init__ directly.

    Example subclass:
        class GPTInference(InferenceJob):
            MODEL = GPT

            @staticmethod
            def forward(state, params, batch, key, deterministic):
                # Custom forward implementation
                ...

    Attributes (set by from_trainer/from_checkpoint):
        state: TrainState with params
        mesh: JAX device mesh
        state_sharding: NamedSharding
        replicas, local_replicas, per_device_batch_size, block_size
        key: PRNG key
    """

    MODEL: type[M]  # Subclasses must define this

    # Set by from_trainer/from_checkpoint/restore_from_path
    state: train_state.TrainState
    mesh: jax.sharding.Mesh
    state_sharding: NamedSharding
    replicas: int
    local_replicas: int
    per_device_batch_size: int
    block_size: int
    model: M

    @property
    def done(self) -> bool:
        """InferenceJob doesn't track completion state."""
        return False

    def run(self) -> None:
        raise NotImplementedError(
            "InferenceJob cannot be run - use for inference only."
        )

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: Any,
        batch: Tuple[jax.Array, Optional[jax.Array], jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        mutable: Optional[list[str] | tuple[str, ...]] = None,
        extra_variables: Optional[dict[str, Any]] = None,
        cache_max_len: Optional[int] = None,
    ) -> Any:
        """Forward pass with optional mutable variable collections (e.g. KV cache).

        Args:
            mutable: List of mutable variable collections (e.g. ['cache']).
                When provided, returns ((logits, loss), mutated_variables).
            extra_variables: Additional variable collections to pass alongside params
                (e.g. {'cache': cache_state} for decode steps).
            cache_max_len: Forwarded to model ``__call__`` so attention layers
                size their KV cache to actual decode need rather than the
                model's full ``block_size``. Only forwarded when not None,
                so models that don't accept it are unaffected.

        Returns:
            (logits, loss, meta) when mutable is None.
            ((logits, loss, meta), mutated_variables) when mutable is provided.
        """
        x, y, padding_mask = batch  # (B, T)

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)

        variables: dict[str, Any] = {"params": params}
        if extra_variables is not None:
            variables.update(extra_variables)

        kwargs: dict[str, Any] = {
            "padding_mask": padding_mask,
            "deterministic": deterministic,
        }
        if dropout_key is not None:
            kwargs["rngs"] = {"dropout": dropout_key}
        if cache_max_len is not None:
            kwargs["cache_max_len"] = cache_max_len

        if mutable is not None:
            (logits, loss), mutated = state.apply_fn(
                variables, x, y, mutable=mutable, **kwargs
            )
            return (logits, loss, {}), mutated
        else:
            logits, loss = state.apply_fn(variables, x, y, **kwargs)
            return logits, loss, {}

    @staticmethod
    def _init_template_params(model: M, block_size: int) -> Any:
        """Return a params pytree (with sharding) for checkpoint restoration."""
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, block_size), dtype=jnp.int32)
        return model.init(key, dummy_input)["params"]

    @classmethod
    def from_trainer(cls, trainer: "BaseTrainer[Any, Any]") -> Self:
        """Create InferenceJob sharing trainer's state.

        The InferenceJob references (not copies) trainer's state, so changes
        to trainer.state are reflected in the InferenceJob.
        """
        job = object.__new__(cls)
        job.spec = trainer.spec
        job.key = trainer.key

        # Share references to trainer's inference state
        job.state = trainer.state
        job.mesh = trainer.mesh
        job.state_sharding = trainer.state_sharding
        job.replicas = trainer.replicas
        job.local_replicas = trainer.local_replicas
        job.per_device_batch_size = trainer.per_device_batch_size
        job.block_size = trainer.args.block_size
        job.model = trainer.model

        return job

    def restore_from_path(self, rel_path: str | Path) -> None:
        """Restore inference state from ``rel_path`` under checkpoints_dir.

        Must be called within a ``configuration(cfg)`` context (as done by
        ``RestoreableJob.from_checkpoint_path``).  Initializes model, mesh,
        sharding, and loads checkpoint.
        """
        cfg = current_config()
        assert cfg is not None, (
            "restore_from_path must be called within a configuration() context"
        )

        # Initialize model from cls.MODEL (configure hydrates from OmegaConf)
        model = configure(self.MODEL)
        self.model = model

        # Get mesh from topology
        assert self.spec.topology is not None, (
            "Topology required for checkpoint loading"
        )
        self.mesh = self.spec.topology.mesh
        self.replicas = self.spec.topology.replicas
        self.local_replicas = self.spec.topology.local_replicas

        # Build a params-only template for checkpoint restoration.
        # We avoid building a full TrainState because the on-disk checkpoint
        # may have a different optimizer state tree (e.g. AdamW) that doesn't
        # match our inference-only template.  Restoring just params sidesteps
        # the mismatch entirely.
        template_params = self._init_template_params(
            model, int(cfg.architecture.block_size)
        )

        # Compute sharding from the params template
        import optax

        template_state = train_state.TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=template_params,
            tx=optax.identity(),
        )
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            flax.linen.get_partition_spec(template_state),
            self.mesh,
            rules=tuple(model.sharding),
        )

        # Restore only params from checkpoint (partial=True skips opt_state)
        params_tree = {"params": template_params}
        restored, metadata = self.get_tree_and_metadata_from_path(
            rel_path, params_tree, partial=True
        )

        # Reconstruct TrainState with restored params
        restored_dict: dict[str, Any] = restored  # type: ignore[assignment]
        state = template_state.replace(params=restored_dict["params"])
        self.state = jax.device_put(state, self.state_sharding)

        self.per_device_batch_size = cfg.training.per_device_batch_size
        self.block_size = cfg.architecture.block_size

    @staticmethod
    def pad(
        seqs: List[List[int]], pad_token: int = 0, pad_to: Optional[int] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """Left-pad sequences to uniform length.

        Args:
            seqs: List of token id lists
            pad_token: Token to use for padding (default 0)
            pad_to: Minimum length to pad to (default None, uses max seq length)

        Returns:
            padded: (batch_size, max_len) jnp array
            mask: (batch_size, max_len) jnp bool array, True for real tokens
        """
        max_len = max(len(s) for s in seqs)
        if pad_to is not None:
            max_len = max(pad_to, max_len)
        padded_seqs = [([pad_token] * (max_len - len(s))) + s for s in seqs]
        padded_masks = [
            ([False] * (max_len - len(s))) + [True for _ in s] for s in seqs
        ]
        return jnp.array(padded_seqs), jnp.array(padded_masks)

    def _autoregress(
        self,
        state: train_state.TrainState,
        key: jax.Array,
        input: jax.Array,
        input_mask: jax.Array,
        num_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jax.Array:
        """Autoregressive generation with KV cache.

        Uses KV cache for O(n) generation instead of O(n²).
        Step 1: Prefill — run full prompt, initialize cache.
        Step 2: Decode — one token at a time via jax.lax.scan.

        Args:
            state: Training state with params and apply_fn
            key: PRNG key
            input: Input token ids (B, T_in)
            input_mask: Attention mask (B, T_in)
            num_tokens: Total sequence length (prompt + generated)
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling threshold

        Returns:
            Generated sequences (B, num_tokens) containing prompt + generated ids
        """

        def top_p_filter_logits(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
            if top_p >= 1.0:
                return logits
            sort_idx = jnp.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumprobs = jnp.cumsum(sorted_probs, axis=-1)
            keep_sorted = cumprobs <= top_p
            keep_sorted = keep_sorted.at[:, 0].set(True)
            keep = jnp.zeros_like(logits, dtype=jnp.bool_)
            keep = keep.at[jnp.arange(logits.shape[0])[:, None], sort_idx].set(
                keep_sorted
            )
            return jnp.where(keep, logits, jnp.array(-jnp.inf, dtype=logits.dtype))

        def sample_token(
            logits: jnp.ndarray, key: jax.Array
        ) -> tuple[jnp.ndarray, jax.Array]:
            if temperature == 0.0:
                return jnp.argmax(logits, axis=-1).astype(jnp.int32), key
            key, subkey = jax_random.split(key)
            scaled = logits / temperature
            if top_p is not None:
                scaled = top_p_filter_logits(scaled, float(top_p))
            tok = jax.random.categorical(subkey, scaled, axis=-1).astype(jnp.int32)
            return tok, key

        B, T_in = input.shape
        forward_fn = self.forward

        # Jit forward so XLA/GSPMD distributes sharded params across devices
        # rather than all-gathering them onto a single device (which OOMs for
        # large vocab/embedding matrices). cache_max_len is static — it
        # determines KV-cache shape allocation inside attention.
        prefill_fn = jax.jit(
            forward_fn,
            static_argnames=("deterministic", "mutable", "cache_max_len"),
        )

        # Step 1: Prefill — initialize cache with full prompt. The cache is
        # sized to ``num_tokens`` (prompt + generated) so attention layers
        # don't allocate the full ``block_size`` they would otherwise reach
        # for; ``cache_max_len`` flows through Module.__call__ → block →
        # attention → _cached_kv. Models without that parameter get None.
        (prefill_logits, _, _), cache = prefill_fn(
            state,
            state.params,
            (input, None, input_mask),
            deterministic=True,
            mutable=("cache",),
            cache_max_len=num_tokens,
        )

        # Sample the first generated token from the last prompt position
        last_logits = prefill_logits[:, -1, :]
        first_token, key = sample_token(last_logits, key)

        # Build output buffer
        out_buf = jnp.zeros((B, num_tokens), dtype=jnp.int32)
        out_buf = out_buf.at[:, :T_in].set(input.astype(jnp.int32))
        out_buf = out_buf.at[:, T_in].set(first_token)

        n_gen = num_tokens - T_in - 1  # remaining tokens after the first generated one
        if n_gen <= 0:
            return out_buf

        # Step 2: Decode loop — one token at a time with cached KV.
        # state is passed explicitly so jit traces it as an input rather than
        # capturing the full params pytree as a 14GB constant.
        def _run_scan(
            state: Any, cache: Any, first_token: Any, out_buf: Any, key: Any
        ) -> Any:
            def decode_step(carry: Any, _step: Any) -> tuple[Any, None]:
                cache_state, last_tok, out, offset, key = carry
                token_input = last_tok[:, None]  # (B, 1)

                (logits, _, _), new_cache = forward_fn(
                    state,
                    state.params,
                    (token_input, None, None),  # type: ignore[arg-type]
                    deterministic=True,
                    mutable=("cache",),
                    extra_variables=cache_state,
                    cache_max_len=num_tokens,
                )

                next_logits = logits[:, -1, :]
                next_token, key = sample_token(next_logits, key)

                out = out.at[:, offset].set(next_token)
                return (new_cache, next_token, out, offset + 1, key), None

            carry = (cache, first_token, out_buf, T_in + 1, key)
            (_, _, final_out, _, _), _ = jax.lax.scan(
                decode_step, carry, jnp.arange(n_gen)
            )
            return final_out

        # Put non-sharded carry elements onto the mesh (replicated) so jit
        # can reconcile shardings with the cache that came out of prefill_fn.
        replicated = NamedSharding(self.mesh, P())  # type: ignore[no-untyped-call]
        first_token = jax.device_put(first_token, replicated)
        out_buf = jax.device_put(out_buf, replicated)
        key = jax.device_put(key, replicated)

        return jax.jit(  # type: ignore[no-any-return]
            _run_scan,
            in_shardings=(
                self.state_sharding,
                None,
                replicated,
                replicated,
                replicated,
            ),
        )(state, cache, first_token, out_buf, key)

    def rollout(
        self,
        inputs: List[Union[str, ChatTemplate]],
        encoding: Tokenizer,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
    ) -> List[Union[str, ChatTemplate]]:
        """Autoregressive rollout of the language model.

        Args:
            inputs: List of raw strings or ChatTemplates to complete.
            encoding: Tokenizer for encoding/decoding.
            max_new_tokens: Maximum number of new tokens to generate. Defaults to block_size.
            temperature: Sampling temperature (0.0 for greedy).
            top_p: Nucleus sampling threshold.
            chunk_size: Number of batches per JIT chunk.

        Returns:
            List of completed strings or ChatTemplates matching input types.
        """
        if max_new_tokens is None:
            max_new_tokens = self.block_size

        # Track which inputs are ChatTemplates for decoding back
        is_chat = [isinstance(inp, list) for inp in inputs]

        # Encode all inputs to strings first
        text_inputs: List[str] = []
        for inp in inputs:
            if isinstance(inp, list):
                # ChatTemplate: format with generation prompt, get text back
                text_inputs.append(
                    encode_chat_template(inp, encoding, prompt=True, tokenize=False)
                )
            else:
                text_inputs.append(inp)

        N = len(text_inputs)
        batch_unit = self.replicas * self.per_device_batch_size

        # Pad count up to a multiple of batch_unit so nothing is dropped
        padded_N = ((N + batch_unit - 1) // batch_unit) * batch_unit
        n_pad = padded_N - N
        if n_pad > 0:
            text_inputs = text_inputs + [text_inputs[-1]] * n_pad

        # Tokenize, left-pad, and build masks
        multihost_utils.sync_global_devices("rollout:pre")
        if jax.process_index() == 0:
            encoded = encoding.encode_batch(text_inputs, allowed_special="all")
            # Remember per-sequence token lengths for stripping left-pad later
            prompt_lengths = [len(seq) for seq in encoded]
            xs, masks = self.pad(encoded)
        else:
            prompt_lengths = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("rollout:post_broadcast")

        # Distribute across processes
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, local_batch, T)
        local_batch = self.local_replicas * self.per_device_batch_size
        xs = xs.reshape(-1, local_batch, xs.shape[-1])
        masks = masks.reshape(-1, local_batch, masks.shape[-1])

        # Create global arrays with sharding
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(xs, self.mesh, data_pspec)
        masks = multihost_utils.host_local_array_to_global_array(
            masks, self.mesh, data_pspec
        )

        # PRNG key
        self.key, key = jax.random.split(self.key)

        # Compute total tokens: prompt length + new tokens, capped by block_size
        max_prompt_length = int(jnp.max(jnp.sum(masks, axis=-1)))
        total_tokens = min(max_prompt_length + max_new_tokens, self.block_size)

        # Chunked autoregressive generation (same pattern as RolloutEvaluation)
        def evaluate_chunk(
            state: Any, xs_chunk: Any, masks_chunk: Any, key: Any
        ) -> Any:
            def reduce(_: Any, batch: Any) -> Any:
                x_batch, mask_batch = batch
                results = self._autoregress(
                    state,
                    key,
                    x_batch,
                    mask_batch,
                    total_tokens,
                    temperature,
                    top_p,
                )
                return None, results

            _, rollouts = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
            return jnp.reshape(rollouts, (-1, rollouts.shape[-1]))

        data_sharding = NamedSharding(self.mesh, data_pspec)
        jitted_chunk = jax.jit(
            evaluate_chunk,
            in_shardings=(self.state_sharding, data_sharding, data_sharding, None),
            out_shardings=None,
        )

        num_batches = xs.shape[0]
        all_results = []
        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            chunk_results = jitted_chunk(
                self.state,
                xs[chunk_start:chunk_end],
                masks[chunk_start:chunk_end],
                key,
            )
            all_results.append(chunk_results)

        results = jnp.concatenate(all_results, axis=0)

        # Gather across hosts
        multihost_utils.sync_global_devices("rollout:pre_gather")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("rollout:post_gather")

        # Flatten and strip left-pad tokens before decoding
        results = jnp.reshape(results, (-1, results.shape[-1]))
        results_list = results.tolist()

        if jax.process_index() == 0:
            assert prompt_lengths is not None
            # Input was left-padded to max_prompt_length; each row starts with
            # (max_prompt_length - prompt_len_i) pad tokens followed by prompt
            # and generated tokens.
            max_prompt_len = max(prompt_lengths)
            stripped = [
                row[max_prompt_len - prompt_lengths[i] :]
                for i, row in enumerate(results_list[:N])
            ]
        else:
            stripped = [row for row in results_list[:N]]

        decoded = encoding.decode_batch(stripped)

        # Convert back to original types
        outputs: List[Union[str, ChatTemplate]] = []
        for i, text in enumerate(decoded):
            if is_chat[i]:
                outputs.append(decode_chat_template(text, encoding))
            else:
                outputs.append(text)

        return outputs
