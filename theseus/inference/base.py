"""
Inference job for running inference on trained models.

Provides abstract base class InferenceJob that can be created from a trainer
or loaded from a checkpoint.
"""

from pathlib import Path
import time
from typing import (
    Any,
    Tuple,
    Generic,
    Literal,
    TypeVar,
    Optional,
    List,
    Union,
    TYPE_CHECKING,
    cast,
)
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from flax.training import train_state
import flax
from loguru import logger


from theseus.model.module import Module
from theseus.job import RestoreableJob
from theseus.base import Axis
from theseus.config import configure, current_config
from theseus.data.datasets.dataset import ChatTemplate, ChatTurn
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
        logger.debug(
            "INFERENCE | init template params dummy_input={}", dummy_input.shape
        )
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

        logger.debug(
            "INFERENCE | from_trainer replicas={} local_replicas={} per_device_batch_size={} block_size={}",
            job.replicas,
            job.local_replicas,
            job.per_device_batch_size,
            job.block_size,
        )
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
        logger.debug("INFERENCE | restore_from_path path={}", rel_path)

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
        logger.debug(
            "INFERENCE | restore topology replicas={} local_replicas={} block_size={}",
            self.replicas,
            self.local_replicas,
            cfg.architecture.block_size,
        )

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
        logger.debug("INFERENCE | restored checkpoint metadata={}", metadata)

        # Reconstruct TrainState with restored params
        restored_dict: dict[str, Any] = restored  # type: ignore[assignment]
        state = template_state.replace(params=restored_dict["params"])
        self.state = jax.device_put(state, self.state_sharding)

        self.per_device_batch_size = cfg.training.per_device_batch_size
        self.block_size = cfg.architecture.block_size
        logger.debug(
            "INFERENCE | restore done per_device_batch_size={} block_size={}",
            self.per_device_batch_size,
            self.block_size,
        )

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
        padded = jnp.array(padded_seqs)
        masks = jnp.array(padded_masks)
        logger.debug(
            "INFERENCE | pad seqs={} max_len={} padded={} masks={}",
            len(seqs),
            max_len,
            padded.shape,
            masks.shape,
        )
        return padded, masks

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

        # Step 1: Prefill — initialize cache with full prompt. The cache is
        # sized to ``num_tokens`` (prompt + generated) so attention layers
        # don't allocate the full ``block_size`` they would otherwise reach
        # for; ``cache_max_len`` flows through Module.__call__ → block →
        # attention → _cached_kv. Models without that parameter get None.
        (prefill_logits, _, _), cache = forward_fn(
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
            def decode_step(carry: Any, step: Any) -> tuple[Any, None]:
                cache_state, last_tok, out, offset, key = carry
                token_input = last_tok[:, None]  # (B, 1)
                del step

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

        return cast(jax.Array, _run_scan(state, cache, first_token, out_buf, key))

    def rollout(
        self,
        inputs: List[Union[str, ChatTemplate, jax.Array, List[int]]],
        encoding: Optional[Tokenizer] = None,
        max_new_tokens: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
        return_type: Literal[
            "decoded",
            "indices",
            "output_decoded",
            "output_indices",
            "raw_indices",
        ] = "decoded",
    ) -> Union[List[Union[str, ChatTemplate]], List[str], List[List[int]]]:
        """Autoregressive rollout of the language model.


        Args:
            inputs: List of raw strings, ChatTemplates, or pre-tokenized 1D
                jax arrays of token ids.
            encoding: Tokenizer for encoding/decoding. Required when any input
                is a string/ChatTemplate or when ``return_type`` is one of the
                decoded variants. May be omitted when all inputs are
                pre-tokenized arrays AND ``return_type`` is an indices variant.
            max_new_tokens: Maximum number of new tokens to generate. Defaults
                to ``block_size - max_prompt_length`` (or ``block_size // 2``
                if ``max_prompt_length`` is also unset).
            max_prompt_length: Length to which prompts are padded. Defaults to
                ``block_size - max_new_tokens``. Inputs longer than this raise.
                Holding this fixed across calls keeps the JIT trace stable —
                varying it triggers recompiles.
            temperature: Sampling temperature (0.0 for greedy).
            top_p: Nucleus sampling threshold.
            chunk_size: Number of batches per JIT chunk.


        return_type:
            - "decoded": full prompt + generated tokens, decoded, with left-pad stripped.
            - "indices": full prompt + generated tokens as ids, with left-pad stripped.
            - "output_decoded": generated portion only, decoded.
            - "output_indices": generated portion only as ids.
            - "raw_indices": full fixed-shape rows, left padding preserved.
            Shape is logically (N, max_prompt_length + max_new_tokens).
        """
        logger.debug(
            "INFERENCE | rollout start inputs={} max_new_tokens={} max_prompt_length={} temperature={} top_p={} chunk_size={} return_type={}",
            len(inputs),
            max_new_tokens,
            max_prompt_length,
            temperature,
            top_p,
            chunk_size,
            return_type,
        )
        valid_modes = (
            "decoded",
            "indices",
            "output_decoded",
            "output_indices",
            "raw_indices",
        )
        if return_type not in valid_modes:
            raise ValueError(
                f"return_type must be one of {valid_modes}, got {return_type!r}"
            )

        needs_decoding = return_type in ("decoded", "output_decoded")
        has_text_input = any(
            isinstance(inp, str)
            or (
                isinstance(inp, list)
                and all(isinstance(turn, ChatTurn) for turn in inp)
            )
            for inp in inputs
        )
        if encoding is None and (needs_decoding or has_text_input):
            raise ValueError(
                "encoding is required when return_type is a decoded variant "
                "or any input is a string/ChatTemplate"
            )

        if not inputs:
            logger.debug("INFERENCE | rollout empty inputs")
            return []

        # Resolve static length budget.
        if max_new_tokens is None and max_prompt_length is None:
            max_new_tokens = self.block_size // 2
            max_prompt_length = self.block_size - max_new_tokens
        elif max_new_tokens is None:
            assert max_prompt_length is not None
            max_new_tokens = self.block_size - max_prompt_length
        elif max_prompt_length is None:
            max_prompt_length = self.block_size - max_new_tokens

        if max_new_tokens <= 0 or max_prompt_length <= 0:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) and max_prompt_length "
                f"({max_prompt_length}) must both be positive"
            )
        if max_new_tokens + max_prompt_length > self.block_size:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) + max_prompt_length "
                f"({max_prompt_length}) exceeds block_size ({self.block_size})"
            )
        logger.debug(
            "INFERENCE | rollout lengths max_prompt_length={} max_new_tokens={} block_size={}",
            max_prompt_length,
            max_new_tokens,
            self.block_size,
        )

        is_chat = [
            isinstance(inp, list) and all(isinstance(turn, ChatTurn) for turn in inp)
            for inp in inputs
        ]

        N = len(inputs)
        batch_unit = self.replicas * self.per_device_batch_size

        padded_N = ((N + batch_unit - 1) // batch_unit) * batch_unit
        n_pad = padded_N - N
        if n_pad > 0:
            inputs = inputs + [inputs[-1]] * n_pad
        logger.debug(
            "INFERENCE | rollout batch N={} padded_N={} n_pad={} batch_unit={}",
            N,
            padded_N,
            n_pad,
            batch_unit,
        )

        # Tokenize, left-pad, and build masks.
        multihost_utils.sync_global_devices("rollout:pre")
        if jax.process_index() == 0:
            encoded: List[List[int]] = [[] for _ in inputs]
            str_buf: List[str] = []
            str_idx: List[int] = []

            for i, inp in enumerate(inputs):
                if isinstance(inp, jax.Array):
                    encoded[i] = [int(x) for x in inp.tolist()]
                elif isinstance(inp, str):
                    str_buf.append(inp)
                    str_idx.append(i)
                elif isinstance(inp, list) and all(
                    isinstance(turn, ChatTurn) for turn in inp
                ):
                    assert encoding is not None
                    chat_inp = cast(ChatTemplate, inp)
                    str_buf.append(
                        encode_chat_template(
                            chat_inp, encoding, prompt=True, tokenize=False
                        )
                    )
                    str_idx.append(i)
                elif isinstance(inp, list):
                    token_inp = cast(List[int], inp)
                    encoded[i] = [int(x) for x in token_inp]
                elif hasattr(inp, "tolist"):
                    encoded[i] = [int(x) for x in inp.tolist()]
                else:
                    raise TypeError(f"unsupported rollout input type: {type(inp)!r}")

            if str_buf:
                assert encoding is not None
                batch_encoded = encoding.encode_batch(str_buf, allowed_special="all")
                for i, ids in zip(str_idx, batch_encoded):
                    encoded[i] = ids

            longest = max(len(seq) for seq in encoded)
            if longest > max_prompt_length:
                raise ValueError(
                    f"input length {longest} exceeds max_prompt_length="
                    f"{max_prompt_length}; truncate prompts before calling rollout"
                )

            prompt_lengths = [len(seq) for seq in encoded]
            xs, masks = self.pad(encoded, pad_to=max_prompt_length)
            logger.debug(
                "INFERENCE | rollout encoded longest={} xs={} masks={}",
                longest,
                xs.shape,
                masks.shape,
            )
        else:
            prompt_lengths = None
            xs, masks = None, None

        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("rollout:post_broadcast")
        logger.debug(
            "INFERENCE | rollout broadcast xs={} masks={}",
            xs.shape,
            masks.shape,
        )

        # Distribute across processes.
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]
        logger.debug(
            "INFERENCE | rollout local split process={}/{} xs={} masks={}",
            jax.process_index(),
            jax.process_count(),
            xs.shape,
            masks.shape,
        )

        local_batch = self.local_replicas * self.per_device_batch_size
        xs = xs.reshape(-1, local_batch, xs.shape[-1])
        masks = masks.reshape(-1, local_batch, masks.shape[-1])
        logger.debug(
            "INFERENCE | rollout reshaped xs={} masks={} local_batch={}",
            xs.shape,
            masks.shape,
            local_batch,
        )

        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(xs, self.mesh, data_pspec)
        masks = multihost_utils.host_local_array_to_global_array(
            masks, self.mesh, data_pspec
        )
        logger.debug(
            "INFERENCE | rollout global xs={} masks={} xs_sharding={} masks_sharding={}",
            xs.shape,
            masks.shape,
            xs.sharding,
            masks.sharding,
        )

        self.key, key = jax.random.split(self.key)
        total_tokens = max_prompt_length + max_new_tokens
        logger.debug("INFERENCE | rollout total_tokens={}", total_tokens)

        def evaluate_chunk(
            state: Any,
            xs_chunk: Any,
            masks_chunk: Any,
            key: Any,
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
            logger.debug(
                "INFERENCE | rollout chunk {}:{} xs_chunk={} masks_chunk={}",
                chunk_start,
                chunk_end,
                xs[chunk_start:chunk_end].shape,
                masks[chunk_start:chunk_end].shape,
            )
            chunk_t0 = time.perf_counter()
            chunk_results = jitted_chunk(
                self.state,
                xs[chunk_start:chunk_end],
                masks[chunk_start:chunk_end],
                key,
            )
            logger.debug(
                "INFERENCE | rollout chunk dispatched {}:{} results={} dispatch_s={:.3f}",
                chunk_start,
                chunk_end,
                chunk_results.shape,
                time.perf_counter() - chunk_t0,
            )
            logger.debug("INFERENCE | rollout chunk block_until_ready start")
            chunk_results = jax.block_until_ready(chunk_results)  # type: ignore[no-untyped-call]
            logger.debug(
                "INFERENCE | rollout chunk ready {}:{} results={} total_s={:.3f}",
                chunk_start,
                chunk_end,
                chunk_results.shape,
                time.perf_counter() - chunk_t0,
            )
            all_results.append(chunk_results)

        concat_t0 = time.perf_counter()
        results = jnp.concatenate(all_results, axis=0)
        logger.debug(
            "INFERENCE | rollout concat dispatched results={} dispatch_s={:.3f}",
            results.shape,
            time.perf_counter() - concat_t0,
        )
        logger.debug("INFERENCE | rollout concat block_until_ready start")
        results = jax.block_until_ready(results)  # type: ignore[no-untyped-call]
        logger.debug(
            "INFERENCE | rollout concat ready results={} total_s={:.3f}",
            results.shape,
            time.perf_counter() - concat_t0,
        )

        sync_t0 = time.perf_counter()
        logger.debug("INFERENCE | rollout pre_gather sync start")
        multihost_utils.sync_global_devices("rollout:pre_gather")
        logger.debug(
            "INFERENCE | rollout pre_gather sync done s={:.3f}",
            time.perf_counter() - sync_t0,
        )
        gather_t0 = time.perf_counter()
        logger.debug(
            "INFERENCE | rollout process_allgather start results={}", results.shape
        )
        results = multihost_utils.process_allgather(results)
        logger.debug(
            "INFERENCE | rollout process_allgather done results={} s={:.3f}",
            results.shape,
            time.perf_counter() - gather_t0,
        )
        sync_t0 = time.perf_counter()
        logger.debug("INFERENCE | rollout post_gather sync start")
        multihost_utils.sync_global_devices("rollout:post_gather")
        logger.debug(
            "INFERENCE | rollout post_gather sync done gathered={} s={:.3f}",
            results.shape,
            time.perf_counter() - sync_t0,
        )

        results = jnp.reshape(results, (-1, results.shape[-1]))
        results_list = results.tolist()
        logger.debug(
            "INFERENCE | rollout flattened results={} rows={}",
            results.shape,
            len(results_list),
        )

        # Keep fixed-shape rows and preserve left-padding. This is what eval wants
        # for action_mask / padding_mask construction.
        if return_type == "raw_indices":
            logger.debug("INFERENCE | rollout return raw_indices rows={}", N)
            return results_list[:N]

        output_only = return_type in ("output_decoded", "output_indices")

        if jax.process_index() == 0:
            assert prompt_lengths is not None

            if output_only:
                out_rows = [row[max_prompt_length:] for row in results_list[:N]]
            else:
                out_rows = [
                    row[max_prompt_length - prompt_lengths[i] :]
                    for i, row in enumerate(results_list[:N])
                ]
        else:
            out_rows = [row for row in results_list[:N]]

        if return_type in ("indices", "output_indices"):
            logger.debug("INFERENCE | rollout return indices rows={}", len(out_rows))
            return out_rows

        assert encoding is not None
        decoded = encoding.decode_batch(out_rows)
        logger.debug("INFERENCE | rollout decoded rows={}", len(decoded))

        if return_type == "output_decoded":
            logger.debug(
                "INFERENCE | rollout return output_decoded rows={}", len(decoded)
            )
            return decoded

        outputs: List[Union[str, ChatTemplate]] = []
        for i, text in enumerate(decoded):
            if is_chat[i]:
                outputs.append(decode_chat_template(text, encoding))
            else:
                outputs.append(text)

        logger.debug("INFERENCE | rollout return decoded rows={}", len(outputs))
        return outputs
