"""
Inference job for running inference on trained models.

Provides abstract base class InferenceJob that can be created from a trainer
or loaded from a checkpoint.
"""

from pathlib import Path
from typing import Any, Tuple, Generic, TypeVar, Optional, List, TYPE_CHECKING, cast
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.sharding import NamedSharding, PartitionSpec as P
from flax.training import train_state
import flax

from omegaconf import OmegaConf
from loguru import logger

from theseus.model.module import Module
from theseus.job import CheckpointedJob
from theseus.base import ExecutionSpec
from theseus.config import configure, configuration

if TYPE_CHECKING:
    from theseus.training.trainers.base import BaseTrainer

C = TypeVar("C")
M = TypeVar("M", bound=Module)


class InferenceJob(CheckpointedJob[C], Generic[C, M]):
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

    # Set by from_trainer/from_checkpoint
    state: train_state.TrainState
    mesh: jax.sharding.Mesh
    state_sharding: NamedSharding
    replicas: int
    local_replicas: int
    per_device_batch_size: int
    block_size: int
    key: jax.Array

    def __init__(self, spec: ExecutionSpec):
        """Direct __init__ not supported - use from_trainer() or from_checkpoint()."""
        raise NotImplementedError(
            f"Cannot instantiate {self.__class__.__name__} directly. "
            "Use from_trainer() or from_checkpoint() instead."
        )

    def _init_from_spec(self, spec: ExecutionSpec) -> None:
        """Internal init called by from_trainer/from_checkpoint."""
        # Call grandparent's __init__ to set up basic attributes
        # We skip CheckpointedJob.__init__ since we handle key ourselves
        self.spec = spec
        # key will be set by from_trainer or from_checkpoint

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
    ) -> Any:
        """Forward pass with optional mutable variable collections (e.g. KV cache).

        Args:
            mutable: List of mutable variable collections (e.g. ['cache']).
                When provided, returns ((logits, loss), mutated_variables).
            extra_variables: Additional variable collections to pass alongside params
                (e.g. {'cache': cache_state} for decode steps).

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

        if mutable is not None:
            (logits, loss), mutated = state.apply_fn(
                variables, x, y, mutable=mutable, **kwargs
            )
            return (logits, loss, {}), mutated
        else:
            logits, loss = state.apply_fn(variables, x, y, **kwargs)
            return logits, loss, {}

    @staticmethod
    def _init_template_state(model: M, block_size: int) -> train_state.TrainState:
        import optax

        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, block_size), dtype=jnp.int32)
        params = model.init(key, dummy_input)["params"]
        state = train_state.TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=params,
            tx=optax.identity(),
        )
        return cast(train_state.TrainState, state)

    @classmethod
    def from_trainer(cls, trainer: "BaseTrainer[Any]") -> Self:
        """Create InferenceJob sharing trainer's state.

        The InferenceJob references (not copies) trainer's state, so changes
        to trainer.state are reflected in the InferenceJob.
        """
        job = object.__new__(cls)
        job._init_from_spec(trainer.spec)

        # Share references to trainer's inference state
        job.state = trainer.state
        job.mesh = trainer.mesh
        job.state_sharding = trainer.state_sharding
        job.replicas = trainer.replicas
        job.local_replicas = trainer.local_replicas
        job.per_device_batch_size = trainer.per_device_batch_size
        job.block_size = trainer.args.block_size
        job.key = trainer.key

        return job

    @classmethod
    def from_checkpoint(
        cls, suffix: str | Path, spec: ExecutionSpec
    ) -> Tuple[Self, Any]:
        """Load InferenceJob from checkpoint using CheckpointedJob infrastructure.

        Uses cls.MODEL for model initialization and sharding.
        Calls get_tree_and_metadata() for checkpoint restoration.

        Args:
            suffix: Checkpoint suffix
            spec: ExecutionSpec with topology

        Returns:
            (job, config) tuple
        """
        path = CheckpointedJob._get_checkpoint_path(spec, suffix)
        logger.debug("CHECKPOINT | loading {} from {}", cls.__name__, path)

        # Load config (we use spec's name/group/project - this is a new job, not a restoration)
        cfg = OmegaConf.load(path / "config.yaml")

        job = object.__new__(cls)
        job._init_from_spec(spec)
        job.key = jax.random.PRNGKey(0)  # Will be overwritten by get_tree_and_metadata

        with configuration(cfg):
            # Initialize model from cls.MODEL (configure hydrates from OmegaConf)
            model = configure(cls.MODEL)

            # Get mesh from topology
            assert spec.topology is not None, "Topology required for checkpoint loading"
            job.mesh = spec.topology.mesh
            job.replicas = spec.topology.replicas
            job.local_replicas = spec.topology.local_replicas

            # Initialize template state (for checkpoint structure)
            template_state = cls._init_template_state(
                model, int(cfg.architecture.block_size)
            )

            # Compute sharding
            job.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
                flax.linen.get_partition_spec(template_state),
                job.mesh,
                rules=tuple(model.sharding),
            )

            # Use CheckpointedJob's restoration infrastructure
            state, metadata = job.get_tree_and_metadata(suffix, template_state)
            job.state = jax.device_put(state, job.state_sharding)

            job.per_device_batch_size = cfg.training.per_device_batch_size
            job.block_size = cfg.architecture.block_size

        logger.debug("CHECKPOINT | loaded {}", spec.name)
        return job, cfg

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
        # large vocab/embedding matrices).
        prefill_fn = jax.jit(forward_fn, static_argnames=("deterministic", "mutable"))

        # Step 1: Prefill — initialize cache with full prompt
        (prefill_logits, _, _), cache = prefill_fn(
            state,
            state.params,
            (input, None, input_mask),
            deterministic=True,
            mutable=("cache",),
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

        return jax.jit(_run_scan)(state, cache, first_token, out_buf, key)  # type: ignore[no-any-return]
