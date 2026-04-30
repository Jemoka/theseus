"""On-policy PPO trainer.

Custom train state caches a frozen reference policy (snapshotted at init) so
the loss can apply a KL penalty against it. The training step is unchanged
relative to BaseTrainer — the loop calls `self.batch()` per step, which we
override to (a) roll out the current policy via the existing `Evaluator`
machinery, (b) compute per-rollout rewards via `self.reward(evals)`, (c) smear
those rewards per-token via reward-to-go with a discount factor, and (d) cache
`old_log_probs` for the importance ratio.

Subclasses (e.g. `GRPOTrainer`) override advantage computation by overriding
`_advantages_from_rewards`.
"""

from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jax_random
import flax
import flax.linen
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

import optax
from loguru import logger

from theseus.base import PyTree, Axis, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.backbone import BackbonedTrainer
from theseus.model.module import Module
from theseus.evaluation.base import Evaluator, RLEvaluatorConfig


@dataclass
class PPOConfig:
    beta: float = field("optimization/ppo/beta", default=0.04)  # KL penalty
    clip_eps: float = field("optimization/ppo/clip_eps", default=0.2)
    discount: float = field("optimization/ppo/discount", default=1.0)
    sample_temperature: float = field(
        "optimization/ppo/sample_temperature", default=1.0
    )
    sample_top_p: float = field("optimization/ppo/sample_top_p", default=1.0)


class PPOTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    base: PyTree[Any]
    beta: float
    clip_eps: float


class PPOTrainer(BaseTrainer[BaseTrainerConfig, M], Generic[M]):
    """PPO trainer scaffold."""

    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [PPOConfig, RLEvaluatorConfig]

    def reward(self, evals: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate per-rollout scores from each RL component into a single
        per-rollout reward.

        ``evals`` is ``{evaluation_name: (N,)}`` - every component is expected
        to produce a per-rollout-slot score array of the same length ``N``.
        The return is shape ``(N,)`` and is the only reward signal used by the
        PPO/GRPO objective. The original component arrays are preserved only for
        logging.

        Default: element-wise sum across components. Subclasses override to
        combine however they like (weighted sum, gating, etc.). Called from
        ``_refill_buffer`` outside the gradient path, so subclasses can freely
        read instance state (configs, schedules, running stats) here.
        """
        stacked = np.stack([np.asarray(v, dtype=np.float32) for v in evals.values()])
        result: np.ndarray = np.sum(stacked, axis=0)
        return result

    # -- state -----------------------------------------------------------------

    def _init_state(self, params: PyTree[jax.Array]) -> None:
        """Build optimizer, scheduler, and sharded PPO train state."""

        self.ppo_config = configure(PPOConfig)
        self.rl_config = configure(RLEvaluatorConfig)

        if self.main_process():
            logger.info(
                "PPO | beta={} clip_eps={} discount={} sample_temperature={} sample_top_p={}",
                self.ppo_config.beta,
                self.ppo_config.clip_eps,
                self.ppo_config.discount,
                self.ppo_config.sample_temperature,
                self.ppo_config.sample_top_p,
            )

        self.scheduler: optax._src.base.Schedule = self._schedule()
        self.tx = self._optimizer()

        beta = self.ppo_config.beta
        clip_eps = self.ppo_config.clip_eps

        def make_state(p: PyTree[jax.Array]) -> PPOTrainState:
            base = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), p)
            return type_cast(
                PPOTrainState,
                PPOTrainState.create(  # type: ignore[no-untyped-call]
                    apply_fn=self.model.apply,
                    params=p,
                    base=base,
                    tx=self.tx,
                    beta=beta,
                    clip_eps=clip_eps,
                ),
            )

        logger.debug("PPO | tracing state shapes via eval_shape")
        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        logger.debug("PPO | jitting sharded state creation")
        self.state = jax.jit(make_state, out_shardings=self.state_sharding)(params)

        self.total_params = (
            sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        )
        if self.main_process():
            logger.info(f"MODEL | Total Parameters: {self.total_params:.2f}m")

    # -- data ------------------------------------------------------------------

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Skip the dataset — PPO generates batches from rollouts."""
        self.strategy = None  # type: ignore[assignment]
        self.train_dl = None  # type: ignore[assignment]
        self.val_dl = None  # type: ignore[assignment]

    def _init_counters_and_eval(self) -> None:
        """Standard counters + an extra Evaluator for the RL components."""
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf")

        # BaseTrainer.evaluator() is typed Optional but always returns a
        # concrete Evaluator here; ignore is on the assignment narrowing.
        self.inference: Evaluator[M] = self.evaluator()  # type: ignore[assignment]
        self.rl_inference: Evaluator[M] = Evaluator.from_trainer(
            self, config=self.rl_config
        )

        # Buffer entries are
        # (x, padding_mask, action_mask, old_log_probs, reward, component_rewards).
        # `old_log_probs` is captured at rollout time under the rollout-generating
        # policy (= π_θ_old in standard PPO notation), so the importance ratio in
        # the surrogate is exactly π_θ_new / π_θ_old. `component_rewards` is the
        # per-rollout-slot score from each RL component, kept only so we can
        # surface the un-aggregated signals as metrics.
        self._rollout_buffer: List[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                Dict[str, float],
            ]
        ] = []
        # Cached JIT for the rollout-time log-prob pass (built lazily on first use).
        self._old_logp_jit: Optional[Any] = None

        if self.main_process():
            logger.info(self.model)
            logger.info(
                "PPO | RL components: {}",
                [e.name for e in self.rl_inference.evaluations],
            )
            logger.debug(
                "PPO | per-step rollouts requested: {}",
                self.per_device_batch_size
                * self.local_replicas
                * self.accumulate_steps,
            )

    # -- rollout collection ----------------------------------------------------

    def _refill_buffer(self) -> None:
        """Run the RL evaluator and stage rollouts as (x, padding_mask,
        action_mask, reward) tuples on every host."""
        cfg = self.ppo_config

        # train_step donates self.state, so the rl_inference's snapshot is
        # stale after step 0 — refresh it before every rollout.
        self.rl_inference.state = self.state

        logger.debug(
            "PPO | refilling rollout buffer (components={}, temperature={}, top_p={})",
            [e.name for e in self.rl_inference.evaluations],
            cfg.sample_temperature,
            cfg.sample_top_p,
        )
        evals_dict, rollouts_per_eval = self.rl_inference.evaluate(
            reduce="none",
            return_intermediates=True,
            temperature=cfg.sample_temperature,
            top_p=cfg.sample_top_p,
        )
        logger.debug("PPO | rollouts done; computing rewards")

        # Reward contract: reward(evals) -> (N,), one scalarized reward per
        # rollout slot. Each RL component contributes a score array of shape
        # (N,) for those same slots; reward() combines those channels into one
        # scalar per slot.
        rewards = np.asarray(self.reward(evals_dict), dtype=np.float32)
        # Keep the un-aggregated per-component scores around so we can report
        # their means as metrics alongside the merged reward. These do not feed
        # the PPO objective.
        component_rewards_arr: Dict[str, np.ndarray] = {
            name: np.asarray(arr, dtype=np.float32) for name, arr in evals_dict.items()
        }

        if not rollouts_per_eval or not rollouts_per_eval[0]:
            raise RuntimeError(
                "PPO | RL evaluator produced no rollouts. Make sure "
                "training/rl/components is non-empty."
            )

        per_eval_counts = [len(rollouts) for rollouts in rollouts_per_eval]
        if len(set(per_eval_counts)) != 1:
            raise NotImplementedError(
                f"PPO | RL components produced different rollout counts "
                f"{per_eval_counts}; scalar reward contract requires equal counts."
            )
        rollout_count = per_eval_counts[0]
        if rewards.shape != (rollout_count,):
            raise ValueError(
                f"PPO | reward returned shape {rewards.shape}, expected "
                f"({rollout_count},) - one scalar reward per rollout slot."
            )

        for name, arr in component_rewards_arr.items():
            if arr.shape != (rollout_count,):
                raise ValueError(
                    f"PPO | component reward '{name}' has shape {arr.shape}, "
                    f"expected ({rollout_count},)."
                )

        # The scalarized reward has one row per rollout slot. Additional RL
        # components are reward channels for those slots; they are not appended
        # as extra training rows.
        rollouts = rollouts_per_eval[0]

        # The selected rollout source should have uniform sequence length T
        # because the eval pads to one ``total_tokens``. Catch violations with a
        # clear message instead of letting np.stack fail cryptically.
        ts = {r[0].shape[-1] for r in rollouts}
        if len(ts) > 1:
            raise NotImplementedError(
                f"PPO | selected rollout source produced mixed sequence "
                f"lengths {sorted(ts)}; expected one padded T."
            )

        # Cache log π_θ_old(y_t | x_<=t) NOW, under the rollout-generating
        # policy. Bound to π_θ_old by construction — the PPO contract.
        x_arr = np.stack([r[0] for r in rollouts]).astype(np.int32)
        am_arr = np.stack([r[1] for r in rollouts]).astype(bool)
        pm_arr = np.stack([r[2] for r in rollouts]).astype(np.int32)
        y_arr = np.roll(x_arr, -1, axis=-1)
        y_arr[:, -1] = 0
        y_safe = np.where(am_arr, y_arr, 0).astype(np.int32)
        logger.debug(
            "PPO | computing old log_probs (B={} T={})", x_arr.shape[0], x_arr.shape[-1]
        )
        old_log_probs = self._rollout_log_probs(x_arr, y_safe, pm_arr)
        old_log_probs = np.where(am_arr, old_log_probs, 0.0).astype(np.float32)

        # Multi-host: rollouts/rewards/log_probs are all replicated (B_global,
        # T) on every host (the eval allgather'd). The SPMD train_step expects
        # each host to provide its UNIQUE shard, which _to_global will stitch
        # back into the global batch. Split here so each host's buffer holds
        # only its slice.
        n_hosts, host_idx = jax.process_count(), jax.process_index()
        my_indices = np.array_split(np.arange(len(rollouts)), n_hosts)[host_idx]

        new_entries: List[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                Dict[str, float],
            ]
        ] = [
            (
                rollouts[i][0],
                rollouts[i][2].astype(bool),  # padding_mask
                rollouts[i][1].astype(bool),  # action_mask
                old_log_probs[i],
                float(rewards[i]),
                {name: float(arr[i]) for name, arr in component_rewards_arr.items()},
            )
            for i in my_indices
        ]
        self._rollout_buffer.extend(new_entries)
        logger.debug(
            "PPO | refill: B_global={} → host {}/{} got {} (T={}, action_tokens/sample={:.1f})",
            len(rollouts),
            host_idx,
            n_hosts,
            len(new_entries),
            x_arr.shape[-1],
            float(am_arr.sum() / max(am_arr.shape[0], 1)),
        )

    def _smear_rewards(
        self, rewards: np.ndarray, action_mask: np.ndarray, discount: float
    ) -> np.ndarray:
        """Distribute terminal sequence rewards across action positions via
        reward-to-go: R_t = γ^(T_a - 1 - t_a) * r where t_a indexes only the
        action positions. With γ=1, every action token gets the sequence reward.
        Non-action positions get 0.
        """
        B, T = action_mask.shape
        out = np.zeros((B, T), dtype=np.float32)
        if discount == 1.0:
            # np.repeat builds [r_0]*n_0 + [r_1]*n_1 + ...; boolean indexing on
            # action_mask iterates row-major (same order), so each row's True
            # slots get filled with that row's reward.
            out[action_mask] = np.repeat(rewards, action_mask.sum(axis=-1))
            return out
        # General case: enumerate action positions per row.
        for b in range(B):
            idxs = np.flatnonzero(action_mask[b])
            if idxs.size == 0:
                continue
            n = idxs.size
            # Largest exponent at the first action position; 0 at the last.
            powers = np.power(discount, np.arange(n - 1, -1, -1, dtype=np.float32))
            out[b, idxs] = rewards[b] * powers
        return out

    # -- rollout-time log-probs ------------------------------------------------

    @staticmethod
    def log_prob_step(
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # dict with x/y/padding_mask, each (S, B, T)
    ) -> jax.Array:
        """Scan log π(y_t | x_<=t) over S micro-batches; mirrors val_step shape."""

        def reduce(_: Any, batch_item: Any) -> Any:
            params: PyTree[jax.Array] = type_cast(PyTree[jax.Array], state.params)
            logits, _, _ = BaseTrainer.forward(
                state, params, batch_item, deterministic=True
            )
            lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            chosen = jnp.take_along_axis(
                lp, batch_item["y"][..., None], axis=-1
            ).squeeze(-1)
            return None, chosen

        _, log_probs = jax.lax.scan(reduce, None, batch)  # (S, B, T)
        return type_cast(jax.Array, log_probs)

    def _rollout_log_probs(
        self,
        x: np.ndarray,
        y: np.ndarray,
        padding_mask: np.ndarray,
    ) -> np.ndarray:
        """Cache log π_θ_old(y_t | x_<=t) for the freshly-drawn rollout batch.

        Inputs are ``(B_global, T)`` numpy, *replicated* on every host (the
        eval ``process_allgather``-s rollouts so every host stores the full
        set). To avoid every host redoing the same forward, we split the
        replicated batch into unique per-host shards, then defer to the same
        ``_reshape_batch`` / ``_to_global`` plumbing the train loop uses.
        After the scan we ``process_allgather`` so every host's buffer ends
        up with matching ``(B_global, T)`` log-probs.
        """
        from jax.experimental import multihost_utils as _mh

        # 1. Split replicated → host-local, then use the parent's standard
        #    (S, B, T) reshape + global-array conversion.
        n_hosts, host_idx = jax.process_count(), jax.process_index()
        local_batch: PyTree[np.ndarray] = type_cast(
            PyTree[np.ndarray],
            {
                "x": np.array_split(x, n_hosts, axis=0)[host_idx],
                "y": np.array_split(y, n_hosts, axis=0)[host_idx],
                "padding_mask": np.array_split(padding_mask, n_hosts, axis=0)[host_idx],
            },
        )
        batch = self._to_global(self._reshape_batch(local_batch))

        # 2. JIT once with the same data sharding as the train step.
        if self._old_logp_jit is None:
            data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore[no-untyped-call]
            self._old_logp_jit = jax.jit(
                self.log_prob_step,
                in_shardings=(self.state_sharding, data_shard),
                out_shardings=data_shard,
            )

        log_probs_g = self._old_logp_jit(self.state, batch)  # (S, B, T) sharded

        # 3. Local shard → numpy, allgather across hosts.
        log_probs_local = _mh.global_array_to_host_local_array(
            log_probs_g,
            self.mesh,
            P(None, Axis.BATCH, None),  # type: ignore[no-untyped-call]
        )
        if n_hosts > 1:
            # tiled=True: concatenate host-local chunks along the leading
            # microbatch axis; flattening happens after gather on host.
            gathered = _mh.process_allgather(log_probs_local, tiled=True)
        else:
            gathered = log_probs_local
        gathered_np = np.asarray(gathered).reshape(-1, gathered.shape[-1])
        logger.debug(
            "PPO | rollout-time log_probs ready (host {}/{}, B_global={} T={})",
            host_idx,
            n_hosts,
            gathered_np.shape[0],
            gathered_np.shape[-1],
        )
        return gathered_np

    # -- batch -----------------------------------------------------------------

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """Generate one PPO training batch by reusing the Evaluator dynamics.

        Returns a numpy dict shaped to feed straight into the parent train()'s
        ``_to_global(_reshape_batch(...))``.
        """
        bsz = self.per_device_batch_size * self.local_replicas * self.accumulate_steps
        while len(self._rollout_buffer) < bsz:
            logger.debug(
                "PPO | buffer have={} need={}, refilling",
                len(self._rollout_buffer),
                bsz,
            )
            self._refill_buffer()

        entries = self._rollout_buffer[:bsz]
        self._rollout_buffer = self._rollout_buffer[bsz:]

        x = np.stack([e[0] for e in entries]).astype(np.int32)  # (B, T)
        padding_mask = np.stack([e[1] for e in entries]).astype(bool)
        action_mask = np.stack([e[2] for e in entries]).astype(bool)
        old_log_probs = np.stack([e[3] for e in entries]).astype(np.float32)
        rewards = np.array([e[4] for e in entries], dtype=np.float32)  # (B,)
        # Per-component per-rollout-slot scores. Component name set is fixed at
        # trainer init time, so the dict keys are stable across calls (JIT-safe).
        component_names = list(entries[0][5].keys()) if entries else []
        component_rewards_per_rollout: Dict[str, np.ndarray] = {
            name: np.array([e[5][name] for e in entries], dtype=np.float32)
            for name in component_names
        }

        # y = next-token target (shift x left by 1); zero outside action_mask
        # since the loss only reads y at action positions anyway.
        y = np.roll(x, -1, axis=-1)
        y[:, -1] = 0
        y_safe = np.where(action_mask, y, 0).astype(np.int32)

        # Per-token reward via reward-to-go with discount γ. (Subclasses like
        # GRPO normalize within group inside _smear_rewards, so this array is
        # really "advantage per token" by the time it leaves.)
        per_token_rewards = self._smear_rewards(
            rewards, action_mask, self.ppo_config.discount
        )
        # Carry raw component scores through the same (S, B, *) sharding plumbing
        # as logging-only scalar channels. They intentionally do not use
        # _smear_rewards, because GRPO normalization and action-token weighting
        # are objective concerns, not component metric concerns.
        component_rewards: Dict[str, np.ndarray] = {
            name: arr[:, None].astype(np.float32)
            for name, arr in component_rewards_per_rollout.items()
        }

        # Stash raw per-rollout reward stats so we can log "is the policy
        # actually learning" without GRPO's z-score eating the signal.
        self._last_raw_reward_mean = float(rewards.mean())
        self._last_raw_reward_max = float(rewards.max())
        if self.main_process():
            logger.info(
                "PPO | raw_reward mean={:.3f} max={:.3f} (B={})",
                self._last_raw_reward_mean,
                self._last_raw_reward_max,
                rewards.shape[0],
            )
            for name, arr in component_rewards_per_rollout.items():
                logger.info(
                    "PPO | component[{}] mean={:.3f} max={:.3f}",
                    name,
                    float(arr.mean()),
                    float(arr.max()),
                )
            logger.debug(
                "PPO | per-token reward stats: mean={:.4f} std={:.4f} action_tokens/sample={:.1f}",
                float(per_token_rewards.mean()),
                float(per_token_rewards.std()),
                float(action_mask.sum() / max(action_mask.shape[0], 1)),
            )

        # Cast: the dict literal's inferred value type widens to include
        # ``Dict[str, np.ndarray]`` (component_rewards), which mypy can't
        # auto-unify with ``PyTree[np.ndarray]`` even though it's structurally
        # valid (PyTree[T] = T | list[...] | tuple[...] | dict[str, PyTree[T]]).
        return type_cast(
            PyTree[np.ndarray],
            {
                "x": x,
                "y": y_safe,
                "padding_mask": padding_mask.astype(np.int32),
                "action_mask": action_mask.astype(np.int32),
                "old_log_probs": old_log_probs,
                "per_token_rewards": per_token_rewards,
                "component_rewards": component_rewards,
            },
        )

    # -- forward / loss --------------------------------------------------------

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        """PPO clipped surrogate loss + KL penalty against the frozen reference."""
        cstate = type_cast(PPOTrainState, state)
        batch_dict = type_cast(Dict[str, jax.Array], batch)

        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"].astype(jnp.bool_)
        action_mask = batch_dict["action_mask"].astype(jnp.bool_)
        old_log_probs = batch_dict["old_log_probs"]
        per_token_rewards = batch_dict["per_token_rewards"]
        component_rewards = type_cast(
            Dict[str, jax.Array], batch_dict.get("component_rewards", {})
        )

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        # Policy forward: returns (logits, loss); we ignore loss.
        policy_logits, _ = cstate.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Reference forward (frozen): reuse the same apply_fn with state.base.
        ref_logits, _ = cstate.apply_fn(
            {"params": cstate.base},
            x,
            y,
            padding_mask=padding_mask,
            deterministic=True,
        )
        ref_logits = jax.lax.stop_gradient(ref_logits)

        # log π(y_t | x) for both policies at every action position.
        policy_logp = jax.nn.log_softmax(policy_logits.astype(jnp.float32), axis=-1)
        new_logp = jnp.take_along_axis(policy_logp, y[..., None], axis=-1).squeeze(-1)

        ref_logp = jax.nn.log_softmax(ref_logits.astype(jnp.float32), axis=-1)
        ref_logp = jnp.take_along_axis(ref_logp, y[..., None], axis=-1).squeeze(-1)

        action_mask_f = action_mask.astype(jnp.float32)
        n_actions = jnp.maximum(action_mask_f.sum(), 1.0)

        # Importance ratio π_new / π_old. Masked to 1.0 outside action
        # positions so the metric isn't misleading; loss-side masking via
        # action_mask_f below is what actually keeps gradients local to
        # generated tokens.
        log_ratio = new_logp - old_log_probs
        ratio = jnp.where(action_mask, jnp.exp(log_ratio), 1.0)

        # Advantages: use raw smeared rewards. (Subclasses normalize by group, etc.)
        advantages = per_token_rewards * action_mask_f

        # Clipped surrogate (PPO objective). Loss = -E[min(ratio*A, clip(ratio)*A)].
        clip_eps = cstate.clip_eps
        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        surrogate = -jnp.minimum(unclipped, clipped)
        surrogate_loss = (surrogate * action_mask_f).sum() / n_actions

        # KL penalty against the frozen reference. We use Schulman's k3
        # estimator: kl ≈ exp(logr) - 1 - logr, where logr = log π_new - log π_ref.
        # Unlike the raw log-ratio (k1), k3 is always non-negative, so a
        # positive β actually penalizes divergence (k1 can flip sign on
        # sampled tokens and accidentally *reward* moving away from ref).
        log_ratio_ref = (new_logp - ref_logp) * action_mask_f
        kl_per_tok = (jnp.exp(log_ratio_ref) - 1.0 - log_ratio_ref) * action_mask_f
        kl = kl_per_tok.sum() / n_actions

        loss = surrogate_loss + cstate.beta * kl

        metrics: Dict[str, Any] = {
            "ppo/surrogate_loss": surrogate_loss,
            "ppo/kl": kl,
            "ppo/reward_mean": (per_token_rewards * action_mask_f).sum() / n_actions,
            "ppo/ratio_mean": (ratio * action_mask_f).sum() / n_actions,
            "ppo/advantage_mean": (advantages * action_mask_f).sum() / n_actions,
            "ppo/advantage_abs_mean": (jnp.abs(advantages) * action_mask_f).sum()
            / n_actions,
            "ppo/n_actions": n_actions,
        }
        # Per-component raw reward means. These are logging-only rollout means;
        # they do not feed the objective and are not action-token weighted.
        for name, ctr in component_rewards.items():
            metrics[f"ppo/component/{name}/reward_mean"] = jnp.mean(
                ctr.astype(jnp.float32)
            )

        return policy_logits, loss, metrics


class BackbonedPPOTrainer(BackbonedTrainer, PPOTrainer[Module]):
    """PPO trainer that initializes from a pretrained HuggingFace backbone.

    Mirrors ``BackbonedContrastiveTrainer``: pulls in BackbonedTrainer's
    `_init_model` (loads HF weights, no `cls.MODEL.gather()`) plus PPOTrainer's
    state/data/forward overrides. The MRO resolves cleanly because both
    parents only override disjoint methods.
    """

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedTrainer (MRO), giving us the HF-style
        # config without MODEL.gather(); we then add PPO/RL configs.
        return super()._config() + [PPOConfig, RLEvaluatorConfig]
