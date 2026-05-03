"""Coordinate-ascent multi-objective reward.

Baseline against `mok_reward`: rather than scalarize the multi-objective
reward via KL-balancing, we pick the lagging component (lowest batch-mean) as
the active coordinate, and zero the advantage of any rollout already
out-performing its GRPO group on some *other* (non-active) component. The
remaining samples are 'free to improve' on the laggard without risking
regression on a coordinate where they already beat their group, so each step
concentrates updates on the worst dimension.

Implemented at the `reward()` level: for each group, masked samples are
assigned the mean active-coord score of the *kept* samples in that group.
After GRPO's group-relative z-scoring, masked samples land at advantage 0
(zero gradient via the surrogate loss), while kept samples are standardized
against the kept-sample mean.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from theseus.config import field


@dataclass
class CoordAscentConfig:
    schedule: str = field("optimization/coord/schedule", default="argmin")


def coord_ascent_reward(
    self: Any, evals: Dict[str, np.ndarray], group_size: int
) -> np.ndarray:
    """Compute coordinate-ascent reward signal.

    Args:
        self: trainer (used only for main_process gating in logging).
        evals: {component_name: (N,)} per-rollout component scores.
        group_size: GRPO group size G; N must be divisible by G.

    Returns:
        (N,) reward array suitable for GRPO `_smear_rewards` z-scoring.
    """
    names: List[str] = sorted(evals.keys())
    stacked = np.stack(
        [np.asarray(evals[k], dtype=np.float32) for k in names]
    )  # (C, N)
    C, N = stacked.shape
    g = int(group_size)
    if N % g != 0:
        raise ValueError(
            f"coord_ascent_reward | rollout count {N} not divisible by group_size {g}"
        )
    n_groups = N // g

    # Active coordinate = lagging component (lowest batch-mean).
    active = int(np.argmin(stacked.mean(axis=1)))

    # Per-group, per-component means.
    grouped = stacked.reshape(C, n_groups, g)
    group_means = grouped.mean(axis=-1, keepdims=True)  # (C, n_g, 1)
    above_mean = grouped > group_means  # (C, n_g, g) bool

    # Mask if above-group-mean on any *non-active* coord.
    if C > 1:
        non_active_mask = np.ones(C, dtype=bool)
        non_active_mask[active] = False
        mask_out = above_mean[non_active_mask].any(axis=0)  # (n_g, g) bool
    else:
        mask_out = np.zeros((n_groups, g), dtype=bool)

    keep = ~mask_out  # (n_g, g)

    active_grouped = grouped[active]  # (n_g, g)

    # Per-group mean of kept samples' active-coord scores. Fall back to the
    # group's overall active mean if every sample is masked (degenerate group
    # → all rewards equal → z-score = 0 → no gradient).
    kept_count = keep.sum(axis=-1)  # (n_g,)
    safe_kept_count = np.maximum(kept_count, 1)
    kept_sum = (active_grouped * keep.astype(np.float32)).sum(axis=-1)  # (n_g,)
    kept_mean = kept_sum / safe_kept_count  # (n_g,)
    fallback_mean = active_grouped.mean(axis=-1)  # (n_g,)
    fill_value = np.where(kept_count > 0, kept_mean, fallback_mean)  # (n_g,)

    out = np.where(
        keep,
        active_grouped,
        fill_value[:, None],
    ).reshape(N)

    if getattr(self, "main_process", lambda: True)():
        masked_frac = float(mask_out.mean())
        fully_masked_groups = int((kept_count == 0).sum())
        logger.debug(
            "coord | active={} masked_frac={:.3f} fully_masked_groups={}/{} "
            "batch_means={}",
            names[active],
            masked_frac,
            fully_masked_groups,
            n_groups,
            {n: float(stacked[i].mean()) for i, n in enumerate(names)},
        )

    return out  # type: ignore[no-any-return]
