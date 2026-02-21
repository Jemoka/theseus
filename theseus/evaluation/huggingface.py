"""
HuggingFace-specific evaluator wiring.
"""

from typing import Any, Optional, Tuple

import jax

from theseus.inference import M
from theseus.evaluation.base import Evaluator
from theseus.inference.huggingface import HFInferenceJob


class HFEvaluator(Evaluator[M]):
    @staticmethod
    def _init_template_state(model: M, block_size: int) -> Any:
        return HFInferenceJob._init_template_state(model, block_size)

    @staticmethod
    def forward(
        state: Any,
        params: Any,
        batch: Tuple[jax.Array, Optional[jax.Array], jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        mutable: Optional[list[str]] = None,
        extra_variables: Optional[dict[str, Any]] = None,
    ) -> Any:
        return HFInferenceJob.forward(
            state=state,
            params=params,
            batch=batch,
            key=key,
            deterministic=deterministic,
            mutable=mutable,
            extra_variables=extra_variables,
        )
