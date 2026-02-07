"""
HuggingFace-specific evaluator wiring.
"""

from typing import Any, Optional, Tuple, cast

import jax

from theseus.inference import M
from theseus.evaluation.base import Evaluator
from theseus.inference_huggingface import HFInferenceJob


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
    ) -> Tuple[jax.Array, jax.Array]:
        return cast(
            Tuple[jax.Array, jax.Array],
            HFInferenceJob.forward(
                state=state,
                params=params,
                batch=batch,
                key=key,
                deterministic=deterministic,
            ),
        )
