from .base import (
    Evaluation,
    RolloutEvaluation,
    EncodingEvaluation,
    PerplexityEvaluation,
    Evaluator,
)

EVALUATIONS: dict[str, type[Evaluation]] = {}

__all__ = [
    "EVALUATIONS",
    "Evaluation",
    "RolloutEvaluation",
    "EncodingEvaluation",
    "PerplexityEvaluation",
    "Evaluator",
]
