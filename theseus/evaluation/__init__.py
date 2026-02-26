from .base import (
    Evaluation,
    RolloutEvaluation,
    EncodingEvaluation,
    PerplexityEvaluation,
    PerplexityComparisonEvaluation,
    Evaluator,
)

EVALUATIONS: dict[str, type[Evaluation]] = {}

__all__ = [
    "EVALUATIONS",
    "Evaluation",
    "RolloutEvaluation",
    "EncodingEvaluation",
    "PerplexityEvaluation",
    "PerplexityComparisonEvaluation",
    "Evaluator",
]
