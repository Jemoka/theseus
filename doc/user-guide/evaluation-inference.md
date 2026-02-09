# Evaluation and Inference

Evaluation in theseus is built as a layer over inference and trainer state.

## Core Abstractions

- `InferenceJob`: model-state + autoregressive/predictive execution utilities.
- `Evaluation`: task-level metric logic.
- `Evaluator`: orchestration over one or more evaluations.

## Evaluation Styles

`theseus/evaluation/base.py` includes patterns for:

- rollout evaluations (generation-based checks),
- encoding evaluations (next-token style checks),
- perplexity-based multiple-choice comparisons.

Each evaluation class focuses on scoring logic and sample formatting, not on low-level distributed execution setup.

## HF Compatibility

HF-compatible paths are provided in:

- `theseus/evaluation/huggingface.py`
- `theseus/inference/huggingface.py`

These reuse the same high-level evaluator flow with HF-specific state/forward wiring.

## Integrating with Trainers

Concrete trainer experiments choose evaluator via `evaluator()` method. Keep this selection explicit in experiment classes.

## Practical Guidance

- Keep prompt formatting deterministic.
- Reuse tokenizer backend and template policy from training where possible.
- Decode/clean predictions before metric checks to avoid tokenization artifacts.
- For long evaluations, validate chunking and memory behavior early.

## Extending Evaluations

For concrete inheritance templates for `RolloutEvaluation`, `EncodingEvaluation`, and `PerplexityEvaluation`, see:

- `user-guide/writing-datasets-evals.md`
