# Model System

The model layer is designed to express architecture intent while staying compatible with distributed execution.

## Base Contract

`theseus.model.module.Module` extends Flax `nn.Module` and adds framework requirements:

- `components()`: declares dependent config-bearing modules/components,
- `gather()`: recursively collects config classes for job config synthesis,
- `sharding`: logical-to-physical sharding rule table.

## Logical Axes First

Models express **logical** axes (e.g. vocab, embedding, attention) independent of physical devices. The trainer later maps these to mesh axes.

This avoids hard-coding cluster-specific layouts inside model code.

## Native Model Path

The GPT baseline (`theseus/model/models/base.py`) demonstrates:

- embedding/decode/unembed decomposition,
- explicit cross-entropy masking with ignore index,
- sharding declarations aligned with logical axes.

## HF-Compatible Path

`HFCompat` (`theseus/model/huggingface.py`) provides a compatibility bridge for HuggingFace causal LM modules.

Key ideas:

- lazily build/cache meta models,
- represent HF parameter tensors in flax parameter trees,
- convert between JAX/torchax tensor views in forward pass,
- preserve buffer mutability for stateful modules.

## Llama Family Example

`theseus/model/models/llama.py` specializes axis mapping by parameter name patterns and reuses HF-compatible trainer/evaluator/inference classes.

## Extension Guidance

When adding a model:

1. decide native vs HF-compat path,
2. define sharding semantics explicitly,
3. keep forward signature compatible with trainer/evaluator expectations,
4. register model in `theseus/model/models/__init__.py`.
