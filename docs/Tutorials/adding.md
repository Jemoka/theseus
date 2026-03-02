# Adding Things

Theseus is structured around four extension points. Each one follows the same pattern: write a class, drop it in the right folder, register it.

| What | Where | How |
|---|---|---|
| [Model](adding-model.md) | `theseus/model/models/` | subclass `Module`, add to `__init__.py` |
| [Experiment](adding-experiment.md) | `theseus/experiments/` | `@job("key")` + subclass `BaseTrainer` |
| [Dataset](adding-dataset.md) | `theseus/data/datasets/` | `@dataset("key")` + subclass a dataset base |
| [Evaluation](adding-evaluation.md) | `theseus/evaluation/datasets/` | `@evaluation("key")` + subclass `RolloutEvaluation` |
