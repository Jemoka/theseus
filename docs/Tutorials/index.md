# Tutorials

Step-by-step guides for common workflows with theseus.

## Running experiments

- [Running Experiments](running.md) — configure a job, run it locally, dispatch it to a remote machine or SLURM cluster, and start an interactive Jupyter REPL.

## Extending theseus

- [Adding Things](adding.md) — overview of the four extension points.
  - [Adding a Model](adding-model.md) — subclass `Module` or extend an existing architecture.
  - [Adding an Experiment](adding-experiment.md) — register a new training job with `@job`.
  - [Adding a Dataset](adding-dataset.md) — register a pretraining or chat dataset with `@dataset`.
  - [Adding an Evaluation](adding-evaluation.md) — register a new evaluation with `@evaluation`.
