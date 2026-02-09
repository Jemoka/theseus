# Core Concepts

This page describes the framework-level mechanics behind every run.

## Specs: Intent vs Allocation

Theseus separates user intent from runtime allocation:

- `JobSpec`: logical identity (`name`, `project`, `group`, optional id).
- `ExecutionSpec`: resolved runtime (`hardware`, `topology`, distributed metadata).

This keeps job semantics stable while hardware allocation changes.

## Job Lifecycle

Jobs follow a synchronized multi-host lifecycle:

1. check `done` idempotency gate,
2. synchronize hosts,
3. execute `run()`,
4. synchronize hosts,
5. execute `finish()`.

Checkpoint-capable jobs extend this with save/restore methods.

## Config Hydration Boundary

A config context (`configuration(...)`) is established before job creation.

Class constructors call `configure(...)` to hydrate typed config objects. This prevents ad-hoc global access and keeps dependencies explicit.

## Registry as Integration Backbone

`theseus.registry` aggregates jobs, datasets, optimizers, and schedules. The registry powers:

- CLI discovery (`theseus jobs`),
- config validation by job key,
- dispatch-time job key validation.

## Data/Model/Trainer Coupling Strategy

The framework deliberately keeps coupling shallow:

- model code declares shape/sharding intent,
- trainer handles orchestration and distributed execution,
- data pipeline handles batch materialization,
- evaluator/inference layers handle metrics and generation.

This reduces cross-cutting change cost when experimenting.
