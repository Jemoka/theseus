# Architecture Philosophy

Theseus is intentionally opinionated. The architecture is optimized for research velocity under imperfect infrastructure constraints.

## Principles

## 1. One Configuration Story

Everything meaningful should be representable in config keys.

Why:

- reproducibility across machines,
- easier diffing/review,
- easier promotion from local to remote.

Consequence:

- dataclass schemas are first-class,
- runtime code should read from hydrated config, not hidden globals.

## 2. Local-First, Cluster-Ready

A run should start locally and then scale to remote with minimal changes.

Why:

- most logic bugs are found cheaply on small runs,
- infrastructure setup should not block experimentation.

Consequence:

- same trainer/model entrypoints for local and remote,
- dispatch only wraps allocation/shipping/bootstrap concerns.

## 3. Composition Over Special Paths

Core loops are generic; specialization is pushed to model/evaluator/experiment classes.

Why:

- fewer hard-coded branches,
- easier to reason about performance and failure modes,
- more reusable extension points.

Consequence:

- extend by subclassing trainer/model/evaluator,
- avoid forking base loops unless semantics truly diverge.

## 4. Explicit Sharding Semantics

Logical axes and physical mesh axes are separate concerns.

Why:

- cleaner portability across topologies,
- easier debugging of model-parallel behavior,
- avoids hard-coding device layout assumptions into layer code.

Consequence:

- model declares logical axis intent,
- trainer/runtime maps to physical axes.

## 5. Recovery Is a Feature, Not an Add-On

Checkpointing is part of the job lifecycle from day one.

Why:

- long-running jobs fail,
- preemption happens,
- recoverability determines operational trust.

Consequence:

- save metadata + config + state together,
- provide explicit CLI recovery commands,
- test restore early.

## Tradeoffs

These choices prioritize reliability and operability over minimal abstraction count. The framework may feel stricter than ad-hoc scripts, but the payoff is lower failure cost at scale.
