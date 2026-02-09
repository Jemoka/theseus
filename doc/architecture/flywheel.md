# Flywheel Data Pipeline

Flywheel is the training-side data strategy layer that turns dataset sources into distributed-ready batches.

## Components

- `strategy.py`: mixture sampling + async fetch orchestration.
- `pmd.py`: memmap token-stream sampling (PMD).
- `padded.py`: fixed-shape samples with explicit masks.

## Why Flywheel Exists

Data is often the first scalability bottleneck. Flywheel isolates sampling and batching behavior so trainer logic remains mostly invariant.

## Strategy Responsibilities

- sample mixture proportions from per-dataset rates,
- fetch from underlying datasets,
- combine and shuffle samples,
- filter invalid/all-zero rows,
- recursively refill when needed,
- provide async queue-backed prefetch for train and val.

## PMD Behavior

PMD loader reads contiguous token files and forms `(x, y)` slices with one-token shift. It is optimized for low-overhead sequential access patterns.

## Padded Behavior

Padded loader reads token + mask matrices. It sets ignored target positions to `-1`, aligning with model loss masking semantics.

## Deterministic Validation

Deterministic key paths are used for stable validation batches. Indexing must always wrap over valid sequence starts to avoid empty slices.

## Trainer Contract

Flywheel returns `(x, y, padding_mask)` arrays. Trainer reshapes into accumulation-aware structure and places arrays on global sharded meshes.
