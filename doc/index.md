# theseus Documentation

`theseus` is a framework for training and evaluating language models with three priorities:

- **fast iteration** on a single machine,
- **reproducible configuration** from dataclasses to YAML,
- **seamless promotion** from local runs to remote clusters.

If you are deciding where to start:

1. Use **Getting Started** for first setup and first run.
2. Use **Tutorials** for end-to-end workflows.
3. Use **User Guide** for day-to-day operations and customization.
4. Use **API Reference** when you need exact signatures/docstrings.

## What Theseus Gives You

- CLI for configure/run/submit/checkpoint/restore lifecycle.
- Python Quick API for notebook/prototyping workflows.
- A composable model abstraction (`theseus.model.module.Module`) with sharding hooks.
- Trainers with distributed setup, batch accumulation, validation, and checkpointing.
- Data tokenization jobs and tokenizer-agnostic chat formatting.
- Remote dispatch system for plain SSH and SLURM.

## Mental Model

A run in `theseus` usually looks like this:

1. **Define intent** with dataclass-backed config keys.
2. **Materialize config** into YAML (`theseus configure`) or mutate in Python (`quick`).
3. **Execute locally** (`theseus run`) until behavior looks right.
4. **Submit remotely** (`theseus submit`) when you need scale.
5. **Recover/resume** with checkpoint tooling.

This separation is deliberate: your model/trainer code should remain mostly identical between laptop and cluster.

## Recommended Learning Path

1. Read `getting-started/installation.md`.
2. Run `getting-started/first-run.md`.
3. Complete `tutorials/cli-end-to-end.md`.
4. Read `user-guide/customization-patterns.md` before extending internals.

## Source Entry Points

- `theseus/cli.py`: all CLI commands.
- `theseus/quick.py`: Python quick API.
- `theseus/config.py`: config build/hydrate system.
- `theseus/job.py`: lifecycle + checkpoint/recovery.
- `theseus/training/trainer.py`: core trainer loop.
- `theseus/model/module.py`: model extension contract.
- `theseus/dispatch/dispatch.py`: remote submission API.

## Styling Note

The docs visual style intentionally mirrors `theseus/web/static/styles.css`:

- IBM Plex Sans / Mono
- `#F7F7F7` background and `#2F3338` foreground
- restrained accent palette (`#37A5BE`, `#26A671`, `#F27200`, `#FF3900`)
- sharp/compact components with low radius
