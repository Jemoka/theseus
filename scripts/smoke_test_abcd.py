#!/usr/bin/env python3
"""Smoke test for ABCDTrainer with refactored training code."""

from theseus.quick import quick

with quick(
    "continual/train/abcd", name="smoke_test_abcd", out_path="/Users/houjun/theseus"
) as j:
    # Tiny model
    j.config.architecture.n_layers = 2
    j.config.architecture.n_embd = 128
    j.config.architecture.n_head = 2
    j.config.architecture.block_size = 128

    # WSDS schedule requires constant_pct
    j.config.optimization.constant_pct = 0.3

    # Minimal multi-phase run: just 2 phases, small token counts
    j.config.training.tokens = [2048, 2048]
    j.config.training.batch_size = 8
    j.config.training.per_device_batch_size = 8
    j.config.training.validation_steps = 8
    j.config.training.validation = True
    j.config.training.evaluate = False
    j.config.eval.evaluations = []  # skip boundary evals (block_size too small for real tasks)

    # Only 2 datasets (matching the 2 token phases)
    j.config.training.dataset = [
        [{"name": "fineweb", "rate": 1.0, "style": "PMD", "suffix": ""}],
        [{"name": "fineweb", "rate": 1.0, "style": "PMD", "suffix": ""}],
    ]

    # Log every step, checkpoint never
    j.config.logging.report_interval = 1
    j.config.logging.checkpoint_interval = 100000
    j.config.logging.validation_interval = 2
    j.config.logging.wandb = False

    print("Starting ABCD smoke test with config:")
    print(
        f"  Model: {j.config.architecture.n_layers} layers, {j.config.architecture.n_embd} embd"
    )
    print(f"  Phases: {j.config.training.tokens}")
    print(f"  Batch size: {j.config.training.batch_size}")
    print()

    j()

    print("\nABCD smoke test completed successfully!")
