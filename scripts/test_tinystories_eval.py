#!/usr/bin/env python3
"""
Test PerplexityEvaluation end-to-end using TinyStoriesEval on a randomly
initialized model via the quick runner.
"""

from theseus.quick import quick

with quick(
    "gpt/train/pretrain", name="test_tinystories_eval", out_path="/Users/houjun/theseus"
) as j:
    # Tiny model
    j.config.architecture.n_layers = 2
    j.config.architecture.n_embd = 64
    j.config.architecture.n_head = 2
    j.config.architecture.block_size = 512  # long enough for TinyStories

    # Minimal training config (just enough to init the trainer)
    j.config.training.batch_size = 8
    j.config.training.per_device_batch_size = 8
    j.config.training.tokens = 8192
    j.config.training.validation = False
    j.config.training.evaluate = True
    j.config.eval.evaluations = ["tinystories"]

    j.config.logging.wandb = False
    j.config.logging.checkpoint_interval = 100000
    j.config.logging.report_interval = 1
    j.config.logging.validation_interval = 100000

    # Create the trainer (initializes model, state, mesh, evaluator)
    trainer = j.create()

    print("Running TinyStories evaluation on randomly initialized model...")
    results = trainer.inference.evaluate()

    score = results["tinystories"]
    ppl = 1.0 / score
    print(f"\nTinyStories 1/ppl : {score:.6f}")
    print(f"TinyStories ppl   : {ppl:.2f}")

    # A random model should have high perplexity (low score)
    assert score > 0, "Score must be positive"
    assert score < 1, "1/ppl should be < 1 for any reasonable ppl > 1"
    print("\nTest passed!")
