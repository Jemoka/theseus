"""Registry sanity tests — verify all expected datasets, evaluations,
and jobs are registered.

Migrated from scripts/test_new_datasets.py (registry portions).
"""

import pytest


class TestDatasetRegistry:
    def test_core_datasets_registered(self):
        from theseus.registry import DATASETS

        expected = [
            "fineweb", "pile", "pes2o", "pg19", "ccaligned",
            "mnli", "qqp", "sst2", "siqa", "mmlu", "squad",
            "cfq", "clutrr", "harmfulqa", "longhealth", "mtob",
            "pile_detoxify", "pile_injected",
            "alpaca", "bbq", "fever", "longbench", "winogrande",
        ]
        for name in expected:
            assert name in DATASETS, f"'{name}' not in DATASETS registry"


class TestEvaluationRegistry:
    def test_core_evals_registered(self):
        from theseus.registry import EVALUATIONS

        expected = [
            "mnli", "qqp", "sst2", "siqa", "mmlu", "squad",
            "cfq", "clutrr", "longhealth", "mtob",
            "pile", "pes2o", "pg19", "tinystories", "fineweb_ppl",
            "mnli_ppl", "qqp_ppl", "sst2_ppl", "siqa_ppl",
            "pile_injected",
            "pg19_2k", "pg19_4k", "pg19_8k", "pg19_16k", "pg19_32k",
        ]
        for name in expected:
            assert name in EVALUATIONS, f"'{name}' not in EVALUATIONS registry"


class TestJobRegistry:
    def test_continual_jobs_registered(self):
        from theseus.registry import JOBS

        expected = [
            "continual/train/abcd",
            "continual/train/abcd_kl",
            "continual/train/benchmark",
            "continual/train/benchmark_mamba",
            "continual/train/benchmark_hybrid",
            "continual/train/benchmark_lora",
            "continual/train/benchmark_mamba_lora",
            "continual/train/benchmark_hybrid_lora",
        ]
        for name in expected:
            assert name in JOBS, f"'{name}' not in JOBS registry"

    def test_base_jobs_registered(self):
        from theseus.registry import JOBS

        expected = ["gpt/train/pretrain"]
        for name in expected:
            assert name in JOBS, f"'{name}' not in JOBS registry"
