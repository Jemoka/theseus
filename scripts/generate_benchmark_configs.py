#!/usr/bin/env python3
"""Generate all YAML configs for the continual learning benchmark paper.

Architectures: transformer, mamba, hybrid, moe (4).
DS/CG: 6 splits × 3 scales × 4 archs × 2 optim × 3 schedules = 432.
IC: 3 splits × 3 scales × 4 archs (full + wsd only) = 36.
Total: 468 configs.

Logging intervals are derived per config from the shortest dataset phase:
    validation_interval = min_phase_steps // 5
    checkpoint_interval = total_steps // 5  (roughly 5 checkpoints per run)

Usage:
    uv run python scripts/generate_benchmark_configs.py
"""

from pathlib import Path

import yaml

# ======================================================================
# Architecture specs by scale
# ======================================================================

_MOE_CFG = {
    "experts": 8,
    "experts_per_embd": 2,
    "capacity_factor": 1.0,
    "capacity_round_to": 128,
    "implementation": "base",
    "bias_update_rate": 0.25,
    "bias_smoothing": 1.0,
}

ARCH_SPECS = {
    "transformer": {
        "700m": {"n_layers": 16, "n_embd": 1664, "n_head": 13},
        "1b": {"n_layers": 16, "n_embd": 2048, "n_head": 16},
        "2b": {"n_layers": 24, "n_embd": 2560, "n_head": 20},
    },
    "mamba": {
        "700m": {
            "n_layers": 32,
            "n_embd": 1664,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
        "1b": {
            "n_layers": 32,
            "n_embd": 2048,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
        "2b": {
            "n_layers": 48,
            "n_embd": 2560,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
    },
    "hybrid": {
        "700m": {
            "n_layers": 20,
            "n_embd": 1664,
            "n_head": 13,
            "mamba_layers": "even",
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
        "1b": {
            "n_layers": 24,
            "n_embd": 2048,
            "n_head": 16,
            "mamba_layers": "even",
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
        "2b": {
            "n_layers": 30,
            "n_embd": 2560,
            "n_head": 20,
            "mamba_layers": "even",
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "n_groups": 1,
            "n_heads": -1,
        },
    },
    "moe": {
        # MoE dimensions are chosen so that *total* params (all experts)
        # approximate the dense-equivalent budget and *active* params
        # (top-2 of 8 experts + shared attention) are ~1/3 of total.
        #   700m ≈ 720M total / 266M active
        #   1b   ≈ 1.04B total / 360M active
        #   2b   ≈ 1.96B total / 649M active
        "700m": {
            "n_layers": 16,
            "n_embd": 768,
            "n_head": 6,
            "layer_norm_eps": 1.0e-05,
            "intermediate_size": -1,
            "moe": dict(_MOE_CFG),
        },
        "1b": {
            "n_layers": 24,
            "n_embd": 768,
            "n_head": 6,
            "layer_norm_eps": 1.0e-05,
            "intermediate_size": -1,
            "moe": dict(_MOE_CFG),
        },
        "2b": {
            "n_layers": 26,
            "n_embd": 1024,
            "n_head": 8,
            "layer_norm_eps": 1.0e-05,
            "intermediate_size": -1,
            "moe": dict(_MOE_CFG),
        },
    },
}

# Chinchilla-optimal token budgets
TOKEN_BUDGETS = {
    "700m": 14_000_000_000,
    "1b": 20_000_000_000,
    "2b": 40_000_000_000,
}

JOB_MAP = {
    ("transformer", "full"): "continual/train/benchmark",
    ("transformer", "lora"): "continual/train/benchmark_lora",
    ("mamba", "full"): "continual/train/benchmark_mamba",
    ("mamba", "lora"): "continual/train/benchmark_mamba_lora",
    ("hybrid", "full"): "continual/train/benchmark_hybrid",
    ("hybrid", "lora"): "continual/train/benchmark_hybrid_lora",
    ("moe", "full"): "continual/train/benchmark_moe",
    ("moe", "lora"): "continual/train/benchmark_moe_lora",
}

# ======================================================================
# Split definitions
# ======================================================================


def _scale_tokens(ratios: list[float], total: int) -> list[int]:
    """Scale ratios to token counts that sum to total."""
    tokens = [int(r * total) for r in ratios]
    # Fix rounding
    tokens[-1] = total - sum(tokens[:-1])
    return tokens


def _split_config(split_name: str, scale: str) -> dict:
    """Return split-specific training config (tokens, datasets, evals, fade)."""
    total = TOKEN_BUDGETS[scale]

    if split_name == "ds_nlu":
        tokens = _scale_tokens([0.98, 0.005, 0.005, 0.005, 0.005], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "fineweb", "rate": 1.0, "style": "PMD"}],
                [{"name": "mnli", "rate": 1.0, "style": "PADDED"}],
                [{"name": "qqp", "rate": 1.0, "style": "PADDED"}],
                [{"name": "sst2", "rate": 1.0, "style": "PADDED"}],
                [{"name": "siqa", "rate": 1.0, "style": "PADDED"}],
            ],
            "evaluations": ["mnli", "qqp", "sst2", "siqa", "fineweb_ppl"],
            "fade": {"overlap": 0.1, "curve": "cosine"},
            "block_size": 2048,
            "skip_first_dataset_validation": True,
        }

    elif split_name == "ds_domain":
        tokens = _scale_tokens([0.5, 0.5], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "fineweb", "rate": 1.0, "style": "PMD"}],
                [{"name": "pes2o", "rate": 1.0, "style": "PMD"}],
            ],
            "evaluations": [
                "fineweb_ppl",
                "pes2o_ppl",
                "pile_ppl",
                "tinystories_ppl",
            ],
            "fade": {"overlap": 0.1, "curve": "cosine"},
            "block_size": 2048,
            "skip_first_dataset_validation": False,
        }

    elif split_name == "ds_multilingual":
        tokens = _scale_tokens([1 / 3, 1 / 3, 1 / 3], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "ccaligned", "rate": 1.0, "style": "PMD", "suffix": "fr"}],
                [{"name": "ccaligned", "rate": 1.0, "style": "PMD", "suffix": "de"}],
                [{"name": "ccaligned", "rate": 1.0, "style": "PMD", "suffix": "zh"}],
            ],
            "evaluations": [
                "ccaligned_fr_ppl",
                "ccaligned_de_ppl",
                "ccaligned_zh_ppl",
            ],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 2048,
            "skip_first_dataset_validation": False,
        }

    elif split_name == "cg_grammar":
        tokens = _scale_tokens([0.5, 0.49, 0.01], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "fineweb", "rate": 1.0, "style": "PMD"}],
                [{"name": "mtob", "rate": 1.0, "style": "PADDED", "suffix": "grammar"}],
                [{"name": "mtob", "rate": 1.0, "style": "PADDED", "suffix": "enkgv"}],
            ],
            "evaluations": ["fineweb_ppl", "mtob"],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 2048,
            "skip_first_dataset_validation": True,
        }

    elif split_name == "cg_cfq":
        tokens = _scale_tokens([0.5, 0.49, 0.01], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "cfq", "rate": 1.0, "style": "PADDED", "suffix": "sparql"}],
                [{"name": "cfq", "rate": 1.0, "style": "PADDED", "suffix": "text"}],
                [{"name": "cfq", "rate": 1.0, "style": "PADDED"}],
            ],
            "evaluations": ["cfq"],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 2048,
            "skip_first_dataset_validation": False,
        }

    elif split_name == "cg_safety":
        tokens = _scale_tokens([0.80, 0.08, 0.02, 0.08, 0.02], total)
        return {
            "tokens": tokens,
            "datasets": [
                [{"name": "pile_detoxify", "rate": 1.0, "style": "PMD"}],
                [
                    {
                        "name": "harmfulqa",
                        "rate": 0.5,
                        "style": "PADDED",
                        "suffix": "red",
                    },
                    {"name": "mmlu", "rate": 0.5, "style": "PADDED"},
                ],
                [
                    {
                        "name": "harmfulqa",
                        "rate": 1.0,
                        "style": "PADDED",
                        "suffix": "blue",
                    }
                ],
                [
                    {
                        "name": "harmfulqa",
                        "rate": 0.5,
                        "style": "PADDED",
                        "suffix": "red",
                    },
                    {"name": "squad", "rate": 0.5, "style": "PADDED"},
                ],
                [
                    {
                        "name": "harmfulqa",
                        "rate": 1.0,
                        "style": "PADDED",
                        "suffix": "blue",
                    }
                ],
            ],
            "evaluations": ["mmlu", "squad", "pile_ppl"],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 2048,
            "skip_first_dataset_validation": True,
        }

    elif split_name == "ic_injected":
        return {
            "tokens": [total],
            "datasets": [[{"name": "pile_injected", "rate": 1.0, "style": "PMD"}]],
            "evaluations": ["pile_injected_ppl", "pile_ppl"],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 32768,
            "skip_first_dataset_validation": False,
        }

    elif split_name == "ic_longqa":
        return {
            "tokens": [total],
            "datasets": [[{"name": "pile", "rate": 1.0, "style": "PMD"}]],
            "evaluations": ["longhealth", "pile_ppl"],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 32768,
            "skip_first_dataset_validation": False,
        }

    elif split_name == "ic_lengthgen":
        return {
            "tokens": [total],
            "datasets": [[{"name": "fineweb", "rate": 1.0, "style": "PMD"}]],
            "evaluations": [
                "pg19_2k_ppl",
                "pg19_4k_ppl",
                "pg19_8k_ppl",
                "pg19_16k_ppl",
                "pg19_32k_ppl",
                "fineweb_ppl",
            ],
            "fade": {"overlap": 0.0, "curve": "linear"},
            "block_size": 2048,
            "skip_first_dataset_validation": False,
        }

    raise ValueError(f"Unknown split: {split_name}")


# ======================================================================
# Config builder
# ======================================================================


def _apply_block_size_suffix(datasets: list, block_size: int) -> list:
    """Append `_{block_size}` to the suffix of every PADDED dataset entry.

    PADDED datasets are pre-tokenized at a fixed sequence length, so different
    block sizes need different on-disk artifacts. PMD datasets are a flat
    token stream sliced at load time and are block-size agnostic.
    """
    out = []
    for stage in datasets:
        new_stage = []
        for entry in stage:
            if entry.get("style") == "PADDED":
                old = entry.get("suffix", "")
                new_suffix = f"{old}_{block_size}" if old else str(block_size)
                new_entry = {**entry, "suffix": new_suffix}
            else:
                new_entry = dict(entry)
            new_stage.append(new_entry)
        out.append(new_stage)
    return out


def build_config(
    split: str,
    arch: str,
    optim: str,
    schedule: str,
    scale: str,
) -> dict:
    """Build a single YAML config dict."""
    split_cfg = _split_config(split, scale)
    split_cfg["datasets"] = _apply_block_size_suffix(
        split_cfg["datasets"], split_cfg["block_size"]
    )
    arch_spec = ARCH_SPECS[arch][scale]

    # Architecture section
    architecture = {
        "dtype": {"param": "float32", "activation": "bfloat16"},
        "rope": True,
        "vocab_size": 100288,
        "dropout": 0.0,
        "bias": True,
        "block_size": split_cfg["block_size"],
        **arch_spec,
    }
    # Mamba doesn't use rope/bias/n_head
    if arch == "mamba":
        architecture.pop("rope", None)
        architecture.pop("bias", None)

    # Optimization section
    optimization = {
        "lr": 0.0003,
        "weight_decay": 0.1,
        "warmup_pct": 0.01,
        "decay_pct": 0.01,
        "constant_pct": 0.02,
        "schedule": schedule.replace("+", "_"),
        "reset_optimizer": schedule == "wsd_reset",
    }
    if optim == "lora":
        optimization["lora"] = {
            "rank": 16,
            "alpha": 16.0,
            "target_modules": ["kernel"],
        }

    batch_size = 64 if split_cfg["block_size"] == 32768 else 512

    # Training section
    training = {
        "batch_size": batch_size,
        "per_device_batch_size": -1,
        "tokens": split_cfg["tokens"],
        "dataset": split_cfg["datasets"],
        "validation": False,
        "evaluate": True,
        "validation_steps": 2048,
        "skip_first_dataset_validation": split_cfg.get(
            "skip_first_dataset_validation", False
        ),
        "fade": split_cfg["fade"],
    }

    # Eval section
    eval_sec = {"evaluations": split_cfg["evaluations"]}

    # Logging intervals derived from the shortest dataset phase so that
    # even sub-1%-token phases (e.g. cg_grammar's kgv tail) get multiple
    # validations and at least one in-phase checkpoint.
    tokens_per_step = batch_size * split_cfg["block_size"]
    phase_steps = [t // tokens_per_step for t in split_cfg["tokens"]]
    min_phase_steps = max(1, min(phase_steps))
    total_steps = sum(phase_steps)
    logging = {
        "report_interval": 32,
        "checkpoint_interval": max(1, total_steps // 5),
        "validation_interval": max(1, min_phase_steps // 5),
        "wandb": True,
        "plots": {"save": True},
    }

    # Job name
    job = JOB_MAP[(arch, optim)]

    # Request
    request = {"chip": "h200", "min_chips": 4, "n_shards": 1}

    config = {
        "architecture": architecture,
        "optimization": optimization,
        "training": training,
        "eval": eval_sec,
        "logging": logging,
        "job": job,
        "request": request,
    }

    return config


# ======================================================================
# All splits and their valid configurations
# ======================================================================

DS_CG_SPLITS = [
    "ds_nlu",
    "ds_domain",
    "ds_multilingual",
    "cg_grammar",
    "cg_cfq",
    "cg_safety",
]
IC_SPLITS = ["ic_injected", "ic_longqa", "ic_lengthgen"]

ARCHITECTURES = ["transformer", "mamba", "hybrid", "moe"]
OPTIMS = ["full", "lora"]
SCHEDULES_LIST = ["wsd", "cosine_rewarm", "wsd_reset"]
SCALES = ["700m", "1b", "2b"]


def generate_all(output_dir: Path) -> int:
    """Generate all benchmark configs. Returns count of files created."""
    count = 0

    for split in DS_CG_SPLITS:
        for scale in SCALES:
            for arch in ARCHITECTURES:
                for optim in OPTIMS:
                    for schedule in SCHEDULES_LIST:
                        fname = f"{arch}_{optim}_{schedule}.yaml"
                        out_path = output_dir / split / scale / fname
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cfg = build_config(split, arch, optim, schedule, scale)
                        with open(out_path, "w") as f:
                            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                        count += 1

    # IC splits: only full + wsd (no LoRA, no schedule variants)
    for split in IC_SPLITS:
        for scale in SCALES:
            for arch in ARCHITECTURES:
                fname = f"{arch}_full_wsd.yaml"
                out_path = output_dir / split / scale / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cfg = build_config(split, arch, "full", "wsd", scale)
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                count += 1

    return count


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "configs" / "continual"
    count = generate_all(output_dir)
    print(f"Generated {count} config files in {output_dir}")
