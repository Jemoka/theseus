#!/usr/bin/env python3
"""Emit a markdown file with one `theseus submit` command per generated config.

The command shape is:
    uv run theseus submit <name>-p1 <stage1.yaml> -s <stage2.yaml> ... -s <training.yaml> \\
      --cluster bonete --chip b200 -n 4 --n_shards 1 \\
      -p continual -g <split>_<scale> training.per_device_batch_size=8

The first positional YAML and every `-s` flag are pipeline stages in order;
tokenization comes first, training is the LAST `-s`. Stages are deduped per
dataset entry (PADDED and PMD both go through their own tokenize config).

Usage:
    uv run python scripts/generate_launch_commands.py
"""

from pathlib import Path

import yaml


CLUSTER = "bonete"
CHIP = "b200"
N_CHIPS = 4
N_SHARDS = 1
PROJECT = "continual"
ENV_FLAGS = ""
EXTRA = "training.per_device_batch_size=8"
DATA_DIR = Path("configs/data/cl100k")

# Per-split stage permutations for `_shuffled` variants. Each value is an index
# permutation applied to both `training.dataset` and `training.tokens`.
#   ds_domain:        fineweb, pes2o                                 -> pes2o, fineweb
#   ds_multilingual:  fr, de, zh                                     -> zh, de, fr
#   cg_safety:        pile_detoxify, red+mmlu, blue, red+squad, blue -> pile_detoxify, red+squad, blue, red+mmlu, blue
SHUFFLE_PERMUTATIONS: dict[str, list[int]] = {
    "ds_domain": [1, 0],
    "ds_multilingual": [2, 1, 0],
    "cg_safety": [0, 3, 2, 1, 4],
}
SHUFFLED_MODELS = ["lact", "moe", "transformer", "mamba"]
SHUFFLED_BASE = "full_wsd"
SHUFFLED_SCALE = "1b"


def _data_config_for(entry: dict) -> str | None:
    """Map a training.dataset entry to its tokenize YAML path (relative to repo).

    PADDED entries have a `_{block_size}` suffix already baked in by the
    benchmark generator; their tokenize config filename mirrors the on-disk
    suffix: configs/data/cl100k/{name}_{suffix}.yaml.

    PMD entries use suffix-agnostic tokenize configs (the PMD stream is a flat
    token buffer, not block-size-specific); we prefer a suffix-specific
    variable tokenize config if it exists, otherwise fall back to the bare
    {name}.yaml. Missing configs return None (caller may flag).
    """
    name = entry["name"]
    suffix = entry.get("suffix", "")
    if suffix:
        cand = DATA_DIR / f"{name}_{suffix}.yaml"
    else:
        cand = DATA_DIR / f"{name}.yaml"
    if cand.exists():
        return str(cand)
    if suffix:
        # Fall back to name-only tokenize config for PMD variants where the
        # suffix-specific file is missing.
        fallback = DATA_DIR / f"{name}.yaml"
        if fallback.exists():
            return str(fallback)
    return None


def _stages_for(training_yaml: Path) -> tuple[list[str], list[str]]:
    """Return (stage_paths, missing_names) for a training config.

    Only PADDED entries become per-run tokenize stages, since their artifacts
    are block-size specific and cheap to materialize. PMD datasets are big
    streaming pretraining corpora (fineweb, pile, ccaligned, etc.) that must
    be tokenized once out-of-band — running them as `-s` stages on every
    training submission would re-check massive corpora on every run.
    """
    cfg = yaml.safe_load(training_yaml.read_text())
    datasets = cfg.get("training", {}).get("dataset", [])
    stages: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for stage in datasets:
        for entry in stage:
            if entry.get("style") != "PADDED":
                continue
            path = _data_config_for(entry)
            if path is None:
                missing.append(
                    f"{entry['name']}"
                    + (f"_{entry['suffix']}" if entry.get("suffix") else "")
                )
                continue
            if path not in seen:
                seen.add(path)
                stages.append(path)
    return stages, missing


def _permute(items: list, perm: list[int]) -> list:
    if sorted(perm) != list(range(len(items))):
        raise ValueError(
            f"permutation {perm} doesn't cover indices 0..{len(items) - 1}"
        )
    return [items[i] for i in perm]


def generate_shuffled_configs(configs_root: Path) -> list[Path]:
    """Materialize `_shuffled` config variants by permuting training stages.

    For each (split, model) in SHUFFLE_PERMUTATIONS x SHUFFLED_MODELS at the
    1b scale, load `{model}_{SHUFFLED_BASE}.yaml`, apply the per-split index
    permutation to `training.dataset` and `training.tokens`, and write to
    `{model}_{SHUFFLED_BASE}_shuffled.yaml`. Returns the list of written paths.
    """
    written: list[Path] = []
    for split, perm in SHUFFLE_PERMUTATIONS.items():
        scale_dir = configs_root / split / SHUFFLED_SCALE
        for model in SHUFFLED_MODELS:
            src = scale_dir / f"{model}_{SHUFFLED_BASE}.yaml"
            if not src.exists():
                print(f"  skip: {src} (not found)")
                continue
            cfg = yaml.safe_load(src.read_text())
            cfg["training"]["dataset"] = _permute(cfg["training"]["dataset"], perm)
            cfg["training"]["tokens"] = _permute(cfg["training"]["tokens"], perm)
            dst = src.with_name(f"{model}_{SHUFFLED_BASE}_shuffled.yaml")
            dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(dst)
    return written


def generate_shuffled_launch_script(repo_root: Path, configs_root: Path) -> Path:
    """Emit a single bash script that submits every shuffled 1b run."""
    out_path = repo_root / "scripts" / "launch_shuffled_1b.sh"
    common = "--cluster bonete --chip b200 -n 4 --n_shards 1 --cpu 32 --mem 512G -p continual"
    extra = "training.per_device_batch_size=2"

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "#",
        "# Launch all 1b `_shuffled` continual runs (lact, moe, transformer, mamba)",
        "# across the splits with permuted stage order:",
        "#   - ds_domain        (pes2o -> fineweb)",
        "#   - ds_multilingual  (zh -> de -> fr)",
        "#   - cg_safety        (pile_detoxify -> red+squad -> blue -> red+mmlu -> blue)",
        "#",
        "# Each job uses:",
        "#   - 4x B200 GPUs on the bonete cluster",
        "#   - 32 CPUs, 512G memory",
        "#   - tensor-parallel shards: 1",
        "#   - per-device batch size: 2",
        "#",
        "# Usage:",
        "#   bash scripts/launch_shuffled_1b.sh        # launch all",
        "#   bash scripts/launch_shuffled_1b.sh --dry  # print commands only",
        "#",
        "# Auto-generated by scripts/generate_launch_commands.py. Do not hand-edit.",
        "",
        "DRY_RUN=false",
        'if [[ "${1:-}" == "--dry" ]]; then',
        "    DRY_RUN=true",
        '    echo "=== DRY RUN — commands will be printed but not executed ==="',
        "    echo",
        "fi",
        "",
        f'COMMON="{common}"',
        f'EXTRA="{extra}"',
        "",
        "run_cmd() {",
        '    echo "+ $*"',
        "    if ! $DRY_RUN; then",
        '        "$@"',
        "    fi",
        "    echo",
        "}",
        "",
    ]

    total = 0
    missing: set[str] = set()
    for split in SHUFFLE_PERMUTATIONS:
        for model in SHUFFLED_MODELS:
            cfg_path = (
                configs_root
                / split
                / SHUFFLED_SCALE
                / f"{model}_{SHUFFLED_BASE}_shuffled.yaml"
            )
            if not cfg_path.exists():
                continue
            data_stages, miss = _stages_for(cfg_path)
            missing.update(miss)
            rel_training = cfg_path.relative_to(repo_root)
            full_stages = data_stages + [str(rel_training)]
            name = cfg_path.stem
            group = f"{split}_{SHUFFLED_SCALE}"

            lines.append(f"# {split} / {model} (shuffled)")
            cmd = [f"run_cmd uv run theseus submit {name}-p1 \\"]
            for i, stage in enumerate(full_stages):
                prefix = "    " if i == 0 else "    -s "
                cmd.append(f"{prefix}{stage} \\")
            cmd.append(f"    $COMMON -g {group} $EXTRA")
            lines.extend(cmd)
            lines.append("")
            total += 1

    lines.append(f'echo "=== All {total} shuffled 1b jobs submitted ==="')
    out_path.write_text("\n".join(lines) + "\n")
    out_path.chmod(0o755)
    if missing:
        print(
            f"  ({len(missing)} unique missing tokenize configs for shuffled: {sorted(missing)})"
        )
    return out_path


def main() -> None:
    repo_root = Path(__file__).parent.parent
    configs_root = repo_root / "configs" / "continual"
    out_path = repo_root / "docs" / "launch_commands.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shuffled = generate_shuffled_configs(configs_root)
    print(f"Wrote {len(shuffled)} shuffled configs")

    sh_path = generate_shuffled_launch_script(repo_root, configs_root)
    print(f"Wrote shuffled launch script to {sh_path}")

    lines: list[str] = [
        "# Launch commands",
        "",
        "Auto-generated by `scripts/generate_launch_commands.py`. Do not hand-edit.",
        "",
        "Stage order is `<first_stage> -s <stage_2> ... -s <training>`. The last",
        "`-s` is the training config; everything before it is tokenization.",
        "",
    ]

    all_missing: set[str] = set()
    total = 0

    splits = sorted(p.name for p in configs_root.iterdir() if p.is_dir())
    for split in splits:
        split_dir = configs_root / split
        scales = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        for scale in scales:
            scale_dir = split_dir / scale
            group = f"{split}_{scale}"
            lines.append(f"## {split} / {scale}")
            lines.append("")
            lines.append("```bash")
            for cfg_path in sorted(scale_dir.glob("*.yaml")):
                name = cfg_path.stem
                rel_training = cfg_path.relative_to(repo_root)
                data_stages, missing = _stages_for(cfg_path)
                all_missing.update(missing)

                # Full stage list, training last.
                full_stages = [
                    str(rel_training.parent.parent.parent.parent / s) if False else s
                    for s in data_stages
                ]
                full_stages = data_stages + [str(rel_training)]

                first = full_stages[0]
                rest = full_stages[1:]
                rest_flags = " ".join(f"-s {s}" for s in rest)
                rest_part = f" {rest_flags}" if rest_flags else ""
                env_part = f" {ENV_FLAGS}" if ENV_FLAGS else ""
                lines.append(
                    f"uv run theseus submit {name}-p1 {first}{rest_part}"
                    f"{env_part} --cluster {CLUSTER} --chip {CHIP} "
                    f"-n {N_CHIPS} --n_shards {N_SHARDS} "
                    f"--cpu 32 --mem 256G "
                    f"-p {PROJECT} -g {group} {EXTRA}"
                )
                total += 1
            lines.append("```")
            lines.append("")

    if all_missing:
        lines.append("## Missing tokenize configs")
        lines.append("")
        lines.append(
            "The following dataset entries had no matching tokenize YAML under "
            "`configs/data/cl100k/`. Launch commands fell back to the bare "
            "`{name}.yaml` where available, but these are worth auditing:"
        )
        lines.append("")
        for m in sorted(all_missing):
            lines.append(f"- `{m}`")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {total} launch commands to {out_path}")
    if all_missing:
        print(f"  ({len(all_missing)} unique missing tokenize configs — see doc)")


if __name__ == "__main__":
    main()
