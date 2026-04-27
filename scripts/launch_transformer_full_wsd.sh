#!/usr/bin/env bash
#
# Launch all 1b full_wsd jobs: transformer first, then moe, then mamba.
# Submission order = queue order, so architecture priority is respected.
#
# Each job uses:
#   - 4x B200 GPUs on the bonete cluster
#   - 32 CPUs, 256G memory
#   - tensor-parallel shards: 1
#   - per-device batch size: 8
#
# Usage:
#   bash scripts/launch_transformer_full_wsd.sh        # launch all
#   bash scripts/launch_transformer_full_wsd.sh --dry   # print commands only
#

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not executed ==="
    echo
fi

COMMON="--cluster bonete --chip b200 -n 4 --n_shards 1 --cpu 32 --mem 256G -p continual"
EXTRA="training.per_device_batch_size=8"

run_cmd() {
    echo "+ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
    echo
}

# =============================================================
#  TRANSFORMER (submitted first — highest queue priority)
# =============================================================

# cg_cfq: compositional generalization (CFQ)
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/data/cl100k/cfq_sparql_2048.yaml \
    -s configs/data/cl100k/cfq_text_2048.yaml \
    -s configs/data/cl100k/cfq_2048.yaml \
    -s configs/continual/cg_cfq/1b/transformer_full_wsd.yaml \
    $COMMON -g cg_cfq_1b $EXTRA

# cg_grammar: compositional generalization (grammar)
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/data/cl100k/mtob_grammar_2048.yaml \
    -s configs/data/cl100k/mtob_enkgv_2048.yaml \
    -s configs/continual/cg_grammar/1b/transformer_full_wsd.yaml \
    $COMMON -g cg_grammar_1b $EXTRA

# cg_safety: safety (harmfulqa + mmlu + squad)
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/data/cl100k/harmfulqa_red_2048.yaml \
    -s configs/data/cl100k/mmlu_2048.yaml \
    -s configs/data/cl100k/harmfulqa_blue_2048.yaml \
    -s configs/data/cl100k/squad_2048.yaml \
    -s configs/continual/cg_safety/1b/transformer_full_wsd.yaml \
    $COMMON -g cg_safety_1b $EXTRA

# ds_domain: domain shift (self-contained config)
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/continual/ds_domain/1b/transformer_full_wsd.yaml \
    $COMMON -g ds_domain_1b $EXTRA

# ds_multilingual: multilingual domain shift
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/continual/ds_multilingual/1b/transformer_full_wsd.yaml \
    $COMMON -g ds_multilingual_1b $EXTRA

# ds_nlu: NLU domain shift
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/data/cl100k/mnli_2048.yaml \
    -s configs/data/cl100k/qqp_2048.yaml \
    -s configs/data/cl100k/sst2_2048.yaml \
    -s configs/data/cl100k/siqa_2048.yaml \
    -s configs/continual/ds_nlu/1b/transformer_full_wsd.yaml \
    $COMMON -g ds_nlu_1b $EXTRA

# ic_injected: in-context injected
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/continual/ic_injected/1b/transformer_full_wsd.yaml \
    $COMMON -g ic_injected_1b $EXTRA

# ic_lengthgen: in-context length generalization
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/continual/ic_lengthgen/1b/transformer_full_wsd.yaml \
    $COMMON -g ic_lengthgen_1b $EXTRA

# ic_longqa: in-context long QA
run_cmd uv run theseus submit transformer_full_wsd-p1 \
    configs/continual/ic_longqa/1b/transformer_full_wsd.yaml \
    $COMMON -g ic_longqa_1b $EXTRA

# =============================================================
#  MOE (submitted second)
# =============================================================

# cg_cfq
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/data/cl100k/cfq_sparql_2048.yaml \
    -s configs/data/cl100k/cfq_text_2048.yaml \
    -s configs/data/cl100k/cfq_2048.yaml \
    -s configs/continual/cg_cfq/1b/moe_full_wsd.yaml \
    $COMMON -g cg_cfq_1b $EXTRA

# cg_grammar
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/data/cl100k/mtob_grammar_2048.yaml \
    -s configs/data/cl100k/mtob_enkgv_2048.yaml \
    -s configs/continual/cg_grammar/1b/moe_full_wsd.yaml \
    $COMMON -g cg_grammar_1b $EXTRA

# cg_safety
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/data/cl100k/harmfulqa_red_2048.yaml \
    -s configs/data/cl100k/mmlu_2048.yaml \
    -s configs/data/cl100k/harmfulqa_blue_2048.yaml \
    -s configs/data/cl100k/squad_2048.yaml \
    -s configs/continual/cg_safety/1b/moe_full_wsd.yaml \
    $COMMON -g cg_safety_1b $EXTRA

# ds_domain
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/continual/ds_domain/1b/moe_full_wsd.yaml \
    $COMMON -g ds_domain_1b $EXTRA

# ds_multilingual
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/continual/ds_multilingual/1b/moe_full_wsd.yaml \
    $COMMON -g ds_multilingual_1b $EXTRA

# ds_nlu
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/data/cl100k/mnli_2048.yaml \
    -s configs/data/cl100k/qqp_2048.yaml \
    -s configs/data/cl100k/sst2_2048.yaml \
    -s configs/data/cl100k/siqa_2048.yaml \
    -s configs/continual/ds_nlu/1b/moe_full_wsd.yaml \
    $COMMON -g ds_nlu_1b $EXTRA

# ic_injected
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/continual/ic_injected/1b/moe_full_wsd.yaml \
    $COMMON -g ic_injected_1b $EXTRA

# ic_lengthgen
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/continual/ic_lengthgen/1b/moe_full_wsd.yaml \
    $COMMON -g ic_lengthgen_1b $EXTRA

# ic_longqa
run_cmd uv run theseus submit moe_full_wsd-p1 \
    configs/continual/ic_longqa/1b/moe_full_wsd.yaml \
    $COMMON -g ic_longqa_1b $EXTRA

# =============================================================
#  MAMBA (submitted last)
# =============================================================

# cg_cfq
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/data/cl100k/cfq_sparql_2048.yaml \
    -s configs/data/cl100k/cfq_text_2048.yaml \
    -s configs/data/cl100k/cfq_2048.yaml \
    -s configs/continual/cg_cfq/1b/mamba_full_wsd.yaml \
    $COMMON -g cg_cfq_1b $EXTRA

# cg_grammar
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/data/cl100k/mtob_grammar_2048.yaml \
    -s configs/data/cl100k/mtob_enkgv_2048.yaml \
    -s configs/continual/cg_grammar/1b/mamba_full_wsd.yaml \
    $COMMON -g cg_grammar_1b $EXTRA

# cg_safety
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/data/cl100k/harmfulqa_red_2048.yaml \
    -s configs/data/cl100k/mmlu_2048.yaml \
    -s configs/data/cl100k/harmfulqa_blue_2048.yaml \
    -s configs/data/cl100k/squad_2048.yaml \
    -s configs/continual/cg_safety/1b/mamba_full_wsd.yaml \
    $COMMON -g cg_safety_1b $EXTRA

# ds_domain
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/continual/ds_domain/1b/mamba_full_wsd.yaml \
    $COMMON -g ds_domain_1b $EXTRA

# ds_multilingual
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/continual/ds_multilingual/1b/mamba_full_wsd.yaml \
    $COMMON -g ds_multilingual_1b $EXTRA

# ds_nlu
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/data/cl100k/mnli_2048.yaml \
    -s configs/data/cl100k/qqp_2048.yaml \
    -s configs/data/cl100k/sst2_2048.yaml \
    -s configs/data/cl100k/siqa_2048.yaml \
    -s configs/continual/ds_nlu/1b/mamba_full_wsd.yaml \
    $COMMON -g ds_nlu_1b $EXTRA

# ic_injected
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/continual/ic_injected/1b/mamba_full_wsd.yaml \
    $COMMON -g ic_injected_1b $EXTRA

# ic_lengthgen
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/continual/ic_lengthgen/1b/mamba_full_wsd.yaml \
    $COMMON -g ic_lengthgen_1b $EXTRA

# ic_longqa
run_cmd uv run theseus submit mamba_full_wsd-p1 \
    configs/continual/ic_longqa/1b/mamba_full_wsd.yaml \
    $COMMON -g ic_longqa_1b $EXTRA

echo "=== All 27 jobs submitted (9 transformer → 9 moe → 9 mamba) ==="
