#!/usr/bin/env bash
#
# Launch lact_full_wsd jobs for lengthgen, multilingual, and domain shift.
#
# Usage:
#   bash scripts/launch_lact_subset_wsd.sh        # launch all
#   bash scripts/launch_lact_subset_wsd.sh --dry   # print commands only
#

#set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not executed ==="
    echo
fi

COMMON="--cluster bonete --chip b200 -n 4 --n_shards 1 --cpu 32 --mem 512G -p continual"
EXTRA="training.per_device_batch_size=2"

run_cmd() {
    echo "+ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
    echo
}

# ds_domain: domain shift (self-contained config)
run_cmd uv run theseus submit lact_full_wsd-p1 \
    configs/continual/ds_domain/1b/lact_full_wsd.yaml \
    $COMMON -g ds_domain_1b $EXTRA

# ds_multilingual: multilingual domain shift
run_cmd uv run theseus submit lact_full_wsd-p1 \
    configs/continual/ds_multilingual/1b/lact_full_wsd.yaml \
    $COMMON -g ds_multilingual_1b $EXTRA

# ic_lengthgen: in-context length generalization
run_cmd uv run theseus submit lact_full_wsd-p1 \
    configs/continual/ic_lengthgen/1b/lact_full_wsd.yaml \
    $COMMON -g ic_lengthgen_1b $EXTRA

echo "=== 3 lact jobs submitted ==="
