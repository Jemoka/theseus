"""Verify that the dictlearn dataset is fully deterministic.

Regenerates function tables, training sequences, and val sequences from
scratch and compares them against the module-level singletons.  Any drift
(due to code reordering, Python version changes, etc.) will cause an
assertion failure.

Usage:
    uv run python scripts/verify_dictlearn_determinism.py
"""

import hashlib
import json

from theseus.data.datasets.dictlearn import (
    FIXED_SEED,
    FUNCTIONS,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
    _build_functions,
    _build_sequences,
    _TRAIN_16,
    _VAL_16,
    _TRAIN_512,
    _VAL_512,
)
from theseus.evaluation.datasets.dictlearn import DictLearnEval16, DictLearnEval512

# -- Regenerate everything from scratch --
funcs_fresh = _build_functions(FIXED_SEED)
train_16_fresh = _build_sequences(FIXED_SEED + 1, TRAIN_SEQUENCES, funcs_fresh, 16)
val_16_fresh = _build_sequences(FIXED_SEED + 2, VAL_SEQUENCES, funcs_fresh, 16)
train_512_fresh = _build_sequences(FIXED_SEED + 1, TRAIN_SEQUENCES, funcs_fresh, 512)
val_512_fresh = _build_sequences(FIXED_SEED + 2, VAL_SEQUENCES, funcs_fresh, 512)

# -- Compare against module-level singletons --
assert FUNCTIONS == funcs_fresh, "Function tables are not deterministic"
assert _TRAIN_16 == train_16_fresh, "Training sequences (16) are not deterministic"
assert _VAL_16 == val_16_fresh, "Val sequences (16) are not deterministic"
assert _TRAIN_512 == train_512_fresh, "Training sequences (512) are not deterministic"
assert _VAL_512 == val_512_fresh, "Val sequences (512) are not deterministic"

# -- Verify evals use the val splits --
for ev, val_seqs, label in [
    (DictLearnEval16(), _VAL_16, "16"),
    (DictLearnEval512(), _VAL_512, "512"),
]:
    assert len(ev) == VAL_SEQUENCES, (
        f"Eval ({label}) length {len(ev)} != VAL_SEQUENCES {VAL_SEQUENCES}"
    )
    for i in range(len(ev)):
        expected = " ".join(str(t) for t in val_seqs[i])
        assert ev.get(i) == expected, f"Eval ({label}) sample {i} does not match"

# -- Verify train/val disjointness --
for train_seqs, val_seqs, label in [
    (_TRAIN_16, _VAL_16, "16"),
    (_TRAIN_512, _VAL_512, "512"),
]:
    train_set = {tuple(s) for s in train_seqs}
    val_overlap = sum(1 for s in val_seqs if tuple(s) in train_set)
    assert val_overlap == 0, f"{val_overlap} val sequences ({label}) found in training"

# -- Fingerprints --
fp_16 = hashlib.sha256(json.dumps(_TRAIN_16[:100]).encode()).hexdigest()[:32]
fp_512 = hashlib.sha256(json.dumps(_TRAIN_512[:100]).encode()).hexdigest()[:32]

print(f"Function tables:        OK ({len(FUNCTIONS)} functions)")
print(f"Training sequences 16:  OK ({len(_TRAIN_16)} sequences)")
print(f"Val sequences 16:       OK ({len(_VAL_16)} sequences)")
print(f"Training sequences 512: OK ({len(_TRAIN_512)} sequences)")
print(f"Val sequences 512:      OK ({len(_VAL_512)} sequences)")
print("Eval == val splits:     OK")
print("Train/val overlap:      0")
print(f"Fingerprint 16  (first 100): {fp_16}")
print(f"Fingerprint 512 (first 100): {fp_512}")
