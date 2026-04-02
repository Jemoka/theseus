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
    _TRAIN,
    _VAL,
)
from theseus.evaluation.datasets.dictlearn import DictLearnEval

# -- Regenerate everything from scratch --
funcs_fresh = _build_functions(FIXED_SEED)
train_fresh = _build_sequences(FIXED_SEED + 1, TRAIN_SEQUENCES, funcs_fresh)
val_fresh = _build_sequences(FIXED_SEED + 2, VAL_SEQUENCES, funcs_fresh)

# -- Compare against module-level singletons --
assert FUNCTIONS == funcs_fresh, "Function tables are not deterministic"
assert _TRAIN == train_fresh, "Training sequences are not deterministic"
assert _VAL == val_fresh, "Val sequences are not deterministic"

# -- Verify eval uses the val split --
ev = DictLearnEval()
assert len(ev) == VAL_SEQUENCES, (
    f"Eval length {len(ev)} != VAL_SEQUENCES {VAL_SEQUENCES}"
)
for i in range(len(ev)):
    expected = " ".join(str(t) for t in _VAL[i])
    assert ev.get(i) == expected, f"Eval sample {i} does not match val split"

# -- Verify train/val disjointness --
train_set = {tuple(s) for s in _TRAIN}
val_overlap = sum(1 for s in _VAL if tuple(s) in train_set)
assert val_overlap == 0, f"{val_overlap} val sequences found in training set"

# -- Fingerprint for cross-platform comparison --
fingerprint = hashlib.sha256(json.dumps(_TRAIN[:100]).encode()).hexdigest()[:32]

print(f"Function tables:      OK ({len(FUNCTIONS)} functions)")
print(f"Training sequences:   OK ({len(_TRAIN)} sequences)")
print(f"Val sequences:        OK ({len(_VAL)} sequences)")
print("Eval == val split:    OK")
print("Train/val overlap:    0")
print(f"Fingerprint (first 100 train seqs): {fingerprint}")
