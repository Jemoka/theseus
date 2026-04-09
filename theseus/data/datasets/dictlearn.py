"""Dictionary learning dataset.

A synthetic dataset for studying whether models can learn compositions of
random lookup-table functions. Each sample is a space-separated integer
sequence designed for use with ``TrivialTokenizer``.

**Sequence format**::

    f1 f2 ... fn <START> v1 <SEP> fn(fn-1(...f1(v1)...))

where:
- ``f1 ... fn`` are randomly chosen function tokens (1-indexed into a table
  of N_FUNCTIONS random permutation-style lookup tables).
- ``v1`` is a randomly chosen value token.
- The final token is the result of composing the functions left-to-right on v1.
- ``<START>`` and ``<SEP>`` are delimiter tokens.

The model must internalize each function's mapping to predict the output.

Constants (hardcoded for reproducibility):

- ``N_FUNCTIONS = 32`` — number of distinct functions.
- ``FIXED_SEED = 7`` — seed for deterministic generation.
- ``TRAIN_SEQUENCES = 100000`` — number of training sequences.
- ``VAL_SEQUENCES = 500`` — number of validation sequences.

Token layout (for a given ``n_values``)::

    Tokens 1..32                    → function tokens
    Tokens 33..32+n_values          → value tokens
    Token  33+n_values              → START delimiter
    Token  34+n_values              → SEP delimiter
    Token  0                        → EOT (end-of-text)
    VOCAB_SIZE = 35 + n_values

Registered variants (seq_length x n_values):

- ``dictlearn_16``          — length 16, 64 values (default)
- ``dictlearn_16_v{N}``     — length 16, N values
- ``dictlearn_512``         — length 512, 64 values (default)
- ``dictlearn_512_v{N}``    — length 512, N values

where N ∈ {32, 64, 128, 256, 512, 1024}.
"""

from __future__ import annotations

from functools import lru_cache
from random import Random
from typing import ClassVar

from theseus.data.datasets.dataset import StringDataset
from theseus.registry import dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FUNCTIONS = 32
FIXED_SEED = 7
TRAIN_SEQUENCES = 100_000
VAL_SEQUENCES = 500

DEFAULT_N_VALUES = 64

# Sweep values to register
N_VALUES_SWEEP = [32, 64, 128, 256, 512, 1024]

# ---------------------------------------------------------------------------
# Deterministic function tables & sequences
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _build_functions(seed: int, n_values: int) -> tuple[dict[int, int], ...]:
    """Build the N_FUNCTIONS lookup tables from *seed* for *n_values*."""
    value_tokens = [i + N_FUNCTIONS + 1 for i in range(n_values)]
    rng = Random(seed)
    functions: list[dict[int, int]] = []
    for _ in range(N_FUNCTIONS):
        func: dict[int, int] = {}
        for v in value_tokens:
            func[v] = rng.choice(value_tokens)
        functions.append(func)
    return tuple(functions)


@lru_cache(maxsize=None)
def _build_sequences(
    seed: int, count: int, n_values: int, seq_length: int
) -> tuple[tuple[int, ...], ...]:
    """Generate *count* sequences of *seq_length* from *seed*."""
    function_tokens = [i + 1 for i in range(N_FUNCTIONS)]
    value_tokens = [i + N_FUNCTIONS + 1 for i in range(n_values)]
    start_token = N_FUNCTIONS + n_values + 1
    sep_token = start_token + 1

    functions = _build_functions(FIXED_SEED, n_values)
    n_func_tokens = seq_length - 4  # START + v1 + SEP + result
    rng = Random(seed)
    sequences: list[tuple[int, ...]] = []
    for _ in range(count):
        seq: list[int] = []
        for _ in range(n_func_tokens):
            seq.append(rng.choice(function_tokens))

        original_value = rng.choice(value_tokens)
        result = original_value
        for f in seq:
            result = functions[f - 1][result]

        seq.append(start_token)
        seq.append(original_value)
        seq.append(sep_token)
        seq.append(result)
        sequences.append(tuple(seq))
    return tuple(sequences)


def vocab_size(n_values: int) -> int:
    """Return the vocab size for a given n_values."""
    return N_FUNCTIONS + n_values + 3  # functions + values + START + SEP + EOT


def _parse_config(config: str | None) -> int:
    """Parse config string like 'v256' into n_values. None → default."""
    if config is None:
        return DEFAULT_N_VALUES
    if config.startswith("v") and config[1:].isdigit():
        return int(config[1:])
    raise ValueError(
        f"Invalid dictlearn config '{config}', expected 'v{{N}}' e.g. 'v256'"
    )


class _DictLearnBase(StringDataset):
    """Base class for dictlearn variants."""

    _seq_length: ClassVar[int]
    _n_values: ClassVar[int] = DEFAULT_N_VALUES

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        n_values = _parse_config(config) if config else self._n_values
        count = TRAIN_SEQUENCES if split == "train" else VAL_SEQUENCES
        seed = FIXED_SEED + 1 if split == "train" else FIXED_SEED + 2
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split '{split}', expected 'train' or 'val'")
        self._sequences = _build_sequences(seed, count, n_values, self._seq_length)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, indx: int) -> str:
        return " ".join(str(t) for t in self._sequences[indx])


# ---------------------------------------------------------------------------
# Register all variants: dictlearn_{seq_length} and dictlearn_{seq_length}_v{N}
# ---------------------------------------------------------------------------

_SEQ_LENGTHS = [16, 512]
_registry: dict[str, type] = {}


def _make_cls(name: str, seq_length: int, n_values: int) -> type:
    cls = type(
        name,
        (_DictLearnBase,),
        {"_seq_length": seq_length, "_n_values": n_values},
    )
    return cls


for _sl in _SEQ_LENGTHS:
    # Default variant: dictlearn_16, dictlearn_512
    _default_name = f"dictlearn_{_sl}"
    _default_cls = _make_cls(f"DictLearn{_sl}", _sl, DEFAULT_N_VALUES)
    _registry[_default_name] = dataset(_default_name)(_default_cls)

    # Sweep variants: dictlearn_16_v32, dictlearn_16_v256, etc.
    for _nv in N_VALUES_SWEEP:
        _variant_name = f"dictlearn_{_sl}_v{_nv}"
        _variant_cls = _make_cls(f"DictLearn{_sl}V{_nv}", _sl, _nv)
        _registry[_variant_name] = dataset(_variant_name)(_variant_cls)

# Export the most common classes for direct import
DictLearn16 = _registry["dictlearn_16"]
DictLearn512 = _registry["dictlearn_512"]
