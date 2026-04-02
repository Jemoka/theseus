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
- ``N_VALUES = 1024`` — size of the value domain.
- ``FIXED_SEED = 7`` — seed for deterministic generation.
- ``MAX_SEQ_LENGTH = 512`` — total tokens per sequence.
- ``TRAIN_SEQUENCES = 100000`` — number of training sequences.
- ``VAL_SEQUENCES = 500`` — number of validation sequences.

Token layout::

    Tokens 1..32         → function tokens
    Tokens 33..1056      → value tokens
    Token  1057          → START delimiter
    Token  1058          → SEP delimiter
    Token  0             → EOT (end-of-text)
    VOCAB_SIZE = 1059
"""

from random import Random

from theseus.data.datasets.dataset import StringDataset
from theseus.registry import dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FUNCTIONS = 32
N_VALUES = 1024
FIXED_SEED = 7
MAX_SEQ_LENGTH = 512
TRAIN_SEQUENCES = 100_000
VAL_SEQUENCES = 500

# Token ranges
FUNCTION_TOKENS = [i + 1 for i in range(N_FUNCTIONS)]
VALUE_TOKENS = [i + N_FUNCTIONS + 1 for i in range(N_VALUES)]

# Special tokens
START_TOKEN = N_FUNCTIONS + N_VALUES + 1
SEP_TOKEN = START_TOKEN + 1
EOT_TOKEN = 0

VOCAB_SIZE = SEP_TOKEN + 1

# Number of function tokens per sequence so that total = MAX_SEQ_LENGTH
# Layout: n_funcs + START + v1 + SEP + result = MAX_SEQ_LENGTH
N_FUNC_TOKENS_PER_SEQ = MAX_SEQ_LENGTH - 4

# ---------------------------------------------------------------------------
# Deterministic function tables & sequences
#
# All RNG usage is contained inside pure functions that create their own
# Random from an explicit seed.  No module-level RNG objects exist, so the
# results are immune to code reordering, import-order changes, or any other
# module-level side-effect interleaving.
# ---------------------------------------------------------------------------


def _build_functions(seed: int) -> list[dict[int, int]]:
    """Build the N_FUNCTIONS lookup tables from *seed*."""
    rng = Random(seed)
    functions: list[dict[int, int]] = []
    for _ in range(N_FUNCTIONS):
        func: dict[int, int] = {}
        for v in VALUE_TOKENS:
            func[v] = rng.choice(VALUE_TOKENS)
        functions.append(func)
    return functions


def _build_sequences(
    seed: int, count: int, functions: list[dict[int, int]]
) -> list[list[int]]:
    """Generate *count* sequences from *seed* using the given function tables."""
    rng = Random(seed)
    sequences: list[list[int]] = []
    for _ in range(count):
        seq: list[int] = []
        for _ in range(N_FUNC_TOKENS_PER_SEQ):
            seq.append(rng.choice(FUNCTION_TOKENS))

        original_value = rng.choice(VALUE_TOKENS)
        result = original_value
        for f in seq:
            result = functions[f - 1][result]

        seq.append(START_TOKEN)
        seq.append(original_value)
        seq.append(SEP_TOKEN)
        seq.append(result)
        sequences.append(seq)
    return sequences


FUNCTIONS = _build_functions(FIXED_SEED)
_TRAIN = _build_sequences(FIXED_SEED + 1, TRAIN_SEQUENCES, FUNCTIONS)
_VAL = _build_sequences(FIXED_SEED + 2, VAL_SEQUENCES, FUNCTIONS)

_SPLITS: dict[str, list[list[int]]] = {"train": _TRAIN, "val": _VAL}


@dataset("dictlearn")
class DictLearn(StringDataset):
    """Dictionary-learning training dataset (see module docstring)."""

    def __init__(self, split: str = "train", config: str | None = None) -> None:
        del config
        if split not in _SPLITS:
            raise ValueError(f"Unknown split '{split}', expected 'train' or 'val'")
        self._sequences = _SPLITS[split]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, indx: int) -> str:
        return " ".join(str(t) for t in self._sequences[indx])
