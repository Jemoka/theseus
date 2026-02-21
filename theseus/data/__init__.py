from typing import Any

from .datasets import DATASETS
from .tokenize import (
    TokenizeBlockwiseDatasetJob,
    TokenizeVariableDatasetJob,
    TokenizeContrastiveDatasetJob,
)
from .tokenizer import (
    Tokenizer,
    TokenizerConfig,
    decode_chat_template,
    encode_chat_template,
    get_chatml_encoder,
    get_tokenizer,
)
from theseus.job import BasicJob

JOBS: dict[str, type[BasicJob[Any]]] = {
    "data/tokenize_blockwise_dataset": TokenizeBlockwiseDatasetJob,
    "data/tokenize_variable_dataset": TokenizeVariableDatasetJob,
    "data/tokenize_contrastive_dataset": TokenizeContrastiveDatasetJob,
}

__all__ = [
    "TokenizeBlockwiseDatasetJob",
    "TokenizeVariableDatasetJob",
    "DATASETS",
    "Tokenizer",
    "TokenizerConfig",
    "get_tokenizer",
    "get_chatml_encoder",
    "encode_chat_template",
    "decode_chat_template",
]
