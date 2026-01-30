from typing import Any

from .datasets import DATASETS
from .tokenize import TokenizeBlockwiseDatasetJob, TokenizeVariableDatasetJob
from .tokenizer import get_chatml_encoder, encode_chat_template, decode_chat_template
from theseus.job import BasicJob

JOBS: dict[str, type[BasicJob[Any]]] = {
    "data/tokenize_blockwise_dataset": TokenizeBlockwiseDatasetJob,
    "data/tokenize_variable_dataset": TokenizeVariableDatasetJob,
}

__all__ = [
    "TokenizeBlockwiseDatasetJob",
    "TokenizeVariableDatasetJob",
    "DATASETS",
    "get_chatml_encoder",
    "encode_chat_template",
    "decode_chat_template",
]
