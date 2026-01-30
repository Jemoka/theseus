from .prepare import TokenizeBlockwiseDatasetJob, TokenizeVariableDatasetJob
from .datasets import DATASETS
from .tokenizer import get_chatml_encoder, encode_chat_template, decode_chat_template

JOBS = {
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
