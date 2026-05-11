"""
so we can shard easily
"""

from enum import Enum


class Axes(Enum):
    VOCAB = "vocab"
    N_EMBD = "n_embd"
    N_EMBD_OUT = "n_embd_out"
    N_EMBD_FF = "n_embd_ff"
    N_ATTN = "n_attn"
    N_FORK = "n_fork"
    N_SCRATCH = "n_scratch"
    BLOCK_SIZE = "block_size"
    N_SSM = "n_ssm"
    N_FW = "n_fw"
