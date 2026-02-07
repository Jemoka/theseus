from typing import Any, Optional

from theseus.base.axis import Axis
from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.huggingface import HFCompat, LogicalAxes


class Llama(HFCompat):
    id: str = field("architecture/huggingface/model")

    @property
    def sharding(self) -> list[tuple[Axes, Optional[Any]]]:
        return [
            (Axes.VOCAB, None),
            (Axes.BLOCK_SIZE, None),
            (Axes.N_EMBD, None),
            (Axes.N_EMBD_FF, Axis.SHARD),
            (Axes.N_EMBD_OUT, Axis.SHARD),
            (Axes.N_ATTN, Axis.SHARD),
        ]

    @classmethod
    def axes(cls, x: str) -> Optional[LogicalAxes]:
        # First dimension sharded
        if (
            ("q_proj.weight" in x)
            or ("k_proj.weight" in x)
            or ("v_proj.weight" in x)
            or ("gate_proj.weight" in x)
            or ("up_proj.weight" in x)
        ):
            if (
                ("q_proj.weight" in x)
                or ("k_proj.weight" in x)
                or ("v_proj.weight" in x)
            ):
                return (Axes.N_ATTN, Axes.N_EMBD)
            return (Axes.N_EMBD_FF, Axes.N_EMBD)

        # Second dimension sharded
        if "o_proj.weight" in x:
            return (Axes.N_EMBD, Axes.N_ATTN)
        if "down_proj.weight" in x:
            return (Axes.N_EMBD, Axes.N_EMBD_FF)
        if ("lm_head.weight" in x) or ("embed_tokens" in x):
            return (Axes.VOCAB, Axes.N_EMBD_OUT)

        return None
