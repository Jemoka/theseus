from typing import Optional

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.huggingface import HFCompat, LogicalAxes


class Llama(HFCompat):
    id: str = field("architecture/model_id")

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
