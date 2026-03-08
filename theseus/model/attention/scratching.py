from typing import Optional, Any

import jax
import flax.linen as nn

from theseus.model.attention.forking import ForkingAttention


class ScratchSparseCrossAttention(ForkingAttention):
    def setup(self) -> None:
        # this doesn't actually need more parameters
        # since we are giving it the q,k,v to attend with
        ...

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        token_index: jax.Array = kwargs.get("query_token_index")  # type: ignore
        if kwargs.get("token_index") is not None and token_index is not None:
            jax.debug.print(
                "token index shape: {}, query_token_index shpe: {}",
                kwargs["token_index"].shape,
                token_index.shape,
            )
            del kwargs["token_index"]
        return super().postprocess_attn(
            y, padding_mask, deterministic, token_index=token_index, **kwargs
        )

    @nn.compact
    def __call__(  # type: ignore[override]
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        # we then do a normal attention operation
        # each of these should be a single "head"
        y = self.attn(
            q[:, :, None, :], k[:, :, None, :], v[:, :, None, :], padding_mask, **kwargs
        )
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        return y[:, :, 0, :]  # get rid of the additional head channel
