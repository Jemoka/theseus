from jax import lax
import jax.numpy as jnp
from typing import Tuple, Optional


class RotaryPosEncoding:
    """RoPE implementation by Róbert Csordás, converted to JAX"""

    def __init__(self, d_model: int, base: int = 10000, seq_dim: int = 1):
        if d_model % 2 != 0:
            raise ValueError("RoPE can only be used with an even number of dimensions")

        self.d_model = d_model
        self.base = base
        self.seq_dim = seq_dim

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, d_model, 2, dtype=jnp.float32) / d_model)
        )
        self.inv_freq = inv_freq

    def rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rot(
        self,
        x: jnp.ndarray,
        sinp: jnp.ndarray,
        cosp: jnp.ndarray,
        seq_dim: int,
        offset: int,
    ) -> jnp.ndarray:
        sin = lax.dynamic_slice_in_dim(sinp, offset, x.shape[seq_dim], axis=seq_dim)
        cos = lax.dynamic_slice_in_dim(cosp, offset, x.shape[seq_dim], axis=seq_dim)
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        sin: jnp.ndarray,
        cos: jnp.ndarray,
        seq_dim: int,
        offset: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(
            k, sin, cos, seq_dim, 0
        )

    def get(
        self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = x.shape[self.seq_dim]

        if t is None:
            t = jnp.arange(x.shape[self.seq_dim], dtype=jnp.float32)

        t = t.astype(self.inv_freq.dtype)

        freqs = jnp.einsum("...i,j->...ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        tgt_shape = [1] * x.ndim
        tgt_shape[self.seq_dim] = seq_len
        tgt_shape[-1] = x.shape[-1]

        # support batch
        tgt_shape[0] = -1

        cos = jnp.cos(emb).reshape(tgt_shape)
        sin = jnp.sin(emb).reshape(tgt_shape)

        return sin, cos

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        pos_offset: int = 0,
        t: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sin, cos = self.get(k, t)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)


class QwenRotaryPosEncoding(RotaryPosEncoding):
    """Qwen RoPE with HF parity (theta default 1e6)."""

    def __init__(self, d_model: int, base: int = 1000000, seq_dim: int = 1):
        super().__init__(d_model=d_model, base=base, seq_dim=seq_dim)

    def get(
        self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (B, T) positions
        positions = x if t is None else t
        if positions.ndim == 1:
            positions = positions[None, :]
        # positions: (B, T)
        freqs = positions[..., None] * self.inv_freq[None, None, :]
        emb = jnp.concatenate((freqs, freqs), axis=-1)  # (B,T,D)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return sin, cos

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        pos_offset: int = 0,
        t: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # q: (B,T,H,D)  k: (B,T,K,D)
        b, tlen, h, d = q.shape
        kvh = k.shape[2]
        if t is None:
            positions = jnp.arange(tlen, dtype=jnp.int32)[None, :].repeat(b, axis=0)
        else:
            positions = t
            if positions.ndim == 1:
                positions = positions[None, :].repeat(b, axis=0)
        positions = positions + pos_offset
        sin, cos = self.get(positions)
        # Broadcast separately for q and k to avoid head misalignment when kv < q heads
        sin_q = sin[:, :, None, :]  # (B,T,1,D)
        cos_q = cos[:, :, None, :]
        sin_k = sin[:, :, None, :].repeat(kvh, axis=2) if kvh != h else sin_q
        cos_k = cos[:, :, None, :].repeat(kvh, axis=2) if kvh != h else cos_q
        q_out = (q * cos_q) + (self.rotate_half(q) * sin_q)
        k_out = (k * cos_k) + (self.rotate_half(k) * sin_k)
        return q_out, k_out
