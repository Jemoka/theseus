from typing import Any, Optional, List, Tuple, Type

import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.model.block.qwen import QwenDecoderBlock
from theseus.model.layers import RMSNorm

try:  # optional torch import for export
    import torch
except Exception:
    torch = None
from theseus.base.axis import Axis


class Qwen(Module):
    n_layers: int = field("architecture/n_layers", default=32)
    n_embd: int = field("architecture/n_embd", default=4096)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)
    intermediate_size: int = field("architecture/intermediate_size", default=22016)
    rope_theta: float = field("architecture/rope_theta", default=1e6)
    rms_norm_eps: float = field("architecture/rms_norm_eps", default=1e-6)
    block_size: int = field("architecture/block_size", default=32768)
    vocab_size: int = field("architecture/vocab_size", default=151936)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    use_sliding_window: bool = field("architecture/use_sliding_window", default=False)
    sliding_window: int = field("architecture/sliding_window", default=-1)
    max_window_layers: int = field("architecture/max_window_layers", default=28)
    bias: bool = field("architecture/bias", default=False)

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.BLOCK_SIZE.value, None),
            (Axes.N_EMBD.value, None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_EMBD_OUT.value, Axis.SHARD),
            (Axes.N_ATTN.value, Axis.SHARD),
        ]

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [QwenDecoderBlock, RMSNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None
        n_kv_head = self.n_head if self.n_kv_head == -1 else self.n_kv_head

        self.wte: Any = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )

        self.drop = nn.Dropout(rate=self.dropout)

        n_kv_head = self.n_head if self.n_kv_head == -1 else self.n_kv_head
        self.layer_types = [
            "sliding"
            if (
                self.use_sliding_window
                and self.sliding_window > 0
                and i >= self.max_window_layers
            )
            else "full"
            for i in range(self.n_layers)
        ]

        self.blocks = [
            QwenDecoderBlock(
                n_layers=self.n_layers,
                n_embd=self.n_embd,
                n_head=self.n_head,
                n_kv_head=n_kv_head,
                intermediate_size=self.intermediate_size,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                rope_theta=self.rope_theta,
                rms_norm_eps=self.rms_norm_eps,
                use_sliding_window=self.use_sliding_window,
                sliding_window=self.sliding_window,
                bias=self.bias,
            )
            for _ in range(self.n_layers)
        ]
        self.ln_f = RMSNorm(ndim=self.n_embd, eps=self.rms_norm_eps)

    def embed(self, idx: jax.Array, deterministic: bool = False) -> Any:
        x = jnp.take(self.wte, idx, axis=0).astype(jnp.float32)
        x = self.drop(x, deterministic=deterministic)
        return x

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        b, t, _ = x.shape
        positions = jnp.arange(t)
        for i, block in enumerate(self.blocks):
            sliding = self.layer_types[i] == "sliding"
            mask = None
            if padding_mask is not None:
                mask = padding_mask
            x = block(
                x,
                padding_mask=mask,
                deterministic=deterministic,
                sliding=sliding,
                positions=positions,
            )
        return x

    def unembed(self, x: jax.Array) -> Any:
        x = self.ln_f(x)
        x_f32 = x.astype(jnp.float32)
        wte = jnp.asarray(self.wte, dtype=jnp.float32)
        logits = jnp.einsum("bth,vh->btv", x_f32, wte)
        return logits

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)
        loss = -jnp.sum(
            jnn.log_softmax(logits_flat, axis=-1)
            * jnn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)
        return loss

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        x = self.embed(idx, deterministic)
        x = self.decode(x, padding_mask=padding_mask, deterministic=deterministic)
        logits = self.unembed(x)
        loss_val = self.loss(logits, targets) if targets is not None else None
        return logits, loss_val

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cpu") -> Any:
        import torch
        from transformers import Qwen2ForCausalLM

        torch_dtype = torch.float32
        hf_model = Qwen2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=None
        )
        hf_model.to(device)
        hf_model.eval()
        cfg = hf_model.config

        rope_theta = 10000.0
        if cfg.rope_parameters is not None and "rope_theta" in cfg.rope_parameters:
            rope_theta = cfg.rope_parameters["rope_theta"]

        model = cls(
            n_layers=cfg.num_hidden_layers,
            n_embd=cfg.hidden_size,
            n_head=cfg.num_attention_heads,
            n_kv_head=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            block_size=cfg.max_position_embeddings,
            vocab_size=cfg.vocab_size,
            dropout=0.0,
            attn_dropout=cfg.attention_dropout,
            rope_theta=rope_theta,
            rms_norm_eps=cfg.rms_norm_eps,
            use_sliding_window=cfg.use_sliding_window,
            sliding_window=cfg.sliding_window,
            max_window_layers=cfg.max_window_layers,
            bias=True,
        )

        dummy = jnp.zeros((1, 1), dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), dummy)["params"]
        params = _from_hf_state_dict(params, hf_model.state_dict(), model.n_layers)
        return params


def _from_hf_state_dict(params: Any, state_dict: Any, n_layers: int) -> Any:
    from flax.core import freeze, unfreeze

    p = unfreeze(params)

    def assign(path: list[str], array: Any) -> None:
        cur = p
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = array

    # Embeddings
    assign(["wte"], state_dict["model.embed_tokens.weight"].cpu().float().numpy())

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"
        # Norms
        assign(
            [block_key, "rms_1", "weight"],
            state_dict[prefix + "input_layernorm.weight"].cpu().float().numpy(),
        )
        assign(
            [block_key, "rms_2", "weight"],
            state_dict[prefix + "post_attention_layernorm.weight"]
            .cpu()
            .float()
            .numpy(),
        )

        # Attention projections
        q_w = state_dict[prefix + "self_attn.q_proj.weight"].cpu().float().numpy().T
        q_b = state_dict[prefix + "self_attn.q_proj.bias"].cpu().float().numpy()
        k_w = state_dict[prefix + "self_attn.k_proj.weight"].cpu().float().numpy().T
        k_b = state_dict[prefix + "self_attn.k_proj.bias"].cpu().float().numpy()
        v_w = state_dict[prefix + "self_attn.v_proj.weight"].cpu().float().numpy().T
        v_b = state_dict[prefix + "self_attn.v_proj.bias"].cpu().float().numpy()
        o_w = state_dict[prefix + "self_attn.o_proj.weight"].cpu().float().numpy().T

        assign([block_key, "attn", "q_proj", "kernel"], q_w)
        assign([block_key, "attn", "q_proj", "bias"], q_b)
        assign([block_key, "attn", "k_proj", "kernel"], k_w)
        assign([block_key, "attn", "k_proj", "bias"], k_b)
        assign([block_key, "attn", "v_proj", "kernel"], v_w)
        assign([block_key, "attn", "v_proj", "bias"], v_b)
        assign([block_key, "attn", "o_proj", "kernel"], o_w)

        # MLP
        gate_w = state_dict[prefix + "mlp.gate_proj.weight"].cpu().float().numpy().T
        up_w = state_dict[prefix + "mlp.up_proj.weight"].cpu().float().numpy().T
        down_w = state_dict[prefix + "mlp.down_proj.weight"].cpu().float().numpy().T
        assign([block_key, "mlp", "gate", "kernel"], gate_w)
        assign([block_key, "mlp", "up", "kernel"], up_w)
        assign([block_key, "mlp", "down", "kernel"], down_w)

    # Final norm
    assign(["ln_f", "weight"], state_dict["model.norm.weight"].cpu().float().numpy())

    return freeze(p)


def _to_hf_state_dict(params: Any, n_layers: int) -> dict[str, "torch.Tensor"]:
    """Convert Theseus Qwen params to a transformers-compatible state_dict."""

    if torch is None:
        raise ImportError("torch is required for HF export")

    from flax.core import unfreeze

    p = unfreeze(params)
    sd: dict[str, torch.Tensor] = {}

    def grab(path: list[str]) -> Any:
        cur = p
        for key in path:
            cur = cur[key]
        return cur

    # Embeddings
    embed = torch.tensor(grab(["wte"]), dtype=torch.float32)
    sd["model.embed_tokens.weight"] = embed
    sd["lm_head.weight"] = embed  # tied weights

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        block_key = f"blocks_{i}"

        sd[prefix + "input_layernorm.weight"] = torch.tensor(
            grab([block_key, "rms_1", "weight"]), dtype=torch.float32
        )
        sd[prefix + "post_attention_layernorm.weight"] = torch.tensor(
            grab([block_key, "rms_2", "weight"]), dtype=torch.float32
        )

        q_w = grab([block_key, "attn", "q_proj", "kernel"]).T
        k_w = grab([block_key, "attn", "k_proj", "kernel"]).T
        v_w = grab([block_key, "attn", "v_proj", "kernel"]).T
        o_w = grab([block_key, "attn", "o_proj", "kernel"]).T
        sd[prefix + "self_attn.q_proj.weight"] = torch.tensor(q_w, dtype=torch.float32)
        sd[prefix + "self_attn.k_proj.weight"] = torch.tensor(k_w, dtype=torch.float32)
        sd[prefix + "self_attn.v_proj.weight"] = torch.tensor(v_w, dtype=torch.float32)
        sd[prefix + "self_attn.o_proj.weight"] = torch.tensor(o_w, dtype=torch.float32)
        sd[prefix + "self_attn.q_proj.bias"] = torch.tensor(
            grab([block_key, "attn", "q_proj", "bias"]), dtype=torch.float32
        )
        sd[prefix + "self_attn.k_proj.bias"] = torch.tensor(
            grab([block_key, "attn", "k_proj", "bias"]), dtype=torch.float32
        )
        sd[prefix + "self_attn.v_proj.bias"] = torch.tensor(
            grab([block_key, "attn", "v_proj", "bias"]), dtype=torch.float32
        )

        gate_w = grab([block_key, "mlp", "gate", "kernel"]).T
        up_w = grab([block_key, "mlp", "up", "kernel"]).T
        down_w = grab([block_key, "mlp", "down", "kernel"]).T
        sd[prefix + "mlp.gate_proj.weight"] = torch.tensor(gate_w, dtype=torch.float32)
        sd[prefix + "mlp.up_proj.weight"] = torch.tensor(up_w, dtype=torch.float32)
        sd[prefix + "mlp.down_proj.weight"] = torch.tensor(down_w, dtype=torch.float32)

    sd["model.norm.weight"] = torch.tensor(
        grab(["ln_f", "weight"]), dtype=torch.float32
    )
    return sd
