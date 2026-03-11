"""SideChannelQwen parity check: with tanh(gate)=0, should match vanilla Qwen exactly.

Usage: uv run python scripts/sidechannel_parity.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello world"
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.utils import logging as hf_logging

from theseus.config import patch, configure
from theseus.model.models.contrib.qwen import Qwen, _from_hf_state_dict
from theseus.model.models.sidechannel.qwen import SideChannelQwen

hf_logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    # --- Load HF model ---
    hf = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    inputs = tok(
        prompt_text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )

    with torch.no_grad():
        logits_hf = hf(**inputs).logits.detach().cpu().numpy()

    cfg = hf.config
    rope_theta = (
        cfg.rope_parameters.get("rope_theta", 10000.0)
        if cfg.rope_parameters
        else 10000.0
    )

    with patch() as th_cfg:
        arch_cfg = {
            "n_layers": cfg.num_hidden_layers,
            "n_embd": cfg.hidden_size,
            "n_head": cfg.num_attention_heads,
            "n_kv_head": cfg.num_key_value_heads,
            "intermediate_size": cfg.intermediate_size,
            "block_size": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "dropout": 0.0,
            "attn_dropout": float(cfg.attention_dropout),
            "rope_theta": float(rope_theta),
            "rms_norm_eps": float(cfg.rms_norm_eps),
            "use_sliding_window": bool(cfg.use_sliding_window),
            "sliding_window": int(cfg.sliding_window)
            if cfg.sliding_window is not None
            else -1,
            "max_window_layers": int(cfg.max_window_layers),
            "bias": False,
            "attention_bias": True,
            "partial_rotary_factor": 1.0,
            "dtype": {"param": "float32", "activation": "float32"},
            "sidechannel": {
                "n_channels": 4,
                "n_latents": 16,
                "perceiver_layers": 1,
                "perceiver_heads": 4,
                "cross_attn_layers": [3, 7, 11, 15],
                "n_head": cfg.num_attention_heads,
                "n_kv_head": cfg.num_key_value_heads,
                "attn_bias": False,
            },
        }
        th_cfg.architecture = OmegaConf.create(arch_cfg)

        # --- Test 1: Vanilla Qwen parity with HF ---
        print("=" * 60)
        print("TEST 1: Vanilla Qwen parity with HF")
        print("=" * 60)

        qwen_model = configure(Qwen)
        dummy = jnp.zeros((1, 1), dtype=jnp.int32)
        abstract = jax.eval_shape(qwen_model.init, jax.random.PRNGKey(0), dummy)
        qwen_params = jax.tree_util.tree_map(
            lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
        )
        qwen_params = _from_hf_state_dict(
            qwen_params, hf.state_dict(), cfg.num_hidden_layers
        )

        idx = jnp.array(inputs["input_ids"].numpy())
        attn_bool = jnp.array(inputs["attention_mask"].numpy(), dtype=bool)
        logits_qwen, _ = qwen_model.apply(
            {"params": qwen_params}, idx, padding_mask=attn_bool, deterministic=True
        )

        attn = inputs["attention_mask"][0].numpy()
        last_tok_idx = int(attn.sum() - 1)
        logits_hf_last = logits_hf[0, last_tok_idx]
        logits_qwen_last = np.array(logits_qwen[0, last_tok_idx])
        max_diff_qwen = np.max(np.abs(logits_hf_last - logits_qwen_last))
        print(f"  Qwen vs HF max diff: {max_diff_qwen}")
        assert max_diff_qwen < 0.01, f"Vanilla Qwen parity failed: {max_diff_qwen}"
        print("  PASS")

        # --- Test 2: SideChannelQwen with no sidechannel input ---
        print()
        print("=" * 60)
        print("TEST 2: SideChannelQwen (gate=0) vs vanilla Qwen")
        print("=" * 60)

        sc_model = configure(SideChannelQwen)
        abstract_sc = jax.eval_shape(sc_model.init, jax.random.PRNGKey(0), dummy)
        sc_params = jax.tree_util.tree_map(
            lambda x: np.zeros(x.shape, x.dtype), abstract_sc["params"]
        )
        # Load base Qwen weights into the SideChannelQwen params
        sc_params = _from_hf_state_dict(
            sc_params, hf.state_dict(), cfg.num_hidden_layers
        )

        # Forward without sidechannel (perceiver encodes zeros, but gate=0 so no effect)
        logits_sc, _ = sc_model.apply(
            {"params": sc_params}, idx, padding_mask=attn_bool, deterministic=True
        )

        logits_sc_last = np.array(logits_sc[0, last_tok_idx])
        max_diff_sc = np.max(np.abs(logits_qwen_last - logits_sc_last))
        mean_diff_sc = np.mean(np.abs(logits_qwen_last - logits_sc_last))
        print(f"  SideChannelQwen vs Qwen max diff: {max_diff_sc}")
        print(f"  SideChannelQwen vs Qwen mean diff: {mean_diff_sc}")
        assert max_diff_sc < 1e-4, (
            f"SideChannelQwen parity failed: gate should be 0 but max diff = {max_diff_sc}"
        )
        print("  PASS: tanh(0) gate preserves pretrained behavior")

        # --- Test 3: SideChannelQwen with sidechannel input ---
        print()
        print("=" * 60)
        print("TEST 3: SideChannelQwen forward with sidechannel input")
        print("=" * 60)

        B, T = idx.shape
        n_channels = 4
        sc_len = 32
        sidechannel = jnp.ones((B, n_channels, sc_len), dtype=jnp.int32)
        sidechannel_mask = jnp.zeros((B, T), dtype=jnp.int32)

        logits_with_sc, _ = sc_model.apply(
            {"params": sc_params},
            idx,
            padding_mask=attn_bool,
            sidechannel=sidechannel,
            sidechannel_mask=sidechannel_mask,
            deterministic=True,
        )
        logits_with_sc_last = np.array(logits_with_sc[0, last_tok_idx])

        # With gate=0, even with sidechannel input, output should be identical
        max_diff_with_sc = np.max(np.abs(logits_qwen_last - logits_with_sc_last))
        print(
            f"  SideChannelQwen (with sc input, gate=0) vs Qwen max diff: {max_diff_with_sc}"
        )
        assert max_diff_with_sc < 1e-4, (
            f"Gate=0 with sidechannel input should produce same output: {max_diff_with_sc}"
        )
        print("  PASS: gate=0 blocks cross-attention contribution even with input")

        # --- Test 4: SideChannelGPT init smoke test ---
        print()
        print("=" * 60)
        print("TEST 4: SideChannelGPT init and forward smoke test")
        print("=" * 60)

        from theseus.model.models.sidechannel.gpt import SideChannelGPT

        # Override config for smaller GPT
        th_cfg.architecture = OmegaConf.create(
            {
                "n_layers": 4,
                "n_embd": 128,
                "n_head": 4,
                "rope": True,
                "block_size": 64,
                "dropout": 0.0,
                "vocab_size": 1000,
                "bias": True,
                "dtype": {"param": "float32", "activation": "float32"},
                "sidechannel": {
                    "n_channels": 2,
                    "n_latents": 8,
                    "perceiver_layers": 1,
                    "perceiver_heads": 2,
                    "cross_attn_layers": [1, 3],
                    "n_head": 4,
                    "n_kv_head": 4,
                    "attn_bias": False,
                },
            }
        )

        gpt_model = configure(SideChannelGPT)
        key = jax.random.PRNGKey(42)
        dummy_idx = jnp.ones((2, 32), dtype=jnp.int32)
        gpt_params = gpt_model.init(key, dummy_idx, deterministic=True)

        dummy_sc = jnp.ones((2, 2, 32), dtype=jnp.int32)
        dummy_sc_mask = jnp.zeros((2, 32), dtype=jnp.int32)
        logits_gpt, _ = gpt_model.apply(
            gpt_params,
            dummy_idx,
            sidechannel=dummy_sc,
            sidechannel_mask=dummy_sc_mask,
            deterministic=True,
        )
        print(f"  SideChannelGPT output shape: {logits_gpt.shape}")
        assert logits_gpt.shape == (2, 32, 1000), (
            f"Unexpected shape: {logits_gpt.shape}"
        )
        print("  PASS: SideChannelGPT forward works")

        # --- Test 5: Forward-backward with random init, loss ~ ln(vocab_size) ---
        print()
        print("=" * 60)
        print("TEST 5: Forward-backward with random init (loss sanity check)")
        print("=" * 60)

        vocab_size = 1000
        expected_loss = float(np.log(vocab_size))
        print(f"  Expected initial loss ~ ln({vocab_size}) = {expected_loss:.4f}")

        # GPT forward-backward
        targets_gpt = jax.random.randint(jax.random.PRNGKey(99), (2, 32), 0, vocab_size)
        dummy_pad = jnp.ones((2, 32), dtype=bool)

        def gpt_loss_fn(params: dict) -> jax.Array:
            _, loss = gpt_model.apply(
                params,
                dummy_idx,
                targets=targets_gpt,
                padding_mask=dummy_pad,
                sidechannel=dummy_sc,
                sidechannel_mask=dummy_sc_mask,
                deterministic=True,
            )
            return loss

        gpt_loss, gpt_grads = jax.value_and_grad(gpt_loss_fn, allow_int=True)(
            gpt_params
        )
        gpt_loss_val = float(gpt_loss)
        print(f"  SideChannelGPT loss: {gpt_loss_val:.4f}")
        assert abs(gpt_loss_val - expected_loss) < 2.0, (
            f"GPT loss {gpt_loss_val} too far from expected {expected_loss}"
        )

        # Check gradients are non-zero (filter out float0 arrays from int params)
        def _grad_norm(g: jax.Array) -> float:
            if hasattr(g, "dtype") and g.dtype == jax.float0:
                return 0.0
            return float(jnp.linalg.norm(g))

        grad_norms = jax.tree_util.tree_map(_grad_norm, gpt_grads)
        flat_norms = jax.tree_util.tree_leaves(grad_norms)
        num_nonzero = sum(1 for n in flat_norms if n > 0)
        print(f"  Non-zero gradient params: {num_nonzero}/{len(flat_norms)}")
        assert num_nonzero > 0, "All gradients are zero!"

        # Check perceiver and cross-attn gate gradients specifically
        gate_grad_norm = float(
            jnp.linalg.norm(gpt_grads["params"]["blocks_1"]["cross_attn"]["gate"])
        )
        perceiver_grad_norm = float(
            jnp.linalg.norm(gpt_grads["params"]["perceiver"]["latent_queries"])
        )
        print(f"  Cross-attn gate grad norm: {gate_grad_norm:.6f}")
        print(f"  Perceiver latent_queries grad norm: {perceiver_grad_norm:.6f}")
        assert gate_grad_norm > 0, "Gate gradient is zero — no learning signal!"
        # Perceiver grad is expected to be 0 at init because tanh(0)=0 blocks
        # gradient flow. Once the gate opens during training, gradients will flow.
        print(
            "  PASS: loss ~ ln(V), gate grad non-zero (will learn to open), "
            "perceiver grad blocked by gate=0 (expected)"
        )

        # SideChannelQwen forward-backward (using pretrained weights)
        # With gate=0, loss should match vanilla Qwen exactly.
        print()
        print("=" * 60)
        print("TEST 6: SideChannelQwen forward-backward with pretrained weights")
        print("=" * 60)

        # Re-set Qwen config
        th_cfg.architecture = OmegaConf.create(arch_cfg)

        targets_qwen = jnp.roll(idx, -1, axis=1)

        # First compute base Qwen loss for reference
        def base_qwen_loss_fn(params: dict) -> jax.Array:
            _, loss = qwen_model.apply(
                {"params": params},
                idx,
                targets=targets_qwen,
                padding_mask=attn_bool,
                deterministic=True,
            )
            return loss

        base_loss_val = float(base_qwen_loss_fn(qwen_params))
        print(f"  Base Qwen loss: {base_loss_val:.4f}")

        # Now compute SideChannelQwen loss (gate=0, should match base)
        def sc_qwen_loss_fn(params: dict) -> jax.Array:
            _, loss = sc_model.apply(
                {"params": params},
                idx,
                targets=targets_qwen,
                padding_mask=attn_bool,
                sidechannel=sidechannel,
                sidechannel_mask=sidechannel_mask,
                deterministic=True,
            )
            return loss

        qwen_loss, qwen_grads = jax.value_and_grad(sc_qwen_loss_fn, allow_int=True)(
            sc_params
        )
        qwen_loss_val = float(qwen_loss)
        print(f"  SideChannelQwen loss: {qwen_loss_val:.4f}")
        assert qwen_loss_val > 0, "Loss is zero or negative!"
        assert not np.isnan(qwen_loss_val), "Loss is NaN!"

        # Parity: SideChannelQwen with gate=0 should match base Qwen
        loss_diff = abs(qwen_loss_val - base_loss_val)
        print(f"  Loss diff (SC vs base): {loss_diff:.6f}")
        assert loss_diff < 0.01, (
            f"SideChannelQwen loss ({qwen_loss_val:.4f}) doesn't match "
            f"base Qwen loss ({base_loss_val:.4f}), diff={loss_diff:.6f}"
        )
        print("  PASS: SideChannelQwen loss matches base Qwen (gate=0 parity)")

        # Check cross-attn gate gradients in SideChannelQwen
        qwen_gate_grads = []
        for layer_idx in [3, 7, 11, 15]:
            key_name = f"blocks_{layer_idx}"
            if key_name in qwen_grads:
                gate_g = float(
                    jnp.linalg.norm(qwen_grads[key_name]["cross_attn"]["gate"])
                )
                qwen_gate_grads.append(gate_g)

        if qwen_gate_grads:
            print(f"  Qwen cross-attn gate grad norms: {qwen_gate_grads}")

        # Verify base Qwen params have non-zero gradients
        qwen_flat_grads = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(_grad_norm, qwen_grads)
        )
        qwen_nonzero = sum(1 for n in qwen_flat_grads if n > 0)
        print(f"  Non-zero gradient params: {qwen_nonzero}/{len(qwen_flat_grads)}")
        assert qwen_nonzero > 0, "All Qwen gradients are zero!"
        print("  PASS: SideChannelQwen backward works, gradients flow")

        print()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)


if __name__ == "__main__":
    main()
