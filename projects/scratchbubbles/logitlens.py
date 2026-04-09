import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from loguru import logger

from theseus.base import Axis
from theseus.config import field
from theseus.data.tokenizer import Tokenizer, get_tokenizer
from theseus.inference.base import InferenceJob
from theseus.model.block.forking import ThoughtBlock
from theseus.model.models.scratchbubbles import Scratchbubbles
from theseus.registry import job
from theseus.training.flywheel.strategy import Strategy, Sampling, DatasetStyle


@dataclass
class LogitLensConfig:
    top_k: int = field("analysis/top_k", default=5)
    block_size: int = field("architecture/block_size", default=512)
    per_device_batch_size: int = field("training/per_device_batch_size", default=-1)
    validation_steps: int = field("training/validation_steps", default=2048)
    datasets: List[Sampling] = field(
        "training/dataset",
        default_factory=lambda: [
            Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)
        ],
    )


@job("scratchbubbles/analysis/logitlens")
class SBLogitLens(InferenceJob[LogitLensConfig, Scratchbubbles]):
    MODEL = Scratchbubbles

    @classmethod
    def config(cls) -> List[Any]:
        return [LogitLensConfig]

    def run(self) -> None:
        # Load a validation batch
        strategy = Strategy(self.spec, self.args.block_size, self.args.datasets)
        val_batch_size = max(
            self.per_device_batch_size * self.local_replicas,
            (
                self.args.validation_steps
                // (self.per_device_batch_size * self.local_replicas)
            )
            * (self.per_device_batch_size * self.local_replicas),
        )
        val_dl = strategy.get_async_batches(
            val_batch_size, split="val", deterministic_key=32
        )
        raw_batch = val_dl.get_batch()

        # Reshape and convert to global arrays
        per = self.per_device_batch_size * self.local_replicas
        pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        batch = jax.tree_util.tree_map(
            lambda arr: multihost_utils.host_local_array_to_global_array(
                arr.reshape(-1, per, arr.shape[-1]), self.mesh, pspec
            ),
            raw_batch,
        )

        data_shard = NamedSharding(self.mesh, pspec)
        top_k = self.args.top_k

        def analyze_one_batch(state: Any, global_batch: Any) -> tuple[Any, Any, Any]:
            batch_dict = cast(Dict[str, jax.Array], global_batch)
            microbatch = {k: v[0] for k, v in batch_dict.items()}

            outputs, mutated = self.model.apply(
                {"params": state.params},
                microbatch["x"],
                microbatch["y"],
                padding_mask=microbatch["padding_mask"],
                deterministic=True,
                capture_intermediates=self._capture_filter,
                mutable=["intermediates"],
            )
            _, loss = outputs

            sample_ids = microbatch["x"][0]
            sample_mask = microbatch["padding_mask"][0]
            default_token_index = jnp.broadcast_to(
                jnp.arange(sample_ids.shape[0], dtype=jnp.int32),
                microbatch["x"].shape,
            )

            intermediates = cast(Mapping[str, Any], mutated["intermediates"])
            layers: Dict[str, Dict[str, jax.Array]] = {}

            embed_capture = intermediates["embed"][0]
            embed_residual, embed_token_index = self._split_capture(
                embed_capture,
                default_token_index,
            )
            embed_logits = self.model.apply(
                {"params": state.params},
                embed_residual[:1],
                method=self.model.unembed,
            )[0]
            embed_top_logits, embed_top_ids = self._top_k_logits(embed_logits, top_k)
            embed_source_index = embed_token_index[0]
            layers["embed"] = {
                "source_index": embed_source_index,
                "source_token_id": jnp.take(
                    sample_ids, embed_source_index, mode="clip"
                ),
                "source_valid": jnp.take(sample_mask, embed_source_index, mode="clip"),
                "top_ids": embed_top_ids[0],
                "top_logits": embed_top_logits[0],
            }

            for layer_idx in range(self.model.n_layers):
                block_capture = intermediates[f"blocks_{layer_idx}"]["__call__"][0]
                block_residual, block_token_index = self._split_capture(
                    block_capture,
                    default_token_index,
                )
                block_logits = self.model.apply(
                    {"params": state.params},
                    block_residual[:1],
                    method=self.model.unembed,
                )[0]
                block_top_logits, block_top_ids = self._top_k_logits(
                    block_logits, top_k
                )
                block_source_index = block_token_index[0]
                layers[f"block_{layer_idx:02d}"] = {
                    "source_index": block_source_index,
                    "source_token_id": jnp.take(
                        sample_ids, block_source_index, mode="clip"
                    ),
                    "source_valid": jnp.take(
                        sample_mask, block_source_index, mode="clip"
                    ),
                    "top_ids": block_top_ids[0],
                    "top_logits": block_top_logits[0],
                }

            return (
                {"input_ids": sample_ids, "input_valid": sample_mask},
                layers,
                {"loss": loss},
            )

        analyze = jax.jit(
            analyze_one_batch,
            in_shardings=(self.state_sharding, data_shard),
            out_shardings=(None, None, None),
        )

        sample_meta, layers, metrics = analyze(self.state, batch)
        sample_meta_np = jax.tree_util.tree_map(np.asarray, sample_meta)
        layers_np = jax.tree_util.tree_map(np.asarray, layers)
        metrics_np = jax.tree_util.tree_map(np.asarray, metrics)

        if not self.main_process():
            return

        try:
            tokenizer = get_tokenizer()
        except Exception as exc:
            tokenizer = None
            logger.warning("TOKENIZER | falling back to token ids only: {}", exc)
        input_ids = cast(np.ndarray, sample_meta_np["input_ids"])
        input_valid = cast(np.ndarray, sample_meta_np["input_valid"])
        loss_value = float(cast(np.ndarray, metrics_np["loss"]))
        decoded_input_tokens = [
            self._format_token(tokenizer, int(token_id))
            for token_id in input_ids.tolist()
        ]

        results_payload: Dict[str, Any] = {
            "job": self.spec.name,
            "model": self.MODEL.__name__,
            "project": self.spec.project,
            "group": self.spec.group,
            "val_loss": loss_value,
            "input": {
                "token_ids": [int(token_id) for token_id in input_ids.tolist()],
                "valid": [bool(is_valid) for is_valid in input_valid.tolist()],
                "tokens": decoded_input_tokens,
            },
            "layers": {},
        }

        logger.info(
            "SCRATCHBUBBLES LOGIT LENS | val_loss={:.6f}",
            loss_value,
        )
        logger.info("INPUT TOKENS | valid_tokens={}", int(input_valid.sum()))
        for pos, (token_id, is_valid) in enumerate(
            zip(input_ids.tolist(), input_valid.tolist())
        ):
            logger.info(
                "INPUT | pos={:03d} | valid={} | token_id={} | token={}",
                pos,
                bool(is_valid),
                int(token_id),
                self._format_token(tokenizer, int(token_id)),
            )

        for layer_name in [
            "embed",
            *[f"block_{i:02d}" for i in range(self.model.n_layers)],
        ]:
            layer = cast(Dict[str, np.ndarray], layers_np[layer_name])
            logger.info("LAYER | {}", layer_name)
            layer_rows: list[dict[str, Any]] = []

            for pos in range(layer["source_index"].shape[0]):
                source_index = int(layer["source_index"][pos])
                source_token_id = int(layer["source_token_id"][pos])
                source_valid = bool(layer["source_valid"][pos])
                top_ids = cast(Sequence[int], layer["top_ids"][pos].tolist())
                top_logits = cast(Sequence[float], layer["top_logits"][pos].tolist())
                topk = [
                    {
                        "token_id": int(pred_id),
                        "logit": float(logit),
                        "token": self._format_token(tokenizer, int(pred_id)),
                    }
                    for pred_id, logit in zip(top_ids, top_logits)
                ]
                layer_rows.append(
                    {
                        "position": pos,
                        "source_position": source_index,
                        "source_valid": source_valid,
                        "source_token_id": source_token_id,
                        "source_token": self._format_token(tokenizer, source_token_id),
                        "topk": topk,
                    }
                )

                preds = ", ".join(
                    f"{int(pred_id)}:{logit:.4f}:{self._format_token(tokenizer, int(pred_id))}"
                    for pred_id, logit in zip(top_ids, top_logits)
                )
                logger.info(
                    "LENS | layer={} | pos={:03d} | source_pos={:03d} | valid={} | "
                    "source_token_id={} | source_token={} | topk=[{}]",
                    layer_name,
                    pos,
                    source_index,
                    source_valid,
                    source_token_id,
                    self._format_token(tokenizer, source_token_id),
                    preds,
                )
            results_payload["layers"][layer_name] = layer_rows

        with self.spec.result(self._result_name(), main_process_only=True) as f:
            if f is not None:
                json.dump(results_payload, f, indent=2)
                logger.info("RESULTS | logit lens JSON saved to {}", f.name)

    @staticmethod
    def _capture_filter(module: Any, method_name: str) -> bool:
        if isinstance(module, Scratchbubbles):
            return method_name == "embed"
        return isinstance(module, ThoughtBlock) and method_name == "__call__"

    @staticmethod
    def _top_k_logits(logits: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
        logits_f32 = logits.astype(jnp.float32)
        return jax.lax.top_k(logits_f32, min(k, logits_f32.shape[-1]))

    @staticmethod
    def _format_token(tokenizer: Tokenizer | None, token_id: int) -> str:
        if tokenizer is None:
            return repr(f"<tok:{token_id}>")
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            text = f"<decode_error:{token_id}>"

        text = (
            text.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        if len(text) > 40:
            text = text[:37] + "..."
        return repr(text)

    @classmethod
    def _split_capture(
        cls, captured: Any, fallback_token_index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if isinstance(captured, tuple):
            residual = cast(jax.Array, captured[0])
            if len(captured) >= 3:
                return residual, cast(jax.Array, captured[2])
            return residual, fallback_token_index
        return cast(jax.Array, captured), fallback_token_index

    @classmethod
    def _result_name(cls) -> str:
        return f"{cls.MODEL.__name__.lower()}_logitlens.json"
