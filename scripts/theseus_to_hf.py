"""Export a Theseus BackbonedTrainer (or BackbonedContrastiveTrainer) checkpoint
to HuggingFace format.

Usage:
    uv run scripts/theseus_to_hf.py SUFFIX --root /path/to/root --name job_name -o /out

SUFFIX is the checkpoint suffix, e.g. 'final/9216', 'best', or 'boundary_0_1'.
"""

import click
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import torch
from flax.training import train_state
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from theseus.base.job import ExecutionSpec
from theseus.job import CheckpointedJob
from theseus.training.backbone import BACKBONES

# Per-implementation conversion functions.
# qwen: (params, n_layers)
# llama / gpt_neox: (params, n_layers, hf_cfg)
from theseus.model.models.qwen import _to_hf_state_dict as _qwen_to_hf
from theseus.model.models.llama import _to_hf_state_dict as _llama_to_hf
from theseus.model.models.gpt_neox import _to_hf_state_dict as _gpt_neox_to_hf


def _call_to_hf(impl: str, params, n_layers: int, hf_cfg):
    if impl == "qwen":
        return _qwen_to_hf(params, n_layers)
    elif impl == "llama":
        return _llama_to_hf(params, n_layers, hf_cfg)
    elif impl == "gpt_neox":
        return _gpt_neox_to_hf(params, n_layers, hf_cfg)
    else:
        raise click.ClickException(f"No _to_hf_state_dict for backbone '{impl}'")


class _Loader(CheckpointedJob):
    """Minimal CheckpointedJob stub: loads a checkpoint without starting a trainer."""

    @classmethod
    def config(cls):
        return object  # unused, but required by the ABC

    @property
    def done(self) -> bool:
        return False

    def run(self) -> None:
        raise NotImplementedError

    def __init__(self, spec: ExecutionSpec):  # type: ignore[override]
        self.spec = spec
        self.key = jax.random.PRNGKey(0)


@click.command()
@click.argument("suffix")
@click.option(
    "--root",
    required=True,
    help="Checkpoint root directory (passed to ExecutionSpec.local)",
)
@click.option("--name", required=True, help="Job name (subfolder inside project/group)")
@click.option("-o", "--output", required=True, help="Output directory for the HF model")
@click.option(
    "--project", default=None, show_default=True, help="Project name (default: general)"
)
@click.option(
    "--group", default=None, show_default=True, help="Group name (default: default)"
)
@click.option(
    "--dpo",
    is_flag=True,
    default=False,
    help="Checkpoint is from a BackbonedContrastiveTrainer (DPO). "
    "Loads ContrastiveTrainState so both policy and reference params are available.",
)
@click.option(
    "--export-base",
    is_flag=True,
    default=False,
    help="DPO only: export the frozen reference (base) model instead of the trained policy.",
)
def main(suffix, root, name, output, project, group, dpo, export_base):
    """Export a Theseus backbone checkpoint to HuggingFace format."""

    if export_base and not dpo:
        raise click.UsageError("--export-base requires --dpo")

    # Build spec and locate checkpoint
    spec = ExecutionSpec.local(root, name=name, project=project, group=group)
    ckpt_path = CheckpointedJob._get_checkpoint_path(spec, suffix)
    click.echo(f"Checkpoint: {ckpt_path}")

    cfg = OmegaConf.load(ckpt_path / "config.yaml")
    impl = str(cfg.architecture.backbone.implementation)
    weights = str(cfg.architecture.backbone.weights)
    click.echo(f"Backbone : {impl}")
    click.echo(f"Weights  : {weights}")

    # Load model architecture and get initial params (needed as a shape template)
    click.echo("Loading pretrained model for template state …")
    model_cls = BACKBONES[impl]
    model, initial_params = model_cls.from_pretrained(weights)

    # Build template state
    if dpo:
        from theseus.training.contrastive import ContrastiveTrainState

        template_state = ContrastiveTrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=initial_params,
            base=jax.tree_util.tree_map(jnp.zeros_like, initial_params),
            tx=optax.identity(),
            beta=0.1,
            label_smooth=0.0,
        )
    else:
        template_state = train_state.TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=initial_params,
            tx=optax.identity(),
        )

    # Restore checkpoint
    click.echo("Restoring checkpoint …")
    loader = _Loader(spec)
    restored_state, metadata = loader.get_tree_and_metadata(suffix, template_state)
    click.echo(
        f"Loaded   : step={metadata.get('steps', '?')}  "
        f"score={metadata.get('score', float('nan')):.4f}"
    )

    # Select which params to export
    if export_base:
        params = restored_state.base  # type: ignore[attr-defined]
        click.echo("Exporting: reference (base) model")
    else:
        params = restored_state.params
        click.echo("Exporting: trained policy model")

    # Convert to HF state dict
    hf_cfg = AutoConfig.from_pretrained(weights)
    n_layers = hf_cfg.num_hidden_layers
    click.echo(f"Converting {n_layers}-layer {impl} params to HF state dict …")
    sd = _call_to_hf(impl, params, n_layers, hf_cfg)

    # Load into HF model
    torch_sd = {k: torch.from_numpy(np.array(jax.device_get(v))) for k, v in sd.items()}
    hf_model = AutoModelForCausalLM.from_config(hf_cfg)
    missing, unexpected = hf_model.load_state_dict(torch_sd, strict=False)
    if missing:
        n = len(missing)
        click.echo(
            f"Warning: {n} missing key(s): {missing[:3]}{'…' if n > 3 else ''}",
            err=True,
        )
    if unexpected:
        n = len(unexpected)
        click.echo(
            f"Warning: {n} unexpected key(s): {unexpected[:3]}{'…' if n > 3 else ''}",
            err=True,
        )

    # Save model + tokenizer
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out)

    tok = AutoTokenizer.from_pretrained(weights)
    tok.save_pretrained(out)

    click.echo(f"Saved to : {out}")


if __name__ == "__main__":
    main()
