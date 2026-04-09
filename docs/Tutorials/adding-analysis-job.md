# The Job System

Every unit of work in theseus is a **job**: a class registered with `@job("key")` that knows how to set itself up, run, and (optionally) save and restore its state. This document covers the full job hierarchy, what each layer adds, what you must implement, and how to pick the right base class.

---

## Hierarchy

```
BasicJob[C]                     config, run, done, __call__
  CheckpointedJob[C]            save/load pytrees + metadata
    RestoreableJob[C]           restore_from_path, from_checkpoint_path
      BaseTrainer[C, M]         training loop, optimizer, data, forward
      InferenceJob[C, M]        model init from checkpoint, rollout
```

Each layer adds one concern. You inherit from the lowest layer that does what you need and implement the rest.

---

## Choosing a base class

| I want to... | Inherit from | I must implement |
|---|---|---|
| Run a one-shot computation (no checkpoints) | `BasicJob` | `config()`, `run()` |
| Save/load pytrees but manage restore myself | `CheckpointedJob` | `config()`, `run()` |
| Save checkpoints and restore from them later | `RestoreableJob` | `config()`, `run()`, `restore_from_path()` |
| Train a model | `BaseTrainer` | `MODEL`, `CONFIG`, `schedule()` |
| Load a trained checkpoint and do read-only work | `InferenceJob` | `MODEL`, `config()`, `run()` |

---

## `BasicJob[C]` — the root

Every job has these. You always implement `config()` and `run()`.

### `config() -> Type[C] | List[Type]`

Returns the config dataclass type(s) for this job. `BasicJob.__init__` calls `configure(config()[0])` to hydrate `self.args: C` from the active OmegaConf context.

If your config needs fields from multiple dataclasses (e.g. model config + your custom config), return a list. The first element becomes `self.args`; the rest register their `field()` paths so OmegaConf knows about them.

```python
@classmethod
def config(cls) -> List[Any]:
    return [MyConfig]
```

### `run()`

Abstract. Your computation goes here. Called on all hosts after device sync.

### `done -> bool`

Idempotency check. Return `True` to skip this job entirely. Default `False`. Override when your job writes a result file and you want to avoid re-running:

```python
@property
def done(self) -> bool:
    return self.spec.result_path("output.json").exists()
```

### `__call__()`

Syncs all devices, calls `run()`, syncs again, calls `finish()`. **Don't override this** — override `run()` instead.

### `local(root_dir, name, project, group) -> Self`

Convenience constructor for single-machine use. Builds a local `ExecutionSpec` and calls `cls(spec)`.

### `main_process() -> bool`

`True` on `jax.process_index() == 0`. Gate all IO (file writes, logging, wandb) behind this.

### `finish()`

Called after `run()` completes. Default tears down wandb. Override to add cleanup.

---

## `CheckpointedJob[C]` — save and load pytrees

Adds the ability to save and load JAX pytrees (model params, optimizer state, etc.) to disk via Orbax.

### The two-form pattern

Every checkpoint operation comes in two forms:

| Form | Takes | Purpose |
|---|---|---|
| `*_from_path(rel_path, ...)` | Arbitrary path under `checkpoints_dir` | Core implementation. Can address *any* checkpoint on disk. |
| `*(suffix, ...)` | Suffix scoped to this job's spec | Thin wrapper. Computes `rel_path = project/group/name/suffix` and delegates. |

This split lets a job load checkpoints belonging to a different job, project, or group. The suffix wrappers are for the common case where a job manages its own checkpoints.

### Path resolution

Checkpoint paths have two parts:

```
checkpoints_dir / rel_path
       ^              ^
  from cluster     project/group/name/suffix
   config          (or arbitrary, for *_from_path)
```

Static helpers:

- `_get_checkpoints_dir(spec)` — Cluster's checkpoint root for this process.
- `_get_checkpoint_rel_path(spec, suffix)` — Computes `project/group/name/suffix`.
- `_get_checkpoint_path(spec, suffix)` — Full absolute path. Legacy helper.

### What's in a checkpoint directory

When you call `save_tree_and_metadata`, the following files are written:

| File | Contents | Written by |
|---|---|---|
| `checkpoint/` | Orbax pytree (sharded arrays on disk) | All processes |
| `rng.npy` | Python, NumPy, and JAX random state | Main process |
| `config.json` | Your metadata dict (e.g. `{"steps": 1000, "score": 0.5}`) | Main process |
| `job.json` | `JobSpec` fields (name, project, group) | Main process |
| `config.yaml` | Full OmegaConf config snapshot | Main process |

### Load

- **`get_tree_and_metadata_from_path(rel_path, template_tree)`** — Restores a pytree from disk using Orbax, guided by `template_tree` for shape and sharding info. Also restores RNG state. Returns `(tree, metadata_dict)`.
- **`get_tree_and_metadata(suffix, template_tree)`** — Suffix wrapper.

### Save

- **`save_tree_and_metadata_from_path(rel_path, tree, metadata)`** — Saves everything listed in the table above. Syncs all devices before and after.
- **`save_tree_and_metadata(suffix, tree, metadata)`** — Suffix wrapper.

### Metadata only

- **`get_metadata_from_path(rel_path)`** — Load just `config.json` without the full pytree.
- **`get_metadata(suffix)`** — Suffix wrapper.

---

## `RestoreableJob[C]` — checkpoint restore protocol

Adds the contract for restoring a job from a checkpoint. This is the layer where `--restore` and idempotent dispatch work.

### What you must implement: `restore_from_path(rel_path)`

This is the **only abstract method**. It is called inside a `configuration(cfg)` context, so `configure()` and `current_config()` work. Your implementation must load whatever state your job needs from the checkpoint at `rel_path`.

**The contract:**

1. You receive `rel_path` — a path relative to `checkpoints_dir`.
2. The merged config (checkpoint config + runtime overrides) is active.
3. `self.args` and `self.spec` are already set (by `__init__`).
4. You must populate whatever instance attributes your `run()` needs.
5. You should call `self.get_tree_and_metadata_from_path(rel_path, template)` to load the pytree.

**Example — a trainer's `restore_from_path`:**

```python
def restore_from_path(self, rel_path):
    # self.state already exists from __init__ (template state)
    old_state = self.state
    state, metadata = self.get_tree_and_metadata_from_path(rel_path, old_state)
    self.state = state

    # Free the old template state to avoid OOM
    jax.tree_util.tree_map(lambda x: x.delete(), old_state)

    # Restore counters from metadata
    self.global_step_counter_ = metadata.get("steps", 0)
    self.best_val_score_ = metadata.get("score", float("-inf"))
```

**Example — an inference job's `restore_from_path`:**

```python
def restore_from_path(self, rel_path):
    cfg = current_config()
    self.model = configure(self.MODEL)

    # Build template state for checkpoint shape info
    template_state = self._init_template_state(self.model, cfg.architecture.block_size)

    # Compute sharding from model's logical rules
    self.state_sharding = flax.linen.logical_to_mesh_sharding(
        flax.linen.get_partition_spec(template_state),
        self.mesh,
        rules=tuple(self.model.sharding),
    )

    # Load and shard
    state, metadata = self.get_tree_and_metadata_from_path(rel_path, template_state)
    self.state = jax.device_put(state, self.state_sharding)
```

### The save side: pairing with `restore_from_path`

If your job saves checkpoints (trainers do, analysis jobs don't), the metadata dict you pass to `save_tree_and_metadata` must contain everything `restore_from_path` reads back. This is a contract between your save and restore:

```python
# Save
self.save_tree_and_metadata(suffix, self.state, {
    "steps": self.global_step_counter_,
    "score": self.best_val_score_,
})
self.register(suffix)  # mark as "latest" for idempotent dispatch

# Restore — reads back the same keys
state, metadata = self.get_tree_and_metadata_from_path(rel_path, self.state)
self.global_step_counter_ = metadata["steps"]
self.best_val_score_ = metadata["score"]
```

### `restore(suffix)`

Suffix wrapper. Computes `rel_path` from `self.spec` and calls `restore_from_path`.

### `register(suffix)`

Writes `suffix` to the `latest` file. Main process only. Call this after saving a checkpoint so that idempotent dispatch knows where to resume.

### `from_checkpoint_path(rel_path, spec, runtime_cfg=None)`

The primary class-level restore entry point. This is what `--restore` and `quick.restore()` call. The full sequence:

1. Load `job.json` from `checkpoints_dir / rel_path` and patch `spec`.
2. Load `config.yaml` from the same directory.
3. Merge `runtime_cfg` on top (if provided) — this is your launch YAML + CLI overrides.
4. Enter `configuration(merged_cfg)`.
5. Resolve the job class from `cfg.job` (or fall back to `cls`).
6. Call `job_cls(spec)` — runs `BasicJob.__init__`, sets `self.args`.
7. Call `job.restore_from_path(rel_path)`.
8. Return `(job, merged_cfg)`.

### `from_checkpoint(suffix, spec, runtime_cfg=None)`

Suffix wrapper. Computes `rel_path` and calls `from_checkpoint_path`.

### `latest(spec) -> str | None`

Read the `latest` file to find the most recent checkpoint suffix. Returns `None` if no checkpoint exists.

### `checkpoints(spec) -> List[str]`

Walk the checkpoint directory and return all available suffixes (directories containing `config.yaml`).

---

## `InferenceJob[C, M]` — load and use a trained model

For jobs that load a trained model checkpoint and run read-only computation. Implements `restore_from_path` with model init, mesh setup, sharding, and checkpoint loading.

### Class attributes

- **`MODEL: type[M]`** — The Flax module class. Must be set by subclasses.

### What `restore_from_path` sets up

After `restore_from_path` completes, these are all populated:

| Attribute | Type | Description |
|---|---|---|
| `self.model` | `M` | Flax module instance (no params) |
| `self.state` | `TrainState` | Params + `apply_fn` |
| `self.mesh` | `jax.sharding.Mesh` | Device mesh |
| `self.state_sharding` | `NamedSharding` | How state is distributed |
| `self.replicas` | `int` | Total data-parallel replicas |
| `self.local_replicas` | `int` | Replicas on this host |
| `self.per_device_batch_size` | `int` | Batch size per device |
| `self.block_size` | `int` | Sequence length |

### `from_trainer(trainer) -> Self`

Create an `InferenceJob` that shares a live trainer's state (by reference, not copy). Uses `object.__new__` to bypass `__init__`. This is how `Evaluator` gets created during training.

### `forward(state, params, batch, ...)` (static)

Default forward pass: unpacks `(x, y, padding_mask)` from `batch`, calls `state.apply_fn`. Supports `mutable` (for KV cache) and `extra_variables`. Override for custom forward logic.

### `rollout(inputs, encoding, ...)`

Autoregressive text generation. Handles tokenization, left-padding, KV-cached decoding, multi-host gather, and detokenization. Works with both raw strings and `ChatTemplate` inputs.

### `pad(seqs, pad_token, pad_to)`

Static. Left-pad a list of token sequences to uniform length. Returns `(padded_array, mask_array)`.

---

## Writing an analysis job (complete example)

An analysis job inherits from `InferenceJob`, sets `MODEL`, and implements `config()` + `run()`. It is always launched with `--restore`.

```python
from dataclasses import dataclass
from typing import Any, List

import jax
from theseus.config import field
from theseus.inference.base import InferenceJob
from theseus.model.models import MyModel
from theseus.registry import job


@dataclass
class MyAnalysisConfig:
    # Only fields YOUR analysis needs.
    # Model architecture comes from the checkpoint's config.yaml.
    num_samples: int = field("analysis/num_samples", default=100)
    block_size: int = field("architecture/block_size", default=512)


@job("my_model/analysis/probe")
class ProbeAnalysis(InferenceJob[MyAnalysisConfig, MyModel]):
    MODEL = MyModel

    @classmethod
    def config(cls) -> List[Any]:
        # Don't include MODEL.gather() — the checkpoint config has it all.
        return [MyAnalysisConfig]

    def run(self) -> None:
        # self.model, self.state, self.mesh are ready.
        logits = self.model.apply(
            {"params": self.state.params},
            some_input,
            deterministic=True,
        )

        if self.main_process():
            # save results
            with self.spec.result("probe_results.json", main_process_only=True) as f:
                json.dump(results, f)
```

Register it, then run:

```bash
theseus run my-probe run.yaml ./output \
    --restore my-project/my-group/my-training-run/best
```

Or from a notebook:

```python
from theseus.quick import quick

with quick("my_model/analysis/probe", "my-probe", out_path="./output") as j:
    j.config.analysis.num_samples = 500
    j.restore("my-project/my-group/my-training-run/best")
    j()
```

---

## Writing a custom restorable job

If `InferenceJob` or `BaseTrainer` don't fit — maybe you need a custom state shape, or you're not working with a Flax model at all — inherit from `RestoreableJob` directly.

You must implement three things: `config()`, `run()`, and `restore_from_path()`.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import jax
from theseus.config import field
from theseus.job import RestoreableJob
from theseus.registry import job


@dataclass
class MyJobConfig:
    lr: float = field("training/lr", default=1e-3)
    checkpoint_interval: int = field("logging/checkpoint_interval", default=1000)


@job("custom/my_job")
class MyCustomJob(RestoreableJob[MyJobConfig]):

    @classmethod
    def config(cls) -> List[Any]:
        return [MyJobConfig]

    def __init__(self, spec):
        super().__init__(spec)
        # Set up your state here. This runs both for fresh starts
        # and before restore_from_path (which overwrites it).
        self.my_state = initialize_something()
        self.step = 0

    def restore_from_path(self, rel_path: str | Path) -> None:
        """Restore from checkpoint. Called inside configuration() context."""
        state, metadata = self.get_tree_and_metadata_from_path(
            rel_path, self.my_state  # template for shape/sharding
        )
        self.my_state = state
        self.step = metadata.get("step", 0)

    def run(self) -> None:
        for step in range(self.step, 10000):
            self.my_state = train_step(self.my_state)
            self.step = step

            if step % self.args.checkpoint_interval == 0:
                self.save_tree_and_metadata(
                    Path("step") / str(step),
                    self.my_state,
                    {"step": step},
                )
                self.register(Path("step") / str(step))
```

### The `restore_from_path` contract

Your implementation **must**:

- Accept `rel_path: str | Path` — a path relative to `checkpoints_dir`.
- Load state using `self.get_tree_and_metadata_from_path(rel_path, template)`.
- Restore all instance attributes that `run()` depends on.

Your implementation **may assume**:

- `self.args` and `self.spec` are set (from `__init__`).
- A `configuration()` context is active (`configure()` and `current_config()` work).
- The config is the checkpoint's `config.yaml` merged with any runtime overrides.

Your implementation **should**:

- Free the old template state if replacing it (to avoid OOM on large models).
- Log what was restored, at least on `main_process()`.

### The save/restore symmetry

Whatever metadata keys you write in `save_tree_and_metadata`, you must read back in `restore_from_path`. Whatever pytree structure you save, you must provide a matching template when loading. This is a contract between your two methods — the checkpoint format is yours to define.

---

## Running restored jobs

### CLI: `--restore`

Both `theseus run` and `theseus submit` accept `--restore <rel_path>`:

```bash
# Local
theseus run my-job run.yaml ./output --restore project/group/name/checkpoint

# Remote
theseus submit my-job run.yaml --restore project/group/name/checkpoint --chip h100 -n 4
```

The launch config (`run.yaml` + CLI overrides) is merged on top of the checkpoint's saved `config.yaml` as `runtime_cfg`. This lets you change hyperparameters (learning rate, batch size, etc.) when resuming.

### Programmatic: `quick` / `init`

```python
from theseus.quick import quick

with quick("custom/my_job", "resumed-run", out_path="./output") as j:
    j.config.training.lr = 1e-4  # override before restore
    j.restore("project/group/name/checkpoint")
    j()
```

### Idempotent dispatch (automatic)

When a `RestoreableJob` is dispatched remotely, the bootstrap script automatically checks for a `latest` checkpoint before starting fresh. If one exists, it calls `from_checkpoint(latest, spec, runtime_cfg=cfg)`. This makes dispatch idempotent — if a job is preempted and restarted, it resumes from where it left off.

You get this for free by calling `self.register(suffix)` after each checkpoint save.

---

## Differences at a glance

| | `BasicJob` | `CheckpointedJob` | `RestoreableJob` | `BaseTrainer` | `InferenceJob` |
|---|---|---|---|---|---|
| `config()` | Required | Required | Required | Automatic | Required |
| `run()` | Required | Required | Required | Automatic (trains) | Required |
| `restore_from_path()` | N/A | N/A | Required | Automatic | Automatic |
| Saves checkpoints | No | Manual | Manual | Automatic | No |
| `--restore` | No | No | Yes | Yes | Yes |
| Idempotent dispatch | No | No | Yes (with `register`) | Yes | N/A |
| `self.model` | No | No | No | Yes | Yes |
| `from_trainer()` | No | No | No | N/A | Yes |
