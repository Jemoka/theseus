# Plot System

Making nice looking plots without blocking the training loop.

---

## Overview

```
Model.sow("plots", key, tensor)   ← called inside forward pass
          │
          ▼  (captured by Flax's mutable="plots" during val step)
          │
trainer.plotter.submit(intermediates, step)
          │
          ▼  (enqueued; training loop continues immediately)
          │
    background thread
          │  model_cls.plot(intermediates) → {name: figure}
          │  wandb.log({name: wandb.Image(fig)}, step=step)
          └  fig.savefig(...)  (if logging.plots.save = true)
```

The design goal is **zero blocking**: the training loop submits intermediate data and moves on. Matplotlib rendering and W&B upload happen on a separate daemon thread.

---

## Step 1 — Models Emit Data via `sow`

Flax's `self.sow(collection, name, value)` stashes a value into a named mutable collection during a forward pass. Theseus uses the `"plots"` collection for visualization data:

```python
class ForkingBlock(Module):
    def __call__(self, x, ...):
        ...
        new_cumulative_scores = ...  # jax.Array, computed during forward
        self.sow("plots", "new_cumulative_scores", new_cumulative_scores)
        return x_out
```

Any sub-module at any depth can call `self.sow("plots", ...)` — Flax aggregates all sowed values into a nested dict keyed by module path.

---

## Step 2 — Val Step Captures the `plots` Collection

During validation the trainer calls `forward()` with `intermediates=True`:

```python
_, loss, meta = cls.forward(
    state, params, batch, deterministic=True, intermediates=True
)
```

`forward()` passes `mutable=["intermediates", "plots"]` to `state.apply_fn`, which tells Flax to capture all `sow` calls and return them as a second output:

```python
(logits, loss), mutated = state.apply_fn(
    {"params": params},
    x, y,
    mutable=["intermediates", "plots"],
)
meta = {
    "intermediates": mutated.get("intermediates", {}),
    "plots":         mutated.get("plots", {}),
}
```

The resulting `meta["plots"]` dict looks like:

```python
{
    "ForkingBlock_0": {"new_cumulative_scores": jax.Array(...)},
    "ForkingBlock_3": {"new_cumulative_scores": jax.Array(...)},
    ...
}
```

---

## Step 3 — Submit to the Plotter

Back in the training loop, the main process hands the captured data off to the background plotter:

```python
if self.main_process():
    plots_meta = jax.device_get(meta)   # pull off device
    self.plotter.submit(plots_meta, step=step)
```

`jax.device_get` ensures the arrays are numpy arrays (off-device) before queuing. `plotter.submit()` enqueues `(plot_fn, step)` via a `Queue(maxsize=8)` — if the queue is full (background thread is falling behind), it blocks here briefly as natural backpressure, but this is rare since plotting runs quickly relative to training steps.

---

## Step 4 — Background Thread Renders Figures

`Plotter` spawns a daemon thread at construction time:

```python
class Plotter:
    def __init__(self, model_cls, save, save_dir):
        self.queue = Queue(maxsize=8)
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
```

The worker initializes matplotlib with the Agg (non-interactive) backend and calls `apply_theme()` once. Then it loops:

```python
def _worker(self):
    matplotlib.use("Agg")
    apply_theme()
    while True:
        plot_fn, step = self.queue.get(timeout=0.5)
        figures = plot_fn()                              # calls model_cls.plot(...)
        for name, fig in figures.items():
            wandb.log({name: wandb.Image(fig)}, step=step)
            if self.save:
                fig.savefig(self.save_dir / f"{name}_step{step}.pdf")
            plt.close(fig)
```

---

## `Module.plot()` — The Override Point

Every model that wants to produce figures overrides the static `plot()` method on `Module`:

```python
class Module(nn):
    @staticmethod
    def plot(intermediates: Any) -> Dict[str, Any]:
        """Override to produce per-validation figures.

        Returns:
            Dict mapping figure name to matplotlib Figure.
            An empty dict means no plots (default).
        """
        return {}
```

The `intermediates` argument is the `meta` dict from Step 2 above — a nested structure of numpy arrays (already pulled off-device).

### Example: `Thoughtbubbles.plot()`

```python
class Thoughtbubbles(GPT):
    @staticmethod
    def plot(intermediates):
        scores = [v["new_cumulative_scores"]
                  for v in intermediates["plots"].values()]
        scores = jnp.exp(jnp.stack(jnp.array(scores))[:, 0, 0])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(np.array(scores).astype(np.float32), ax=ax)
        return {"analysis/cum_scores": fig}
```

`plot()` is a `@staticmethod` deliberately — it receives only data arrays, never `self`, so it can be called from the background thread without any thread-safety concerns around the model or its params.

---

## Trainer Setup: Opt-In Detection

The trainer detects whether the model has a real `plot()` override at construction time:

```python
model_cls = self.MODEL if self.MODEL.plot is not Module.plot else None
self.plotter = Plotter(
    model_cls=model_cls,
    save=plots_cfg.save,
    save_dir=save_dir,
)
```

If `model_cls is None`, `plotter.submit()` is a no-op — no data is queued, no thread is woken up. Models that don't need plots pay zero cost.

---

## Config: Saving Figures to Disk

Set `logging.plots.save: true` in your config YAML to additionally save every figure as a PDF:

```yaml
logging:
  plots:
    save: true
```

Figures are saved under `{cluster.root}/{project}/{group}/{run_name}/`, one file per figure per validation step: `analysis_cum_scores_step4096.pdf`.

---

## `apply_theme()` — Publication-Ready Defaults

`apply_theme()` applies a consistent visual style to all figures produced by `plot()`:

- **Font**: Times New Roman / DejaVu Serif serif stack
- **Palette**: muted teal / orange / green / mauve / violet / red
- **Spine**: top and right spines removed; horizontal grid lines
- **Tick locator**: `MaxNLocator(nbins="auto")` patched onto every new `Axes` so tick density adapts to figure size
- **Sub-figure labels**: optional `axes=` argument labels panels (a), (b), (c)...

Call it directly for notebook/script use:

```python
from theseus.plot import apply_theme
apply_theme()
```

---

## Error Propagation

Exceptions inside the worker thread are caught and stored:

```python
except Exception as e:
    self.error = e
```

On the next call to `plotter.plot()` or `plotter.submit()`, the stored error is re-raised on the main thread. This surfaces plotting bugs during training rather than silently swallowing them.

---

## Lifecycle

```
Plotter.__init__()        ← called once at trainer startup
plotter.submit(meta, step) ← called every val step (main process only)
plotter.close()           ← called at end of training; drains queue, joins thread
```

`close()` is also called from `__del__` as a safety net, but explicit `close()` at the end of training is preferred since it ensures all queued figures are flushed before the process exits.
