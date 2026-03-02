# Design

Architecture and design documents for theseus internals.

## Documents

- [Dispatch Infrastructure](dispatch.md) — how `theseus submit` and `theseus repl` get code onto remote machines and run it.
- [Config System](config.md) — how `field()`, `build()`, and `configuration()` turn annotated dataclasses into a unified OmegaConf schema.
- [Mock System](mock.md) — how `Mocker` lets you run model analysis code on CPU without real parameters.
- [Plot System](plot.md) — how models expose figures via `sow` and `plot()`, and how the trainer logs them asynchronously to W&B.
