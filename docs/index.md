# Getting Started

Theseus is a heterogeneous pretraining infrastructure and architecture experimentation platform.

## Installation

First, clone the repo

```bash
git clone https://github.com/Jemoka/theseus.git
```

For dependencies, we recommend using uv to perform installation; this depends on who your compute partner is...

- cuda13: `uv sync --group all --group cuda13`
- cuda12: `uv sync --group all --group cuda12`
- TPUs: `uv sync --group all --group tpu`
- CPU: `uv sync --group all --group cpu`

## Running

You have two options:

1. **Run theseus on the machine its installed in**: You can install it directly onto your infrastructure, and thus sync with dependency group relating to the hardware; this enables [running experiments locally](Tutorials/running.md/#locally)
2. **Run theseus on a remote machine or SLURM**: You can leverage the remote dispatch infrastructure to [run on a remote cluster](Tutorials/running.md/#dispatch-remotely); if you do, you have to install the `cpu` variant on your head node / local machine, and use the dispatch CLI there.

For most use-cases, the first option will do just fine. For specific cases where remote infrastructure is not a normal shell (e.g., SLURM), the remote dispatch would work better.

<!-- ## Quick Start -->
<!-- Use the CLI. -->

<!-- ```bash -->
<!-- # List available jobs -->
<!-- theseus jobs -->

<!-- # Generate a config for data tokenization -->
<!-- theseus configure data/tokenize_variable_dataset tokenize.yaml \ -->
<!--     data.name=fineweb data.max_samples=1000000 -->

<!-- # Run the tokenization locally -->
<!-- theseus run tokenize-fineweb tokenize.yaml ./output -->

<!-- # Generate a config for pretraining -->
<!-- theseus configure gpt/train/pretrain train.yaml \ -->
<!--     --chip h100 -n 8 -->

<!-- # Run training locally -->
<!-- theseus run my-gpt-run train.yaml ./output -->
<!-- ``` -->

<!-- ### Quick Start, but You Have Infra -->

<!-- Set up `~/.theseus.yaml` (see `examples/dispatch.yaml`), then submit jobs to remote clusters: -->

<!-- ```bash -->
<!-- theseus submit my-run train.yaml --chip h100 -n 8 -->
<!-- ``` -->

<!-- ## Quickish Start -->

<!-- For programmatic configuration and rapid prototyping: -->

<!-- ```python -->
<!-- from theseus.quick import quick -->
<!-- from theseus.registry import JOBS -->

<!-- with quick("gpt/train/pretrain", "/path/to/output", "my-run") as j: -->
<!--     j.config.training.per_device_batch_size = 16 -->
<!--     j.config.logging.checkpoint_interval = 4096 -->
<!--     j()  # run locally -->

<!-- # Or save config for later submission: -->
<!-- with quick("gpt/train/pretrain", "/path/to/output", "my-run") as j: -->
<!--     j.config.training.per_device_batch_size = 16 -->
<!--     j.save("config.yaml", chip="h100", n_chips=8) -->
<!-- ``` -->

<!-- ## Not Quick Start at All -->

<!-- When you (or Claude) manage to find some time to chill you can actually extend this package. The package is organized based around `theseus.job.BasicJob`s. They can be extended with checkpointing and recovery tools. -->

<!-- The main entrypoint to start hacking: -->

<!-- 1. take a look at how to compose a model together in `theseus.model.models.base` -->
<!-- 2. bodge together anything you want to change and make a new model in the models folder (be sure to add it to `theseus.model.models.__init__`) -->
<!-- 3. write an experiment, which is a `RestoreableJob`. A very basic one can just inherit the normal trainer, and then that's about it. see `theseus.experiments.gpt` to get started (be sure to add it to `theseus.experiments.__init__`) -->

<!-- ```python -->
<!-- # theseus/experiments/my_model.py -->
<!-- from theseus.training.base import BaseTrainer, BaseTrainerConfig -->
<!-- from theseus.model.models import MyModel -->

<!-- class PretrainMyModel(BaseTrainer[BaseTrainerConfig, MyModel]): -->
<!--     MODEL = MyModel -->
<!--     CONFIG = BaseTrainerConfig -->

<!--     @classmethod -->
<!--     def schedule(cls): -->
<!--         return "wsd" -->
<!-- ``` -->

<!-- ## JuiceFS Integration -->
<!-- When you are on many remote computors but bursty you may go "aw schucks I need to copy like 50TB of pretraining data around that's so lame!" -->

<!-- Don't worry, we gotchu. If you use the `submit` API, we have a way to ship your root directory around by using a thing called [JuiceFS](https://juicefs.com/en/), which is a distributed filesystem. -->

<!-- In your `~/.theseus.yaml`, add the `mount` field to your cluster config: -->

<!-- ```yaml -->
<!-- clusters: -->
<!--   hpc: -->
<!--     root: /mnt/juicefs/theseus -->
<!--     work: /scratch/theseus -->
<!--     mount: redis://:password@redis.example.com:6379/0 -->
<!--     cache_size: 100G -->
<!--     cache_dir: /scratch/juicefs-cache -->
<!-- ``` -->

<!-- ## Features -->

<!-- - **CLI & Programmatic API**: Configure and run jobs via `theseus` CLI or the `quick()` Python API -->
<!-- - **Remote Dispatch**: Submit jobs to SLURM clusters or plain SSH hosts via `~/.theseus.yaml` -->
<!-- - **Checkpointing & Recovery**: Jobs are `RestoreableJob`s with built-in checkpoint/restore support -->
<!-- - **Data Pipelines**: Tokenize datasets (blockwise or streaming) with `data/tokenize_*` jobs -->
<!-- - **JuiceFS Integration**: Distributed filesystem support for sharing data across clusters -->
<!-- - **Multi-backend**: CUDA 11/12/13, TPU, and CPU via `uv sync --group` -->
<!-- - **Extensible**: Add models in `theseus.model.models`, experiments in `theseus.experiments`, and datasets in `theseus.data.datasets` -->
<!-- - **Dataclass Configs**: Type-safe configuration via dataclasses with OmegaConf, easy configuration with `theseus.config.field` dataclass extension, and Hydra-style cheeky cli overrides (`model.hidden_size=1024`) -->
