import random
from pathlib import Path

from abc import abstractmethod
from loguru import logger
from typing import Generic, TypeVar, Dict, Any, Tuple, Type, Self, List, Union

import jax
import json
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils

from omegaconf import OmegaConf

from theseus.base import _BaseJob, ExecutionSpec, PyTree, JobSpec
from theseus.config import current_config, configure, configuration


C = TypeVar("C")


class BasicJob(_BaseJob, Generic[C]):
    @classmethod
    def config(cls) -> Union[Type[C], List[Type[Any]]]:
        raise NotImplementedError()

    def __init__(self, spec: ExecutionSpec):
        cfg_type = self.config()

        self.args: C

        if isinstance(cfg_type, list):
            self.args = configure(cfg_type[0])
        else:
            self.args = configure(cfg_type)

        self.spec = spec

    def main_process(self) -> bool:
        res: bool = jax.process_index() == 0
        return res

    @property
    def done(self) -> bool:
        """Check if job is already complete (idempotency check)"""
        return False

    @abstractmethod
    def run(self) -> None:
        """Run the job, assuming all hosts have setup"""
        raise NotImplementedError()

    def finish(self) -> None:
        # Finalize wandb run (if active) so sequential stages or callers
        # get a clean slate.  Safe to call even when wandb was never imported.
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    def __call__(self) -> None:
        if self.done:
            logger.info(f"JOB {self.spec.name} | already done, skipping")
            return
        logger.info(f"JOB {self.spec.name} | starting")
        logger.debug(f"JOB {self.spec.name} | pre-start sync")
        multihost_utils.sync_global_devices(f"{self.spec.name}:start")
        logger.debug(f"JOB {self.spec.name} | syncronized, starting")
        self.run()
        logger.debug(f"JOB {self.spec.name} | finishd, waiting for everyone...")
        multihost_utils.sync_global_devices(f"{self.spec.name}:finish")
        self.finish()

    @classmethod
    def local(
        cls,
        root_dir: str,
        name: str = "local",
        project: str | None = None,
        group: str | None = None,
    ) -> Self:
        spec = ExecutionSpec.local(
            root_dir,
            name=name,
            project=project,
            group=group,
        )
        return cls(spec)


class CheckpointedJob(BasicJob[C], Generic[C]):
    """Job with checkpoint save/load support.

    Checkpoint path resolution
    --------------------------
    Paths are split into two parts:

        checkpoints_dir / rel_path

    where ``checkpoints_dir`` comes from the cluster config and ``rel_path``
    is ``project/group/job_name/suffix``.

    The ``*_from_path`` methods accept an arbitrary ``rel_path``, which lets
    a job load/save checkpoints belonging to a *different* job.  The plain
    ``get_tree_and_metadata`` / ``save_tree_and_metadata`` methods derive
    ``rel_path`` from ``self.spec`` automatically — they exist for backwards
    compatibility and are thin wrappers around the ``*_from_path`` variants.

    ``_get_checkpoint_path`` is a legacy static helper used by external
    callers (scripts, inference, RestoreableJob) that returns the full
    absolute path.  It is kept for backwards compatibility.
    """

    def __init__(self, spec: ExecutionSpec):
        super().__init__(spec)
        self.key = jax.random.PRNGKey(0)

    # -- path helpers --------------------------------------------------

    @staticmethod
    def _get_checkpoints_dir(spec: ExecutionSpec) -> Path:
        """Resolve the cluster's checkpoints directory for the current process."""
        return spec.hardware.hosts[jax.process_index()].cluster.checkpoints_dir

    @staticmethod
    def _get_checkpoint_rel_path(spec: ExecutionSpec, suffix: str | Path) -> Path:
        """Return ``project/group/name/suffix`` (no checkpoints_dir prefix)."""
        project = spec.project or "general"
        group = spec.group if spec.group else "default"
        return Path(project) / group / spec.name / str(suffix)

    @staticmethod
    def _get_checkpoint_path(spec: ExecutionSpec, suffix: str | Path) -> Path:
        """Full absolute path: ``checkpoints_dir/project/group/name/suffix``.

        Kept for backwards compatibility with external callers.  Prefer the
        ``*_from_path`` instance methods for new code.
        """
        return CheckpointedJob._get_checkpoints_dir(
            spec
        ) / CheckpointedJob._get_checkpoint_rel_path(spec, suffix)

    # -- load ----------------------------------------------------------

    def get_tree_and_metadata_from_path(
        self, rel_path: str | Path, template_tree: PyTree[Any]
    ) -> Tuple[PyTree[Any], Dict[str, Any]]:
        """Load tree and metadata from ``rel_path`` under checkpoints_dir."""
        path = self._get_checkpoints_dir(self.spec) / Path(rel_path)

        try:
            rng_state = np.load(path / "rng.npy", allow_pickle=True).item()
            random.setstate(rng_state["python_random"])
            np.random.set_state(rng_state["numpy_random"])
            self.key = jax.random.PRNGKey(rng_state["jax_random"])
        except EOFError:
            self.key = jax.random.PRNGKey(0)

        checkpointer = ocp.StandardCheckpointer()

        def _to_sharded_struct(x: Any) -> Any:
            if isinstance(x, jax.Array):
                return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)  # type: ignore
            return x

        sharded_target = jax.tree_util.tree_map(_to_sharded_struct, template_tree)
        restored = checkpointer.restore(path / "checkpoint", target=sharded_target)

        with open(path / "config.json", "r") as df:
            data = json.load(df)

        return restored, data

    def get_tree_and_metadata(
        self, suffix: str | Path, template_tree: PyTree[Any]
    ) -> Tuple[PyTree[Any], Dict[str, Any]]:
        """Load from this job's own checkpoint. Wrapper for backwards compat."""
        return self.get_tree_and_metadata_from_path(
            self._get_checkpoint_rel_path(self.spec, suffix), template_tree
        )

    # -- save ----------------------------------------------------------

    def save_tree_and_metadata_from_path(
        self, rel_path: str | Path, tree: PyTree[Any], metadata: Dict[str, Any]
    ) -> None:
        """Save tree and metadata to ``rel_path`` under checkpoints_dir."""
        path = self._get_checkpoints_dir(self.spec) / Path(rel_path)
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)

        multihost_utils.sync_global_devices("save:pre")

        if self.main_process():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("CHECKPOINT | created checkpoint directory")

            rng_state = {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "jax_random": int(self.key[0]),
            }
            np.save(path / "rng.npy", rng_state)  # type: ignore
            logger.debug("CHECKPOINT | saved random state")

            with open(path / "config.json", "w") as df:
                json.dump(metadata, df)
            logger.debug("CHECKPOINT | saved configuration")

            job_spec_data = {
                field_name: getattr(self.spec, field_name)
                for field_name in JobSpec.model_fields
            }
            with open(path / "job.json", "w") as df:
                json.dump(job_spec_data, df)
            logger.debug("CHECKPOINT | saved job spec")

            cfg = current_config()
            if cfg is not None:
                with open(path / "config.yaml", "w") as df:
                    df.write(OmegaConf.to_yaml(cfg))
                logger.debug("CHECKPOINT | saved config.yaml")

        multihost_utils.sync_global_devices("save:mid")

        checkpointer = ocp.StandardCheckpointer()

        def make_global_array(x: PyTree[Any]) -> PyTree[Any]:
            if isinstance(x, jax.Array):
                if len(x.sharding.device_set) == 1:
                    broadcasted: PyTree[Any] = multihost_utils.broadcast_one_to_all(x)
                    return broadcasted
            return x

        state_to_save = jax.tree_util.tree_map(make_global_array, tree)

        checkpointer.save(path / "checkpoint", state_to_save, force=True)
        checkpointer.wait_until_finished()
        logger.debug("CHECKPOINT | saved training state")

        multihost_utils.sync_global_devices("save:post")

    def save_tree_and_metadata(
        self, suffix: str | Path, tree: PyTree[Any], metadata: Dict[str, Any]
    ) -> None:
        """Save to this job's own checkpoint. Wrapper for backwards compat."""
        self.save_tree_and_metadata_from_path(
            self._get_checkpoint_rel_path(self.spec, suffix), tree, metadata
        )


class RestoreableJob(CheckpointedJob[C], Generic[C]):
    @abstractmethod
    def restore(self, suffix: Path) -> None:
        """Restore job state from checkpoint with given suffix"""
        raise NotImplementedError()

    def register(self, suffix: str | Path) -> None:
        """Register this checkpoint as the latest, for idempotent restore."""
        if not self.main_process():
            return
        path = self._get_checkpoint_path(self.spec, "latest")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(suffix))
        logger.debug(f"CHECKPOINT | registered {suffix} as latest")

    @classmethod
    def latest(cls, spec: ExecutionSpec) -> str | None:
        """Get the latest checkpoint suffix, or None if no checkpoint exists."""
        path = CheckpointedJob._get_checkpoint_path(spec, "latest")
        if not path.exists():
            return None
        return path.read_text().strip()

    @classmethod
    def from_checkpoint(
        cls,
        suffix: str | Path,
        spec: ExecutionSpec,
        runtime_cfg: Any | None = None,
    ) -> Tuple[Self, Any]:
        """loads and instantiates a checkpointed job from disk

        Args:
            suffix: checkpoint suffix to restore from
            spec: execution spec to use for locating checkpoint
            runtime_cfg: config values from the current launch to overlay onto
                the checkpoint config before job initialization

        Returns:
            Tuple[Self, Any]: restored job instance and configuration
        """

        # use the current spec to identify paths
        path = CheckpointedJob._get_checkpoint_path(spec, suffix)
        logger.debug("CHECKPOINT | restoring checkpointed job at {}", path)

        # Load job spec (only JobSpec fields, not ExecutionSpec)
        # Uses JobSpec.model_fields to be extensible - new fields added to JobSpec
        # will automatically be included without modifying this code
        with open(path / "job.json", "r") as df:
            job_spec_data = json.load(df)

        # Create new ExecutionSpec with loaded JobSpec fields
        for k, v in job_spec_data.items():
            setattr(spec, k, v)
        new_spec_obj = spec
        logger.debug("CHECKPOINT | restored job spec")

        # load config now from config.yaml
        cfg = OmegaConf.load(path / "config.yaml")
        if runtime_cfg is not None:
            cfg = OmegaConf.merge(cfg, runtime_cfg)

        # instantiate job within configuration context
        with configuration(cfg):
            # if there is a job field, then we need to use that to instantiate
            if "job" in cfg:
                from theseus.registry import JOBS

                job_cls = JOBS.get(cfg.job)
                if job_cls is None:
                    logger.warning(
                        f"Unknown job type '{cfg.job}' in configuration, defaulting to {cls.__name__}"
                    )
                    job_cls = cls
                elif not issubclass(job_cls, cls):
                    logger.warning(
                        f"Configured job type '{cfg.job}' is not a subclass of {cls.__name__}, defaulting to {cls.__name__}"
                    )
                    job_cls = cls
            else:
                logger.warning(
                    f"No job type specified in configuration, defaulting to {cls.__name__}"
                )
                job_cls = cls

            job = job_cls(new_spec_obj)
            job.restore(Path(suffix))

        logger.debug(f"CHECKPOINT | restored checkpointed job {new_spec_obj.name}")

        return job, cfg

    @classmethod
    def checkpoints(cls, spec: ExecutionSpec) -> List[str]:
        """given the execution spec, list available checkpoints to restore from"""

        path = CheckpointedJob._get_checkpoint_path(spec, "")
        if not path.exists():
            return []

        suffixes = []
        stack = [path]
        while stack:
            current = stack.pop()
            if (current / "config.yaml").exists():
                suffixes.append(str(current.relative_to(path)))
            else:
                for item in current.iterdir():
                    if item.is_dir():
                        stack.append(item)

        return suffixes
