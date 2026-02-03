import random
from pathlib import Path

from abc import abstractmethod, abstractproperty
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

    @abstractproperty
    def done(self) -> bool:
        """Check if job is already complete (idempotency check)"""
        raise NotImplementedError()

    @abstractmethod
    def run(self) -> None:
        """Run the job, assuming all hosts have setup"""
        raise NotImplementedError()

    def finish(self) -> None: ...

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
    def __init__(self, spec: ExecutionSpec):
        super().__init__(spec)
        self.key = jax.random.PRNGKey(0)

    @staticmethod
    def _get_checkpoint_path(spec: ExecutionSpec, suffix: str | Path) -> Path:
        """
        Compute checkpoint path based on process index, project, group, and job name.

        Directory structure: checkpoints_dir/project/group/job_name/suffix/
        - project: defaults to "misc" if None
        - group: defaults to "default" if None or empty string
        - job_name: required (self.spec.name)
        - suffix: provided by caller (e.g., "step_1000", "final", or nested "best/model")
        """
        process_index = jax.process_index()
        checkpoint_dir = spec.hardware.hosts[process_index].cluster.checkpoints_dir

        # Handle project: use default "theseus" if None
        project = spec.project or "general"

        # Handle group: use "default" if None or empty string
        group = spec.group if spec.group else "default"

        # Build path: checkpoints_dir/project/group/job_name/suffix/
        path = checkpoint_dir / project / group / spec.name / str(suffix)

        return path

    def get_tree_and_metadata(
        self, suffix: str | Path, template_tree: PyTree[Any]
    ) -> Tuple[PyTree[Any], Dict[str, Any]]:
        path = self._get_checkpoint_path(self.spec, suffix)

        try:
            rng_state = np.load(path / "rng.npy", allow_pickle=True).item()
            random.setstate(rng_state["python_random"])
            np.random.set_state(rng_state["numpy_random"])
            self.key = jax.random.PRNGKey(rng_state["jax_random"])
        except EOFError:
            self.key = jax.random.PRNGKey(0)

        # Load checkpoint using Orbax
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(path / "checkpoint", target=template_tree)

        # Load metadata
        with open(path / "config.json", "r") as df:
            data = json.load(df)

        return restored, data

    def save_tree_and_metadata(
        self, suffix: str | Path, tree: PyTree[Any], metadata: Dict[str, Any]
    ) -> None:
        path = self._get_checkpoint_path(self.spec, suffix)
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)

        multihost_utils.sync_global_devices("save:pre")

        # Write directly to shared filesystem path (multi-host safe)
        if self.main_process():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("CHECKPOINT | created checkpoint directory")

            # Save random state
            rng_state = {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "jax_random": int(self.key[0]),  # Save seed
            }
            np.save(path / "rng.npy", rng_state)  # type: ignore
            logger.debug("CHECKPOINT | saved random state")

            # Save config
            with open(path / "config.json", "w") as df:
                json.dump(
                    metadata,
                    df,
                )
            logger.debug("CHECKPOINT | saved configuration")

            # Save job spec (only JobSpec fields, not ExecutionSpec)
            # Uses JobSpec.model_fields to be extensible - new fields added to JobSpec
            # will automatically be included without modifying this code
            job_spec_data = {
                field_name: getattr(self.spec, field_name)
                for field_name in JobSpec.model_fields
            }
            with open(path / "job.json", "w") as df:
                json.dump(job_spec_data, df)
            logger.debug("CHECKPOINT | saved job spec")

            # Save current OmegaConf configuration as YAML
            cfg = current_config()
            if cfg is not None:
                with open(path / "config.yaml", "w") as df:
                    df.write(OmegaConf.to_yaml(cfg))
                logger.debug("CHECKPOINT | saved config.yaml")

        multihost_utils.sync_global_devices("save:mid")

        # Save checkpoint - convert host-local arrays to global arrays for multi-host
        # This handles replicated scalars like 'step' that have SingleDeviceSharding
        checkpointer = ocp.StandardCheckpointer()

        # Convert any host-local arrays to globally replicated arrays
        def make_global_array(x: PyTree[Any]) -> PyTree[Any]:
            if isinstance(x, jax.Array):
                # If it's a host-local single-device array, make it globally replicated
                if len(x.sharding.device_set) == 1:
                    broadcasted: PyTree[Any] = multihost_utils.broadcast_one_to_all(x)
                    return broadcasted
            return x

        state_to_save = jax.tree_util.tree_map(make_global_array, tree)

        checkpointer.save(path / "checkpoint", state_to_save, force=True)
        checkpointer.wait_until_finished()
        logger.debug("CHECKPOINT | saved training state")

        multihost_utils.sync_global_devices("save:post")


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
        cls, suffix: str | Path, spec: ExecutionSpec
    ) -> Tuple[Self, Any]:
        """loads and instantiates a checkpointed job from disk

        Args:
            suffix: checkpoint suffix to restore from
            spec: execution spec to use for locating checkpoint

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
