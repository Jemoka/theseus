"""
Remote job dispatch: solve hardware → ship code → run job on remote.

Usage:
    from omegaconf import OmegaConf
    from theseus.base.hardware import HardwareRequest
    from theseus.base.chip import SUPPORTED_CHIPS
    from theseus.base.job import JobSpec
    from theseus.dispatch import dispatch, load_dispatch_config

    # Load dispatch configuration
    dispatch_config = load_dispatch_config("dispatch.yaml")

    # Define job config (must have 'job' key pointing to registered job)
    cfg = OmegaConf.load("experiment.yaml")

    # Define job specification
    spec = JobSpec(
        name="my-training-run",
        project="my-project",
        group="experiment-1",
    )

    # Define hardware requirements
    hardware = HardwareRequest(
        chip=SUPPORTED_CHIPS["h100"],
        min_chips=8,
    )

    # Dispatch! Returns SlurmResult or RunResult
    result = dispatch(
        cfg=cfg,
        spec=spec,
        hardware=hardware,
        dispatch_config=dispatch_config,
        dirty=True,  # include uncommitted changes
    )

    if result.ok:
        print(f"Job dispatched: {result}")
    else:
        print(f"Failed: {result.stderr}")
"""

import json
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from theseus.base.hardware import Cluster, HardwareRequest, HardwareResult
from theseus.base.job import JobSpec
from theseus.dispatch.config import (
    DispatchConfig,
    JuiceFSMount,
    SlurmHostConfig,
    RemoteInventory,
)
from theseus.dispatch.slurm import SlurmJob, SlurmResult, submit_packed
from theseus.dispatch.ssh import RunResult, run
from theseus.dispatch.solve import solve_or_raise, SolveResult
from theseus.dispatch.sync import ship, ship_dirty
from theseus.registry import JOBS


# Path to bootstrap template
BOOTSTRAP_TEMPLATE = Path(__file__).parent / "bootstrap.py"


def _serialize_hardware(result: HardwareResult) -> str:
    """Serialize HardwareResult to JSON string."""
    data = {
        "chip": result.chip.name if result.chip else None,
        "total_chips": result.total_chips,
        "hosts": [
            {
                "name": h.name,
                "cluster": {
                    "name": h.cluster.name,
                    "root": h.cluster.root,
                    "work": h.cluster.work,
                },
                "resources": {chip.name: count for chip, count in h.resources.items()},
            }
            for h in result.hosts
        ],
    }
    return json.dumps(data)


def _generate_bootstrap(
    cfg: DictConfig,
    hardware: HardwareResult,
    spec: JobSpec,
) -> str:
    """Generate bootstrap.py script with embedded data."""
    template = BOOTSTRAP_TEMPLATE.read_text()

    # Fill in placeholders
    config_yaml = OmegaConf.to_yaml(cfg)
    hardware_json = _serialize_hardware(hardware)

    script = template.replace("__CONFIG_YAML__", config_yaml)
    script = script.replace("__HARDWARE_JSON__", hardware_json)
    script = script.replace("__JOB_NAME__", spec.name)
    script = script.replace("__PROJECT__", spec.project or "")
    script = script.replace("__GROUP__", spec.group or "")

    return script


def _get_work_dir(cluster_work: str, spec: JobSpec) -> str:
    """Compute work directory path."""
    project = spec.project or "default"
    group = spec.group or "default"
    return f"{cluster_work}/{project}/{group}/{spec.name}"


def dispatch(
    cfg: DictConfig,
    spec: JobSpec,
    hardware: HardwareRequest,
    dispatch_config: DispatchConfig,
    dirty: bool = False,
    timeout: float = 60.0,
) -> SlurmResult | RunResult:
    """Dispatch a job to remote infrastructure.

    This function:
    1. Validates the job exists in registry
    2. Solves for hardware allocation
    3. Generates a bootstrap script with embedded config
    4. Ships code and runs on remote (via SLURM or plain SSH)

    Args:
        cfg: Job configuration (must have cfg.job pointing to a registered job)
        spec: Job specification (name, project, group)
        hardware: Hardware requirements
        dispatch_config: Remote host/cluster configuration
        dirty: Include uncommitted changes (default: False)
        timeout: SSH timeout in seconds

    Returns:
        SlurmResult for SLURM clusters, RunResult for plain SSH hosts

    Raises:
        RuntimeError: If job not found in registry or no hardware available
    """
    # 1. Validate job exists
    job_key = cfg.job
    if job_key not in JOBS:
        raise RuntimeError(
            f"Job '{job_key}' not in registry. Available: {list(JOBS.keys())}"
        )

    # 2. Solve for hardware
    solve_result = solve_or_raise(hardware, dispatch_config, timeout=timeout)
    assert solve_result.result is not None
    assert solve_result.host_config is not None

    # 3. Get cluster info
    inventory = RemoteInventory(dispatch_config)
    cluster_config = dispatch_config.clusters[solve_result.host_config.cluster]
    cluster = inventory.get_cluster(solve_result.host_config.cluster)
    work_dir = _get_work_dir(cluster.work, spec)

    # JuiceFS mount info if configured
    juicefs_mount: JuiceFSMount | None = None
    if cluster_config.mount:
        juicefs_mount = JuiceFSMount(
            redis_url=cluster_config.mount,
            mount_point=cluster.root,
            cache_size=cluster_config.cache_size,
            cache_dir=cluster_config.cache_dir,
        )

    # 4. Generate bootstrap script
    bootstrap_script = _generate_bootstrap(cfg, solve_result.result, spec)

    # 5. Write bootstrap.py to temp file in repo root (will be included in snapshot)
    repo_root = _find_repo_root()
    bootstrap_path = repo_root / "_bootstrap_dispatch.py"

    try:
        bootstrap_path.write_text(bootstrap_script)

        # 6. Submit based on host type
        if solve_result.is_slurm:
            return _dispatch_slurm(
                solve_result,
                work_dir,
                cluster,
                juicefs_mount,
                dirty,
                timeout,
            )
        else:
            return _dispatch_plain(
                solve_result,
                work_dir,
                cluster,
                juicefs_mount,
                dirty,
                timeout,
            )
    finally:
        # 7. Cleanup temp bootstrap file
        if bootstrap_path.exists():
            bootstrap_path.unlink()


def _find_repo_root() -> Path:
    """Find git repository root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Not in a git repository")
    return Path(result.stdout.strip())


def _dispatch_slurm(
    solve_result: SolveResult,
    work_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    dirty: bool,
    timeout: float,
) -> SlurmResult:
    """Dispatch job via SLURM."""
    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, SlurmHostConfig)
    assert solve_result.result is not None

    host_config = solve_result.host_config

    # Build SlurmJob
    job = SlurmJob(
        name="theseus-dispatch",
        command="python _bootstrap_dispatch.py",
        partition=solve_result.partition,
        nodes=len(solve_result.result.hosts),
        gpus_per_node=solve_result.result.total_chips
        // max(len(solve_result.result.hosts), 1),
        account=host_config.account,
        qos=host_config.qos,
        uv_groups=host_config.uv_groups,
        payload_extract_to=work_dir,
        output=f"{cluster.log_dir}/slurm-%j.out",
        juicefs_mount=juicefs_mount,
    )

    return submit_packed(
        job,
        host_config.ssh,
        dirty=dirty,
        timeout=timeout,
    )


def _dispatch_plain(
    solve_result: SolveResult,
    work_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    dirty: bool,
    timeout: float,
) -> RunResult:
    """Dispatch job via plain SSH using bootstrap.sh (non-blocking with nohup)."""
    assert solve_result.host_config is not None
    assert not isinstance(solve_result.host_config, SlurmHostConfig)

    host_config = solve_result.host_config
    ssh_alias = host_config.ssh
    log_dir = cluster.log_dir
    log_file = f"{log_dir}/dispatch.log"

    # Build bootstrap job (reusing SlurmJob but for SSH mode)
    job = SlurmJob(
        name="theseus-dispatch",
        command="python _bootstrap_dispatch.py",
        is_slurm=False,  # SSH mode - no SBATCH directives
        uv_groups=host_config.uv_groups,
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
    )

    # Generate bootstrap script (without SBATCH directives since partition=None)
    script = job.to_script()

    # Ship code first
    if dirty:
        ship_result = ship_dirty(ssh_alias, work_dir, timeout=timeout)
    else:
        ship_result = ship(ssh_alias, work_dir, timeout=timeout)

    if not ship_result.ok:
        return ship_result

    # Write bootstrap script to remote
    bootstrap_remote = f"{work_dir}/_bootstrap.sh"
    write_cmd = f"cat > {bootstrap_remote} << 'BOOTSTRAP_EOF'\n{script}BOOTSTRAP_EOF"
    write_result = run(write_cmd, ssh_alias, timeout=timeout)
    if not write_result.ok:
        return write_result

    # Run bootstrap script with nohup (non-blocking)
    run_cmd = (
        f"mkdir -p {log_dir} && "
        f"chmod +x {bootstrap_remote} && "
        f"nohup {bootstrap_remote} > {log_file} 2>&1 &"
    )
    result = run(run_cmd, ssh_alias, timeout=timeout)

    # Include log path in stdout for user reference
    if result.ok:
        return RunResult(
            returncode=result.returncode,
            stdout=f"Job started in background. Logs: {ssh_alias}:{log_file}\n{result.stdout}",
            stderr=result.stderr,
        )
    return result
