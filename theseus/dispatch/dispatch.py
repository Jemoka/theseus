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
from pathlib import Path

from loguru import logger
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
                    "log": h.cluster.log,
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
    project = spec.project or "general"
    group = spec.group or "default"
    return f"{cluster_work}/{project}/{group}/{spec.name}"


def dispatch(
    cfg: DictConfig,
    spec: JobSpec,
    hardware: HardwareRequest,
    dispatch_config: DispatchConfig,
    dirty: bool = False,
    check_availability: bool = True,
    mem: str | None = None,
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
        hardware: Hardware requirements (includes cluster preferences)
        dispatch_config: Remote host/cluster configuration
        dirty: Include uncommitted changes (default: False)
        check_availability: Check real-time GPU availability (default: True)
        mem: Memory override for SLURM jobs (e.g., "64G", "128G")
        timeout: SSH timeout in seconds

    Returns:
        SlurmResult for SLURM clusters, RunResult for plain SSH hosts

    Raises:
        RuntimeError: If job not found in registry or no hardware available
    """
    logger.info(
        f"DISPATCH | starting dispatch for job '{spec.name}' (project={spec.project}, group={spec.group})"
    )
    logger.debug(
        f"DISPATCH | hardware request: {hardware.min_chips}x {hardware.chip.name}"
    )

    # 1. Validate job exists
    job_key = cfg.job
    if job_key not in JOBS:
        logger.error(f"DISPATCH | job '{job_key}' not in registry")
        raise RuntimeError(
            f"Job '{job_key}' not in registry. Available: {list(JOBS.keys())}"
        )
    logger.debug(f"DISPATCH | validated job '{job_key}' exists in registry")

    # 2. Solve for hardware
    logger.debug("DISPATCH | solving for hardware allocation...")
    solve_result = solve_or_raise(
        hardware,
        dispatch_config,
        check_availability=check_availability,
        timeout=timeout,
    )
    assert solve_result.result is not None
    assert solve_result.host_config is not None
    logger.info(
        f"DISPATCH | solved: host={solve_result.host_name}, is_slurm={solve_result.is_slurm}, chips={solve_result.result.total_chips}"
    )

    # 3. Get cluster info
    inventory = RemoteInventory(dispatch_config)
    cluster_config = dispatch_config.clusters[solve_result.host_config.cluster]
    cluster = inventory.get_cluster(solve_result.host_config.cluster)
    work_dir = _get_work_dir(cluster.work, spec)
    # Shared dir for scripts visible to all nodes (login + compute)
    share_dir = cluster_config.share or f"{cluster.work}/.dispatch"
    logger.debug(
        f"DISPATCH | work_dir={work_dir}, share_dir={share_dir}, cluster={cluster.name}"
    )

    # JuiceFS mount info if configured
    juicefs_mount: JuiceFSMount | None = None
    if cluster_config.mount:
        juicefs_mount = JuiceFSMount(
            redis_url=cluster_config.mount,
            mount_point=cluster.root,
            cache_size=cluster_config.cache_size,
            cache_dir=cluster_config.cache_dir,
        )
        logger.debug(f"DISPATCH | JuiceFS mount configured: {cluster.root}")

    # 4. Generate bootstrap script (Python script that runs the job)
    logger.debug("DISPATCH | generating bootstrap script")
    bootstrap_py_content = _generate_bootstrap(cfg, solve_result.result, spec)

    # 5. Submit based on host type
    if solve_result.is_slurm:
        logger.info(
            f"DISPATCH | submitting via SLURM to {solve_result.host_config.ssh}"
        )
        return _dispatch_slurm(
            solve_result,
            spec,
            dispatch_config,
            work_dir,
            share_dir,
            cluster,
            juicefs_mount,
            bootstrap_py_content,
            mem,
            dirty,
            timeout,
        )
    else:
        logger.info(f"DISPATCH | submitting via SSH to {solve_result.host_config.ssh}")
        return _dispatch_plain(
            solve_result,
            work_dir,
            cluster,
            juicefs_mount,
            spec,
            bootstrap_py_content,
            dirty,
            timeout,
        )


def _dispatch_slurm(
    solve_result: SolveResult,
    spec: JobSpec,
    dispatch_config: DispatchConfig,
    work_dir: str,
    share_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    bootstrap_py_content: str,
    mem: str | None,
    dirty: bool,
    timeout: float,
) -> SlurmResult:
    """Dispatch job via SLURM."""
    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, SlurmHostConfig)
    assert solve_result.result is not None

    host_config = solve_result.host_config
    gpus_per_node = solve_result.result.total_chips // max(
        len(solve_result.result.hosts), 1
    )

    # Look up GPU type from gres_mapping using chip name
    chip_name = solve_result.result.chip.name if solve_result.result.chip else None
    gpu_type = dispatch_config.gres_mapping.get(chip_name) if chip_name else None

    # Use mem override if provided, otherwise fall back to host config
    job_mem = mem or host_config.mem

    logger.debug(
        f"DISPATCH | SLURM dispatch: partition={solve_result.partition}, nodes={len(solve_result.result.hosts)}, gpus_per_node={gpus_per_node}, gpu_type={gpu_type}, mem={job_mem}"
    )

    # Build job name from spec
    project = spec.project or "general"
    group = spec.group or "default"
    job_name = f"{project}-{group}-{spec.name}"

    # Build SlurmJob with embedded bootstrap Python script
    job = SlurmJob(
        name=job_name,
        command="python _bootstrap_dispatch.py",
        partition=solve_result.partition,
        nodes=len(solve_result.result.hosts),
        gpus_per_node=gpus_per_node,
        gpu_type=gpu_type,
        mem=job_mem,  # uses CLI override, then config value, then defaults to 64G in slurm.py
        account=host_config.account,
        qos=host_config.qos,
        exclude=host_config.exclude,
        uv_groups=host_config.uv_groups,
        payload_extract_to=work_dir,
        output=f"{cluster.log_dir}/{job_name}-%j.out",
        juicefs_mount=juicefs_mount,
        bootstrap_py=bootstrap_py_content,
    )

    result = submit_packed(
        job,
        host_config.ssh,
        share_dir=share_dir,
        dirty=dirty,
        timeout=timeout,
    )

    if result.ok:
        logger.info(
            f"DISPATCH | SLURM job submitted successfully, job_id={result.job_id}"
        )
    else:
        logger.error(f"DISPATCH | SLURM submission failed: {result.ssh_result.stderr}")

    return result


def _dispatch_plain(
    solve_result: SolveResult,
    work_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    spec: JobSpec,
    bootstrap_py_content: str,
    dirty: bool,
    timeout: float,
) -> RunResult:
    """Dispatch job via plain SSH using bootstrap.sh (non-blocking with nohup)."""
    from datetime import datetime

    assert solve_result.host_config is not None
    assert not isinstance(solve_result.host_config, SlurmHostConfig)

    host_config = solve_result.host_config
    ssh_alias = host_config.ssh
    log_dir = cluster.log_dir

    # Build log filename with job metadata and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = spec.project or "general"
    group = spec.group or "default"
    log_file = f"{log_dir}/{project}_{group}_{spec.name}_{timestamp}.log"

    logger.debug(f"DISPATCH | SSH dispatch: host={ssh_alias}, work_dir={work_dir}")

    # Build bootstrap job (reusing SlurmJob but for SSH mode)
    job = SlurmJob(
        name="theseus-dispatch",
        command="python _bootstrap_dispatch.py",
        is_slurm=False,  # SSH mode - no SBATCH directives
        uv_groups=host_config.uv_groups,
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
        bootstrap_py=bootstrap_py_content,
    )

    # Generate bootstrap script (without SBATCH directives since partition=None)
    script = job.to_script()

    # Ship code first
    logger.debug(f"DISPATCH | shipping code to {ssh_alias}:{work_dir} (dirty={dirty})")
    if dirty:
        ship_result = ship_dirty(ssh_alias, work_dir, timeout=timeout)
    else:
        ship_result = ship(ssh_alias, work_dir, timeout=timeout)

    if not ship_result.ok:
        logger.error(f"DISPATCH | failed to ship code: {ship_result.stderr}")
        return ship_result

    logger.debug("DISPATCH | code shipped successfully")

    # Write bootstrap.sh script to remote
    bootstrap_remote = f"{work_dir}/_bootstrap.sh"
    logger.debug(f"DISPATCH | writing bootstrap.sh to {bootstrap_remote}")
    write_cmd = f"cat > {bootstrap_remote} << 'BOOTSTRAP_EOF'\n{script}BOOTSTRAP_EOF"
    write_result = run(write_cmd, ssh_alias, timeout=timeout)
    if not write_result.ok:
        logger.error(
            f"DISPATCH | failed to write bootstrap script: {write_result.stderr}"
        )
        return write_result

    # Write _bootstrap_dispatch.py to remote (not included in git archive)
    bootstrap_py_remote = f"{work_dir}/_bootstrap_dispatch.py"
    logger.debug(f"DISPATCH | writing _bootstrap_dispatch.py to {bootstrap_py_remote}")
    write_py_cmd = f"cat > {bootstrap_py_remote} << 'BOOTSTRAP_PY_EOF'\n{bootstrap_py_content}BOOTSTRAP_PY_EOF"
    write_py_result = run(write_py_cmd, ssh_alias, timeout=timeout)
    if not write_py_result.ok:
        logger.error(
            f"DISPATCH | failed to write bootstrap Python script: {write_py_result.stderr}"
        )
        return write_py_result

    # Run bootstrap script with nohup (non-blocking)
    logger.debug(f"DISPATCH | launching job via nohup, logs at {log_file}")
    run_cmd = (
        f"mkdir -p {log_dir} && "
        f"chmod +x {bootstrap_remote} && "
        f"nohup {bootstrap_remote} > {log_file} 2>&1 &"
    )
    result = run(run_cmd, ssh_alias, timeout=timeout)

    # Include log path in stdout for user reference
    if result.ok:
        logger.info(f"DISPATCH | SSH job started in background on {ssh_alias}")
        logger.debug(f"DISPATCH | logs at {ssh_alias}:{log_file}")
        return RunResult(
            returncode=result.returncode,
            stdout=f"Job started in background. Logs: {ssh_alias}:{log_file}\n{result.stdout}",
            stderr=result.stderr,
        )
    else:
        logger.error(f"DISPATCH | failed to launch job: {result.stderr}")
    return result
