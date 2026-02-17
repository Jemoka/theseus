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
import re
import time
from dataclasses import dataclass
from datetime import datetime
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
from theseus.dispatch.ssh import RunResult, TunnelResult, forward_port, run
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
    target_desc = (
        "cpu-only"
        if hardware.min_chips == 0
        else f"{hardware.min_chips}x {(hardware.chip.name if hardware.chip else 'any-gpu')}"
    )
    logger.debug(f"DISPATCH | hardware request: {target_desc}")

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
    gpus_per_node = None
    if solve_result.result.total_chips > 0:
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
        root_dir=cluster.root,
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
        cpus_per_task=2,
        time="14-0",
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
        root_dir=cluster.root,
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


@dataclass
class ReplResult:
    """Result of launching an interactive Jupyter REPL session."""

    ok: bool
    is_slurm: bool
    selected_host: str
    ssh_host: str
    log_path: str
    job_id: int | None = None
    allocated_hostname: str | None = None
    remote_pid: int | None = None
    remote_port: int | None = None
    token: str | None = None
    remote_url: str | None = None
    local_port: int | None = None
    local_url: str | None = None
    tunnel_pid: int | None = None
    cluster_name: str | None = None
    cluster_root: str | None = None
    cluster_mount: str | None = None
    work_dir: str | None = None
    mailbox_job_id: str | None = None
    stderr: str = ""


def _repl_command(sync_enabled: bool) -> str:
    base = "--with jupyter jupyter lab --ip 0.0.0.0 --no-browser --port 8888 --ServerApp.port_retries=50"
    if not sync_enabled:
        return base

    # Keep payload free of double quotes for safe embedding in:
    # MAIN_COMMAND="__COMMAND__" within bootstrap.sh.
    inner = (
        "set -euo pipefail;"
        "export THESEUS_REPL_WORKDIR=\\$PWD;"
        "uv run --with jupyter jupyter lab --ip 0.0.0.0 --no-browser --port 8888 --ServerApp.port_retries=50 & "
        "THESEUS_REPL_NOTEBOOK_PID=\\$!;"
        "export THESEUS_REPL_NOTEBOOK_PID;"
        "if [[ -z \\${THESEUS_REPL_NOTEBOOK_PID:-} ]]; then echo [repl-sync] notebook-pid-unavailable; fi;"
        "export THESEUS_REPL_MAILBOX_JOB_ID=\\${THESEUS_REPL_MAILBOX_JOB_ID:-\\${SLURM_JOB_ID:-\\$THESEUS_REPL_NOTEBOOK_PID}};"
        "echo \\$THESEUS_REPL_MAILBOX_JOB_ID > .theseus_repl_mailbox_job_id || true;"
        "uv run python -m theseus.dispatch.mailbox.sidecar & "
        "set +u; THESEUS_REPL_SIDECAR_PID=\\$!; set -u;"
        "echo \\$THESEUS_REPL_SIDECAR_PID > .theseus_repl_sidecar.pid || true;"
        "if [[ -z \\${THESEUS_REPL_SIDECAR_PID:-} ]]; then echo [repl-sync] sidecar-pid-unavailable; fi;"
        "cleanup(){ "
        "if [[ -n \\${THESEUS_REPL_NOTEBOOK_PID:-} ]]; then "
        "kill \\$THESEUS_REPL_NOTEBOOK_PID 2>/dev/null || true; "
        "wait \\$THESEUS_REPL_NOTEBOOK_PID 2>/dev/null || true; "
        "fi; "
        "if [[ -n \\${THESEUS_REPL_SIDECAR_PID:-} ]]; then "
        "kill \\$THESEUS_REPL_SIDECAR_PID 2>/dev/null || true; "
        "wait \\$THESEUS_REPL_SIDECAR_PID 2>/dev/null || true; "
        "fi; "
        "};"
        "trap cleanup EXIT INT TERM;"
        "set +e; wait \\$THESEUS_REPL_NOTEBOOK_PID; THESEUS_REPL_RC=\\$?; set -e; "
        "exit \\$THESEUS_REPL_RC"
    )
    return f"bash -lc '{inner}'"


def _parse_jupyter_startup(text: str) -> tuple[str | None, int | None, str | None]:
    """Extract URL, port, and token from Jupyter startup logs."""
    urls = [u.rstrip(".,);]") for u in re.findall(r"https?://[^\s]+", text)]
    remote_url = None
    token = None

    # Prefer a lab URL that explicitly carries token query params.
    for candidate in urls:
        if "/lab" in candidate and "token=" in candidate:
            remote_url = candidate
            token_match = re.search(r"[?&]token=([^&\s]+)", candidate)
            if token_match:
                token = token_match.group(1)
            break

    # Fall back to first non-localhost lab URL, then any lab URL.
    if remote_url is None:
        for candidate in urls:
            if "/lab" in candidate and "127.0.0.1" not in candidate:
                remote_url = candidate
                break
    if remote_url is None:
        for candidate in urls:
            if "/lab" in candidate:
                remote_url = candidate
                break

    port_match = re.search(r":(\d+)", remote_url or "")
    remote_port = int(port_match.group(1)) if port_match else None

    return remote_url, remote_port, token


def _wait_for_jupyter_log(
    host: str,
    log_path: str,
    timeout: float,
    ssh_timeout: float,
) -> tuple[str | None, int | None, str | None, str | None]:
    """Poll a remote log file until Jupyter startup metadata appears."""
    start = time.time()
    missing_file_grace_until = start + min(timeout, 20.0)
    while (time.time() - start) < timeout:
        read_result = run(f"tail -n 200 {log_path}", host, timeout=ssh_timeout)
        if not read_result.ok:
            stderr = (read_result.stderr or "").strip()
            if (
                "No such file or directory" in stderr
                and time.time() < missing_file_grace_until
            ):
                time.sleep(1.0)
                continue
            return (
                None,
                None,
                None,
                f"failed reading remote log '{log_path}' on {host}: {stderr or 'unknown error'}",
            )

        if read_result.stdout.strip():
            text = read_result.stdout
            remote_url, remote_port, token = _parse_jupyter_startup(text)
            if remote_url or token:
                return remote_url, remote_port, token, None

            # bootstrap script emits this on termination; if seen before Jupyter URL,
            # fail fast instead of waiting for timeout.
            if "[bootstrap] cleaning up on" in text:
                return (
                    None,
                    None,
                    None,
                    f"bootstrap exited before Jupyter became ready; check log {host}:{log_path}",
                )
        time.sleep(2.0)
    return None, None, None, None


def _resolve_remote_notebook_pid(
    host: str, port: int, ssh_timeout: float
) -> int | None:
    """Resolve the remote notebook PID bound to the selected port."""
    pid_cmd = (
        f"(lsof -ti :{port} 2>/dev/null | head -n 1) || "
        f"(ss -ltnp 2>/dev/null | grep ':{port} ' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n 1)"
    )
    pid_result = run(pid_cmd, host, timeout=ssh_timeout)
    if not pid_result.ok:
        return None
    for line in pid_result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _read_remote_mailbox_job_id(
    host: str, work_dir: str, ssh_timeout: float
) -> str | None:
    path = f"{work_dir}/.theseus_repl_mailbox_job_id"
    read_result = run(f"cat {path}", host, timeout=ssh_timeout)
    if not read_result.ok:
        return None
    value = read_result.stdout.strip()
    return value or None


def dispatch_repl(
    spec: JobSpec,
    hardware: HardwareRequest,
    dispatch_config: DispatchConfig,
    local_port: int,
    dirty: bool = False,
    check_availability: bool = True,
    mem: str | None = None,
    timeout: float = 60.0,
    startup_timeout: float = 180.0,
    slurm_wait_timeout: float | None = None,
    sync_enabled: bool = False,
) -> ReplResult:
    """Dispatch an interactive Jupyter session on selected infrastructure."""
    solve_result = solve_or_raise(
        hardware,
        dispatch_config,
        check_availability=check_availability,
        timeout=timeout,
    )
    assert solve_result.result is not None
    assert solve_result.host_name is not None
    assert solve_result.host_config is not None

    inventory = RemoteInventory(dispatch_config)
    cluster_config = dispatch_config.clusters[solve_result.host_config.cluster]
    cluster = inventory.get_cluster(solve_result.host_config.cluster)
    work_dir = _get_work_dir(cluster.work, spec)
    share_dir = cluster_config.share or f"{cluster.work}/.dispatch"

    juicefs_mount: JuiceFSMount | None = None
    if cluster_config.mount:
        juicefs_mount = JuiceFSMount(
            redis_url=cluster_config.mount,
            mount_point=cluster.root,
            cache_size=cluster_config.cache_size,
            cache_dir=cluster_config.cache_dir,
        )

    if solve_result.is_slurm:
        return _dispatch_repl_slurm(
            solve_result=solve_result,
            spec=spec,
            dispatch_config=dispatch_config,
            work_dir=work_dir,
            share_dir=share_dir,
            cluster=cluster,
            juicefs_mount=juicefs_mount,
            mem=mem,
            dirty=dirty,
            timeout=timeout,
            startup_timeout=startup_timeout,
            slurm_wait_timeout=slurm_wait_timeout,
            sync_enabled=sync_enabled,
        )

    return _dispatch_repl_plain(
        solve_result=solve_result,
        spec=spec,
        work_dir=work_dir,
        cluster=cluster,
        juicefs_mount=juicefs_mount,
        local_port=local_port,
        dirty=dirty,
        timeout=timeout,
        startup_timeout=startup_timeout,
        sync_enabled=sync_enabled,
    )


def _dispatch_repl_plain(
    solve_result: SolveResult,
    spec: JobSpec,
    work_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    local_port: int,
    dirty: bool,
    timeout: float,
    startup_timeout: float,
    sync_enabled: bool,
) -> ReplResult:
    assert solve_result.host_name is not None
    assert solve_result.host_config is not None
    assert not isinstance(solve_result.host_config, SlurmHostConfig)

    host_name = solve_result.host_name
    host_config = solve_result.host_config
    ssh_alias = host_config.ssh
    log_dir = cluster.log_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = spec.project or "general"
    group = spec.group or "default"
    log_file = f"{log_dir}/{project}_{group}_{spec.name}_{timestamp}.log"

    job = SlurmJob(
        name="theseus-repl",
        command=_repl_command(sync_enabled),
        root_dir=cluster.root,
        is_slurm=False,
        uv_groups=host_config.uv_groups,
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
    )
    script = job.to_script()

    if dirty:
        ship_result = ship_dirty(ssh_alias, work_dir, timeout=timeout)
    else:
        ship_result = ship(ssh_alias, work_dir, timeout=timeout)
    if not ship_result.ok:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=ship_result.stderr,
        )

    bootstrap_remote = f"{work_dir}/_bootstrap_repl.sh"
    write_cmd = f"cat > {bootstrap_remote} << 'BOOTSTRAP_EOF'\n{script}BOOTSTRAP_EOF"
    write_result = run(write_cmd, ssh_alias, timeout=timeout)
    if not write_result.ok:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=write_result.stderr,
        )

    run_cmd = (
        f"mkdir -p {log_dir} && "
        f"chmod +x {bootstrap_remote} && "
        f"nohup {bootstrap_remote} > {log_file} 2>&1 & echo $!"
    )
    launch_result = run(run_cmd, ssh_alias, timeout=timeout)
    if not launch_result.ok:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=launch_result.stderr,
        )

    launcher_pid = None
    for line in reversed(launch_result.stdout.strip().splitlines()):
        line = line.strip()
        if line.isdigit():
            launcher_pid = int(line)
            break

    remote_url, remote_port, token, log_wait_error = _wait_for_jupyter_log(
        ssh_alias, log_file, timeout=startup_timeout, ssh_timeout=timeout
    )
    if log_wait_error:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            remote_pid=launcher_pid,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=log_wait_error,
        )
    if remote_port is None:
        remote_port = 8888
    remote_pid = _resolve_remote_notebook_pid(ssh_alias, remote_port, timeout)
    if remote_pid is None:
        remote_pid = launcher_pid
    mailbox_job_id = None
    if sync_enabled:
        mailbox_job_id = _read_remote_mailbox_job_id(ssh_alias, work_dir, timeout)
    if mailbox_job_id is None and remote_pid is not None:
        mailbox_job_id = str(remote_pid)

    tunnel_result: TunnelResult = forward_port(
        ssh_alias, local_port=local_port, remote_port=remote_port
    )
    if not tunnel_result.ok:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            remote_pid=remote_pid,
            remote_port=remote_port,
            token=token,
            remote_url=remote_url,
            local_port=local_port,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=tunnel_result.stderr or "failed to start SSH tunnel",
        )

    local_url = f"http://localhost:{local_port}/lab"
    if token:
        local_url = f"{local_url}?token={token}"

    return ReplResult(
        ok=True,
        is_slurm=False,
        selected_host=host_name,
        ssh_host=ssh_alias,
        log_path=log_file,
        remote_pid=remote_pid,
        remote_port=remote_port,
        token=token,
        remote_url=remote_url,
        local_port=local_port,
        local_url=local_url,
        tunnel_pid=tunnel_result.pid,
        cluster_name=cluster.name,
        cluster_root=cluster.root,
        cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
        work_dir=work_dir,
        mailbox_job_id=mailbox_job_id,
    )


def _dispatch_repl_slurm(
    solve_result: SolveResult,
    spec: JobSpec,
    dispatch_config: DispatchConfig,
    work_dir: str,
    share_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    mem: str | None,
    dirty: bool,
    timeout: float,
    startup_timeout: float,
    slurm_wait_timeout: float | None,
    sync_enabled: bool,
) -> ReplResult:
    from theseus.dispatch.slurm import cancel, wait_until_running

    assert solve_result.host_name is not None
    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, SlurmHostConfig)
    assert solve_result.result is not None

    host_name = solve_result.host_name
    host_config = solve_result.host_config
    ssh_alias = host_config.ssh
    cluster_cfg = dispatch_config.clusters[host_config.cluster]

    gpus_per_node = solve_result.result.total_chips // max(
        len(solve_result.result.hosts), 1
    )
    chip_name = solve_result.result.chip.name if solve_result.result.chip else None
    gpu_type = dispatch_config.gres_mapping.get(chip_name) if chip_name else None
    job_mem = mem or host_config.mem

    project = spec.project or "general"
    group = spec.group or "default"
    job_name = f"{project}-{group}-{spec.name}"
    output_template = f"{cluster.log_dir}/{job_name}-%j.out"

    job = SlurmJob(
        name=job_name,
        command=_repl_command(sync_enabled),
        root_dir=cluster.root,
        partition=solve_result.partition,
        nodes=len(solve_result.result.hosts),
        gpus_per_node=gpus_per_node,
        gpu_type=gpu_type,
        mem=job_mem,
        account=host_config.account,
        qos=host_config.qos,
        exclude=host_config.exclude,
        uv_groups=host_config.uv_groups,
        payload_extract_to=work_dir,
        output=output_template,
        juicefs_mount=juicefs_mount,
        cpus_per_task=2,
        time="14-0",
    )

    submit_result: SlurmResult = submit_packed(
        job,
        ssh_alias,
        share_dir=share_dir,
        dirty=dirty,
        timeout=timeout,
    )
    if not submit_result.ok or submit_result.job_id is None:
        stderr = submit_result.ssh_result.stderr if submit_result.ssh_result else ""
        return ReplResult(
            ok=False,
            is_slurm=True,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=output_template,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=cluster_cfg.mount,
            work_dir=work_dir,
            stderr=stderr or "SLURM submit failed",
        )

    job_id = submit_result.job_id
    try:
        allocated_hostname, _ = wait_until_running(
            job_id,
            ssh_alias,
            poll_interval=5.0,
            timeout=slurm_wait_timeout,
        )
    except KeyboardInterrupt:
        logger.warning(
            f"DISPATCH | interrupted while waiting for SLURM allocation; cancelling job {job_id}"
        )
        cancel_result = cancel(job_id, ssh_alias, timeout=timeout)
        if not cancel_result.ok:
            logger.warning(
                f"DISPATCH | failed to cancel interrupted SLURM job {job_id}: {cancel_result.stderr}"
            )
        raise
    log_file = output_template.replace("%j", str(job_id))
    if allocated_hostname is None:
        return ReplResult(
            ok=False,
            is_slurm=True,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            job_id=job_id,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=cluster_cfg.mount,
            work_dir=work_dir,
            stderr=f"Timed out waiting for SLURM allocation for job {job_id}",
        )

    remote_url, remote_port, token, log_wait_error = _wait_for_jupyter_log(
        ssh_alias, log_file, timeout=startup_timeout, ssh_timeout=timeout
    )
    if log_wait_error:
        return ReplResult(
            ok=False,
            is_slurm=True,
            selected_host=host_name,
            ssh_host=ssh_alias,
            log_path=log_file,
            job_id=job_id,
            allocated_hostname=allocated_hostname,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=cluster_cfg.mount,
            work_dir=work_dir,
            stderr=log_wait_error,
        )
    if remote_port is None:
        remote_port = 8888

    return ReplResult(
        ok=True,
        is_slurm=True,
        selected_host=host_name,
        ssh_host=ssh_alias,
        log_path=log_file,
        job_id=job_id,
        allocated_hostname=allocated_hostname,
        remote_port=remote_port,
        token=token,
        remote_url=remote_url,
        cluster_name=cluster.name,
        cluster_root=cluster.root,
        cluster_mount=cluster_cfg.mount,
        work_dir=work_dir,
        mailbox_job_id=str(job_id),
    )
