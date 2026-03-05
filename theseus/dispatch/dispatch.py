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
from collections.abc import Callable
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
    PlainHostConfig,
    SlurmHostConfig,
    TPUHostConfig,
    VolcanoHostConfig,
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


def _build_stages(
    cfgs: list[DictConfig],
    hardware: HardwareResult,
    spec: JobSpec,
) -> tuple[dict[str, str], str]:
    """Build bootstrap .py files and command string for one or more stages.

    Single-stage (backward compat): produces ``_bootstrap_dispatch.py`` with the
    standard command ``python _bootstrap_dispatch.py``.

    Multi-stage: produces ``_bootstrap_dispatch_stage{N}.py`` per stage and a
    ``bash -c`` command that chains them with ``&&``.

    Returns:
        (bootstrap_pys dict mapping filename → content, command string)
    """
    n = len(cfgs)
    if n == 1:
        content = _generate_bootstrap(cfgs[0], hardware, spec)
        return {"_bootstrap_dispatch.py": content}, "python _bootstrap_dispatch.py"

    bootstrap_pys: dict[str, str] = {}
    commands: list[str] = []
    for i, cfg in enumerate(cfgs, 1):
        stage_name = f"{spec.name}_stage{i}"
        stage_spec = JobSpec(name=stage_name, project=spec.project, group=spec.group)
        filename = f"_bootstrap_dispatch_stage{i}.py"
        content = _generate_bootstrap(cfg, hardware, stage_spec)
        bootstrap_pys[filename] = content
        commands.append(f"python {filename}")

    # Chain with && so failure in any stage stops the pipeline
    command = "bash -c '" + " && ".join(commands) + "'"
    return bootstrap_pys, command


def dispatch(
    cfg: DictConfig,
    spec: JobSpec,
    hardware: HardwareRequest,
    dispatch_config: DispatchConfig,
    dirty: bool = False,
    check_availability: bool = True,
    mem: str | None = None,
    timeout: float = 60.0,
    extra_uv_groups: list[str] | None = None,
    extra_cfgs: list[DictConfig] | None = None,
    tpu_version_override: str | None = None,
    tpu_spot_override: bool | None = None,
    tpu_preemptible_override: bool | None = None,
    volcano_image_override: str | None = None,
    volcano_namespace_override: str | None = None,
) -> SlurmResult | RunResult:
    """Dispatch a job to remote infrastructure.

    This function:
    1. Validates the job exists in registry
    2. Solves for hardware allocation
    3. Generates a bootstrap script with embedded config
    4. Ships code and runs on remote (via SLURM, plain SSH, GCloud TPU, or Volcano)

    Args:
        cfg: Job configuration (must have cfg.job pointing to a registered job)
        spec: Job specification (name, project, group)
        hardware: Hardware requirements (includes cluster preferences)
        dispatch_config: Remote host/cluster configuration
        dirty: Include uncommitted changes (default: False)
        check_availability: Check real-time GPU availability (default: True)
        mem: Memory override for SLURM jobs (e.g., "64G", "128G")
        timeout: SSH timeout in seconds
        extra_cfgs: Additional job configs for multi-stage pipelines.
            When provided, all configs run sequentially in a single allocation.
            Job names are suffixed: name_stage1, name_stage2, ...
        tpu_version_override: Override TPU software version for TPU dispatches
        tpu_spot_override: Override spot setting for TPU dispatches
        tpu_preemptible_override: Override preemptible setting for TPU dispatches
        volcano_image_override: Override container image for Volcano dispatches
        volcano_namespace_override: Override namespace for Volcano dispatches

    Returns:
        SlurmResult for SLURM clusters, RunResult for plain SSH / TPU hosts

    Raises:
        RuntimeError: If job not found in registry or no hardware available
    """
    all_cfgs = [cfg] + (extra_cfgs or [])
    n_stages = len(all_cfgs)

    logger.info(
        f"DISPATCH | starting dispatch for job '{spec.name}' "
        f"(project={spec.project}, group={spec.group}, stages={n_stages})"
    )
    target_desc = (
        "cpu-only"
        if hardware.min_chips == 0
        else f"{hardware.min_chips}x {(hardware.chip.name if hardware.chip else 'any-gpu')}"
    )
    logger.debug(f"DISPATCH | hardware request: {target_desc}")

    # 1. Validate all stage jobs exist in registry
    for i, stage_cfg in enumerate(all_cfgs):
        job_key = stage_cfg.job
        if job_key not in JOBS:
            stage_label = f"stage {i + 1} " if n_stages > 1 else ""
            logger.error(f"DISPATCH | {stage_label}job '{job_key}' not in registry")
            raise RuntimeError(
                f"Job '{job_key}' not in registry. Available: {list(JOBS.keys())}"
            )
    logger.debug(f"DISPATCH | validated {n_stages} job(s) exist in registry")

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

    # 4. Generate bootstrap script(s) (Python scripts that run the job stages)
    logger.debug(f"DISPATCH | generating {n_stages} bootstrap script(s)")
    bootstrap_pys, command = _build_stages(all_cfgs, solve_result.result, spec)

    # 5. Submit based on host type
    if isinstance(solve_result.host_config, VolcanoHostConfig):
        if juicefs_mount is not None:
            logger.warning(
                f"DISPATCH | cluster '{solve_result.host_config.cluster}' has a "
                f"JuiceFS mount configured, but Volcano dispatch uses a PVC "
                f"('{solve_result.host_config.pvc_name}') for storage — the "
                f"JuiceFS mount will be ignored"
            )
        logger.info(f"DISPATCH | submitting via Volcano to '{solve_result.host_name}'")
        return _dispatch_volcano(
            solve_result,
            work_dir,
            cluster,
            spec,
            bootstrap_pys,
            command,
            dirty,
            timeout,
            extra_uv_groups=extra_uv_groups or [],
            volcano_image_override=volcano_image_override,
            volcano_namespace_override=volcano_namespace_override,
        )
    elif isinstance(solve_result.host_config, TPUHostConfig):
        logger.info(
            f"DISPATCH | submitting via GCloud TPU to '{solve_result.host_name}'"
        )
        return _dispatch_tpu(
            solve_result,
            work_dir,
            cluster,
            juicefs_mount,
            spec,
            bootstrap_pys,
            command,
            dirty,
            timeout,
            extra_uv_groups=extra_uv_groups or [],
            tpu_version_override=tpu_version_override,
            tpu_spot_override=tpu_spot_override,
            tpu_preemptible_override=tpu_preemptible_override,
        )
    elif solve_result.is_slurm:
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
            bootstrap_pys,
            command,
            mem,
            dirty,
            timeout,
            extra_uv_groups=extra_uv_groups or [],
        )
    else:
        logger.info(f"DISPATCH | submitting via SSH to {solve_result.host_config.ssh}")
        return _dispatch_plain(
            solve_result,
            work_dir,
            cluster,
            juicefs_mount,
            spec,
            bootstrap_pys,
            command,
            dirty,
            timeout,
            extra_uv_groups=extra_uv_groups or [],
        )


def _dispatch_slurm(
    solve_result: SolveResult,
    spec: JobSpec,
    dispatch_config: DispatchConfig,
    work_dir: str,
    share_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    bootstrap_pys: dict[str, str],
    command: str,
    mem: str | None,
    dirty: bool,
    timeout: float,
    extra_uv_groups: list[str] | None = None,
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

    # SLURM needs bootstrap_pys for heredoc embedding (self-contained sbatch
    # script).  stage_files enables per-stage autobatch in bootstrap.sh.
    job = SlurmJob(
        name=job_name,
        command=command,
        root_dir=cluster.root,
        partition=solve_result.partition,
        nodes=len(solve_result.result.hosts),
        gpus_per_node=gpus_per_node,
        gpu_type=gpu_type,
        mem=job_mem,  # uses CLI override, then config value, then defaults to 64G in slurm.py
        account=host_config.account,
        qos=host_config.qos,
        exclude=host_config.exclude,
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
        payload_extract_to=work_dir,
        output=f"{cluster.log_dir}/{job_name}-%j.out",
        juicefs_mount=juicefs_mount,
        stage_files=list(bootstrap_pys.keys()),
        bootstrap_pys=bootstrap_pys,
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
    bootstrap_pys: dict[str, str],
    command: str,
    dirty: bool,
    timeout: float,
    extra_uv_groups: list[str] | None = None,
) -> RunResult:
    """Dispatch job via plain SSH using bootstrap.sh (non-blocking with nohup)."""
    assert solve_result.host_config is not None
    assert not isinstance(solve_result.host_config, SlurmHostConfig)
    assert isinstance(solve_result.host_config, PlainHostConfig)

    host_config = solve_result.host_config
    ssh_alias = host_config.ssh
    log_dir = cluster.log_dir

    # Build log filename with job metadata and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = spec.project or "general"
    group = spec.group or "default"
    log_file = f"{log_dir}/{project}_{group}_{spec.name}_{timestamp}.log"

    logger.debug(f"DISPATCH | SSH dispatch: host={ssh_alias}, work_dir={work_dir}")

    # bootstrap_pys omitted — files are written to disk separately below.
    # stage_files gives bootstrap.sh per-stage autobatch iteration.
    job = SlurmJob(
        name="theseus-dispatch",
        command=command,
        root_dir=cluster.root,
        is_slurm=False,
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
        stage_files=list(bootstrap_pys.keys()),
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

    # Write bootstrap Python script(s) to remote (not included in git archive)
    for filename, content in bootstrap_pys.items():
        remote_path = f"{work_dir}/{filename}"
        logger.debug(f"DISPATCH | writing {filename} to {remote_path}")
        write_py_cmd = (
            f"cat > {remote_path} << 'BOOTSTRAP_PY_EOF'\n{content}BOOTSTRAP_PY_EOF"
        )
        write_py_result = run(write_py_cmd, ssh_alias, timeout=timeout)
        if not write_py_result.ok:
            logger.error(
                f"DISPATCH | failed to write {filename}: {write_py_result.stderr}"
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


def _dispatch_volcano(
    solve_result: SolveResult,
    work_dir: str,
    cluster: Cluster,
    spec: JobSpec,
    bootstrap_pys: dict[str, str],
    command: str,
    dirty: bool,
    timeout: float,
    extra_uv_groups: list[str] | None = None,
    volcano_image_override: str | None = None,
    volcano_namespace_override: str | None = None,
) -> RunResult:
    """Dispatch job to a Kubernetes Volcano cluster.

    Ships code to PVC, writes bootstrap scripts, renders the Volcano Job
    YAML template, and submits via ``kubectl apply``.
    """
    import dataclasses
    import subprocess

    from theseus.dispatch import volcano as volcano_mod
    from theseus.dispatch.sync import snapshot

    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, VolcanoHostConfig)

    # Apply CLI overrides
    host_config = solve_result.host_config
    if volcano_image_override:
        host_config = dataclasses.replace(host_config, image=volcano_image_override)
    if volcano_namespace_override:
        host_config = dataclasses.replace(
            host_config, namespace=volcano_namespace_override
        )

    namespace = host_config.namespace
    kubeconfig = host_config.kubeconfig
    context = host_config.context

    # Build job name from spec
    project_name = spec.project or "general"
    group = spec.group or "default"
    # K8s names must be DNS-compatible
    job_name = f"{project_name}-{group}-{spec.name}".lower().replace("_", "-")
    # Truncate to 63 chars (K8s limit)
    job_name = job_name[:63].rstrip("-")

    # Remote subdir on PVC
    remote_subdir = f"{project_name}/{group}/{spec.name}"

    logger.debug(
        f"DISPATCH | Volcano dispatch: job={job_name}, namespace={namespace}, "
        f"pvc={host_config.pvc_name}, work_dir={work_dir}"
    )

    # bootstrap_pys omitted — files are shipped to PVC via ship_and_write_to_pvc.
    # stage_files gives bootstrap.sh per-stage autobatch iteration.
    job = SlurmJob(
        name="theseus-dispatch",
        command=command,
        root_dir=cluster.root,
        is_slurm=False,
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
        workdir=f"{host_config.pvc_mount_path}/{remote_subdir}",
        stage_files=list(bootstrap_pys.keys()),
    )
    script = job.to_script()

    # 2. Snapshot code and ship everything to PVC via a single helper vcjob
    logger.debug(
        f"DISPATCH | shipping code + bootstrap to PVC '{host_config.pvc_name}' (dirty={dirty})"
    )
    if dirty:
        stash_result = subprocess.run(
            ["git", "stash", "create"],
            capture_output=True,
            text=True,
        )
        ref = stash_result.stdout.strip() or "HEAD"
    else:
        ref = "HEAD"
    tarball = snapshot(".", ref)

    ship_result = volcano_mod.ship_and_write_to_pvc(
        tarball=tarball,
        script=script,
        bootstrap_pys=bootstrap_pys,
        pvc_name=host_config.pvc_name,
        remote_subdir=remote_subdir,
        queue=host_config.queue,
        namespace=namespace,
        pvc_mount_path=host_config.pvc_mount_path,
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )
    if not ship_result.ok:
        logger.error(f"DISPATCH | failed to ship to PVC: {ship_result.stderr}")
        return ship_result

    # 4. Render Volcano Job YAML
    bootstrap_cmd = (
        f"cd {host_config.pvc_mount_path}/{remote_subdir} && bash _bootstrap.sh"
    )
    rendered_yaml = volcano_mod.render_volcano_job(
        job_name=job_name,
        host_config=host_config,
        bootstrap_command=bootstrap_cmd,
        work_dir=work_dir,
    )

    # 5. Submit via kubectl apply
    logger.info(f"DISPATCH | submitting Volcano Job '{job_name}'")
    apply_result = volcano_mod.apply_job(
        yaml_content=rendered_yaml,
        namespace=namespace,
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )

    if apply_result.ok:
        logger.info(f"DISPATCH | Volcano Job '{job_name}' submitted successfully")
        return RunResult(
            returncode=0,
            stdout=(
                f"Volcano Job '{job_name}' submitted to namespace '{namespace}'.\n"
                f"Monitor with: kubectl get vcjob {job_name} -n {namespace}\n"
                f"Logs: kubectl logs -l volcano.sh/job-name={job_name} -n {namespace} --all-containers -f\n"
                f"{apply_result.stdout}"
            ),
            stderr=apply_result.stderr,
        )
    else:
        logger.error(f"DISPATCH | Volcano Job submission failed: {apply_result.stderr}")
        return apply_result


def _dispatch_tpu(
    solve_result: SolveResult,
    work_dir: str,
    cluster: Cluster,
    juicefs_mount: JuiceFSMount | None,
    spec: JobSpec,
    bootstrap_pys: dict[str, str],
    command: str,
    dirty: bool,
    timeout: float,
    extra_uv_groups: list[str] | None = None,
    tpu_version_override: str | None = None,
    tpu_spot_override: bool | None = None,
    tpu_preemptible_override: bool | None = None,
) -> RunResult:
    """Dispatch job to a Google Cloud TPU VM.

    Mirrors the plain SSH dispatch path but uses ``gcloud compute tpus tpu-vm``
    commands instead of regular SSH.  Code is shipped identically to **all**
    workers in the TPU pod so that ``jax.distributed.initialize()`` can
    coordinate them.
    """
    from theseus.dispatch import tpu as tpu_mod

    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, TPUHostConfig)

    host_config = solve_result.host_config
    tpu_name = solve_result.host_name
    assert tpu_name is not None
    zone = host_config.zone
    project = host_config.project
    internal_ip = host_config.internal_ip
    log_dir = cluster.log_dir

    # Apply CLI overrides
    version = tpu_version_override or host_config.version
    spot = tpu_spot_override if tpu_spot_override is not None else host_config.spot
    preemptible = (
        tpu_preemptible_override
        if tpu_preemptible_override is not None
        else host_config.preemptible
    )

    # Build log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = spec.project or "general"
    group = spec.group or "default"
    log_file = f"{log_dir}/{project_name}_{group}_{spec.name}_{timestamp}.log"

    logger.debug(
        f"DISPATCH | TPU dispatch: tpu={tpu_name}, zone={zone}, work_dir={work_dir}"
    )

    # ------------------------------------------------------------------ #
    # 1. Ensure TPU VM exists and is READY
    # ------------------------------------------------------------------ #
    tpu_state = tpu_mod.get_status(tpu_name, zone, project, timeout=30.0)
    if tpu_state is None:
        # TPU does not exist — prompt the user before incurring cost
        from rich.console import Console
        from rich.prompt import Confirm

        _console = Console()
        _console.print()
        _console.print(
            f"[yellow]TPU VM [bold]'{tpu_name}'[/bold] does not exist.[/yellow]"
        )
        _console.print(f"[yellow]  type : {host_config.accelerator_type}[/yellow]")
        _console.print(f"[yellow]  zone : {zone}[/yellow]")
        _console.print(f"[yellow]  version : {version}[/yellow]")
        if spot:
            _console.print(
                "[yellow]  pricing: [bold]spot[/bold] (may be preempted)[/yellow]"
            )
        elif preemptible:
            _console.print(
                "[yellow]  pricing: [bold]preemptible[/bold] (may be preempted, 24h limit)[/yellow]"
            )
        else:
            _console.print("[yellow]  pricing: [bold]on-demand[/bold][/yellow]")
        _console.print()
        _console.print(
            "[bold red]Creating this TPU VM will incur Google Cloud costs.[/bold red]"
        )
        if not Confirm.ask("[bold]Create TPU VM and continue?[/bold]", default=False):
            return RunResult(
                returncode=1,
                stdout="",
                stderr="TPU VM creation cancelled by user",
            )

        logger.info(f"DISPATCH | creating TPU VM '{tpu_name}'")
        create_result = tpu_mod.create(
            name=tpu_name,
            zone=zone,
            accelerator_type=host_config.accelerator_type,
            version=version,
            project=project,
            spot=spot,
            preemptible=preemptible,
            network=host_config.network,
            subnetwork=host_config.subnetwork,
            service_account=host_config.service_account,
            metadata=host_config.metadata or None,
            timeout=600.0,
        )
        if not create_result.ok:
            logger.error(f"DISPATCH | TPU VM creation failed: {create_result.stderr}")
            return create_result

        if not tpu_mod.wait_ready(tpu_name, zone, project, timeout=600.0):
            return RunResult(
                returncode=1,
                stdout="",
                stderr=f"TPU VM '{tpu_name}' did not become READY in time",
            )

    elif tpu_state in ("PREEMPTED", "TERMINATED"):
        # Preempted/terminated TPUs won't recover on their own — delete and recreate.
        logger.info(
            f"DISPATCH | TPU VM '{tpu_name}' is {tpu_state}, deleting and recreating..."
        )
        del_result = tpu_mod.delete(tpu_name, zone, project, timeout=300.0)
        if not del_result.ok:
            logger.error(
                f"DISPATCH | failed to delete {tpu_state} TPU: {del_result.stderr}"
            )
            return del_result

        create_result = tpu_mod.create(
            name=tpu_name,
            zone=zone,
            accelerator_type=host_config.accelerator_type,
            version=version,
            project=project,
            spot=spot,
            preemptible=preemptible,
            network=host_config.network,
            subnetwork=host_config.subnetwork,
            service_account=host_config.service_account,
            metadata=host_config.metadata or None,
            timeout=600.0,
        )
        if not create_result.ok:
            logger.error(f"DISPATCH | TPU VM recreation failed: {create_result.stderr}")
            return create_result

        if not tpu_mod.wait_ready(tpu_name, zone, project, timeout=600.0):
            return RunResult(
                returncode=1,
                stdout="",
                stderr=f"TPU VM '{tpu_name}' did not become READY after recreation",
            )

    elif tpu_state != "READY":
        logger.info(
            f"DISPATCH | TPU VM '{tpu_name}' exists but state={tpu_state}, waiting..."
        )
        if not tpu_mod.wait_ready(tpu_name, zone, project, timeout=300.0):
            return RunResult(
                returncode=1,
                stdout="",
                stderr=f"TPU VM '{tpu_name}' did not become READY (state was {tpu_state})",
            )
    else:
        logger.info(f"DISPATCH | TPU VM '{tpu_name}' is READY")

    # ------------------------------------------------------------------ #
    # 2. Build bootstrap script (same as plain SSH, with TPU env var)
    # ------------------------------------------------------------------ #
    # bootstrap_pys omitted — files are SCP'd to all workers separately below.
    # stage_files gives bootstrap.sh per-stage autobatch iteration.
    job = SlurmJob(
        name="theseus-dispatch",
        command=command,
        root_dir=cluster.root,
        is_slurm=False,
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
        stage_files=list(bootstrap_pys.keys()),
        env={"THESEUS_TPU_MODE": "1"},
    )
    script = job.to_script()

    # ------------------------------------------------------------------ #
    # 3. Ship code to ALL workers (identical tarball)
    # ------------------------------------------------------------------ #
    logger.debug(f"DISPATCH | shipping code to {tpu_name}:{work_dir} (dirty={dirty})")
    if dirty:
        ship_result = tpu_mod.ship_dirty(
            tpu_name, work_dir, zone, project, internal_ip, timeout=timeout
        )
    else:
        ship_result = tpu_mod.ship(
            tpu_name, work_dir, zone, project, internal_ip, timeout=timeout
        )

    if not ship_result.ok:
        logger.error(f"DISPATCH | failed to ship code to TPU: {ship_result.stderr}")
        return ship_result

    logger.debug("DISPATCH | code shipped to all TPU workers")

    # ------------------------------------------------------------------ #
    # 4. Write bootstrap scripts to ALL workers via SCP
    # ------------------------------------------------------------------ #
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path as _Path

        # Write bootstrap.sh locally and SCP to all workers
        bootstrap_sh_local = _Path(tmpdir) / "_bootstrap.sh"
        bootstrap_sh_local.write_text(script)

        logger.debug("DISPATCH | writing bootstrap.sh to all workers")
        scp_sh = tpu_mod.copy_to(
            bootstrap_sh_local,
            tpu_name,
            f"{work_dir}/_bootstrap.sh",
            zone,
            project,
            worker="all",
            internal_ip=internal_ip,
            timeout=timeout,
        )
        if not scp_sh.ok:
            logger.error(f"DISPATCH | failed to write bootstrap.sh: {scp_sh.stderr}")
            return scp_sh

        # Write bootstrap Python script(s) and SCP each to all workers
        for filename, content in bootstrap_pys.items():
            py_local = _Path(tmpdir) / filename
            py_local.write_text(content)

            logger.debug(f"DISPATCH | writing {filename} to all workers")
            scp_py = tpu_mod.copy_to(
                py_local,
                tpu_name,
                f"{work_dir}/{filename}",
                zone,
                project,
                worker="all",
                internal_ip=internal_ip,
                timeout=timeout,
            )
            if not scp_py.ok:
                logger.error(f"DISPATCH | failed to write {filename}: {scp_py.stderr}")
                return scp_py

    # ------------------------------------------------------------------ #
    # 5. Launch bootstrap on ALL workers (non-blocking via nohup)
    # ------------------------------------------------------------------ #
    logger.debug(f"DISPATCH | launching job on all TPU workers, logs at {log_file}")
    run_cmd = (
        f"mkdir -p {log_dir} && "
        f"chmod +x {work_dir}/_bootstrap.sh && "
        f"nohup {work_dir}/_bootstrap.sh > {log_file} 2>&1 &"
    )
    result = tpu_mod.run(
        run_cmd,
        tpu_name,
        zone,
        project,
        worker="all",
        internal_ip=internal_ip,
        timeout=timeout,
    )

    if result.ok:
        logger.info(f"DISPATCH | TPU job started on all workers of '{tpu_name}'")
        logger.debug(f"DISPATCH | logs at {tpu_name}:{log_file}")
        return RunResult(
            returncode=result.returncode,
            stdout=(
                f"Job started on TPU VM '{tpu_name}' ({host_config.accelerator_type}).\n"
                f"Logs: {tpu_name}:{log_file}\n"
                f"To check: gcloud compute tpus tpu-vm ssh {tpu_name} "
                f"--zone={zone} --worker=0 --command='tail -f {log_file}'\n"
                f"{result.stdout}"
            ),
            stderr=result.stderr,
        )
    else:
        logger.error(f"DISPATCH | failed to launch job on TPU: {result.stderr}")
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
    run_fn: Callable[..., RunResult] | None = None,
) -> tuple[str | None, int | None, str | None, str | None]:
    """Poll a remote log file until Jupyter startup metadata appears."""
    _run = run_fn or run
    start = time.time()
    missing_file_grace_until = start + min(timeout, 20.0)
    while (time.time() - start) < timeout:
        read_result = _run(f"tail -n 200 {log_path}", host, timeout=ssh_timeout)
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
    host: str,
    port: int,
    ssh_timeout: float,
    run_fn: Callable[..., RunResult] | None = None,
) -> int | None:
    """Resolve the remote notebook PID bound to the selected port."""
    _run = run_fn or run
    pid_cmd = (
        f"(lsof -ti :{port} 2>/dev/null | head -n 1) || "
        f"(ss -ltnp 2>/dev/null | grep ':{port} ' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n 1)"
    )
    pid_result = _run(pid_cmd, host, timeout=ssh_timeout)
    if not pid_result.ok:
        return None
    for line in pid_result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _read_remote_mailbox_job_id(
    host: str,
    work_dir: str,
    ssh_timeout: float,
    run_fn: Callable[..., RunResult] | None = None,
) -> str | None:
    _run = run_fn or run
    path = f"{work_dir}/.theseus_repl_mailbox_job_id"
    read_result = _run(f"cat {path}", host, timeout=ssh_timeout)
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
    extra_uv_groups: list[str] | None = None,
    tpu_version_override: str | None = None,
    tpu_spot_override: bool | None = None,
    tpu_preemptible_override: bool | None = None,
) -> ReplResult:
    """Dispatch an interactive Jupyter session on selected infrastructure."""
    solve_result = solve_or_raise(
        hardware,
        dispatch_config,
        check_availability=check_availability,
        timeout=timeout,
        interactive=True,
    )
    assert solve_result.result is not None
    assert solve_result.host_name is not None
    assert solve_result.host_config is not None
    assert not isinstance(solve_result.host_config, VolcanoHostConfig)

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

    if isinstance(solve_result.host_config, TPUHostConfig):
        from theseus.dispatch.tpu import parse_accelerator_type

        _, n_chips = parse_accelerator_type(solve_result.host_config.accelerator_type)
        if n_chips > 4:
            return ReplResult(
                ok=False,
                is_slurm=False,
                selected_host=solve_result.host_name or "",
                ssh_host=f"tpu:{solve_result.host_name}",
                log_path="",
                stderr=(
                    f"REPL is only supported on single-host TPUs (<=4 chips), "
                    f"but '{solve_result.host_config.accelerator_type}' has {n_chips} chips"
                ),
            )
        return _dispatch_repl_tpu(
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
            extra_uv_groups=extra_uv_groups or [],
            tpu_version_override=tpu_version_override,
            tpu_spot_override=tpu_spot_override,
            tpu_preemptible_override=tpu_preemptible_override,
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
            extra_uv_groups=extra_uv_groups or [],
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
        extra_uv_groups=extra_uv_groups or [],
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
    extra_uv_groups: list[str] | None = None,
) -> ReplResult:
    assert solve_result.host_name is not None
    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, PlainHostConfig)

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
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
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


def _dispatch_repl_tpu(
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
    extra_uv_groups: list[str] | None = None,
    tpu_version_override: str | None = None,
    tpu_spot_override: bool | None = None,
    tpu_preemptible_override: bool | None = None,
) -> ReplResult:
    """Dispatch a Jupyter REPL on a single-host TPU VM (4 chips only)."""
    from theseus.dispatch import tpu as tpu_mod

    assert solve_result.host_config is not None
    assert isinstance(solve_result.host_config, TPUHostConfig)

    host_config = solve_result.host_config
    tpu_name = solve_result.host_name
    assert tpu_name is not None
    zone = host_config.zone
    project = host_config.project
    internal_ip = host_config.internal_ip
    log_dir = cluster.log_dir

    # Apply CLI overrides
    version = tpu_version_override or host_config.version
    spot = tpu_spot_override if tpu_spot_override is not None else host_config.spot
    preemptible = (
        tpu_preemptible_override
        if tpu_preemptible_override is not None
        else host_config.preemptible
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = spec.project or "general"
    group = spec.group or "default"
    log_file = f"{log_dir}/{project_name}_{group}_{spec.name}_{timestamp}.log"

    # Display name for ReplResult (user-facing identifier)
    ssh_host = f"tpu:{tpu_name}"

    def _fail(stderr: str) -> ReplResult:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=tpu_name,
            ssh_host=ssh_host,
            log_path=log_file,
            cluster_name=cluster.name,
            cluster_root=cluster.root,
            cluster_mount=juicefs_mount.redis_url if juicefs_mount else None,
            work_dir=work_dir,
            stderr=stderr,
        )

    # Build a run_fn that wraps tpu.run for the helpers.
    def _tpu_run(cmd: str, _host: str, timeout: float | None = None) -> RunResult:
        return tpu_mod.run(
            cmd,
            tpu_name,
            zone,
            project,
            worker="0",
            internal_ip=internal_ip,
            timeout=timeout,
        )

    # ------------------------------------------------------------------ #
    # 1. Ensure TPU VM exists and is READY
    # ------------------------------------------------------------------ #
    tpu_state = tpu_mod.get_status(tpu_name, zone, project, timeout=30.0)
    if tpu_state is None:
        from rich.console import Console
        from rich.prompt import Confirm

        _console = Console()
        _console.print()
        _console.print(
            f"[yellow]TPU VM [bold]'{tpu_name}'[/bold] does not exist.[/yellow]"
        )
        _console.print(f"[yellow]  type : {host_config.accelerator_type}[/yellow]")
        _console.print(f"[yellow]  zone : {zone}[/yellow]")
        _console.print(f"[yellow]  version : {version}[/yellow]")
        if spot:
            _console.print(
                "[yellow]  pricing: [bold]spot[/bold] (may be preempted)[/yellow]"
            )
        elif preemptible:
            _console.print(
                "[yellow]  pricing: [bold]preemptible[/bold] (may be preempted, 24h limit)[/yellow]"
            )
        else:
            _console.print("[yellow]  pricing: [bold]on-demand[/bold][/yellow]")
        _console.print()
        _console.print(
            "[bold red]Creating this TPU VM will incur Google Cloud costs.[/bold red]"
        )
        if not Confirm.ask("[bold]Create TPU VM and continue?[/bold]", default=False):
            return _fail("TPU VM creation cancelled by user")

        logger.info(f"REPL | creating TPU VM '{tpu_name}'")
        create_result = tpu_mod.create(
            name=tpu_name,
            zone=zone,
            accelerator_type=host_config.accelerator_type,
            version=version,
            project=project,
            spot=spot,
            preemptible=preemptible,
            network=host_config.network,
            subnetwork=host_config.subnetwork,
            service_account=host_config.service_account,
            metadata=host_config.metadata or None,
            timeout=600.0,
        )
        if not create_result.ok:
            return _fail(f"TPU VM creation failed: {create_result.stderr}")

        if not tpu_mod.wait_ready(tpu_name, zone, project, timeout=600.0):
            return _fail(f"TPU VM '{tpu_name}' did not become READY in time")

    elif tpu_state != "READY":
        logger.info(f"REPL | TPU VM '{tpu_name}' state={tpu_state}, waiting...")
        if not tpu_mod.wait_ready(tpu_name, zone, project, timeout=300.0):
            return _fail(
                f"TPU VM '{tpu_name}' did not become READY (state was {tpu_state})"
            )
    else:
        logger.info(f"REPL | TPU VM '{tpu_name}' is READY")

    # ------------------------------------------------------------------ #
    # 2. Build bootstrap script for Jupyter
    # ------------------------------------------------------------------ #
    job = SlurmJob(
        name="theseus-repl",
        command=_repl_command(sync_enabled),
        root_dir=cluster.root,
        is_slurm=False,
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
        juicefs_mount=juicefs_mount,
        workdir=work_dir,
    )
    script = job.to_script()

    # ------------------------------------------------------------------ #
    # 3. Ship code to TPU workers (single-host REPL)
    # ------------------------------------------------------------------ #
    if dirty:
        ship_result = tpu_mod.ship_dirty(
            tpu_name, work_dir, zone, project, internal_ip, timeout=timeout
        )
    else:
        ship_result = tpu_mod.ship(
            tpu_name, work_dir, zone, project, internal_ip, timeout=timeout
        )
    if not ship_result.ok:
        return _fail(f"failed to ship code: {ship_result.stderr}")

    # ------------------------------------------------------------------ #
    # 4. Write bootstrap script to worker 0 via SCP
    # ------------------------------------------------------------------ #
    import tempfile as _tmpmod

    with _tmpmod.TemporaryDirectory() as tmpdir:
        from pathlib import Path as _Path

        bootstrap_local = _Path(tmpdir) / "_bootstrap_repl.sh"
        bootstrap_local.write_text(script)

        scp_result = tpu_mod.copy_to(
            bootstrap_local,
            tpu_name,
            f"{work_dir}/_bootstrap_repl.sh",
            zone,
            project,
            worker="0",
            internal_ip=internal_ip,
            timeout=timeout,
        )
        if not scp_result.ok:
            return _fail(f"failed to write bootstrap: {scp_result.stderr}")

    # ------------------------------------------------------------------ #
    # 5. Launch Jupyter on worker 0
    # ------------------------------------------------------------------ #
    run_cmd = (
        f"mkdir -p {log_dir} && "
        f"chmod +x {work_dir}/_bootstrap_repl.sh && "
        f"nohup {work_dir}/_bootstrap_repl.sh > {log_file} 2>&1 & echo $!"
    )
    launch_result = tpu_mod.run(
        run_cmd,
        tpu_name,
        zone,
        project,
        worker="0",
        internal_ip=internal_ip,
        timeout=timeout,
    )
    if not launch_result.ok:
        return _fail(f"failed to launch Jupyter: {launch_result.stderr}")

    launcher_pid = None
    for line in reversed(launch_result.stdout.strip().splitlines()):
        line = line.strip()
        if line.isdigit():
            launcher_pid = int(line)
            break

    # ------------------------------------------------------------------ #
    # 6. Wait for Jupyter startup (reuse helper with _tpu_run)
    # ------------------------------------------------------------------ #
    remote_url, remote_port, token, log_wait_error = _wait_for_jupyter_log(
        ssh_host,
        log_file,
        timeout=startup_timeout,
        ssh_timeout=timeout,
        run_fn=_tpu_run,
    )
    if log_wait_error:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=tpu_name,
            ssh_host=ssh_host,
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

    remote_pid = _resolve_remote_notebook_pid(
        ssh_host,
        remote_port,
        timeout,
        run_fn=_tpu_run,
    )
    if remote_pid is None:
        remote_pid = launcher_pid

    mailbox_job_id = None
    if sync_enabled:
        mailbox_job_id = _read_remote_mailbox_job_id(
            ssh_host,
            work_dir,
            timeout,
            run_fn=_tpu_run,
        )
    if mailbox_job_id is None and remote_pid is not None:
        mailbox_job_id = str(remote_pid)

    # ------------------------------------------------------------------ #
    # 7. Port forwarding via gcloud SSH tunnel
    # ------------------------------------------------------------------ #
    tunnel_result = tpu_mod.forward_port(
        tpu_name,
        zone,
        local_port=local_port,
        remote_port=remote_port,
        project=project,
        worker="0",
        internal_ip=internal_ip,
    )
    if not tunnel_result.ok:
        return ReplResult(
            ok=False,
            is_slurm=False,
            selected_host=tpu_name,
            ssh_host=ssh_host,
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
            stderr=tunnel_result.stderr or "failed to start gcloud SSH tunnel",
        )

    local_url = f"http://localhost:{local_port}/lab"
    if token:
        local_url = f"{local_url}?token={token}"

    return ReplResult(
        ok=True,
        is_slurm=False,
        selected_host=tpu_name,
        ssh_host=ssh_host,
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
    extra_uv_groups: list[str] | None = None,
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
        uv_groups=host_config.uv_groups + (extra_uv_groups or []),
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
