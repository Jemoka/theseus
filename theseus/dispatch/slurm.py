"""
SLURM dispatch utilities
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from theseus.dispatch.ssh import run, RunResult

if TYPE_CHECKING:
    from theseus.dispatch.config import JuiceFSMount

# Path to templates
BOOTSTRAP_TEMPLATE = Path(__file__).parent / "bootstrap.sh"
SBATCH_TEMPLATE = Path(__file__).parent / "sbatch.sh"


@dataclass
class SlurmJob:
    """Configuration for an sbatch script."""

    # Required
    name: str
    command: str

    # Resource allocation
    partition: str | None = None
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int | None = None
    cpus_per_task: int | None = None
    gpus: int | None = None
    gpus_per_node: int | None = None
    gpu_type: str | None = None  # e.g., "a100", "h100"
    mem: str | None = None  # e.g., "64G"
    mem_per_cpu: str | None = None
    time: str | None = None  # e.g., "24:00:00"

    # Paths
    output: str | None = None  # stdout log path
    error: str | None = None  # stderr log path (defaults to output if not set)
    workdir: str | None = None

    # Environment
    env: dict[str, str] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)  # modules to load
    uv_groups: list[str] = field(default_factory=list)  # uv sync --group flags

    # Job dependencies and scheduling
    dependency: str | None = None  # e.g., "afterok:12345"
    exclusive: bool = False
    constraint: str | None = None  # e.g., "gpu80g"
    account: str | None = None
    qos: str | None = None

    # Extras
    extra_directives: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)  # run before main command

    # Packed payload - base64 encoded tarball extracted at runtime
    payload: str | None = None  # base64-encoded tarball
    payload_extract_to: str = "$SLURM_TMPDIR/code"  # where to extract

    # JuiceFS mount configuration - will mount before running
    juicefs_mount: JuiceFSMount | None = None

    # Set to False for plain SSH (skips SBATCH directives)
    is_slurm: bool = True

    def pack(self, tarball: bytes) -> "SlurmJob":
        """Return a new SlurmJob with the given tarball as payload.

        Args:
            tarball: gzip-compressed tarball bytes (from sync.snapshot())

        Returns:
            New SlurmJob with payload set
        """
        import dataclasses

        return dataclasses.replace(
            self,
            payload=base64.b64encode(tarball).decode("ascii"),
        )

    def _sbatch_directives(self) -> str:
        """Generate #SBATCH directive lines."""
        lines = []

        lines.append(f"#SBATCH --job-name={self.name}")

        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --nodes={self.nodes}")
        lines.append(f"#SBATCH --ntasks={self.ntasks}")

        if self.ntasks_per_node:
            lines.append(f"#SBATCH --ntasks-per-node={self.ntasks_per_node}")
        if self.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")

        if self.gpus:
            gpu_spec = (
                f"{self.gpu_type}:{self.gpus}" if self.gpu_type else str(self.gpus)
            )
            lines.append(f"#SBATCH --gpus={gpu_spec}")
        elif self.gpus_per_node:
            gpu_spec = (
                f"{self.gpu_type}:{self.gpus_per_node}"
                if self.gpu_type
                else str(self.gpus_per_node)
            )
            lines.append(f"#SBATCH --gpus-per-node={gpu_spec}")

        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        elif self.mem_per_cpu:
            lines.append(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}")

        if self.time:
            lines.append(f"#SBATCH --time={self.time}")

        if self.output:
            lines.append(f"#SBATCH --output={self.output}")
            lines.append(f"#SBATCH --error={self.error or self.output}")

        if self.workdir:
            lines.append(f"#SBATCH --chdir={self.workdir}")

        if self.dependency:
            lines.append(f"#SBATCH --dependency={self.dependency}")
        if self.exclusive:
            lines.append("#SBATCH --exclusive")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        for directive in self.extra_directives:
            lines.append(f"#SBATCH {directive}")

        return "\n".join(lines)

    def to_bootstrap_script(self) -> str:
        """Generate the bootstrap.sh script that runs on each node."""
        template = BOOTSTRAP_TEMPLATE.read_text()
        script = template

        # JuiceFS mount
        if self.juicefs_mount:
            mount_opts = []
            if self.juicefs_mount.cache_size:
                mount_opts.append(f"--cache-size {self.juicefs_mount.cache_size}")
            if self.juicefs_mount.cache_dir:
                mount_opts.append(f"--cache-dir {self.juicefs_mount.cache_dir}")
            opts_str = " ".join(mount_opts)
            juicefs_str = f"""
if ! mountpoint -q {self.juicefs_mount.mount_point}; then
    echo "[bootstrap] mounting JuiceFS at {self.juicefs_mount.mount_point}..."
    mkdir -p {self.juicefs_mount.mount_point}
    juicefs mount -d {opts_str} {self.juicefs_mount.redis_url} {self.juicefs_mount.mount_point}
fi
# Track mount point for cleanup on exit/preemption
JUICEFS_MOUNT_POINT="{self.juicefs_mount.mount_point}"
"""
        else:
            juicefs_str = ""
        script = script.replace("__JUICEFS_MOUNT__", juicefs_str)

        # Modules
        if self.modules:
            modules_str = "\n".join(f"module load {mod}" for mod in self.modules)
        else:
            modules_str = ""
        script = script.replace("__MODULES__", modules_str)

        # Environment variables
        if self.env:
            env_str = "\n".join(f"export {k}={v}" for k, v in self.env.items())
        else:
            env_str = ""
        script = script.replace("__ENV_VARS__", env_str)

        # Payload extraction (for SLURM with embedded payload)
        if self.payload:
            payload_extract = f"""
echo "[bootstrap] extracting code payload..."
mkdir -p {self.payload_extract_to}
base64 -d <<'__PAYLOAD_EOF__' | tar -xzf - -C {self.payload_extract_to}
{self.payload}
__PAYLOAD_EOF__
"""
        else:
            payload_extract = ""
        script = script.replace("__PAYLOAD_EXTRACT__", payload_extract)

        # Working directory - use payload_extract_to for SLURM, workdir for SSH
        if self.payload:
            workdir = self.payload_extract_to
        else:
            workdir = self.workdir or "."
        script = script.replace("__WORKDIR__", workdir)

        # UV sync command with optional groups
        if self.uv_groups:
            groups_flags = " ".join(f"--group {g}" for g in self.uv_groups)
            uv_sync_cmd = (
                f"uv sync --frozen {groups_flags} 2>/dev/null || uv sync {groups_flags}"
            )
        else:
            uv_sync_cmd = "uv sync --frozen 2>/dev/null || uv sync"
        script = script.replace("__UV_SYNC__", uv_sync_cmd)

        # Setup commands
        if self.setup_commands:
            setup_str = "\n".join(self.setup_commands)
        else:
            setup_str = ""
        script = script.replace("__SETUP_COMMANDS__", setup_str)

        # Command (use uv run to execute in the synced environment)
        script = script.replace("__COMMAND__", f"uv run {self.command}")

        return script

    def to_sbatch_script(self, bootstrap_script_path: str) -> str:
        """Generate the sbatch wrapper script that calls srun on bootstrap.sh."""
        template = SBATCH_TEMPLATE.read_text()

        script = template.replace("__SBATCH_DIRECTIVES__", self._sbatch_directives())

        # Working directory for sbatch --chdir
        workdir = self.payload_extract_to if self.payload else (self.workdir or ".")
        script = script.replace("__WORKDIR__", workdir)

        # Path to bootstrap script
        script = script.replace("__BOOTSTRAP_SCRIPT__", bootstrap_script_path)

        return script

    def to_script(self) -> str:
        """Generate script for backward compatibility.

        For SLURM: returns bootstrap script (sbatch wrapper generated separately)
        For SSH: returns bootstrap script
        """
        return self.to_bootstrap_script()


@dataclass
class SlurmResult:
    """Result of a SLURM job submission."""

    job_id: int | None
    ssh_result: RunResult

    @property
    def ok(self) -> bool:
        return self.job_id is not None


@dataclass
class JobStatus:
    """Status of a SLURM job from squeue."""

    job_id: int
    partition: str
    name: str
    user: str
    state: str  # PD=pending, R=running, CG=completing, CD=completed, F=failed, etc.
    time_elapsed: str
    nodes: int
    nodelist: str  # node names or reason if pending

    @property
    def is_running(self) -> bool:
        return self.state == "R"

    @property
    def is_pending(self) -> bool:
        return self.state == "PD"

    @property
    def is_completed(self) -> bool:
        return self.state in ("CD", "CG")

    @property
    def is_failed(self) -> bool:
        return self.state in ("F", "CA", "TO", "NF", "OOM")


@dataclass
class JobInfo:
    """Detailed info about a SLURM job from sacct."""

    job_id: str  # can include step suffixes like "12345.batch"
    name: str
    partition: str
    state: str
    exit_code: str
    elapsed: str
    max_rss: str  # max memory used
    nodelist: str


@dataclass
class QueueResult:
    """Result of a queue query."""

    jobs: list[JobStatus]
    ssh_result: RunResult

    @property
    def ok(self) -> bool:
        return self.ssh_result.ok


@dataclass
class StatusResult:
    """Result of a job status query."""

    job: JobStatus | None
    ssh_result: RunResult

    @property
    def ok(self) -> bool:
        return self.ssh_result.ok and self.job is not None


@dataclass
class JobInfoResult:
    """Result of a job info query."""

    steps: list[JobInfo]  # main job + steps like .batch, .extern
    ssh_result: RunResult

    @property
    def ok(self) -> bool:
        return self.ssh_result.ok and len(self.steps) > 0

    @property
    def main(self) -> JobInfo | None:
        """Get the main job entry (without step suffix)."""
        for step in self.steps:
            if "." not in step.job_id:
                return step
        return self.steps[0] if self.steps else None


def submit(
    job: SlurmJob,
    host: str,
    script_path: str | None = None,
    timeout: float | None = None,
) -> SlurmResult:
    """Submit a SLURM job to a remote host via SSH.

    Creates two scripts on remote:
    - bootstrap.sh: runs on each node via srun (setup + command)
    - sbatch wrapper: contains SBATCH directives, calls srun bootstrap.sh

    Args:
        job: SlurmJob configuration
        host: SSH host with SLURM access
        script_path: Optional remote path prefix for scripts (uses mktemp if not provided)
        timeout: SSH timeout in seconds

    Returns:
        SlurmResult with job_id and SSH result
    """
    import re

    logger.info(f"SLURM | submitting job '{job.name}' to {host}")
    logger.debug(
        f"SLURM | job config: nodes={job.nodes}, partition={job.partition}, gpus_per_node={job.gpus_per_node}"
    )

    # Generate both scripts
    bootstrap_content = job.to_bootstrap_script()

    if script_path:
        remote_sbatch = script_path
        remote_bootstrap = script_path.replace(".sbatch", "_bootstrap.sh")
    else:
        # Use mktemp on remote for both scripts
        logger.debug(f"SLURM | creating temp script paths on {host}")
        mktemp_result = run("mktemp --suffix=.sbatch", host, timeout=timeout)
        if not mktemp_result.ok:
            logger.error(f"SLURM | failed to create temp file: {mktemp_result.stderr}")
            return SlurmResult(job_id=None, ssh_result=mktemp_result)
        remote_sbatch = mktemp_result.stdout.strip()
        remote_bootstrap = remote_sbatch.replace(".sbatch", "_bootstrap.sh")

    logger.debug(
        f"SLURM | script paths: sbatch={remote_sbatch}, bootstrap={remote_bootstrap}"
    )

    # Generate sbatch wrapper that calls srun on bootstrap
    sbatch_content = job.to_sbatch_script(remote_bootstrap)

    # Write bootstrap script to remote
    logger.debug(f"SLURM | writing bootstrap script to {remote_bootstrap}")
    write_bootstrap = (
        f"cat > {remote_bootstrap} << 'BOOTSTRAP_EOF'\n{bootstrap_content}BOOTSTRAP_EOF"
    )
    write_result = run(write_bootstrap, host, timeout=timeout)
    if not write_result.ok:
        logger.error(f"SLURM | failed to write bootstrap script: {write_result.stderr}")
        return SlurmResult(job_id=None, ssh_result=write_result)

    # Make bootstrap executable
    chmod_result = run(f"chmod +x {remote_bootstrap}", host, timeout=timeout)
    if not chmod_result.ok:
        logger.error(f"SLURM | failed to chmod bootstrap script: {chmod_result.stderr}")
        return SlurmResult(job_id=None, ssh_result=chmod_result)

    # Write sbatch wrapper to remote
    logger.debug(f"SLURM | writing sbatch script to {remote_sbatch}")
    write_sbatch = f"cat > {remote_sbatch} << 'SBATCH_EOF'\n{sbatch_content}SBATCH_EOF"
    write_result = run(write_sbatch, host, timeout=timeout)
    if not write_result.ok:
        logger.error(f"SLURM | failed to write sbatch script: {write_result.stderr}")
        return SlurmResult(job_id=None, ssh_result=write_result)

    # Submit with sbatch
    logger.debug(f"SLURM | submitting sbatch {remote_sbatch}")
    submit_result = run(f"sbatch {remote_sbatch}", host, timeout=timeout)

    # Parse job ID from output like "Submitted batch job 12345"
    job_id = None
    if submit_result.ok:
        match = re.search(r"Submitted batch job (\d+)", submit_result.stdout)
        if match:
            job_id = int(match.group(1))
            logger.info(f"SLURM | job submitted successfully, job_id={job_id}")
    else:
        logger.error(f"SLURM | sbatch failed: {submit_result.stderr}")

    return SlurmResult(job_id=job_id, ssh_result=submit_result)


def submit_packed(
    job: SlurmJob,
    host: str,
    repo_path: str | None = None,
    dirty: bool = False,
    script_path: str | None = None,
    timeout: float | None = None,
) -> SlurmResult:
    """Submit a SLURM job with code packed into the script.

    The code tarball is embedded in the sbatch script and extracted
    at runtime on the compute node (to $SLURM_TMPDIR/code by default).

    Args:
        job: SlurmJob configuration
        host: SSH host with SLURM access
        repo_path: Local git repo to pack (default: cwd)
        dirty: Include uncommitted changes (default: False)
        script_path: Optional remote path for script
        timeout: SSH timeout in seconds

    Returns:
        SlurmResult with job_id and SSH result
    """
    from theseus.dispatch.sync import snapshot
    import subprocess

    repo_path_resolved = repo_path or "."
    logger.info(f"SLURM | packing code from {repo_path_resolved} (dirty={dirty})")

    if dirty:
        # Capture working tree state without stashing
        logger.debug("SLURM | capturing dirty working tree via git stash create")
        stash_result = subprocess.run(
            ["git", "stash", "create"],
            cwd=repo_path_resolved,
            capture_output=True,
            text=True,
        )
        ref = stash_result.stdout.strip() or "HEAD"
    else:
        ref = "HEAD"

    logger.debug(f"SLURM | creating snapshot at ref={ref}")
    tarball = snapshot(repo_path_resolved, ref)
    logger.debug(f"SLURM | tarball size: {len(tarball)} bytes")
    packed_job = job.pack(tarball)

    return submit(packed_job, host, script_path, timeout)


def _parse_squeue_line(line: str) -> JobStatus | None:
    """Parse a pipe-delimited squeue output line."""
    parts = line.strip().split("|")
    if len(parts) < 8:
        return None
    try:
        return JobStatus(
            job_id=int(parts[0]),
            partition=parts[1],
            name=parts[2],
            user=parts[3],
            state=parts[4],
            time_elapsed=parts[5],
            nodes=int(parts[6]) if parts[6].isdigit() else 0,
            nodelist=parts[7],
        )
    except (ValueError, IndexError):
        return None


def status(job_id: int, host: str, timeout: float | None = None) -> StatusResult:
    """Check the status of a SLURM job.

    Args:
        job_id: SLURM job ID
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        StatusResult with parsed JobStatus
    """
    logger.debug(f"SLURM | checking status of job {job_id} on {host}")
    # Use parseable format with pipe delimiter
    result = run(
        f"squeue -j {job_id} -o '%i|%P|%j|%u|%t|%M|%D|%R' --noheader",
        host,
        timeout=timeout,
    )

    job = None
    if result.ok and result.stdout.strip():
        job = _parse_squeue_line(result.stdout.strip())
        if job:
            logger.debug(f"SLURM | job {job_id} state={job.state}, nodes={job.nodes}")

    return StatusResult(job=job, ssh_result=result)


def cancel(job_id: int, host: str, timeout: float | None = None) -> RunResult:
    """Cancel a SLURM job.

    Args:
        job_id: SLURM job ID to cancel
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        RunResult from scancel
    """
    logger.info(f"SLURM | cancelling job {job_id} on {host}")
    result = run(f"scancel {job_id}", host, timeout=timeout)
    if result.ok:
        logger.debug(f"SLURM | job {job_id} cancelled successfully")
    else:
        logger.warning(f"SLURM | failed to cancel job {job_id}: {result.stderr}")
    return result


def job_info(job_id: int, host: str, timeout: float | None = None) -> JobInfoResult:
    """Get detailed info about a SLURM job (including completed jobs).

    Args:
        job_id: SLURM job ID
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        JobInfoResult with parsed job steps
    """
    logger.debug(f"SLURM | fetching job info for {job_id} from {host}")
    result = run(
        f"sacct -j {job_id} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList -P --noheader",
        host,
        timeout=timeout,
    )

    steps: list[JobInfo] = []
    if result.ok and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 8:
                steps.append(
                    JobInfo(
                        job_id=parts[0],
                        name=parts[1],
                        partition=parts[2],
                        state=parts[3],
                        exit_code=parts[4],
                        elapsed=parts[5],
                        max_rss=parts[6],
                        nodelist=parts[7],
                    )
                )
        logger.debug(f"SLURM | job {job_id} has {len(steps)} steps")

    return JobInfoResult(steps=steps, ssh_result=result)


def queue(
    host: str, user: str | None = None, timeout: float | None = None
) -> QueueResult:
    """List jobs in the SLURM queue.

    Args:
        host: SSH host with SLURM access
        user: Filter by user (default: all users)
        timeout: SSH timeout in seconds

    Returns:
        QueueResult with list of parsed JobStatus
    """
    logger.debug(f"SLURM | querying queue on {host} (user={user})")
    cmd = "squeue -o '%i|%P|%j|%u|%t|%M|%D|%R' --noheader"
    if user:
        cmd += f" -u {user}"

    result = run(cmd, host, timeout=timeout)

    jobs: list[JobStatus] = []
    if result.ok and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            job = _parse_squeue_line(line)
            if job:
                jobs.append(job)
        logger.debug(f"SLURM | found {len(jobs)} jobs in queue")

    return QueueResult(jobs=jobs, ssh_result=result)


@dataclass
class NodeGres:
    """GRES (generic resource) info for a node."""

    name: str  # e.g., "gpu"
    type: str | None  # e.g., "a100", "h100"
    configured: int
    allocated: int

    @property
    def available(self) -> int:
        return self.configured - self.allocated


@dataclass
class NodeInfo:
    """Detailed info about a SLURM node."""

    name: str
    state: str  # e.g., "idle", "allocated", "mixed", "down"
    cpus_total: int
    cpus_allocated: int
    memory_total: int  # MB
    memory_allocated: int  # MB
    gres: list[NodeGres]
    partitions: list[str]
    features: list[str]

    @property
    def cpus_available(self) -> int:
        return self.cpus_total - self.cpus_allocated

    @property
    def memory_available(self) -> int:
        return self.memory_total - self.memory_allocated

    def get_gres(self, name: str) -> NodeGres | None:
        """Get GRES by name (e.g., 'gpu')."""
        for g in self.gres:
            if g.name == name:
                return g
        return None


@dataclass
class PartitionInfo:
    """Info about a SLURM partition."""

    name: str
    state: str  # e.g., "up", "down"
    nodes: list[str]
    total_cpus: int
    total_nodes: int


def partitions(host: str, timeout: float | None = None) -> list[PartitionInfo]:
    """List all SLURM partitions.

    Args:
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        List of PartitionInfo
    """
    logger.debug(f"SLURM | listing partitions on {host}")
    result = run(
        "sinfo -o '%P|%a|%D|%C|%N' --noheader",
        host,
        timeout=timeout,
    )

    parts: list[PartitionInfo] = []
    if result.ok and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            fields = line.strip().split("|")
            if len(fields) >= 5:
                name = fields[0].rstrip("*")  # remove default marker
                # %C format is "allocated/idle/other/total"
                cpus_parts = fields[3].split("/")
                total_cpus = int(cpus_parts[-1]) if cpus_parts[-1].isdigit() else 0
                # Expand node list
                nodes = _expand_nodelist(fields[4])
                parts.append(
                    PartitionInfo(
                        name=name,
                        state=fields[1],
                        total_nodes=int(fields[2]) if fields[2].isdigit() else 0,
                        total_cpus=total_cpus,
                        nodes=nodes,
                    )
                )
        logger.debug(f"SLURM | found {len(parts)} partitions")

    return parts


def partition_nodes(
    partition: str, host: str, timeout: float | None = None
) -> list[str]:
    """List all nodes in a SLURM partition.

    Args:
        partition: Partition name
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        List of node names
    """
    logger.debug(f"SLURM | listing nodes in partition '{partition}' on {host}")
    result = run(
        f"sinfo -p {partition} -N -h -o '%n'",
        host,
        timeout=timeout,
    )

    if not result.ok:
        logger.warning(f"SLURM | failed to list partition nodes: {result.stderr}")
        return []

    nodes = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    logger.debug(f"SLURM | partition '{partition}' has {len(nodes)} nodes")
    return nodes


def _expand_nodelist(nodelist: str) -> list[str]:
    """Expand SLURM nodelist notation like 'node[01-03,05]' to individual names."""
    import re

    if not nodelist or nodelist == "(null)":
        return []

    nodes: list[str] = []

    # Split by comma, but not inside brackets
    parts = re.split(r",(?![^\[]*\])", nodelist)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for range notation: prefix[01-03,05]
        match = re.match(r"^(.+?)\[([^\]]+)\](.*)$", part)
        if match:
            prefix, ranges, suffix = match.groups()
            for r in ranges.split(","):
                if "-" in r:
                    start, end = r.split("-", 1)
                    width = len(start)
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{str(i).zfill(width)}{suffix}")
                else:
                    nodes.append(f"{prefix}{r}{suffix}")
        else:
            nodes.append(part)

    return nodes


def _parse_gres(cfg_tres: str, alloc_tres: str) -> list[NodeGres]:
    """Parse GRES from CfgTRES and AllocTRES strings."""
    import re

    gres_list: list[NodeGres] = []

    # Parse configured GRES: gres/gpu=4 or gres/gpu:a100=4
    cfg_pattern = re.compile(r"gres/(\w+)(?::(\w+))?=(\d+)")
    alloc_pattern = re.compile(r"gres/(\w+)(?::(\w+))?=(\d+)")

    cfg_gres: dict[str, tuple[str | None, int]] = {}
    for match in cfg_pattern.finditer(cfg_tres):
        name, gtype, count = match.groups()
        cfg_gres[name] = (gtype, int(count))

    alloc_gres: dict[str, int] = {}
    for match in alloc_pattern.finditer(alloc_tres):
        name, _, count = match.groups()
        alloc_gres[name] = int(count)

    for name, (gtype, configured) in cfg_gres.items():
        gres_list.append(
            NodeGres(
                name=name,
                type=gtype,
                configured=configured,
                allocated=alloc_gres.get(name, 0),
            )
        )

    return gres_list


def node_info(
    nodename: str, host: str, timeout: float | None = None
) -> NodeInfo | None:
    """Get detailed info about a SLURM node.

    Args:
        nodename: Name of the node
        host: SSH host with SLURM access
        timeout: SSH timeout in seconds

    Returns:
        NodeInfo or None if node not found
    """
    import re

    logger.debug(f"SLURM | fetching node info for '{nodename}' on {host}")
    result = run(f"scontrol show node {nodename}", host, timeout=timeout)

    if not result.ok or "not found" in result.stdout.lower():
        logger.debug(f"SLURM | node '{nodename}' not found")
        return None

    output = result.stdout

    def extract(pattern: str, default: str = "") -> str:
        match = re.search(pattern, output)
        return match.group(1) if match else default

    state = extract(r"State=(\S+)")
    cpus_total = int(extract(r"CPUTot=(\d+)", "0"))
    cpus_alloc = int(extract(r"CPUAlloc=(\d+)", "0"))
    mem_total = int(extract(r"RealMemory=(\d+)", "0"))
    mem_alloc = int(extract(r"AllocMem=(\d+)", "0"))
    partitions_str = extract(r"Partitions=(\S+)")
    features_str = extract(r"AvailableFeatures=(\S+)")
    cfg_tres = extract(r"CfgTRES=(\S+)")
    alloc_tres = extract(r"AllocTRES=(\S*)")

    return NodeInfo(
        name=nodename,
        state=state.split("+")[0] if state else "unknown",  # remove flags like "+DRAIN"
        cpus_total=cpus_total,
        cpus_allocated=cpus_alloc,
        memory_total=mem_total,
        memory_allocated=mem_alloc,
        gres=_parse_gres(cfg_tres, alloc_tres),
        partitions=partitions_str.split(",") if partitions_str else [],
        features=features_str.split(",")
        if features_str and features_str != "(null)"
        else [],
    )


def nodes_info(
    nodenames: list[str], host: str, timeout: float | None = None
) -> dict[str, NodeInfo]:
    """Get info about multiple nodes in parallel.

    Args:
        nodenames: List of node names
        host: SSH host with SLURM access
        timeout: SSH timeout per node

    Returns:
        Dict mapping nodename -> NodeInfo (excludes failed lookups)
    """
    import concurrent.futures

    logger.debug(f"SLURM | fetching info for {len(nodenames)} nodes on {host}")
    results: dict[str, NodeInfo] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(nodenames), 16)
    ) as executor:
        future_to_node = {
            executor.submit(node_info, name, host, timeout): name for name in nodenames
        }
        for future in concurrent.futures.as_completed(future_to_node):
            name = future_to_node[future]
            try:
                info = future.result()
                if info:
                    results[name] = info
            except Exception as e:
                logger.debug(f"SLURM | failed to get info for node '{name}': {e}")

    logger.debug(f"SLURM | retrieved info for {len(results)}/{len(nodenames)} nodes")
    return results


def available_gpus(
    partition: str, host: str, gpu_type: str | None = None, timeout: float | None = None
) -> list[tuple[str, int]]:
    """Find nodes with available GPUs in a partition.

    Args:
        partition: Partition name
        host: SSH host with SLURM access
        gpu_type: Optional GPU type filter (e.g., "a100")
        timeout: SSH timeout

    Returns:
        List of (nodename, available_gpu_count) tuples, sorted by availability descending
    """
    logger.debug(
        f"SLURM | checking available GPUs in partition '{partition}' (type={gpu_type})"
    )
    node_names = partition_nodes(partition, host, timeout=timeout)
    if not node_names:
        logger.debug(f"SLURM | no nodes found in partition '{partition}'")
        return []

    infos = nodes_info(node_names, host, timeout=timeout)

    available: list[tuple[str, int]] = []
    for name, info in infos.items():
        gpu = info.get_gres("gpu")
        if gpu and gpu.available > 0:
            if gpu_type is None or gpu.type == gpu_type:
                available.append((name, gpu.available))

    total_available = sum(count for _, count in available)
    logger.debug(
        f"SLURM | found {total_available} available GPUs across {len(available)} nodes"
    )
    return sorted(available, key=lambda x: x[1], reverse=True)


def wait(
    job_id: int, host: str, poll_interval: float = 10.0, timeout: float | None = None
) -> JobInfoResult:
    """Wait for a SLURM job to complete.

    Args:
        job_id: SLURM job ID
        host: SSH host with SLURM access
        poll_interval: Seconds between status checks
        timeout: Total timeout in seconds (None = wait forever)

    Returns:
        JobInfoResult with final job state
    """
    import time

    logger.info(
        f"SLURM | waiting for job {job_id} to complete (poll_interval={poll_interval}s)"
    )
    start = time.time()

    while True:
        # First check squeue (for running/pending jobs)
        st = status(job_id, host)
        if st.job is None:
            # Job not in queue, check sacct for completion info
            logger.info(
                f"SLURM | job {job_id} no longer in queue, fetching final status"
            )
            return job_info(job_id, host)

        if st.job.is_failed:
            logger.warning(f"SLURM | job {job_id} failed with state={st.job.state}")
            return job_info(job_id, host)

        # Check timeout
        if timeout and (time.time() - start) > timeout:
            elapsed = time.time() - start
            logger.warning(
                f"SLURM | wait timed out after {elapsed:.1f}s for job {job_id}"
            )
            return job_info(job_id, host)

        logger.debug(
            f"SLURM | job {job_id} state={st.job.state}, waiting {poll_interval}s..."
        )
        time.sleep(poll_interval)
