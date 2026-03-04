"""
Google Cloud TPU VM utilities for remote dispatch.

Wraps gcloud compute tpus tpu-vm commands to provide an SSH-like interface
for creating, managing, and executing commands on TPU VMs.

Assumes the user has gcloud installed and authenticated locally.
"""

import json
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from theseus.dispatch.ssh import RunResult, TunnelResult


def parse_accelerator_type(accel_type: str) -> tuple[str, int]:
    """Parse TPU accelerator type into (chip_name, total_chips).

    Examples:
        "v4-32" -> ("tpu-v4", 32)
        "v5e-16" -> ("tpu-v5e", 16)
        "v3-8" -> ("tpu-v3", 8)
    """
    match = re.match(r"(v\d+[a-z]*)-(\d+)", accel_type)
    if not match:
        raise ValueError(
            f"Invalid TPU accelerator type: '{accel_type}'. "
            f"Expected format like 'v4-32', 'v5e-16', etc."
        )
    version = match.group(1)
    chips = int(match.group(2))
    return f"tpu-{version}", chips


# ---------------------------------------------------------------------------
# Core gcloud wrappers
# ---------------------------------------------------------------------------


def run(
    cmd: str,
    tpu_name: str,
    zone: str,
    project: str | None = None,
    worker: str = "all",
    internal_ip: bool = False,
    timeout: float | None = None,
) -> RunResult:
    """Execute a command on a TPU VM via ``gcloud compute tpus tpu-vm ssh``.

    Uses ``--worker=all`` by default so the same command runs on every worker
    in a TPU pod simultaneously.
    """
    gcloud_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--worker={worker}",
        f"--command={cmd}",
    ]
    if project:
        gcloud_cmd.append(f"--project={project}")
    if internal_ip:
        gcloud_cmd.append("--internal-ip")

    cmd_preview = cmd[:80] + "..." if len(cmd) > 80 else cmd
    logger.debug(f"TPU | running on {tpu_name} (worker={worker}): {cmd_preview}")

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.debug("TPU | command succeeded")
        else:
            logger.debug(
                f"TPU | command failed (rc={result.returncode}): "
                f"{result.stderr[:200] if result.stderr else 'no stderr'}"
            )
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"TPU | command timed out after {timeout}s on {tpu_name}")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"gcloud ssh timed out after {timeout}s",
        )


def copy_to(
    local_path: str | Path,
    tpu_name: str,
    remote_path: str,
    zone: str,
    project: str | None = None,
    worker: str = "all",
    internal_ip: bool = False,
    timeout: float | None = None,
) -> RunResult:
    """Copy a local file or directory to a TPU VM via ``gcloud compute tpus tpu-vm scp``.

    Copies to all workers by default so every host gets identical files.
    """
    local_path = Path(local_path)
    gcloud_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "scp",
        str(local_path),
        f"{tpu_name}:{remote_path}",
        f"--zone={zone}",
        f"--worker={worker}",
    ]
    if local_path.is_dir():
        gcloud_cmd.append("--recurse")
    if project:
        gcloud_cmd.append(f"--project={project}")
    if internal_ip:
        gcloud_cmd.append("--internal-ip")

    logger.debug(f"TPU | copying {local_path} to {tpu_name}:{remote_path} (worker={worker})")

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.debug("TPU | copy_to succeeded")
        else:
            logger.warning(f"TPU | copy_to failed: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"TPU | scp timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"gcloud scp timed out after {timeout}s",
        )


# ---------------------------------------------------------------------------
# TPU lifecycle management
# ---------------------------------------------------------------------------


def create(
    name: str,
    zone: str,
    accelerator_type: str,
    version: str,
    project: str | None = None,
    spot: bool = False,
    preemptible: bool = False,
    network: str | None = None,
    subnetwork: str | None = None,
    service_account: str | None = None,
    metadata: dict[str, str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Create a TPU VM.

    .. warning:: This incurs GCP costs.  The dispatch layer prompts the user
       for confirmation before calling this function.
    """
    gcloud_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "create",
        name,
        f"--zone={zone}",
        f"--accelerator-type={accelerator_type}",
        f"--version={version}",
    ]
    if project:
        gcloud_cmd.append(f"--project={project}")
    if spot:
        gcloud_cmd.append("--spot")
    if preemptible:
        gcloud_cmd.append("--preemptible")
    if network:
        gcloud_cmd.append(f"--network={network}")
    if subnetwork:
        gcloud_cmd.append(f"--subnetwork={subnetwork}")
    if service_account:
        gcloud_cmd.append(f"--service-account={service_account}")
    if metadata:
        pairs = ",".join(f"{k}={v}" for k, v in metadata.items())
        gcloud_cmd.append(f"--metadata={pairs}")

    logger.info(
        f"TPU | creating TPU VM '{name}' ({accelerator_type}) in {zone}"
        f"{' (spot)' if spot else ''}{' (preemptible)' if preemptible else ''}"
    )

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info(f"TPU | created TPU VM '{name}' successfully")
        else:
            logger.error(f"TPU | failed to create TPU VM: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"TPU | create timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"gcloud create timed out after {timeout}s",
        )


def delete(
    name: str,
    zone: str,
    project: str | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Delete a TPU VM.

    Uses ``--quiet`` to skip interactive confirmation from gcloud itself.
    """
    gcloud_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete",
        name,
        f"--zone={zone}",
        "--quiet",
    ]
    if project:
        gcloud_cmd.append(f"--project={project}")

    logger.info(f"TPU | deleting TPU VM '{name}' in {zone}")

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info(f"TPU | deleted TPU VM '{name}'")
        else:
            logger.error(f"TPU | failed to delete TPU VM: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"TPU | delete timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"gcloud delete timed out after {timeout}s",
        )


def describe(
    name: str,
    zone: str,
    project: str | None = None,
    timeout: float | None = None,
) -> dict | None:
    """Get TPU VM description as parsed JSON.  Returns *None* if not found."""
    gcloud_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe",
        name,
        f"--zone={zone}",
        "--format=json",
    ]
    if project:
        gcloud_cmd.append(f"--project={project}")

    logger.debug(f"TPU | describing TPU VM '{name}' in {zone}")

    try:
        result = subprocess.run(
            gcloud_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def get_status(
    name: str,
    zone: str,
    project: str | None = None,
    timeout: float | None = None,
) -> str | None:
    """Get TPU VM state (e.g. ``READY``, ``CREATING``).  Returns *None* if not found."""
    info = describe(name, zone, project, timeout)
    if info is None:
        return None
    return info.get("state")


def wait_ready(
    name: str,
    zone: str,
    project: str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 15.0,
) -> bool:
    """Block until the TPU VM reaches ``READY`` state.

    Returns ``True`` on success, ``False`` on timeout or terminal state.
    """
    logger.info(f"TPU | waiting for '{name}' to become READY (timeout={timeout}s)")
    start = time.time()
    while (time.time() - start) < timeout:
        state = get_status(name, zone, project, timeout=30.0)
        if state == "READY":
            logger.info(f"TPU | '{name}' is READY")
            return True
        if state in ("TERMINATED", "FAILED", "DELETING"):
            logger.error(f"TPU | '{name}' entered terminal state: {state}")
            return False
        logger.debug(f"TPU | '{name}' state={state}, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    logger.error(f"TPU | timed out waiting for '{name}' to become READY")
    return False


# ---------------------------------------------------------------------------
# Port forwarding (mirrors ssh.forward_port for TPU)
# ---------------------------------------------------------------------------


def forward_port(
    tpu_name: str,
    zone: str,
    local_port: int,
    remote_port: int,
    project: str | None = None,
    worker: str = "0",
    internal_ip: bool = False,
) -> TunnelResult:
    """Start a background SSH tunnel via ``gcloud compute tpus tpu-vm ssh``.

    Forwards *local_port* on the dispatching machine to *remote_port* on the
    TPU VM worker.  Only runs on a single worker (default ``0``) since REPL
    sessions are single-host.
    """
    if local_port <= 0 or remote_port <= 0:
        return TunnelResult(
            returncode=-1,
            pid=None,
            command=[],
            stderr="local_port and remote_port must be positive integers",
        )

    # Fail fast if local port already in use.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", local_port))
        except OSError as e:
            return TunnelResult(
                returncode=-1,
                pid=None,
                command=[],
                stderr=f"local port {local_port} is already in use: {e}",
            )

    def _listener_pids(port: int) -> set[int]:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return set()
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--worker={worker}",
        f"--ssh-flag=-L {local_port}:localhost:{remote_port}",
        "--ssh-flag=-N",
    ]
    if project:
        cmd.append(f"--project={project}")
    if internal_ip:
        cmd.append("--internal-ip")

    logger.debug(
        f"TPU | forwarding local:{local_port} -> {tpu_name}(worker={worker}):{remote_port}"
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception as e:
        return TunnelResult(returncode=-1, pid=None, command=cmd, stderr=str(e))

    # Wait for tunnel listener to appear locally.
    wait_seconds = float(os.environ.get("THESEUS_SSH_TUNNEL_WAIT_SECONDS", "20.0"))
    wait_seconds = max(5.0, wait_seconds)
    deadline = time.time() + wait_seconds
    sleep_s = 0.1
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read() or ""
            return TunnelResult(returncode=rc, pid=None, command=cmd, stderr=stderr.strip())

        listeners = _listener_pids(local_port)
        if proc.pid in listeners:
            logger.debug(f"TPU | tunnel established (pid={proc.pid})")
            return TunnelResult(returncode=0, pid=proc.pid, command=cmd, stderr="")
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 1.5, 1.0)

    # Timed out.
    try:
        proc.terminate()
    except Exception:
        pass
    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read() or ""
        except Exception:
            stderr = ""
    return TunnelResult(
        returncode=-1,
        pid=None,
        command=cmd,
        stderr=stderr.strip()
        or f"gcloud ssh tunnel did not open local listener in time ({wait_seconds:.1f}s)",
    )


# ---------------------------------------------------------------------------
# Code shipping (mirrors sync.ship / sync.ship_dirty for TPU)
# ---------------------------------------------------------------------------


def ship(
    tpu_name: str,
    remote_path: str,
    zone: str,
    project: str | None = None,
    internal_ip: bool = False,
    repo_path: str | Path | None = None,
    ref: str = "HEAD",
    timeout: float | None = None,
) -> RunResult:
    """Ship a code snapshot to **all** TPU VM workers.

    Creates a tarball via ``git archive``, SCPs it to every worker, then
    extracts in-place.  This guarantees identical code across all hosts.
    """
    from theseus.dispatch.sync import snapshot

    logger.info(f"TPU | shipping code to {tpu_name}:{remote_path}")
    tarball = snapshot(repo_path, ref)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        f.write(tarball)
        local_tarball = f.name

    try:
        # SCP tarball to all workers
        logger.debug(f"TPU | copying {len(tarball)} byte tarball to all workers")
        scp_result = copy_to(
            local_tarball,
            tpu_name,
            "~/_theseus_snapshot.tar.gz",
            zone,
            project,
            worker="all",
            internal_ip=internal_ip,
            timeout=timeout,
        )
        if not scp_result.ok:
            logger.error(f"TPU | failed to copy tarball: {scp_result.stderr}")
            return scp_result

        # Extract on all workers
        extract_cmd = (
            f"mkdir -p {remote_path} && "
            f"tar -xzf ~/_theseus_snapshot.tar.gz -C {remote_path} -m && "
            f"rm -f ~/_theseus_snapshot.tar.gz"
        )
        logger.debug("TPU | extracting tarball on all workers")
        extract_result = run(
            extract_cmd,
            tpu_name,
            zone,
            project,
            worker="all",
            internal_ip=internal_ip,
            timeout=timeout,
        )
        if extract_result.ok:
            logger.info(f"TPU | code shipped successfully to {tpu_name}")
        else:
            logger.error(f"TPU | extraction failed: {extract_result.stderr}")
        return extract_result
    finally:
        os.unlink(local_tarball)


def ship_dirty(
    tpu_name: str,
    remote_path: str,
    zone: str,
    project: str | None = None,
    internal_ip: bool = False,
    repo_path: str | Path | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Ship code including uncommitted changes to all TPU VM workers."""
    logger.info(f"TPU | shipping dirty code to {tpu_name}:{remote_path}")
    repo_path_resolved = Path(repo_path) if repo_path else Path.cwd()

    stash_result = subprocess.run(
        ["git", "stash", "create"],
        cwd=repo_path_resolved,
        capture_output=True,
        text=True,
    )
    ref = stash_result.stdout.strip() or "HEAD"
    if ref != "HEAD":
        logger.debug(f"TPU | dirty ref: {ref[:12]}")
    else:
        logger.debug("TPU | no uncommitted changes, using HEAD")

    return ship(
        tpu_name,
        remote_path,
        zone,
        project,
        internal_ip,
        repo_path_resolved,
        ref,
        timeout,
    )
