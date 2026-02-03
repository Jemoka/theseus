"""
ssh utilities for remote dispatch
"""

import subprocess
import shlex
import re
from pathlib import Path
from dataclasses import dataclass

from loguru import logger


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell execution."""
    return shlex.quote(s)


@dataclass
class RunResult:
    """Result of a remote command execution"""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def hosts() -> list[str]:
    """Parse ~/.ssh/config to list accessible hosts.

    Returns a list of host names/aliases defined in the SSH config.
    Excludes wildcard patterns and special entries.
    """
    config_path = Path.home() / ".ssh" / "config"

    if not config_path.exists():
        logger.debug("SSH | no ~/.ssh/config found")
        return []

    hosts_list: list[str] = []
    content = config_path.read_text()

    # Match "Host" lines, handling multiple hosts per line
    host_pattern = re.compile(r"^\s*Host\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE)

    for match in host_pattern.finditer(content):
        # Split on whitespace to handle "Host foo bar baz"
        entries = match.group(1).split()
        for entry in entries:
            # Skip wildcards and special patterns
            if "*" in entry or "?" in entry or "!" in entry:
                continue
            hosts_list.append(entry)

    logger.debug(f"SSH | found {len(hosts_list)} hosts in ssh config")
    return hosts_list


def run(cmd: str | list[str], host: str, timeout: float | None = None) -> RunResult:
    """Execute a command remotely on a host via SSH.

    Runs in a login shell to ensure environment variables are loaded.

    Args:
        cmd: Command to execute (string or list of args)
        host: SSH host (name, alias, or user@host)
        timeout: Optional timeout in seconds

    Returns:
        RunResult with returncode, stdout, and stderr
    """
    if isinstance(cmd, list):
        cmd = " ".join(cmd)

    # Wrap in login shell to get full environment (bashrc, profile, etc.)
    # Using $SHELL -l -c ensures we use the user's default shell (bash, zsh, etc.)
    wrapped_cmd = f"$SHELL -l -c {_shell_quote(cmd)}"
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", host, wrapped_cmd]

    # Truncate cmd for logging if too long
    cmd_preview = cmd[:80] + "..." if len(cmd) > 80 else cmd
    logger.debug(f"SSH | running on {host}: {cmd_preview}")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.debug(
                f"SSH | command failed (rc={result.returncode}): {result.stderr[:200] if result.stderr else 'no stderr'}"
            )
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired as e:
        logger.warning(f"SSH | command timed out after {timeout}s on {host}")
        return RunResult(
            returncode=-1,
            stdout=e.stdout or ""
            if isinstance(e.stdout, str)
            else (e.stdout.decode() if e.stdout else ""),
            stderr=f"Command timed out after {timeout}s",
        )


def run_many(
    cmd: str | list[str], hosts: list[str], timeout: float | None = None
) -> dict[str, RunResult]:
    """Execute a command on multiple hosts in parallel.

    Args:
        cmd: Command to execute
        hosts: List of SSH hosts
        timeout: Optional timeout in seconds per host

    Returns:
        Dict mapping host -> RunResult
    """
    import concurrent.futures

    logger.debug(f"SSH | running command on {len(hosts)} hosts in parallel")
    results: dict[str, RunResult] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        future_to_host = {
            executor.submit(run, cmd, host, timeout): host for host in hosts
        }
        for future in concurrent.futures.as_completed(future_to_host):
            host = future_to_host[future]
            try:
                results[host] = future.result()
            except Exception as e:
                logger.warning(f"SSH | exception running on {host}: {e}")
                results[host] = RunResult(returncode=-1, stdout="", stderr=str(e))

    success_count = sum(1 for r in results.values() if r.ok)
    logger.debug(f"SSH | run_many completed: {success_count}/{len(hosts)} succeeded")
    return results


def copy_to(
    local_path: str | Path, host: str, remote_path: str, timeout: float | None = None
) -> RunResult:
    """Copy a local file/directory to a remote host via scp.

    Args:
        local_path: Local file or directory path
        host: SSH host
        remote_path: Destination path on remote host
        timeout: Optional timeout in seconds

    Returns:
        RunResult with returncode, stdout, and stderr
    """
    local_path = Path(local_path)
    logger.debug(f"SSH | copying {local_path} to {host}:{remote_path}")
    scp_cmd = ["scp", "-o", "BatchMode=yes"]

    if local_path.is_dir():
        scp_cmd.append("-r")

    scp_cmd.extend([str(local_path), f"{host}:{remote_path}"])

    try:
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.debug("SSH | copy_to succeeded")
        else:
            logger.warning(f"SSH | copy_to failed: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"SSH | scp timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"SCP timed out after {timeout}s",
        )


def copy_from(
    host: str, remote_path: str, local_path: str | Path, timeout: float | None = None
) -> RunResult:
    """Copy a remote file/directory to local via scp.

    Args:
        host: SSH host
        remote_path: Source path on remote host
        local_path: Local destination path
        timeout: Optional timeout in seconds

    Returns:
        RunResult with returncode, stdout, and stderr
    """
    logger.debug(f"SSH | copying {host}:{remote_path} to {local_path}")
    scp_cmd = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-r",
        f"{host}:{remote_path}",
        str(local_path),
    ]

    try:
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.debug("SSH | copy_from succeeded")
        else:
            logger.warning(f"SSH | copy_from failed: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"SSH | scp timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"SCP timed out after {timeout}s",
        )


def is_reachable(host: str, timeout: float = 5.0) -> bool:
    """Check if a host is reachable via SSH.

    Args:
        host: SSH host to check
        timeout: Connection timeout in seconds

    Returns:
        True if host is reachable
    """
    logger.debug(f"SSH | checking if {host} is reachable")
    result = run("echo ok", host, timeout=timeout)
    reachable = result.ok and "ok" in result.stdout
    logger.debug(f"SSH | {host} reachable={reachable}")
    return reachable
