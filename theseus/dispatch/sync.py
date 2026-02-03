"""
Code synchronization utilities for shipping repos to remote hosts.
"""

import subprocess
from pathlib import Path

from loguru import logger

from theseus.dispatch.ssh import RunResult


def snapshot(repo_path: str | Path | None = None, ref: str = "HEAD") -> bytes:
    """Create a tarball snapshot of the repo using git archive.

    Only includes tracked files, respects .gitignore, excludes .git/.

    Args:
        repo_path: Path to git repo (default: current working directory)
        ref: Git ref to archive (default: HEAD, can be branch/tag/commit)

    Returns:
        Tarball bytes (gzip compressed)
    """
    logger.debug(f"SYNC | creating snapshot of {repo_path or 'cwd'} at ref={ref}")
    cmd = ["git", "archive", "--format=tar.gz", ref]

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    logger.debug(f"SYNC | snapshot created: {len(result.stdout)} bytes")
    return result.stdout


def ship(
    host: str,
    remote_path: str,
    repo_path: str | Path | None = None,
    ref: str = "HEAD",
    timeout: float | None = None,
) -> RunResult:
    """Ship a snapshot of the repo to a remote host.

    Creates a tarball of tracked files and extracts on remote.

    Args:
        host: SSH host
        remote_path: Destination directory on remote (will be created)
        repo_path: Local git repo path (default: cwd)
        ref: Git ref to archive (default: HEAD)
        timeout: SSH timeout

    Returns:
        RunResult from extraction
    """
    logger.info(f"SYNC | shipping to {host}:{remote_path} (ref={ref})")
    tarball = snapshot(repo_path, ref)

    # Create remote directory and extract tarball via stdin
    # Using -m to avoid timestamp issues with NFS/distributed filesystems
    extract_cmd = f"mkdir -p {remote_path} && tar -xzf - -C {remote_path} -m"

    logger.debug(f"SYNC | extracting {len(tarball)} bytes on remote")
    result = _run_with_stdin(extract_cmd, host, tarball, timeout=timeout)

    if result.ok:
        logger.info(f"SYNC | shipped successfully to {host}:{remote_path}")
    else:
        logger.error(f"SYNC | ship failed: {result.stderr}")

    return result


def ship_dirty(
    host: str,
    remote_path: str,
    repo_path: str | Path | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Ship repo including uncommitted changes.

    Uses git stash to capture working directory state, then archives.
    Useful for development/testing when you want to ship uncommitted code.

    Args:
        host: SSH host
        remote_path: Destination directory on remote
        repo_path: Local git repo path (default: cwd)
        timeout: SSH timeout

    Returns:
        RunResult from extraction
    """
    logger.info(f"SYNC | shipping dirty to {host}:{remote_path}")
    repo_path = Path(repo_path) if repo_path else Path.cwd()

    # Create tarball of tracked files including working tree changes
    # git stash create makes a commit object without actually stashing
    logger.debug("SYNC | capturing dirty state via git stash create")
    stash_result = subprocess.run(
        ["git", "stash", "create"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    # If there are changes, stash create returns a commit hash
    # If no changes, it returns empty (use HEAD)
    ref = stash_result.stdout.strip() or "HEAD"
    if ref != "HEAD":
        logger.debug(f"SYNC | dirty ref: {ref[:12]}")
    else:
        logger.debug("SYNC | no uncommitted changes, using HEAD")

    return ship(host, remote_path, repo_path, ref, timeout)


def ship_files(
    host: str,
    remote_path: str,
    files: list[str | Path],
    base_path: str | Path | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Ship specific files to a remote host.

    Args:
        host: SSH host
        remote_path: Destination directory on remote
        files: List of file paths to include
        base_path: Base path for relative file paths (default: cwd)
        timeout: SSH timeout

    Returns:
        RunResult from extraction
    """
    logger.info(f"SYNC | shipping {len(files)} files to {host}:{remote_path}")
    base_path = Path(base_path) if base_path else Path.cwd()

    # Create tarball of specified files
    cmd = ["tar", "-czf", "-", "-C", str(base_path)]
    for f in files:
        # Make paths relative to base
        p = Path(f)
        if p.is_absolute():
            p = p.relative_to(base_path)
        cmd.append(str(p))

    logger.debug(f"SYNC | creating tarball of {len(files)} files")
    result = subprocess.run(cmd, capture_output=True, check=True)
    tarball = result.stdout
    logger.debug(f"SYNC | tarball size: {len(tarball)} bytes")

    extract_cmd = f"mkdir -p {remote_path} && tar -xzf - -C {remote_path} -m"
    return _run_with_stdin(extract_cmd, host, tarball, timeout=timeout)


def _run_with_stdin(
    cmd: str, host: str, stdin_data: bytes, timeout: float | None = None
) -> RunResult:
    """Run a command on remote host with stdin data."""
    import shlex

    # Wrap in login shell for env vars
    wrapped_cmd = f"$SHELL -l -c {shlex.quote(cmd)}"
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", host, wrapped_cmd]

    try:
        result = subprocess.run(
            ssh_cmd,
            input=stdin_data,
            capture_output=True,
            timeout=timeout,
        )
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout.decode() if result.stdout else "",
            stderr=result.stderr.decode() if result.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
        )


def sync(
    host: str,
    remote_path: str,
    repo_path: str | Path | None = None,
    exclude: list[str] | None = None,
    delete: bool = False,
    timeout: float | None = None,
) -> RunResult:
    """Sync repo to remote using rsync (requires rsync on both ends).

    More efficient for incremental updates than ship().

    Args:
        host: SSH host
        remote_path: Destination directory on remote
        repo_path: Local path to sync (default: cwd)
        exclude: Additional patterns to exclude
        delete: Remove remote files not in source
        timeout: Timeout in seconds

    Returns:
        RunResult from rsync
    """
    logger.info(f"SYNC | rsync to {host}:{remote_path} (delete={delete})")
    repo_path = Path(repo_path) if repo_path else Path.cwd()

    cmd = [
        "rsync",
        "-az",  # archive mode, compress
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.mypy_cache",
        "--exclude=.ruff_cache",
        "--exclude=.venv",
        "--exclude=*.egg-info",
        "--exclude=.env",
        "--exclude=*.pt",
        "--exclude=*.pth",
        "--exclude=*.ckpt",
        "--exclude=*.safetensors",
        "--exclude=wandb/",
    ]

    if exclude:
        for pattern in exclude:
            cmd.append(f"--exclude={pattern}")
        logger.debug(f"SYNC | additional excludes: {exclude}")

    if delete:
        cmd.append("--delete")

    # Ensure trailing slash on source to sync contents, not directory
    src = str(repo_path).rstrip("/") + "/"
    dst = f"{host}:{remote_path}"

    cmd.extend([src, dst])

    logger.debug(f"SYNC | rsync from {src} to {dst}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("SYNC | rsync completed successfully")
        else:
            logger.error(f"SYNC | rsync failed: {result.stderr}")
        return RunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"SYNC | rsync timed out after {timeout}s")
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"rsync timed out after {timeout}s",
        )
