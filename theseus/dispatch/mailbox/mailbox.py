from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import difflib
from loguru import logger

MAILBOX_FILE_MODE = 0o666
MAILBOX_DIR_MODE = 0o777


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"command failed: {' '.join(cmd)}: {stderr}")
    return result.stdout


def require_git_repo(cwd: Path) -> Path:
    try:
        root = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd).strip()
    except RuntimeError as exc:
        raise RuntimeError(
            "mailbox sync requires running inside a git repository"
        ) from exc
    if not root:
        raise RuntimeError("failed to resolve git repository root")
    return Path(root)


def git_stash_create(repo_root: Path) -> str | None:
    out = _run(["git", "stash", "create"], cwd=repo_root).strip()
    return out or None


def _tracked_files(repo_root: Path) -> list[str]:
    out = _run(["git", "ls-files", "-z"], cwd=repo_root)
    files = [p for p in out.split("\x00") if p]
    files.sort()
    return files


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(MAILBOX_DIR_MODE)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    try:
        path.chmod(MAILBOX_FILE_MODE)
    except Exception:
        pass


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(MAILBOX_DIR_MODE)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    try:
        path.chmod(MAILBOX_FILE_MODE)
    except Exception:
        pass


def _snapshot_tracked(repo_root: Path, snapshot_dir: Path) -> list[str]:
    files = _tracked_files(repo_root)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for rel in files:
        src = repo_root / rel
        dst = snapshot_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    _write_json_atomic(snapshot_dir / ".manifest.json", {"files": files})
    return files


def _load_manifest(snapshot_dir: Path) -> list[str]:
    manifest_path = snapshot_dir / ".manifest.json"
    if not manifest_path.exists():
        files: list[str] = []
        for p in snapshot_dir.rglob("*"):
            if p.is_file() and p.name != ".manifest.json":
                files.append(p.relative_to(snapshot_dir).as_posix())
        files.sort()
        return files
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [str(x) for x in data.get("files", [])]
    files.sort()
    return files


def _read_text_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8", errors="surrogateescape")
    except Exception:
        return None


def _diff_snapshots(old_dir: Path, new_dir: Path) -> str:
    old_files = set(_load_manifest(old_dir)) if old_dir.exists() else set()
    new_files = set(_load_manifest(new_dir)) if new_dir.exists() else set()
    all_files = sorted(old_files | new_files)

    chunks: list[str] = []
    for rel in all_files:
        old_text = _read_text_or_none(old_dir / rel) if old_dir.exists() else None
        new_text = _read_text_or_none(new_dir / rel) if new_dir.exists() else None

        # Skip binary or unreadable text files.
        if (old_dir / rel).exists() and old_text is None:
            continue
        if (new_dir / rel).exists() and new_text is None:
            continue

        old_lines = [] if old_text is None else old_text.splitlines(keepends=True)
        new_lines = [] if new_text is None else new_text.splitlines(keepends=True)
        if old_lines == new_lines:
            continue

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
            lineterm="",
        )
        chunks.extend(list(diff))

    if not chunks:
        return ""
    return "\n".join(chunks) + "\n"


def _mount_line_for(path: Path) -> str | None:
    result = subprocess.run(["mount"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    needle_a = f" on {path} "
    needle_b = f" on {path} ("
    for line in result.stdout.splitlines():
        if needle_a in line or needle_b in line:
            return line
    return None


def is_local_juicefs_mounted(mount_dir: Path) -> bool:
    line = _mount_line_for(mount_dir)
    return line is not None and "juicefs" in line.lower()


def ensure_local_mount(mount_dir: Path, backend: str) -> None:
    line = _mount_line_for(mount_dir)
    if line is not None:
        if "juicefs" not in line.lower():
            raise RuntimeError(
                f"mount path {mount_dir} exists but is not a JuiceFS mount: {line}"
            )
        return

    if shutil.which("juicefs") is None:
        raise RuntimeError("juicefs not found in PATH and mount is not available")

    mount_dir.mkdir(parents=True, exist_ok=True)
    mount_result = subprocess.run(
        ["juicefs", "mount", "-d", backend, str(mount_dir)],
        capture_output=True,
        text=True,
    )
    if mount_result.returncode != 0:
        raise RuntimeError(
            "failed to mount JuiceFS: "
            + (mount_result.stderr or "unknown error").strip()
        )

    line = _mount_line_for(mount_dir)
    if line is None or "juicefs" not in line.lower():
        raise RuntimeError(f"JuiceFS mount did not become active at {mount_dir}")


def mailbox_root_for(local_mount: Path, cluster_root: str) -> Path:
    # Canonical semantics: top-level dispatch.mount points to the mounted
    # THESEUS root, so mailbox is directly under it.
    return local_mount / "mailbox"


def _active_path(mailbox_root: Path) -> Path:
    return mailbox_root / ".active"


def _read_active(mailbox_root: Path) -> list[dict[str, Any]]:
    path = _active_path(mailbox_root)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                entries.append(item)
        except json.JSONDecodeError:
            continue
    return entries


def read_active(mailbox_root: Path) -> list[dict[str, Any]]:
    return _read_active(mailbox_root)


def _write_active(mailbox_root: Path, entries: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n"
    _write_text_atomic(_active_path(mailbox_root), text)


def deactivate_active_entry(
    *,
    mailbox_root: Path,
    job_id: str,
    workdir: str | None = None,
    reason: str = "stopped",
) -> bool:
    entries = _read_active(mailbox_root)
    changed = False
    out: list[dict[str, Any]] = []
    for entry in entries:
        same_job = str(entry.get("job_id", "")).strip() == job_id
        same_work = (
            workdir is None
            or str(entry.get("workdir", "")).strip() == workdir
            or str(entry.get("workdir", "")).strip() == ""
        )
        if same_job and same_work and entry.get("status", "active") == "active":
            changed = True
            continue
        out.append(entry)
    if changed:
        _write_active(mailbox_root, out)
    return changed


def prune_stale_active_entries(
    *, mailbox_root: Path, max_stale_seconds: float = 180.0
) -> int:
    entries = _read_active(mailbox_root)
    if not entries:
        return 0

    now = datetime.now(timezone.utc)
    changed = False
    removed = 0
    out: list[dict[str, Any]] = []
    for entry in entries:
        if (
            entry.get("status", "active") != "active"
            or entry.get("mode") != "repl-sync"
        ):
            out.append(entry)
            continue

        job_id = str(entry.get("job_id", "")).strip()
        if not job_id:
            out.append(entry)
            continue

        sidecar_state = mailbox_root / "jobs" / job_id / "state" / "sidecar.json"
        last_seen_dt: datetime | None = None
        if sidecar_state.exists():
            try:
                state = json.loads(sidecar_state.read_text(encoding="utf-8"))
                last_seen_dt = _parse_iso(str(state.get("last_seen", "")))
            except Exception:
                last_seen_dt = None
        if last_seen_dt is None:
            last_seen_dt = _parse_iso(str(entry.get("last_heartbeat", "")))
        if last_seen_dt is None:
            last_seen_dt = _parse_iso(str(entry.get("started_at", "")))

        if last_seen_dt is None:
            out.append(entry)
            continue

        age = (now - last_seen_dt).total_seconds()
        if age > max_stale_seconds:
            changed = True
            removed += 1
            continue
        out.append(entry)

    if changed:
        _write_active(mailbox_root, out)
        return removed
    return 0


def register_synced_repl(
    *,
    local_mount: Path,
    cluster_root: str,
    cluster_name: str,
    job_id: str,
    ssh_alias: str,
    host_key: str,
    workdir: str,
    backend: str,
    repo_root: Path,
    initialize_shadow: bool = True,
) -> Path:
    mailbox_root = mailbox_root_for(local_mount, cluster_root)
    mailbox_root.mkdir(parents=True, exist_ok=True)
    try:
        mailbox_root.chmod(MAILBOX_DIR_MODE)
    except Exception:
        pass

    jobs_dir = mailbox_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    try:
        jobs_dir.chmod(MAILBOX_DIR_MODE)
    except Exception:
        pass

    job_dir = jobs_dir / job_id
    for d in (
        job_dir,
        job_dir / "inbox",
        job_dir / "ack",
        job_dir / "applied",
        job_dir / "failed",
    ):
        d.mkdir(parents=True, exist_ok=True)
        try:
            d.chmod(MAILBOX_DIR_MODE)
        except Exception:
            pass

    base_rev = git_stash_create(repo_root)
    if initialize_shadow:
        _snapshot_tracked(repo_root, job_dir / "shadow")

    state_path = job_dir / "state.json"
    state: dict[str, Any]
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}
    state.update(
        {
            "job_id": job_id,
            "cluster": cluster_name,
            "cluster_root": cluster_root,
            "backend": backend,
            "workdir": workdir,
            "shadow_version": int(state.get("shadow_version", 0))
            + (1 if initialize_shadow else 0),
            "base_rev": base_rev,
            "updated_at": _now_iso(),
        }
    )
    _write_json_atomic(state_path, state)

    entries = _read_active(mailbox_root)
    key = (job_id, cluster_name, workdir)
    new_entry = {
        "job_id": job_id,
        "cluster": cluster_name,
        "cluster_root": cluster_root,
        "backend": backend,
        "host": host_key,
        "ssh_alias": ssh_alias,
        "workdir": workdir,
        "mode": "repl-sync",
        "status": "active",
        "last_heartbeat": _now_iso(),
        "started_at": _now_iso(),
    }

    replaced = False
    out_entries: list[dict[str, Any]] = []
    for entry in entries:
        entry_key = (
            str(entry.get("job_id", "")),
            str(entry.get("cluster", "")),
            str(entry.get("workdir", "")),
        )
        if entry_key == key:
            out_entries.append(new_entry)
            replaced = True
        else:
            out_entries.append(entry)
    if not replaced:
        out_entries.append(new_entry)
    _write_active(mailbox_root, out_entries)

    return mailbox_root


@dataclass
class UpdateSummary:
    job_id: str
    status: str
    detail: str


def publish_updates(
    *,
    local_mount: Path,
    repo_root: Path,
    sender_host: str,
    include_clusters: set[str] | None = None,
    exclude_clusters: set[str] | None = None,
) -> tuple[list[UpdateSummary], Path]:
    mailbox_root = mailbox_root_for(local_mount, "")
    logger.debug("MAILBOX | publish start")
    pruned = prune_stale_active_entries(mailbox_root=mailbox_root)
    if pruned > 0:
        logger.info(f"MAILBOX | pruned {pruned} stale active entries")
    entries = [
        e
        for e in _read_active(mailbox_root)
        if e.get("mode") == "repl-sync" and e.get("status", "active") == "active"
    ]
    if include_clusters:
        entries = [
            e for e in entries if str(e.get("cluster", "")).strip() in include_clusters
        ]
    if exclude_clusters:
        entries = [
            e
            for e in entries
            if str(e.get("cluster", "")).strip() not in exclude_clusters
        ]
    logger.debug(f"MAILBOX | active synced entries={len(entries)} at {mailbox_root}")

    if not entries:
        logger.info("MAILBOX | no active synced entries; nothing to send")
        return ([], mailbox_root)

    backends = sorted(
        {str(e.get("backend", "")).strip() for e in entries if e.get("backend")}
    )
    if len(backends) > 1:
        raise RuntimeError(
            "active synced REPLs use multiple backends; filter by cluster and retry: "
            + ", ".join(backends)
        )
    current_ref = git_stash_create(repo_root) or "HEAD"
    logger.debug(f"MAILBOX | current ref for update: {current_ref}")

    summaries: list[UpdateSummary] = []
    for entry in entries:
        job_id = str(entry.get("job_id", "")).strip()
        if not job_id:
            continue
        logger.debug(f"MAILBOX | preparing patch for job_id={job_id}")
        job_dir = mailbox_root / "jobs" / job_id
        jobs_dir = mailbox_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        try:
            jobs_dir.chmod(MAILBOX_DIR_MODE)
        except Exception:
            pass
        inbox = job_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        try:
            inbox.chmod(MAILBOX_DIR_MODE)
        except Exception:
            pass

        state_path = job_dir / "state.json"
        shadow_version = 0
        state: dict[str, Any] = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                shadow_version = int(state.get("shadow_version", 0))
            except Exception:
                state = {}
                shadow_version = 0

        base_ref = str(state.get("base_rev") or "HEAD").strip() or "HEAD"
        check = subprocess.run(
            ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            logger.warning(
                f"MAILBOX | invalid base_rev='{base_ref}' for job_id={job_id}, falling back to HEAD"
            )
            base_ref = "HEAD"

        diff_result = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                base_ref,
                "--",
                ".",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if diff_result.returncode != 0:
            raise RuntimeError(
                f"failed to build patch from {base_ref} for {job_id}: {diff_result.stderr.strip()}"
            )
        patch_text = diff_result.stdout
        logger.debug(
            f"MAILBOX | patch size for job_id={job_id}: {len(patch_text.encode('utf-8'))} bytes"
        )
        if not patch_text.strip():
            summaries.append(
                UpdateSummary(job_id=job_id, status="unchanged", detail="no changes")
            )
            continue

        mail_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        diff_path = inbox / f"{mail_id}.diff"
        meta_path = inbox / f"{mail_id}.meta.json"
        ready_path = inbox / f"{mail_id}.ready"

        _write_text_atomic(diff_path, patch_text)
        meta = {
            "mail_id": mail_id,
            "job_id": job_id,
            "created_at": _now_iso(),
            "sender_host": sender_host,
            "base_shadow_version": shadow_version,
            "new_shadow_version": shadow_version + 1,
            "base_rev": base_ref,
            "new_rev": current_ref,
        }
        _write_json_atomic(meta_path, meta)
        _write_text_atomic(ready_path, "ready\n")

        new_state = {
            "job_id": job_id,
            "cluster": entry.get("cluster"),
            "cluster_root": entry.get("cluster_root"),
            "backend": entry.get("backend"),
            "workdir": entry.get("workdir"),
            "shadow_version": shadow_version + 1,
            "base_rev": current_ref,
            "updated_at": _now_iso(),
        }
        _write_json_atomic(state_path, new_state)

        summaries.append(
            UpdateSummary(job_id=job_id, status="sent", detail=f"mail_id={mail_id}")
        )

    return summaries, mailbox_root


def pick_cluster_for_update(mailbox_root: Path) -> tuple[str, str]:
    entries = [
        e
        for e in _read_active(mailbox_root)
        if e.get("mode") == "repl-sync" and e.get("status", "active") == "active"
    ]
    clusters = sorted(
        {str(e.get("cluster", "")).strip() for e in entries if e.get("cluster")}
    )
    roots = sorted(
        {
            str(e.get("cluster_root", "")).strip()
            for e in entries
            if e.get("cluster_root")
        }
    )
    if not clusters or not roots:
        raise RuntimeError("no active synced REPL entries found in mailbox")
    if len(clusters) != 1 or len(roots) != 1:
        raise RuntimeError(
            "active synced REPL entries span multiple clusters/roots; narrow targets first"
        )
    return clusters[0], roots[0]
