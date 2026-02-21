from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MAILBOX_FILE_MODE = 0o777
MAILBOX_DIR_MODE = 0o777


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(MAILBOX_DIR_MODE)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
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


def _deactivate_active(
    mailbox_root: Path, job_id: str, workdir: str, reason: str
) -> None:
    active_path = mailbox_root / ".active"
    if not active_path.exists():
        return
    try:
        raw_lines = active_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    changed = False
    out = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if not isinstance(entry, dict):
            out.append(line)
            continue
        same_job = str(entry.get("job_id", "")).strip() == job_id
        same_workdir = str(entry.get("workdir", "")).strip() == workdir
        if same_job and same_workdir and entry.get("status", "active") == "active":
            changed = True
            continue
        out.append(json.dumps(entry, sort_keys=True))
    if changed:
        _write_text_atomic(active_path, "\n".join(out) + "\n")


def _build_logger(workdir: Path) -> Callable[[str], None]:
    log_path = workdir / ".theseus_repl_sidecar.log"

    def _log(msg: str) -> None:
        line = f"{_now_iso()} {msg}"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        print(f"[repl-sidecar] {msg}")

    return _log


def _dir_is_writable(path: Path) -> bool:
    """Best-effort check that directory is writable for this process."""
    if not path.exists() or not path.is_dir():
        return False
    try:
        probe = path / f".theseus_write_probe_{os.getpid()}"
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok\n")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


_STOP = False


def _handle_stop(signum, _frame) -> None:  # type: ignore[no-untyped-def]
    global _STOP
    _STOP = True


for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _handle_stop)


def _job_id() -> str:
    return (
        os.environ.get("THESEUS_REPL_MAILBOX_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or str(os.getpid())
    )


def main() -> int:
    # Keep mailbox artifacts world-readable/writable for cross-host users.
    os.umask(0)

    root_env = os.environ.get("THESEUS_ROOT", "").strip()
    if not root_env:
        print("[repl-sidecar] THESEUS_ROOT is not set; exiting")
        return 2
    root = Path(root_env).expanduser()

    job_id = _job_id().strip()
    if not job_id:
        print("[repl-sidecar] job id is empty; exiting")
        return 2

    workdir = Path(os.environ.get("THESEUS_REPL_WORKDIR", os.getcwd())).resolve()
    log = _build_logger(workdir)
    mailbox_root = root / "mailbox"
    jobs_dir = mailbox_root / "jobs"
    job_dir = jobs_dir / job_id
    inbox = job_dir / "inbox"
    ack_dir = job_dir / "ack"
    applied_dir = job_dir / "applied"
    failed_dir = job_dir / "failed"
    state_dir = job_dir / "state"

    for p in (
        mailbox_root,
        jobs_dir,
        job_dir,
        inbox,
        ack_dir,
        applied_dir,
        failed_dir,
        state_dir,
    ):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Shared filesystems can deny mkdir/chmod on parent-owned paths while
            # still allowing writes to existing subdirectories.
            if _dir_is_writable(p):
                log(f"mkdir denied for existing writable path {p}; continuing")
            else:
                log(f"mailbox path not writable ({p}); sync apply disabled")
                _deactivate_active(
                    mailbox_root, job_id, str(workdir), "mailbox-permission-denied"
                )
                return 0
        try:
            p.chmod(MAILBOX_DIR_MODE)
        except PermissionError:
            # Non-fatal: ownership may differ across hosts even when mode is already usable.
            log(f"chmod denied for {p}; continuing")

    log(f"started job_id={job_id} inbox={inbox}")

    poll_interval = float(os.environ.get("THESEUS_REPL_MAILBOX_POLL_INTERVAL", "1.0"))
    poll_interval = max(0.5, poll_interval)
    parent_pid = os.getppid()
    notebook_pid_raw = os.environ.get("THESEUS_REPL_NOTEBOOK_PID", "").strip()
    notebook_pid = int(notebook_pid_raw) if notebook_pid_raw.isdigit() else None
    exit_reason = "stopped"
    while not _STOP:
        # Exit when the REPL parent process is gone to avoid stray sidecars.
        current_ppid = os.getppid()
        if current_ppid == 1 or current_ppid != parent_pid:
            log(
                f"parent exited (initial={parent_pid}, current={current_ppid}); stopping"
            )
            exit_reason = "parent-exited"
            break
        if notebook_pid is not None:
            try:
                os.kill(notebook_pid, 0)
            except ProcessLookupError:
                log(f"notebook pid exited ({notebook_pid}); stopping")
                exit_reason = "notebook-exited"
                break
            except PermissionError:
                # Don't fail if process exists but isn't signalable by this user.
                pass

        _write_json_atomic(
            state_dir / "sidecar.json",
            {
                "job_id": job_id,
                "pid": os.getpid(),
                "workdir": str(workdir),
                "hostname": os.uname().nodename,
                "last_seen": _now_iso(),
            },
        )

        claimed = None
        for ready in sorted(inbox.glob("*.ready")):
            mail_id = ready.name[: -len(".ready")]
            processing = inbox / f"{mail_id}.processing.{os.getpid()}"
            try:
                ready.replace(processing)
                claimed = (mail_id, processing)
                break
            except FileNotFoundError:
                continue
            except OSError:
                continue

        if claimed is None:
            time.sleep(poll_interval)
            continue

        mail_id, processing = claimed
        diff_path = inbox / f"{mail_id}.diff"
        meta_path = inbox / f"{mail_id}.meta.json"

        try:
            patch_text = diff_path.read_text(encoding="utf-8")
        except Exception as exc:
            log(f"failed to read patch mail_id={mail_id}: {exc}")
            _write_json_atomic(
                failed_dir / f"{mail_id}.fail.json",
                {
                    "job_id": job_id,
                    "mail_id": mail_id,
                    "status": "failed",
                    "error": f"read diff failed: {exc}",
                    "timestamp": _now_iso(),
                },
            )
            processing.unlink(missing_ok=True)
            continue

        result = subprocess.run(
            ["patch", "-p1", "--forward", "--batch"],
            cwd=str(workdir),
            input=patch_text,
            text=True,
            capture_output=True,
        )

        payload = {
            "job_id": job_id,
            "mail_id": mail_id,
            "timestamp": _now_iso(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        if result.returncode == 0:
            log(f"applied mail_id={mail_id}")
            _write_json_atomic(ack_dir / f"{mail_id}.ack.json", payload)
            try:
                diff_path.replace(applied_dir / f"{mail_id}.diff")
            except OSError:
                pass
            try:
                meta_path.replace(applied_dir / f"{mail_id}.meta.json")
            except OSError:
                pass
        else:
            log(f"failed mail_id={mail_id} rc={result.returncode}")
            _write_json_atomic(failed_dir / f"{mail_id}.fail.json", payload)
            try:
                diff_path.replace(failed_dir / f"{mail_id}.diff")
            except OSError:
                pass
            try:
                meta_path.replace(failed_dir / f"{mail_id}.meta.json")
            except OSError:
                pass

        processing.unlink(missing_ok=True)

    _deactivate_active(mailbox_root, job_id, str(workdir), exit_reason)
    log(f"exiting reason={exit_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
