#!/bin/bash
#
# Theseus bootstrap script - runs on each node (via srun for SLURM, directly for SSH)
#
# There's a series of placeholders (filled by slurm.py)
# 

set -euo pipefail

# Source .bashrc if it exists
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

echo "[bootstrap] starting on $(hostname)"

# Track JuiceFS mount point for cleanup
JUICEFS_MOUNT_POINT=""
# Track work directory for cleanup
BOOTSTRAP_WORKDIR=""

cleanup() {
    local exit_code=$?
    echo "[bootstrap] cleaning up on $(hostname) (exit code: $exit_code)..."

    # Unmount JuiceFS gracefully if mounted
    if [[ -n "$JUICEFS_MOUNT_POINT" ]] && mountpoint -q "$JUICEFS_MOUNT_POINT" 2>/dev/null; then
        echo "[bootstrap] unmounting JuiceFS at $JUICEFS_MOUNT_POINT..."
        juicefs umount "$JUICEFS_MOUNT_POINT" 2>/dev/null || \
        juicefs umount --force "$JUICEFS_MOUNT_POINT" 2>/dev/null || \
        echo "[bootstrap] WARNING: failed to unmount JuiceFS"
    fi

    # Clean up work directory on success
    if [[ $exit_code -eq 0 ]] && [[ -n "$BOOTSTRAP_WORKDIR" ]] && [[ -d "$BOOTSTRAP_WORKDIR" ]]; then
        echo "[bootstrap] removing work directory: $BOOTSTRAP_WORKDIR"
        rm -rf "$BOOTSTRAP_WORKDIR"
    elif [[ $exit_code -ne 0 ]] && [[ -n "$BOOTSTRAP_WORKDIR" ]]; then
        echo "[bootstrap] preserving work directory for debugging: $BOOTSTRAP_WORKDIR"
    fi

    exit $exit_code
}

# Trap signals for cleanup on preemption/crash
trap cleanup EXIT TERM INT HUP

# ============================================================================
# UV Installation
# ============================================================================

ensure_uv() {
    if command -v uv &> /dev/null; then
        return 0
    fi

    echo "[bootstrap] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo "[bootstrap] ERROR: uv installation failed"
        exit 1
    fi
    echo "[bootstrap] uv installed successfully"
}

ensure_juicefs() {
    if command -v juicefs &> /dev/null; then
        return 0
    fi

    # Check if we have a local copy from previous install
    if [[ -x "$HOME/.local/bin/juicefs" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi

    echo "[bootstrap] juicefs not found, installing..."

    # Only supported on Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "[bootstrap] ERROR: juicefs auto-install only supported on Linux"
        exit 1
    fi

    # Detect architecture
    local arch=$(uname -m)
    local jfs_arch
    case "$arch" in
        x86_64)
            jfs_arch="amd64"
            ;;
        aarch64|arm64)
            jfs_arch="arm64"
            ;;
        *)
            echo "[bootstrap] ERROR: unsupported architecture: $arch"
            exit 1
            ;;
    esac
    echo "[bootstrap] detected architecture: $arch -> $jfs_arch"

    # Get latest release tag
    JFS_LATEST_TAG=$(curl -s https://api.github.com/repos/juicedata/juicefs/releases/latest | grep 'tag_name' | cut -d '"' -f 4 | tr -d 'v')
    if [[ -z "$JFS_LATEST_TAG" ]]; then
        echo "[bootstrap] ERROR: failed to get juicefs latest release"
        exit 1
    fi

    # Download and extract
    local tmpdir=$(mktemp -d)
    cd "$tmpdir"
    local download_url="https://github.com/juicedata/juicefs/releases/download/v${JFS_LATEST_TAG}/juicefs-${JFS_LATEST_TAG}-linux-${jfs_arch}.tar.gz"
    echo "[bootstrap] downloading juicefs v${JFS_LATEST_TAG} for ${jfs_arch}..."
    wget -q "$download_url"
    tar -zxf "juicefs-${JFS_LATEST_TAG}-linux-${jfs_arch}.tar.gz"

    # Install to ~/.local/bin
    mkdir -p "$HOME/.local/bin"
    mv juicefs "$HOME/.local/bin/"
    cd - > /dev/null
    rm -rf "$tmpdir"

    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v juicefs &> /dev/null; then
        echo "[bootstrap] ERROR: juicefs installation failed"
        exit 1
    fi
    echo "[bootstrap] juicefs installed successfully"
}

ensure_uv
ensure_juicefs

ROOT_PLACEHOLDER="__THESEUS_RUNTIME_ROOT__"

# Resolve runtime root placeholders in generated paths.
resolve_runtime_root_tokens() {
    local value="$1"
    if [[ "$value" == *"$ROOT_PLACEHOLDER"* ]]; then
        if [[ -z "${THESEUS_DISPATCH_ROOT_OVERRIDE:-}" ]]; then
            echo "[bootstrap] ERROR: this script requires --root PATH at runtime"
            exit 2
        fi
        value="${value//$ROOT_PLACEHOLDER/${THESEUS_DISPATCH_ROOT_OVERRIDE}}"
    fi
    printf '%s\n' "$value"
}

# Allow runtime path overrides when running standalone bootstrap scripts.
print_bootstrap_usage() {
    cat <<'EOF'
Usage: bootstrap.sh [--root PATH] [--work PATH] [--log PATH]

Options:
  --root PATH  Override cluster root path.
  --work PATH  Override cluster work path.
  --log PATH   Override cluster log path.
  -h, --help   Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            if [[ $# -lt 2 ]]; then
                echo "[bootstrap] ERROR: --root requires a value"
                exit 2
            fi
            export THESEUS_DISPATCH_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        --work)
            if [[ $# -lt 2 ]]; then
                echo "[bootstrap] ERROR: --work requires a value"
                exit 2
            fi
            export THESEUS_DISPATCH_WORK_OVERRIDE="$2"
            shift 2
            ;;
        --log)
            if [[ $# -lt 2 ]]; then
                echo "[bootstrap] ERROR: --log requires a value"
                exit 2
            fi
            export THESEUS_DISPATCH_LOG_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            print_bootstrap_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "[bootstrap] ERROR: unknown argument: $1"
            print_bootstrap_usage
            exit 2
            ;;
    esac
done

if [[ "${THESEUS_DISPATCH_REQUIRE_ROOT:-0}" == "1" ]] && [[ -z "${THESEUS_DISPATCH_ROOT_OVERRIDE:-}" ]]; then
    echo "[bootstrap] ERROR: root path is required; pass --root PATH"
    exit 2
fi

# ============================================================================
# JuiceFS Mount (if configured)
# ============================================================================

__JUICEFS_MOUNT__

# ============================================================================
# Environment Setup
# ============================================================================

__MODULES__

__ENV_VARS__

# JAX/XLA GPU allocator defaults for large-model runs.
# Keep overridable by honoring pre-set environment values.
: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.85}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

# ============================================================================
# Working Directory & Payload Extraction
# ============================================================================

BOOTSTRAP_WORKDIR="${THESEUS_DISPATCH_WORK_OVERRIDE:-__WORKDIR__}"
BOOTSTRAP_WORKDIR="$(resolve_runtime_root_tokens "$BOOTSTRAP_WORKDIR")"
__PAYLOAD_EXTRACT__

cd "$BOOTSTRAP_WORKDIR"
echo "[bootstrap] working directory: $(pwd)"

# ============================================================================
# Python Environment
# ============================================================================

# Sync dependencies with uv (reads pyproject.toml)
if [[ -f "pyproject.toml" ]]; then
    echo "[bootstrap] syncing dependencies with uv..."
    uv python install 3.11 --force # because otherwise it freezes?
    __UV_SYNC__
fi

# ============================================================================
# User Setup Commands
# ============================================================================

__SETUP_COMMANDS__

# ============================================================================
# Run Command
# ============================================================================

echo "[bootstrap] running command..."
MAIN_COMMAND="__COMMAND__"
BATCH_SIZE_CANDIDATES=(256 128 64 32 16 10 8 4 3 2 1)

should_search_batch_size() {
    if [[ -n "${THESEUS_DISPATCH_BATCH_SIZE:-}" ]]; then
        return 1
    fi

    if [[ ! -f "_bootstrap_dispatch.py" ]]; then
        return 1
    fi

    grep -Eq '^[[:space:]]*per_device_batch_size:[[:space:]]*-1([[:space:]]|$)' "_bootstrap_dispatch.py"
}

run_command_with_autobatch_search() {
    local candidates_csv
    candidates_csv="$(IFS=,; echo "${BATCH_SIZE_CANDIDATES[*]}")"

    export THESEUS_DISPATCH_MAIN_COMMAND="$MAIN_COMMAND"
    export THESEUS_DISPATCH_BATCH_CANDIDATES="$candidates_csv"

    uv run python - <<'PY'
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
from datetime import datetime

RESOURCE_MARKER = "RESOURCE_EXHAUSTED"
INITIAL_STABILITY_TIMEOUT = float(
    os.environ.get("THESEUS_DISPATCH_INITIAL_STABILITY_TIMEOUT", "420")
)
MIN_TIMEOUT_SECONDS = 5.0
MIN_STABLE_CHECK_SECONDS = float(
    os.environ.get("THESEUS_DISPATCH_MIN_STABLE_CHECK_SECONDS", "60")
)
PROBE_COOLDOWN_SECONDS = float(
    os.environ.get("THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS", "8")
)


def _put_event(events: mp.Queue, payload: tuple) -> None:
    try:
        events.put(payload)
    except Exception:
        pass


def _signal_child_group(child_pid: int | None, sig: int) -> None:
    if child_pid is None:
        return
    try:
        os.killpg(child_pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _cooldown_between_allocates() -> None:
    if PROBE_COOLDOWN_SECONDS <= 0:
        return
    print(
        "[bootstrap] AUTO_BATCH cooldown "
        f"{PROBE_COOLDOWN_SECONDS:.1f}s between allocation attempts"
    )
    time.sleep(PROBE_COOLDOWN_SECONDS)


def _run_command_worker(command: str, env: dict[str, str], events: mp.Queue) -> None:
    child: subprocess.Popen[str] | None = None

    def _forward_terminate(signum: int, _frame: object) -> None:
        nonlocal child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_terminate)
    signal.signal(signal.SIGINT, _forward_terminate)

    start = time.monotonic()
    saw_resource = False
    returncode = 1

    try:
        child = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            start_new_session=True,
        )
        _put_event(events, ("child_pid", child.pid))

        assert child.stdout is not None
        for line in child.stdout:
            if RESOURCE_MARKER in line:
                saw_resource = True
            _put_event(events, ("line", line))

        returncode = child.wait()
    except Exception as exc:
        _put_event(events, ("line", f"[bootstrap] AUTO_BATCH worker error: {exc}\n"))
    finally:
        elapsed = time.monotonic() - start
        _put_event(events, ("done", returncode, elapsed, saw_resource))


def _run_attempt(
    command: str,
    base_env: dict[str, str],
    batch_size: int,
    timeout_s: float | None,
    probe_mode: bool,
) -> dict[str, object]:
    env = dict(base_env)
    env["THESEUS_DISPATCH_BATCH_SIZE"] = str(batch_size)
    if probe_mode:
        env["THESEUS_DISPATCH_DISABLE_WANDB"] = "1"
    else:
        env.pop("THESEUS_DISPATCH_DISABLE_WANDB", None)

    events: mp.Queue = mp.Queue()
    worker = mp.Process(target=_run_command_worker, args=(command, env, events))
    worker.start()

    started = time.monotonic()
    saw_resource = False
    child_pid: int | None = None
    done_payload: tuple[int, float, bool] | None = None
    worker_dead_since: float | None = None

    try:
        while True:
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                event = None

            if event is not None:
                kind = event[0]
                if kind == "line":
                    line = str(event[1])
                    if RESOURCE_MARKER in line:
                        saw_resource = True
                    sys.stdout.write(line)
                    sys.stdout.flush()
                elif kind == "child_pid":
                    child_pid = int(event[1])
                elif kind == "done":
                    done_payload = (
                        int(event[1]),
                        float(event[2]),
                        bool(event[3]),
                    )
                    saw_resource = saw_resource or done_payload[2]

            elapsed = time.monotonic() - started
            if timeout_s is not None and elapsed >= timeout_s and worker.is_alive():
                print(
                    "[bootstrap] AUTO_BATCH probe timeout; "
                    f"terminating batch_size={batch_size}"
                )
                _signal_child_group(child_pid, signal.SIGTERM)
                worker.terminate()
                worker.join(timeout=10)
                _signal_child_group(child_pid, signal.SIGKILL)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
                return {
                    "kind": "stable",
                    "elapsed": elapsed,
                    "returncode": None,
                    "timed_out": True,
                }

            if done_payload is not None and not worker.is_alive():
                returncode, done_elapsed, worker_saw_resource = done_payload
                saw_resource = saw_resource or worker_saw_resource
                if returncode == 0:
                    return {
                        "kind": "completed",
                        "elapsed": done_elapsed,
                        "returncode": returncode,
                        "timed_out": False,
                    }
                if saw_resource:
                    return {
                        "kind": "resource",
                        "elapsed": done_elapsed,
                        "returncode": returncode,
                        "timed_out": False,
                    }
                return {
                    "kind": "error",
                    "elapsed": done_elapsed,
                    "returncode": returncode,
                    "timed_out": False,
                }

            if done_payload is None and not worker.is_alive():
                if worker_dead_since is None:
                    worker_dead_since = time.monotonic()
                    continue
                if (time.monotonic() - worker_dead_since) < 2.0:
                    continue
                _signal_child_group(child_pid, signal.SIGKILL)
                return {
                    "kind": "error",
                    "elapsed": elapsed,
                    "returncode": worker.exitcode,
                    "timed_out": False,
                }
            else:
                worker_dead_since = None
    finally:
        if worker.is_alive():
            _signal_child_group(child_pid, signal.SIGTERM)
            worker.terminate()
            worker.join(timeout=5)
            _signal_child_group(child_pid, signal.SIGKILL)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        events.close()
        events.join_thread()


def _select_batch_size(
    command: str,
    base_env: dict[str, str],
    candidates: list[int],
) -> tuple[int | None, int]:
    upper_idx: int | None = None
    upper_crash_time: float | None = None
    lower_idx: int | None = None

    for idx, batch_size in enumerate(candidates):
        timeout_s = (
            max(MIN_TIMEOUT_SECONDS, MIN_STABLE_CHECK_SECONDS, (upper_crash_time or 0.0) * 1.5)
            if upper_crash_time is not None
            else INITIAL_STABILITY_TIMEOUT
        )
        print(
            f"[bootstrap] AUTO_BATCH probing batch_size={batch_size} "
            f"(timeout={timeout_s:.1f}s)"
        )
        attempt = _run_attempt(
            command=command,
            base_env=base_env,
            batch_size=batch_size,
            timeout_s=timeout_s,
            probe_mode=True,
        )
        _cooldown_between_allocates()

        kind = str(attempt["kind"])
        elapsed = float(attempt["elapsed"])
        if kind == "resource":
            upper_idx = idx
            upper_crash_time = max(upper_crash_time or 0.0, elapsed, 0.1)
            print(
                "[bootstrap] AUTO_BATCH RESOURCE_EXHAUSTED at "
                f"batch_size={batch_size} after {elapsed:.1f}s"
            )
            continue

        if kind in {"stable", "completed"}:
            lower_idx = idx
            if bool(attempt["timed_out"]):
                print(
                    "[bootstrap] AUTO_BATCH stable lower bound found at "
                    f"batch_size={batch_size} (no crash for {elapsed:.1f}s)"
                )
            else:
                print(
                    "[bootstrap] AUTO_BATCH batch_size="
                    f"{batch_size} exited successfully during probe"
                )
            break

        print(
            "[bootstrap] ERROR: probe failed without RESOURCE_EXHAUSTED "
            f"for batch_size={batch_size} (returncode={attempt['returncode']})"
        )
        return None, int(attempt["returncode"] or 1)

    if lower_idx is None:
        print("[bootstrap] ERROR: no viable per-device batch size found in candidate list")
        return None, 1

    if upper_idx is None:
        selected = candidates[lower_idx]
        print(
            "[bootstrap] AUTO_BATCH selected batch_size="
            f"{selected} (no crashing upper bound observed)"
        )
        return selected, 0

    while lower_idx - upper_idx > 1:
        mid_idx = (upper_idx + lower_idx) // 2
        batch_size = candidates[mid_idx]
        timeout_s = max(
            MIN_TIMEOUT_SECONDS,
            MIN_STABLE_CHECK_SECONDS,
            (upper_crash_time or 0.0) * 1.5,
        )

        print(
            "[bootstrap] AUTO_BATCH binary probe batch_size="
            f"{batch_size} (timeout={timeout_s:.1f}s)"
        )
        attempt = _run_attempt(
            command=command,
            base_env=base_env,
            batch_size=batch_size,
            timeout_s=timeout_s,
            probe_mode=True,
        )
        _cooldown_between_allocates()

        kind = str(attempt["kind"])
        elapsed = float(attempt["elapsed"])
        if kind == "resource":
            upper_idx = mid_idx
            upper_crash_time = max(upper_crash_time or 0.0, elapsed, 0.1)
            print(
                "[bootstrap] AUTO_BATCH binary result: "
                f"batch_size={batch_size} crashes (RESOURCE_EXHAUSTED)"
            )
            continue

        if kind in {"stable", "completed"}:
            lower_idx = mid_idx
            print(
                "[bootstrap] AUTO_BATCH binary result: "
                f"batch_size={batch_size} is stable"
            )
            continue

        print(
            "[bootstrap] ERROR: binary probe failed without RESOURCE_EXHAUSTED "
            f"for batch_size={batch_size} (returncode={attempt['returncode']})"
        )
        return None, int(attempt["returncode"] or 1)

    selected = candidates[lower_idx]
    print(f"[bootstrap] AUTO_BATCH selected final batch_size={selected}")
    return selected, 0


def main() -> int:
    command = os.environ.get("THESEUS_DISPATCH_MAIN_COMMAND", "").strip()
    if not command:
        print("[bootstrap] ERROR: missing THESEUS_DISPATCH_MAIN_COMMAND")
        return 1

    candidates_raw = os.environ.get("THESEUS_DISPATCH_BATCH_CANDIDATES", "")
    candidates = [int(x.strip()) for x in candidates_raw.split(",") if x.strip()]
    if not candidates:
        print("[bootstrap] ERROR: no batch-size candidates provided")
        return 1

    start_time = datetime.now().isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    run_id = f"{timestamp}_{slurm_job_id}" if slurm_job_id else timestamp

    base_env = dict(os.environ)
    base_env["THESEUS_DISPATCH_RUN_ID"] = run_id
    base_env["THESEUS_DISPATCH_START_TIME"] = start_time

    print(
        "[bootstrap] AUTO_BATCH enabled; searching per-device batch size "
        f"across {candidates}"
    )
    selected, rc = _select_batch_size(command, base_env, candidates)
    if selected is None:
        return rc

    print(
        "[bootstrap] AUTO_BATCH launching real run with "
        f"training.per_device_batch_size={selected}"
    )
    _cooldown_between_allocates()
    final = _run_attempt(
        command=command,
        base_env=base_env,
        batch_size=selected,
        timeout_s=None,
        probe_mode=False,
    )

    if final["kind"] == "completed":
        return 0

    if final["kind"] == "resource":
        print(
            "[bootstrap] ERROR: real run still failed with RESOURCE_EXHAUSTED "
            f"at batch_size={selected}"
        )
    else:
        print(
            "[bootstrap] ERROR: real run failed with returncode="
            f"{final['returncode']}"
        )
    return int(final["returncode"] or 1)


if __name__ == "__main__":
    # `spawn` cannot re-import a heredoc main module (`<stdin>`), so prefer
    # `fork` when available in this bootstrap context.
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    else:
        mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
PY
}

if should_search_batch_size; then
    run_command_with_autobatch_search
else
    bash -lc "$MAIN_COMMAND"
fi
