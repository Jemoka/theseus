#!/bin/bash
#
# Theseus bootstrap script - runs on each node (via srun for SLURM, directly for SSH)
#
# Placeholders (filled by slurm.py):
#   __WORKDIR__            - working directory to cd into
#   __MODULES__            - module load commands
#   __ENV_VARS__           - export statements
#   __JUICEFS_MOUNT__      - JuiceFS mount commands
#   __PAYLOAD_EXTRACT__    - base64 payload extraction (SLURM only)
#   __UV_SYNC__            - uv sync command with groups
#   __SETUP_COMMANDS__     - pre-run setup commands
#   __COMMAND__            - the actual command to run

set -euo pipefail

echo "[bootstrap] starting on $(hostname)"

# Track JuiceFS mount point for cleanup
JUICEFS_MOUNT_POINT=""

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

    # Get latest release tag
    JFS_LATEST_TAG=$(curl -s https://api.github.com/repos/juicedata/juicefs/releases/latest | grep 'tag_name' | cut -d '"' -f 4 | tr -d 'v')
    if [[ -z "$JFS_LATEST_TAG" ]]; then
        echo "[bootstrap] ERROR: failed to get juicefs latest release"
        exit 1
    fi

    # Download and extract
    local tmpdir=$(mktemp -d)
    cd "$tmpdir"
    wget -q "https://github.com/juicedata/juicefs/releases/download/v${JFS_LATEST_TAG}/juicefs-${JFS_LATEST_TAG}-linux-amd64.tar.gz"
    tar -zxf "juicefs-${JFS_LATEST_TAG}-linux-amd64.tar.gz"

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

# ============================================================================
# JuiceFS Mount (if configured)
# ============================================================================

__JUICEFS_MOUNT__

# ============================================================================
# Environment Setup
# ============================================================================

__MODULES__

__ENV_VARS__

# ============================================================================
# Working Directory & Payload Extraction
# ============================================================================

__PAYLOAD_EXTRACT__

cd __WORKDIR__
echo "[bootstrap] working directory: $(pwd)"

# ============================================================================
# Python Environment
# ============================================================================

# Sync dependencies with uv (reads pyproject.toml)
if [[ -f "pyproject.toml" ]]; then
    echo "[bootstrap] syncing dependencies with uv..."
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
__COMMAND__
