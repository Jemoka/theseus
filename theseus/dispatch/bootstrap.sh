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
: "${TF_GPU_ALLOCATOR:=cuda_malloc_async}"
: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.96}"
export TF_GPU_ALLOCATOR
export XLA_PYTHON_CLIENT_MEM_FRACTION

# ============================================================================
# Working Directory & Payload Extraction
# ============================================================================

__PAYLOAD_EXTRACT__

BOOTSTRAP_WORKDIR="__WORKDIR__"
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
__COMMAND__
