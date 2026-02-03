#!/bin/bash
#
# Theseus bootstrap script for SLURM jobs.
# Users can edit this file to customize job setup.
#
# Placeholders (filled by slurm.py):
#   __SBATCH_DIRECTIVES__  - #SBATCH lines
#   __MODULES__            - module load commands
#   __ENV_VARS__           - export statements
#   __SETUP_COMMANDS__     - pre-run setup commands
#   __PAYLOAD__            - base64-encoded code tarball (optional)
#   __PAYLOAD_DIR__        - extraction directory
#   __COMMAND__            - the actual command to run

__SBATCH_DIRECTIVES__

set -euo pipefail

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

ensure_uv

# ============================================================================
# Environment Setup
# ============================================================================

__MODULES__

__ENV_VARS__

# ============================================================================
# Payload Extraction
# ============================================================================

__PAYLOAD_EXTRACT__

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
