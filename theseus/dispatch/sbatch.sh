#!/bin/bash
#
# Theseus SLURM sbatch wrapper - calls srun to run bootstrap on all nodes
#
# Placeholders (filled by slurm.py):
#   __SBATCH_DIRECTIVES__  - #SBATCH lines
#   __WORKDIR__            - working directory
#   __BOOTSTRAP_SCRIPT__   - path to bootstrap.sh

__SBATCH_DIRECTIVES__

set -euo pipefail

echo "[sbatch] starting job on $(hostname)"
echo "[sbatch] allocated nodes: $SLURM_JOB_NODELIST"
echo "[sbatch] tasks: $SLURM_NTASKS"

cd __WORKDIR__

# Run bootstrap.sh on all nodes via srun
# Each node will do its own setup (uv, juicefs, etc.) and run the command
srun --export=ALL bash __BOOTSTRAP_SCRIPT__
