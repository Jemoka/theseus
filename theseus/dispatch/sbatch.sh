#!/bin/bash
__SBATCH_DIRECTIVES__

set -euo pipefail

echo "[sbatch] starting job on $(hostname)"
echo "[sbatch] allocated nodes: $SLURM_JOB_NODELIST"
echo "[sbatch] tasks: $SLURM_NTASKS"

# Run bootstrap.sh on all nodes via srun
# bootstrap.sh handles: extract payload, cd to workdir, setup env, run command
srun --export=ALL bash __BOOTSTRAP_SCRIPT__
