#!/bin/bash
__SBATCH_DIRECTIVES__

set -euo pipefail

echo "[sbatch] starting job on $(hostname)"
echo "[sbatch] allocated nodes: $SLURM_JOB_NODELIST"
echo "[sbatch] tasks: $SLURM_NTASKS"

# How long srun waits for the step to finish after shutdown starts.
: "${THESEUS_SRUN_WAIT_SECONDS:=120}"

# Run bootstrap.sh on all nodes via srun
# bootstrap.sh handles: extract payload, cd to workdir, setup env, run command
srun --wait="$THESEUS_SRUN_WAIT_SECONDS" bash -l __BOOTSTRAP_SCRIPT__
rc=$?
echo "[sbatch] srun exited rc=$rc"
exit $rc
