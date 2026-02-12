#!/bin/bash
#SBATCH --job-name=calib_analysis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH -A thes2143
#SBATCH -p c23ms

set -euo pipefail

module purge
module load GCC/14.3.0

export PATH="$HOME/.local/bin:$PATH"
HOME_REAL="$(readlink -f "$HOME")"
export MAMBA_ROOT_PREFIX="$HOME_REAL/.micromamba"
MM="micromamba -r $MAMBA_ROOT_PREFIX run -n fenicsx"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/analysis_out

# avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --mpi=none -n 1 $MM python scripts/analyze_posteriors.py