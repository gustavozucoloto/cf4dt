#!/bin/bash
#SBATCH --job-name=cryobot_dt_uq_exponential
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=3GB
#SBATCH -A thes2143
#SBATCH -p c23ms

set -euo pipefail

module purge
module load GCC/14.3.0
# IMPORTANT: do NOT load OpenMPI here when using conda-forge dolfinx/mpi4py (often MPICH)
# module load OpenMPI/5.0.8

# ---------- micromamba locations ----------
export PATH="$HOME/.local/bin:$PATH"
HOME_REAL="$(readlink -f "$HOME")"
export MAMBA_ROOT_PREFIX="$HOME_REAL/.micromamba"

# Hard fail early if env missing
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/fenicsx" ]; then
  echo "ERROR: fenicsx env not found at: $MAMBA_ROOT_PREFIX/envs/fenicsx"
  echo "Hint: check with: micromamba -r $MAMBA_ROOT_PREFIX env list"
  exit 1
fi

MM="micromamba -r $MAMBA_ROOT_PREFIX run -n fenicsx"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

MODEL="exponential"
GP_FILE="data/gp_${MODEL}.joblib"
POST_FILE="data/posterior_${MODEL}.npy"
OUT_PREFIX="outputs/uq_${MODEL}"
DATA_CSV="data/artificial_Qlc_data.csv"

if [ ! -f "$GP_FILE" ]; then
  echo "ERROR: missing $GP_FILE; build GP emulator first."
  exit 1
fi

if [ ! -f "$POST_FILE" ]; then
  echo "ERROR: missing $POST_FILE; run calibration first."
  exit 1
fi

if [ ! -f "$DATA_CSV" ]; then
  echo "ERROR: missing $DATA_CSV; generate artificial data first."
  exit 1
fi

echo "=== UQ plots (${MODEL}) with artificial data overlay ==="
# Use --mpi=none so SLURM doesn't attempt PMI/PMIx setup
srun --mpi=none -n 1 $MM python scripts/run_uq.py \
  --model "$MODEL" \
  --gp "$GP_FILE" \
  --posterior "$POST_FILE" \
  --out-prefix "$OUT_PREFIX" \
  --data-csv "$DATA_CSV" \
  --use-data-ts

echo "Done."
