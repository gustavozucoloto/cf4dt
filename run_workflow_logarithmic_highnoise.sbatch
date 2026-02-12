#!/bin/bash
#SBATCH --job-name=cryobot_dt_logarithmic_highnoise
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=5GB
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
mkdir -p logs data outputs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

TRUTH_DATA="data/artificial_Qlc_data.csv"
MODEL="logarithmic"
JOB_TAG="${SLURM_JOB_ID:-$$}"
GP_FILE="data/gp_${MODEL}_highnoise_${JOB_TAG}.joblib"
POST_FILE="data/posterior_${MODEL}_highnoise_${JOB_TAG}.npy"
OUT_PREFIX="outputs/uq_${MODEL}_highnoise_${JOB_TAG}"

if [ ! -f "$TRUTH_DATA" ]; then
  echo "ERROR: missing $TRUTH_DATA; generate artificial data first."
  exit 1
fi

# White-noise settings (override with sbatch --export=ALL,...)
WN_LEVEL="${WN_LEVEL:-1e-4}"
WN_BOUNDS_MIN="${WN_BOUNDS_MIN:-1e-8}"
WN_BOUNDS_MAX="${WN_BOUNDS_MAX:-1e-1}"

echo "=== Step 2: Training GP emulator (${MODEL}, HIGH noise) -> ${GP_FILE} ==="
srun --mpi=none -n 1 $MM python scripts/build_gp_emulator.py \
  --model "$MODEL" \
  --data "$TRUTH_DATA" \
  --out "$GP_FILE" \
  --n-theta 40 \
  --subset 88 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --white-noise-level "$WN_LEVEL" \
  --white-noise-bounds "$WN_BOUNDS_MIN" "$WN_BOUNDS_MAX"

echo "=== Step 3: Bayesian calibration (emcee) -> ${POST_FILE} ==="
srun --mpi=none -n 1 $MM python scripts/run_bayesian_calibration.py \
  --model "$MODEL" \
  --prior truncnorm \
  --data "$TRUTH_DATA" \
  --gp "$GP_FILE" \
  --out "$POST_FILE" \
  --nwalkers 32 \
  --nsteps 6000 \
  --burn 1500 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "=== Step 4: UQ plots -> ${OUT_PREFIX}* ==="
srun --mpi=none -n 1 $MM python scripts/run_uq.py \
  --model "$MODEL" \
  --gp "$GP_FILE" \
  --posterior "$POST_FILE" \
  --out-prefix "$OUT_PREFIX"

echo "All done for ${MODEL} (HIGH-noise)."
