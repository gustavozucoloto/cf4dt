#!/bin/bash
#SBATCH --job-name=cryobot_calib_uq
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=5GB
#SBATCH -A thes2143
#SBATCH -p c23ms

# Array mapping (12 tasks total):
#  task_id = level*6 + base, where level=0 quick, level=1 medium
#  base in 0..5 uses the same (model,kernel) mapping as run_batch_train_gps.sbatch

set -euo pipefail

module purge
module load GCC/14.3.0
# IMPORTANT: do NOT load OpenMPI here when using conda-forge dolfinx/mpi4py (often MPICH)
# module load OpenMPI/5.0.8

# ---------- micromamba locations ----------
export PATH="$HOME/.local/bin:$PATH"
HOME_REAL="$(readlink -f "$HOME")"
export MAMBA_ROOT_PREFIX="$HOME_REAL/.micromamba"

if [ ! -d "$MAMBA_ROOT_PREFIX/envs/fenicsx" ]; then
  echo "ERROR: fenicsx env not found at: $MAMBA_ROOT_PREFIX/envs/fenicsx"
  exit 1
fi

MM="micromamba -r $MAMBA_ROOT_PREFIX run -n fenicsx"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# If these leak in from an interactive allocation, srun can error out.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# Avoid cross-job cache collisions
TAG="${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export XDG_CACHE_HOME="$SLURM_TMPDIR/xdg_cache_$TAG"
  export MPLCONFIGDIR="$SLURM_TMPDIR/mplconfig_$TAG"
  export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/pycache_$TAG"
else
  export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache/xdg_cache_$TAG"
  export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/mplconfig_$TAG"
  export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/.cache/pycache_$TAG"
fi
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

TRUTH_DATA="data/artificial_Qlc_data.csv"
if [ ! -f "$TRUTH_DATA" ]; then
  echo "ERROR: missing $TRUTH_DATA; generate artificial data first."
  exit 1
fi

TID="${SLURM_ARRAY_TASK_ID:-0}"
LEVEL=$(( TID / 6 ))
BASE=$(( TID % 6 ))

if [ "$LEVEL" -eq 0 ]; then
  CALIB_KIND="quick"
  NWALKERS="${NWALKERS_QUICK:-8}"
  NSTEPS="${NSTEPS_QUICK:-500}"
  BURN="${BURN_QUICK:-100}"
  THIN="${THIN_QUICK:-5}"
  NPOST="${NPOST_QUICK:-100}"
elif [ "$LEVEL" -eq 1 ]; then
  CALIB_KIND="medium"
  NWALKERS="${NWALKERS_MEDIUM:-32}"
  NSTEPS="${NSTEPS_MEDIUM:-1000}"
  BURN="${BURN_MEDIUM:-250}"
  THIN="${THIN_MEDIUM:-5}"
  NPOST="${NPOST_MEDIUM:-200}"
else
  echo "ERROR: SLURM_ARRAY_TASK_ID must be 0..11"; exit 2
fi

case "$BASE" in
  0) MODEL="powerlaw";    KERNEL="rbf";        KTAG="rbf";;
  1) MODEL="powerlaw";    KERNEL="matern_2p5"; KTAG="matern25";;
  2) MODEL="exponential"; KERNEL="rbf";        KTAG="rbf";;
  3) MODEL="exponential"; KERNEL="matern_2p5"; KTAG="matern25";;
  4) MODEL="logarithmic"; KERNEL="rbf";        KTAG="rbf";;
  5) MODEL="logarithmic"; KERNEL="matern_2p5"; KTAG="matern25";;
  *) echo "ERROR: base index must be 0..5"; exit 2;;
esac

GP_FILE="data/gp_${MODEL}_${KTAG}.joblib"
POST_FILE="data/posterior_${MODEL}_${KTAG}_${CALIB_KIND}.npy"
OUT_PREFIX="outputs/uq_${MODEL}_${KTAG}_${CALIB_KIND}"

if [ ! -f "$GP_FILE" ]; then
  echo "ERROR: missing ${GP_FILE}. Did GP training finish?"
  exit 1
fi

echo "=== Calibration+UQ: model=${MODEL}, kernel=${KERNEL}, kind=${CALIB_KIND} ==="
echo "GP:   ${GP_FILE}"
echo "POST: ${POST_FILE}"

srun --mpi=none -n 1 $MM python scripts/run_bayesian_calibration.py \
  --model "$MODEL" \
  --prior truncnorm \
  --data "$TRUTH_DATA" \
  --gp "$GP_FILE" \
  --out "$POST_FILE" \
  --nwalkers "$NWALKERS" \
  --nsteps "$NSTEPS" \
  --burn "$BURN" \
  --thin "$THIN" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

srun --mpi=none -n 1 $MM python scripts/run_uq.py \
  --model "$MODEL" \
  --gp "$GP_FILE" \
  --posterior "$POST_FILE" \
  --out-prefix "$OUT_PREFIX" \
  --n-post "$NPOST"

echo "DONE: ${MODEL} ${KTAG} ${CALIB_KIND}"
