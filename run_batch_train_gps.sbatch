#!/bin/bash
#SBATCH --job-name=cryobot_train_gps
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=5GB
#SBATCH -A thes2143
#SBATCH -p c23ms

# Array mapping (6 tasks total):
#  0: powerlaw    rbf
#  1: powerlaw    matern_2p5
#  2: exponential rbf
#  3: exponential matern_2p5
#  4: logarithmic rbf
#  5: logarithmic matern_2p5

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

# Avoid cross-job FEniCS/FFCx JIT cache collisions
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

N_THETA="${N_THETA:-40}"
SUBSET="${SUBSET:-88}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) MODEL="powerlaw";    KERNEL="rbf";        KTAG="rbf";;
  1) MODEL="powerlaw";    KERNEL="matern_2p5"; KTAG="matern25";;
  2) MODEL="exponential"; KERNEL="rbf";        KTAG="rbf";;
  3) MODEL="exponential"; KERNEL="matern_2p5"; KTAG="matern25";;
  4) MODEL="logarithmic"; KERNEL="rbf";        KTAG="rbf";;
  5) MODEL="logarithmic"; KERNEL="matern_2p5"; KTAG="matern25";;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID must be 0..5"; exit 2;;
esac

GP_FILE="data/gp_${MODEL}_${KTAG}.joblib"

echo "=== Training GP: model=${MODEL}, kernel=${KERNEL} -> ${GP_FILE} ==="
srun --mpi=none -n 1 $MM python scripts/build_gp_emulator.py \
  --model "$MODEL" \
  --kernel "$KERNEL" \
  --data "$TRUTH_DATA" \
  --out "$GP_FILE" \
  --n-theta "$N_THETA" \
  --subset "$SUBSET" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "DONE: ${GP_FILE}"
