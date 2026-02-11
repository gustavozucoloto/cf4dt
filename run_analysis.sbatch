#!/bin/bash
#SBATCH --job-name=cryobot_uq_only
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
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

echo "HOME_REAL:      $HOME_REAL"
echo "MAMBA_ROOT:     $MAMBA_ROOT_PREFIX"

# Hard fail early if env missing
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/fenicsx" ]; then
  echo "ERROR: fenicsx env not found at: $MAMBA_ROOT_PREFIX/envs/fenicsx"
  echo "Hint: check with: micromamba -r $MAMBA_ROOT_PREFIX env list"
  exit 1
fi

MM="micromamba -r $MAMBA_ROOT_PREFIX run -n fenicsx"

# Use SLURM submit directory as project root
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

# Threads for numpy/scipy/sklearn where applicable
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job started on: $(hostname)"
echo "Project dir:    $PWD"
echo "CPUs per task:  ${SLURM_CPUS_PER_TASK:-1}"
echo "MAMBA_ROOT:     $MAMBA_ROOT_PREFIX"
echo

echo "=== Sanity check: Python + MPI vendor ==="
srun --mpi=none -n 1 $MM python - <<'PY'
from mpi4py import MPI
print("Python OK")
print("MPI vendor:", MPI.get_vendor())
print("MPI library:", MPI.Get_library_version().splitlines()[0])
print("COMM_WORLD size:", MPI.COMM_WORLD.Get_size())
PY
echo

# ---------- Check required inputs ----------
TRUTH_DATA="${TRUTH_DATA:-data/artificial_Qlc_data.csv}"

if [ ! -f "$TRUTH_DATA" ]; then
  echo "ERROR: Missing data CSV: $TRUTH_DATA"
  exit 1
fi

for MODEL in powerlaw exponential logarithmic; do
  GP_FILE="data/gp_${MODEL}.joblib"
  POST_FILE="data/posterior_${MODEL}.npy"
  if [ ! -f "$GP_FILE" ]; then
    echo "ERROR: Missing GP emulator file: $GP_FILE"
    exit 1
  fi
  if [ ! -f "$POST_FILE" ]; then
    echo "ERROR: Missing posterior samples file: $POST_FILE"
    exit 1
  fi
done

# ---------- STEP 4: UQ only ----------
for MODEL in powerlaw exponential logarithmic; do
  GP_FILE="data/gp_${MODEL}.joblib"
  POST_FILE="data/posterior_${MODEL}.npy"
  OUT_PREFIX="outputs/uq_${MODEL}"

  echo "============================================================"
  echo "MODEL: ${MODEL}"
  echo "============================================================"

  echo "=== Step 4: UQ plots -> ${OUT_PREFIX}* ==="
  srun --mpi=none -n 1 $MM python scripts/run_uq.py \
    --model "$MODEL" \
    --gp "$GP_FILE" \
    --posterior "$POST_FILE" \
    --out-prefix "$OUT_PREFIX" \
    --data-csv "$TRUTH_DATA" \
    --use-data-ts
  echo
done

# ---------- STEP 5: Calibration quality analysis ----------
echo "=== Step 5: Calibration quality analysis ==="
srun --mpi=none -n 1 $MM python run_calibration_analysis.py \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
echo

# ---------- STEP 6: GP emulator k-fold verification ----------
echo "=== Step 6: GP emulator k-fold verification ==="
srun --mpi=none -n 1 $MM python scripts/verify_gp_emulator_kfold.py \
  --n-theta 10 \
  --subset 20 \
  --kfold 5 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
echo

echo "All done."
