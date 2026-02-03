#!/bin/bash
#SBATCH --job-name=cryobot_dt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
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

# Helper: run inside micromamba env with explicit root
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
# Use --mpi=none so SLURM doesn't attempt PMI/PMIx setup (prevents the PMI1 error)
srun --mpi=none -n 1 $MM python - <<'PY'
from mpi4py import MPI
print("Python OK")
print("MPI vendor:", MPI.get_vendor())
print("MPI library:", MPI.Get_library_version().splitlines()[0])
print("COMM_WORLD size:", MPI.COMM_WORLD.Get_size())
PY
echo

# ---------- STEP 1: generate synthetic truth data ----------
TRUTH_DATA="data/artificial_Qlc_data.csv"

echo "=== Step 1: Generating artificial data -> ${TRUTH_DATA} ==="
srun --mpi=none -n 1 $MM python scripts/generate_artificial_data.py \
  --out "$TRUTH_DATA"
echo

# ---------- STEPS 2-4: loop over both parameterized models ----------
for MODEL in powerlaw exponential; do
  GP_FILE="data/gp_${MODEL}.joblib"
  POST_FILE="data/posterior_${MODEL}.npy"
  OUT_PREFIX="outputs/uq_${MODEL}"

  echo "============================================================"
  echo "MODEL: ${MODEL}"
  echo "============================================================"

  echo "=== Step 2: Training GP emulator -> ${GP_FILE} ==="
  srun --mpi=none -n 1 $MM python scripts/build_gp_emulator.py \
    --model "$MODEL" \
    --data "$TRUTH_DATA" \
    --out "$GP_FILE"
  echo

  echo "=== Step 3: Bayesian calibration (emcee) -> ${POST_FILE} ==="
  srun --mpi=none -n 1 $MM python scripts/run_bayesian_calibration.py \
    --model "$MODEL" \
    --data "$TRUTH_DATA" \
    --gp "$GP_FILE" \
    --out "$POST_FILE"
  echo

  echo "=== Step 4: UQ plots -> ${OUT_PREFIX}* ==="
  srun --mpi=none -n 1 $MM python scripts/run_uq.py \
    --model "$MODEL" \
    --gp "$GP_FILE" \
    --posterior "$POST_FILE" \
    --out-prefix "$OUT_PREFIX"
  echo
done

echo "All done."
