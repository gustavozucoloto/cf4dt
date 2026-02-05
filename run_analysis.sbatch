#!/bin/bash
#SBATCH --job-name=cryobot_analysis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4GB
#SBATCH -A thes2143
#SBATCH -p c23ms

set -euo pipefail

module purge
module load GCC/14.3.0

# ---------- micromamba configuration ----------
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
mkdir -p logs outputs

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

echo "=== Running Calibration Quality Evaluation ==="
echo "Input data:       data/artificial_Qlc_data.csv"
echo "GP models:        data/gp_powerlaw.joblib, data/gp_exponential.joblib"
echo "Posterior files:  data/posterior_powerlaw.npy, data/posterior_exponential.npy"
echo "Output directory: outputs/"
echo "Parallel CPUs:    ${SLURM_CPUS_PER_TASK:-1}"
echo

srun --mpi=none -n 1 $MM python run_calibration_analysis.py --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

echo
echo "=== Analysis Complete ==="
echo "Generated outputs:"
ls -lh outputs/*.png 2>/dev/null || echo "No PNG outputs found"
ls -lh outputs/*.csv 2>/dev/null || echo "No CSV outputs found"
