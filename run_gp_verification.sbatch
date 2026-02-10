#!/usr/bin/env bash
#SBATCH --job-name=gp_verify
#SBATCH --output=logs/gp_verify_%j.out
#SBATCH --error=logs/gp_verify_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
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

micromamba run -p ~/.micromamba/envs/fenicsx \
  python run_calibration_analysis.py --n-jobs=16

micromamba run -p ~/.micromamba/envs/fenicsx \
  python scripts/verify_gp_emulator_kfold.py --n-theta 10 --subset 20 --kfold 5 --n-jobs 16