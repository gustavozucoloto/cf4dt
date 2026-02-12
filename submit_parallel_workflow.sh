#!/bin/bash
set -euo pipefail

# Submits the 2-stage workflow:
#  1) Train 6 GPs in parallel (3 models × 2 kernels)
#  2) After training succeeds, run 12 calibration+UQ tasks (quick+medium)

TRAIN_PARALLELISM="${TRAIN_PARALLELISM:-6}"
CALIB_PARALLELISM="${CALIB_PARALLELISM:-12}"

EXPORT_SPEC="NONE"

# Optional tunables passed into the job environment (avoid exporting SLURM_* vars from interactive allocations)
if [ -n "${N_THETA:-}" ]; then EXPORT_SPEC+=" ,N_THETA=${N_THETA}"; fi
if [ -n "${SUBSET:-}" ]; then EXPORT_SPEC+=" ,SUBSET=${SUBSET}"; fi

if [ -n "${NWALKERS_QUICK:-}" ]; then EXPORT_SPEC+=",NWALKERS_QUICK=${NWALKERS_QUICK}"; fi
if [ -n "${NSTEPS_QUICK:-}" ]; then EXPORT_SPEC+=",NSTEPS_QUICK=${NSTEPS_QUICK}"; fi
if [ -n "${BURN_QUICK:-}" ]; then EXPORT_SPEC+=",BURN_QUICK=${BURN_QUICK}"; fi
if [ -n "${THIN_QUICK:-}" ]; then EXPORT_SPEC+=",THIN_QUICK=${THIN_QUICK}"; fi
if [ -n "${NPOST_QUICK:-}" ]; then EXPORT_SPEC+=",NPOST_QUICK=${NPOST_QUICK}"; fi

if [ -n "${NWALKERS_MEDIUM:-}" ]; then EXPORT_SPEC+=",NWALKERS_MEDIUM=${NWALKERS_MEDIUM}"; fi
if [ -n "${NSTEPS_MEDIUM:-}" ]; then EXPORT_SPEC+=",NSTEPS_MEDIUM=${NSTEPS_MEDIUM}"; fi
if [ -n "${BURN_MEDIUM:-}" ]; then EXPORT_SPEC+=",BURN_MEDIUM=${BURN_MEDIUM}"; fi
if [ -n "${THIN_MEDIUM:-}" ]; then EXPORT_SPEC+=",THIN_MEDIUM=${THIN_MEDIUM}"; fi
if [ -n "${NPOST_MEDIUM:-}" ]; then EXPORT_SPEC+=",NPOST_MEDIUM=${NPOST_MEDIUM}"; fi

echo "Submitting GP training array (6 tasks, max ${TRAIN_PARALLELISM} in flight)..."
TRAIN_JOBID=$(sbatch --export="${EXPORT_SPEC}" --parsable --array=0-5%"${TRAIN_PARALLELISM}" run_batch_train_gps.sbatch)
echo "TRAIN_JOBID=${TRAIN_JOBID}"

echo "Submitting calibration+UQ array (12 tasks, max ${CALIB_PARALLELISM} in flight), dependent on training..."
CALIB_JOBID=$(sbatch --export="${EXPORT_SPEC}" --parsable --dependency=afterok:"${TRAIN_JOBID}" --array=0-11%"${CALIB_PARALLELISM}" run_batch_calib_uq.sbatch)
echo "CALIB_JOBID=${CALIB_JOBID}"

echo "Done. Monitor with: squeue -u ${USER}"
