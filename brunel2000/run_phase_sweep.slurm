#!/bin/bash
#SBATCH --job-name=brunel_phase_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=30

set -euo pipefail

mkdir -p logs

# Optional overrides from sbatch --export=ALL,...
OUT_DIR=${OUT_DIR:-outputs/brunel2000_phase_sweep}
CPUS=${SLURM_CPUS_PER_TASK:-30}
MEM_GB=${MEM_GB:-120}

# Keep BLAS/OpenMP single-threaded per worker process.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

NUM_WORKERS=${CPUS//2}  # Each worker uses 2 threads for BLAS/OpenMP, so we can run CPUS/2 workers in parallel.
N_TRIALS=${N_TRIALS:-1}
PAPER_PROFILE=${PAPER_PROFILE:-both_key}
SAVE_TRIAL_PLOTS=${SAVE_TRIAL_PLOTS:-true}
RASTER_DPI=${RASTER_DPI:-90}
TRACE_FORMAT=${TRACE_FORMAT:-pdf}

echo "Running Brunel phase sweep on $(hostname)"
echo "Output dir: ${OUT_DIR}"
echo "CPUs: ${CPUS}, MEM_GB: ${MEM_GB}"
echo "Workers: ${NUM_WORKERS}, Trials: ${N_TRIALS}, Paper profile: ${PAPER_PROFILE}"
echo "Save trial plots: ${SAVE_TRIAL_PLOTS}, raster dpi: ${RASTER_DPI}, trace format: ${TRACE_FORMAT}"

micromamba run -n ml-py312 python -m brunel2000.sweep \
  exec.output_dir="${OUT_DIR}" \
  exec.num_workers="${NUM_WORKERS}" \
  exec.save_trial_plots=$([[ "${SAVE_TRIAL_PLOTS}" == "1" || "${SAVE_TRIAL_PLOTS}" == "true" ]] && echo true || echo false) \
  exec.raster_dpi="${RASTER_DPI}" \
  exec.trace_format="${TRACE_FORMAT}" \
  sim.n_trials="${N_TRIALS}" \
  paper.profile="${PAPER_PROFILE}" \
  sim.device=cpu

micromamba run -n ml-py312 python -m brunel2000.plot_phase_transition \
  input_dir="${OUT_DIR}" \
  output_dir="${OUT_DIR}/plots"

echo "Done. Plots are under ${OUT_DIR}/plots"
