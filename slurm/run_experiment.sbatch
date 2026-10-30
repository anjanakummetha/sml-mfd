#!/usr/bin/env bash
#SBATCH --job-name=sml-mfd
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail

# Submit from the repository root. Set the required paths in the sbatch
# environment and select the GPU partition with --partition at submission.
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${SML_DATA_DIR:?Set SML_DATA_DIR to the monthly Parquet folder}"
: "${SML_MODE:?Set SML_MODE to 60 or 120}"
: "${SML_JOBS:?Set SML_JOBS to the corresponding job-list CSV}"
: "${SML_OUTPUT_DIR:?Set SML_OUTPUT_DIR to a scratch results folder}"
: "${SML_PYTHON:?Set SML_PYTHON to the sml environment Python executable}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a job array}"
[[ "${SML_MODE}" == 60 || "${SML_MODE}" == 120 ]] || { echo "SML_MODE must be 60 or 120" >&2; exit 2; }
[[ -d "${SML_DATA_DIR}" && -f "${SML_JOBS}" && -x "${SML_PYTHON}" ]] || {
  echo "Check SML_DATA_DIR, SML_JOBS, and SML_PYTHON" >&2
  exit 2
}

"${SML_PYTHON}" -m src.experiment_runner run \
  --mode "${SML_MODE}" \
  --jobs-csv "${SML_JOBS}" \
  --data-dir "${SML_DATA_DIR}" \
  --manifest config/feature_manifest.json \
  --output-dir "${SML_OUTPUT_DIR}" \
  --device cuda
