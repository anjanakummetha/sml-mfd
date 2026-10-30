#!/usr/bin/env bash
#SBATCH --job-name=sml-mfd-daehyun
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
# Configure the correct NVWulf GPU partition/account and Python environment before submission.
# Example: sbatch --array=0-1099 --gpus=1 slurm/run_experiment.sbatch
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${SML_DATA_DIR:?Set SML_DATA_DIR to the monthly Parquet folder}"
: "${SML_MODE:?Set SML_MODE to 60 or 120}"
: "${SML_JOBS:?Set SML_JOBS to the corresponding job-list CSV}"
python -m src.experiment_runner run \
  --mode "${SML_MODE}" \
  --jobs-csv "${SML_JOBS}" \
  --data-dir "${SML_DATA_DIR}" \
  --manifest config/feature_manifest.json \
  --output-dir results/daehyun \
  --device cuda
