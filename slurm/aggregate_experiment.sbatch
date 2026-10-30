#!/usr/bin/env bash
#SBATCH --job-name=sml-aggregate
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SML_OUTPUT_DIR:?Set SML_OUTPUT_DIR to the production results folder}"
: "${SML_MODE:?Set SML_MODE to 60 or 120}"
: "${SML_PYTHON:?Set SML_PYTHON to the sml environment Python executable}"

"${SML_PYTHON}" -m src.evaluate_oos \
  --output-dir "${SML_OUTPUT_DIR}" \
  --mode "${SML_MODE}" \
  --expected-investors 50

test -s "${SML_OUTPUT_DIR}/mfd_${SML_MODE}.csv"
test -s "${SML_OUTPUT_DIR}/aggregated_${SML_MODE}.csv"
test -s "${SML_OUTPUT_DIR}/accuracy_overall_${SML_MODE}.csv"
