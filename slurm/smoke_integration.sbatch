#!/usr/bin/env bash
#SBATCH --job-name=sml-integration-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=smoke-integration-%j.out
#SBATCH --error=smoke-integration-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SML_DATA_DIR:?Set SML_DATA_DIR to the monthly Parquet directory}"
: "${SML_OUTPUT_DIR:?Set SML_OUTPUT_DIR to a scratch output directory}"
SML_PYTHON="${SML_PYTHON:-python}"

for track in accounting market; do
  for investor_id in 1 2; do
    "${SML_PYTHON}" -m src.experiment_runner run \
      --data-dir "${SML_DATA_DIR}" \
      --manifest config/feature_manifest.json \
      --output-dir "${SML_OUTPUT_DIR}" \
      --mode smoke --track "${track}" \
      --investor-id "${investor_id}" \
      --forecast-year 2005 --device cuda
  done
done

"${SML_PYTHON}" -m src.evaluate_oos \
  --output-dir "${SML_OUTPUT_DIR}" \
  --mode smoke --expected-investors 2
