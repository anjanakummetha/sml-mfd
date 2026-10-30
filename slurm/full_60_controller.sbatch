#!/usr/bin/env bash
# Submit bounded waves so the 1100 investor-year fits respect NVwulf's
# 100-submitted-job b40x4 QOS limit. This CPU-only controller uses one slot.
#SBATCH --job-name=sml-full-60
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${SML_DATA_DIR:?Set SML_DATA_DIR}"
: "${SML_OUTPUT_DIR:?Set SML_OUTPUT_DIR}"
: "${SML_PYTHON:?Set SML_PYTHON}"
: "${SML_JOBS:?Set SML_JOBS}"
: "${SML_GPU_PARTITION:?Set SML_GPU_PARTITION}"

total=$(( $(wc -l < "${SML_JOBS}") - 1 ))
[[ "${total}" -eq 1100 ]] || { echo "Expected 1100 jobs, got ${total}" >&2; exit 2; }
batch_size="${SML_BATCH_SIZE:-90}"
max_parallel="${SML_MAX_PARALLEL:-2}"
[[ "${batch_size}" =~ ^[1-9][0-9]*$ && "${batch_size}" -le 90 ]] || {
  echo "SML_BATCH_SIZE must be between 1 and 90" >&2; exit 2;
}
[[ "${max_parallel}" =~ ^[12]$ ]] || {
  echo "SML_MAX_PARALLEL must be 1 or 2 while the controller holds one job slot" >&2; exit 2;
}
export SML_MODE=60 SML_DATA_DIR SML_OUTPUT_DIR SML_PYTHON SML_JOBS

for ((start=0; start<total; start+=batch_size)); do
  end=$((start + batch_size - 1))
  if (( end >= total )); then end=$((total - 1)); fi
  echo "Submitting investor-year tasks ${start}-${end} at $(date -Is)"
  wave_job="$(sbatch --wait --parsable \
    --partition="${SML_GPU_PARTITION}" \
    --time=00:15:00 \
    --array="${start}-${end}%${max_parallel}" \
    --output="${SML_OUTPUT_DIR}/logs/slurm-%A_%a.out" \
    --error="${SML_OUTPUT_DIR}/logs/slurm-%A_%a.err" \
    --export=ALL slurm/run_experiment.sbatch)"
  wave_job="${wave_job%%;*}"
  printf 'wave=%s tasks=%s-%s\n' "${wave_job}" "${start}" "${end}" \
    >> "${SML_OUTPUT_DIR}/submitted-jobs-60.txt"
  "${SML_PYTHON}" - "${SML_JOBS}" "${SML_OUTPUT_DIR}" "${start}" "${end}" <<'PY'
import csv
import sys
from pathlib import Path

jobs, output, start, end = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with jobs.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows[start:end + 1]:
    prediction = output / "predictions" / "60" / row["track"] / row["forecast_year"] / f"investor_{int(row['investor']):03d}.csv"
    if not prediction.is_file() or prediction.stat().st_size == 0:
        raise SystemExit(f"Missing prediction after wave: {prediction}")
print(f"Verified predictions for tasks {start}-{end}")
PY
done

aggregate_job="$(sbatch --wait --parsable \
  --partition="${SML_CPU_PARTITION:-${SML_GPU_PARTITION}}" \
  --output="${SML_OUTPUT_DIR}/logs/aggregate-%j.out" \
  --error="${SML_OUTPUT_DIR}/logs/aggregate-%j.err" \
  --export=ALL slurm/aggregate_experiment.sbatch)"
aggregate_job="${aggregate_job%%;*}"
printf 'aggregate=%s\n' "${aggregate_job}" >> "${SML_OUTPUT_DIR}/submitted-jobs-60.txt"
test -s "${SML_OUTPUT_DIR}/mfd_60.csv"
test -s "${SML_OUTPUT_DIR}/aggregated_60.csv"
test -s "${SML_OUTPUT_DIR}/accuracy_overall_60.csv"
echo "Production output ready: ${SML_OUTPUT_DIR}/mfd_60.csv"
