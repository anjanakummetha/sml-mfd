#!/usr/bin/env bash
# Submit the primary 60-month, 50-investor-per-track run from the repo root.
# Requires SML_DATA_DIR, SML_OUTPUT_DIR, SML_PYTHON and SML_GPU_PARTITION.
set -euo pipefail

: "${SML_DATA_DIR:?Set SML_DATA_DIR to the monthly Parquet directory}"
: "${SML_OUTPUT_DIR:?Set SML_OUTPUT_DIR to a new scratch results directory}"
: "${SML_PYTHON:?Set SML_PYTHON to the sml environment Python executable}"
: "${SML_GPU_PARTITION:?Set SML_GPU_PARTITION to an allowed GPU partition}"

[[ -f config/feature_manifest.json && -f slurm/run_experiment.sbatch ]] || {
  echo "Run this script from the repository root" >&2
  exit 2
}
[[ -d "${SML_DATA_DIR}" && -x "${SML_PYTHON}" ]] || {
  echo "Check SML_DATA_DIR and SML_PYTHON" >&2
  exit 2
}
[[ ! -e "${SML_OUTPUT_DIR}/predictions/60" ]] || {
  echo "Production predictions already exist at ${SML_OUTPUT_DIR}; use a new output directory" >&2
  exit 2
}
for year in {2000..2015}; do
  for month in {01..12}; do
    [[ -s "${SML_DATA_DIR}/${year}-${month}.parquet" ]] || {
      echo "Missing monthly data: ${year}-${month}.parquet" >&2
      exit 2
    }
  done
done

"${SML_PYTHON}" - "${SML_DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

monthly = Path(sys.argv[1])
source_manifest = monthly.parent / "feature_manifest.json"
if not source_manifest.is_file():
    raise SystemExit(f"Missing data manifest: {source_manifest}")
source = json.loads(source_manifest.read_text())
repo = json.loads(Path("config/feature_manifest.json").read_text())
keys = ("accounting_table6_pure_fundamentals", "market_table8_pure_market")
if any(source[key] != repo[key] for key in keys):
    raise SystemExit("Repository feature manifest differs from the data manifest")
required = {"id", "eom", "ret_exc_lead1m", "ret_lead1m", "me"}
required.update(source[keys[0]])
required.update(source[keys[1]])
total_rows = 0
paths = sorted(monthly.glob("????-??.parquet"))
if len(paths) != 192:
    raise SystemExit(f"Expected 192 monthly files, found {len(paths)}")
for path in paths:
    parquet = pq.ParquetFile(path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"{path.name} is missing {sorted(missing)}")
    total_rows += parquet.metadata.num_rows
print(f"Preflight passed: 192 monthly files, {total_rows:,} stock-month rows")
PY

mkdir -p "${SML_OUTPUT_DIR}/logs"
export SML_MODE=60 SML_DATA_DIR SML_OUTPUT_DIR SML_PYTHON
max_parallel="${SML_MAX_PARALLEL_PER_TRACK:-2}"
[[ "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SML_MAX_PARALLEL_PER_TRACK must be positive" >&2
  exit 2
}

for track in accounting market; do
  jobs="${SML_OUTPUT_DIR}/jobs-${track}-60.csv"
  "${SML_PYTHON}" -m src.experiment_runner make-jobs \
    --mode 60 --tracks "${track}" --investors 50 --out "${jobs}"
  count=$(( $(wc -l < "${jobs}") - 1 ))
  [[ "${count}" -eq 550 ]] || { echo "Expected 550 ${track} jobs, got ${count}" >&2; exit 2; }
  export SML_JOBS="${jobs}"
  job_id="$(sbatch --parsable \
    --partition="${SML_GPU_PARTITION}" \
    --time=00:15:00 \
    --array="0-$((count - 1))%${max_parallel}" \
    --output="${SML_OUTPUT_DIR}/logs/slurm-%A_%a.out" \
    --error="${SML_OUTPUT_DIR}/logs/slurm-%A_%a.err" \
    --export=ALL slurm/run_experiment.sbatch)"
  job_id="${job_id%%;*}"
  printf '%s array job: %s\n' "${track}" "${job_id}"
  if [[ "${track}" == accounting ]]; then
    accounting_job="${job_id}"
  else
    market_job="${job_id}"
  fi
done

# The dependency ensures no partial result is mistaken for a full experiment.
cpu_partition="${SML_CPU_PARTITION:-${SML_GPU_PARTITION}}"
aggregate_job="$(sbatch --parsable \
  --partition="${cpu_partition}" \
  --dependency="afterok:${accounting_job}:${market_job}" \
  --output="${SML_OUTPUT_DIR}/logs/aggregate-%j.out" \
  --error="${SML_OUTPUT_DIR}/logs/aggregate-%j.err" \
  --export=ALL slurm/aggregate_experiment.sbatch)"
aggregate_job="${aggregate_job%%;*}"
printf 'aggregate job: %s\nFinal output: %s/mfd_60.csv\n' \
  "${aggregate_job}" "${SML_OUTPUT_DIR}"
printf 'accounting=%s\nmarket=%s\naggregate=%s\n' \
  "${accounting_job}" "${market_job}" "${aggregate_job}" \
  > "${SML_OUTPUT_DIR}/submitted-jobs-60.txt"
