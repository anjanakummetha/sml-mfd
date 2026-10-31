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
[[ ! -e "${SML_OUTPUT_DIR}/predictions/60" || "${SML_RESUME:-0}" == 1 ]] || {
  echo "Production predictions already exist at ${SML_OUTPUT_DIR}; set SML_RESUME=1 to reuse verified results" >&2
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
export SML_MODE=60 SML_DATA_DIR SML_OUTPUT_DIR SML_PYTHON SML_GPU_PARTITION
export SML_JOBS="${SML_OUTPUT_DIR}/jobs-all-60.csv"
"${SML_PYTHON}" -m src.experiment_runner make-jobs \
  --mode 60 --tracks accounting market --investors 50 --out "${SML_JOBS}"
count=$(( $(wc -l < "${SML_JOBS}") - 1 ))
[[ "${count}" -eq 1100 ]] || { echo "Expected 1100 jobs, got ${count}" >&2; exit 2; }

# NVwulf's b40x4 QOS allows 100 submitted jobs per user. The controller
# occupies one slot and submits at most 90 array tasks at a time.
controller_job="$(sbatch --parsable \
  --partition="${SML_GPU_PARTITION}" \
  --output="${SML_OUTPUT_DIR}/logs/controller-%j.out" \
  --error="${SML_OUTPUT_DIR}/logs/controller-%j.err" \
  --export=ALL slurm/full_60_controller.sbatch)"
controller_job="${controller_job%%;*}"
if [[ "${SML_RESUME:-0}" == 1 ]]; then
  printf 'controller=%s (resume)\n' "${controller_job}" >> "${SML_OUTPUT_DIR}/submitted-jobs-60.txt"
else
  printf 'controller=%s\n' "${controller_job}" > "${SML_OUTPUT_DIR}/submitted-jobs-60.txt"
fi
printf 'Controller job: %s\nFinal output: %s/mfd_60.csv\n' \
  "${controller_job}" "${SML_OUTPUT_DIR}"
