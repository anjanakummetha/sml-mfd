# SML machine forecast disagreement

The project uses two separate feature tracks: accounting and market. The full
2000–2015 JKP preprocessing script is `src/full_preprocess_colab.py`. Its output
is one ranked Parquet file per signal month plus `feature_manifest.json`; raw
data and generated results are kept outside GitHub.

## NVwulf production run

Run from the repository root in NVwulf Cluster Shell Access after the 192
monthly Parquet files are present on scratch. Use the same feature manifest
that was generated with those files. The primary run is the **60-month** window;
the 120-month window is a separate optional robustness run.

```bash
export SML_DATA_DIR=/lustre/nvwulf/scratch/akummetha/sml-data/monthly
export SML_PYTHON=/lustre/nvwulf/home/akummetha/conda/envs/sml/bin/python
export SML_OUTPUT_DIR=/lustre/nvwulf/scratch/akummetha/sml-full-60-20261030
export SML_GPU_PARTITION=b40x4
export SML_MAX_PARALLEL=2
bash slurm/submit_full_60.sh
```

The launcher checks every month from 2000-01 through 2015-12 and creates one
1,100-row job list (50 investors × 11 forecast years × 2 tracks). NVwulf's
`b40x4` QOS permits 100 submitted jobs per user, so a CPU-only controller
submits waves of at most 90 GPU array tasks and waits for each wave before
submitting the next. It then submits the aggregation job. The controller and
wave IDs are written to `$SML_OUTPUT_DIR/submitted-jobs-60.txt`. Inspect
progress with `squeue -u "$USER"`, `sacct -j JOB_ID -X`, and the controller log
in `$SML_OUTPUT_DIR/logs`.

The final stock-month file is `$SML_OUTPUT_DIR/mfd_60.csv`. Its distinct
disagreement columns are `accounting_disagreement_sd`,
`market_disagreement_sd`, and `cross_track_disagreement_abs`. The legacy `mfd`
column duplicates the cross-track absolute gap; it is **not** the paper's
pooled-investor MFD. The file also includes the two track-mean forecasts,
following-month excess and raw returns, and signal-month market value `me` for
Task 7. Monthly and overall accuracy files are saved beside it.

`slurm/smoke_integration.sbatch` runs a small GPU check with two investors per
track and 50 trees. It is not the 50-investor production experiment.
