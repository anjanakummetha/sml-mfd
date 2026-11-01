# Machine forecast disagreement

We compare forecasts from accounting and market data to see whether disagreement
predicts next month's stock returns. `src/full_preprocess_colab.py` prepares the
2000–2015 JKP panel as 192 monthly Parquet files. `config/feature_manifest.json`
lists the features in each track. Data and results stay off GitHub.

On NVwulf, run this from the repository root after copying the monthly files to
scratch, with `feature_manifest.json` beside the `monthly/` directory:

```bash
export SML_DATA_DIR=/lustre/nvwulf/scratch/akummetha/sml-data/monthly
export SML_PYTHON=/lustre/nvwulf/home/akummetha/conda/envs/sml/bin/python
export SML_OUTPUT_DIR=/lustre/nvwulf/scratch/akummetha/sml-full-60-$(date +%Y%m%d-%H%M%S)
export SML_GPU_PARTITION=b40x4
export SML_MAX_PARALLEL=2
bash slurm/submit_full_60.sh
```

This runs 50 investors per track across 11 forecast years, using a 60-month
training window. Check `submitted-jobs-60.txt` and `logs/` in the output
directory for progress. To resume an interrupted run, keep the same output
directory, set `SML_RESUME=1`, and rerun the launcher.

The final stock-month file is `mfd_60.csv`. It contains the two track-average
forecasts, within-track disagreement (`accounting_disagreement_sd` and
`market_disagreement_sd`), and their absolute gap (`cross_track_disagreement_abs`).
The `mfd` column repeats that gap; it is not the paper's pooled-investor measure.
The same directory has monthly and overall accuracy files. The 120-month window
is a separate, optional check.
