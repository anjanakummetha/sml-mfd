# Machine forecast disagreement

We compared forecasts from accounting and market data to see whether disagreement
predicts next month's stock returns. `src/full_preprocess_colab.py` prepares the
2000–2015 JKP panel as 192 monthly Parquet files. `config/feature_manifest.json`
lists the features in each track. 

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

## Backtester (Task 7)

`src/backtester.py` sorts stocks into portfolios on the disagreement scores in `mfd_60.csv` and reports
returns, Newey-West t-statistics, Sharpe ratios, drawdowns, Fama-French alphas and turnover. Percentile
ranges, score combinations, equal or value weighting, and holding periods of 1–12 months (overlapping
portfolios) are all set in the config files in `config/backtest/`. It runs on a laptop; the cluster
environment is not needed.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-backtest.txt
mkdir -p data && cp "$SML_OUTPUT_DIR/mfd_60.csv" data/          # data/ is git-ignored
.venv/bin/python -m src.backtester config/backtest/baseline.json   # results in results/backtest/
.venv/bin/python tests/test_backtester.py                          # engine checks
.venv/bin/streamlit run dashboard/app.py                           # research dashboard
```

The dashboard shows the portfolio-sort, factor-regression, subperiod and turnover tables for any
specification, plus the engine's checks. `config/backtest/datasets.json` lists the input files; a
dataset appears only when its file is in `data/`. The Fama-French factors in `config/factors/` are
from Kenneth French's data library. The last-year reproduction test needs last year's
`full_bbl_predictions.csv` saved as `data/lastyear_bbl_predictions.csv`; without it that test is skipped.
