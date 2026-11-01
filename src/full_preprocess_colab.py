"""Prepare the 2000–2015 JKP panel as monthly Parquet files in Colab."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("/content/drive/MyDrive/Colab Notebooks/jkp_month_2000_2015.csv")
OUTPUT = Path("/content/drive/MyDrive/Colab Notebooks/jkp_full_preprocessed_2000_2015_v1")
CHUNK_SIZE = 100_000

# Paper Table 6 characteristics using only accounting fundamentals or accounting
# histories. The JKP CSV already contains these characteristics; they are not
# recomputed from Compustat. Book enterprise value is not market enterprise value.
ACCOUNTING_CANDIDATES = """
at_gr1 sale_gr1 capx_gr1 inv_gr1 debt_gr3 sale_gr3 capx_gr3
inv_gr1a lti_gr1a sti_gr1a coa_gr1a col_gr1a cowc_gr1a ncoa_gr1a
ncol_gr1a nncoa_gr1a fnl_gr1a nfna_gr1a tax_gr1a be_gr1a
ebit_sale gp_at cop_at ope_be ni_be ebit_bev netis_at eqnetis_at
dbnetis_at oaccruals_at oaccruals_ni taccruals_at taccruals_ni
noa_at opex_at at_turnover sale_bev rd_sale cash_at sale_emp_gr1
emp_gr1 ni_inc8q noa_gr1a ppeinv_gr1a lnoa_gr1a capx_gr2
saleq_gr1 niq_be niq_at niq_be_chg1 niq_at_chg1 rd5_at
dsale_dinv dsale_drec dgp_dsale dsale_dsga saleq_su niq_su
capex_abn op_atl1 gp_atl1 ope_bel1 cop_atl1 pi_nix ocf_at
op_at ocf_at_chg1 at_be ocfq_saleq_std tangibility
earnings_variability aliq_at f_score o_score ni_ar1 ni_ivol
""".split()

# Paper Table 8 price, returns, trading, shares, dividends and market-factor
# characteristics. market_equity is the paper's size measure; raw me is retained
# separately for evaluation weights.
MARKET_CANDIDATES = """
market_equity div12m_me chcsho_12m eqnpo_12m
ret_1_0 ret_3_1 ret_6_1 ret_9_1 ret_12_1 ret_12_7 ret_60_12
seas_1_1an seas_1_1na seas_2_5an seas_2_5na seas_6_10an
seas_6_10na seas_11_15an seas_11_15na seas_16_20an seas_16_20na
beta_60m resff3_12_1 resff3_6_1
ivol_capm_21d iskew_capm_21d coskew_21d beta_dimson_21d
ivol_ff3_21d iskew_ff3_21d ivol_hxz4_21d iskew_hxz4_21d
rmax5_21d rmax1_21d rvol_21d rskew_21d zero_trades_21d
dolvol_126d dolvol_var_126d turnover_126d turnover_var_126d
zero_trades_126d zero_trades_252d ami_126d ivol_capm_252d
prc_highprc_252d betadown_252d bidaskhl_21d corr_1260d
betabab_1260d rmax5_rvol_21d
""".split()

# These are documented rather than silently assigned. Table 6 contains some
# market-scaled fundamentals, while Table 8 contains some accounting/market
# composites. Including them would break the project's strict track separation.
MIXED_OR_UNASSIGNED = {
    "market_scaled_fundamentals": """
        at_me be_me debt_me netdebt_me sale_me ni_me ocf_me fcf_me
        eqpo_me eqnpo_me rd_me ival_me bev_mev ebitda_mev
    """.split(),
    "accounting_market_composites": """
        aliq_mat eq_dur mispricing_mgmt mispricing_perf qmj qmj_prof
        qmj_growth qmj_safety z_score kz_index
    """.split(),
    "listing_history_not_pure_fundamentals_or_prices": ["age"],
}

REQUIRED = """
id eom excntry obs_main crsp_shrcd crsp_exchcd sic prc me ret_exc ret
""".split()
OPTIONAL_METADATA = ["permno", "gvkey", "ff49", "gics"]
OUTPUT_METADATA = [
    "id", "eom", "ret_exc_lead1m", "ret_lead1m", "ret_exc", "ret",
    "me", "prc", "sic", "crsp_shrcd", "crsp_exchcd",
    *OPTIONAL_METADATA,
]


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    started = time.time()
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Mount Drive and locate the source CSV: {SOURCE}")
    header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
    absent_required = sorted(set(REQUIRED) - set(header))
    if absent_required:
        raise ValueError(f"Required source columns missing: {absent_required}")
    if len(set(header)) != len(header):
        raise ValueError("Duplicate source column names")
    if len(ACCOUNTING_CANDIDATES) != len(set(ACCOUNTING_CANDIDATES)):
        raise ValueError("Duplicate accounting feature name")
    if len(MARKET_CANDIDATES) != len(set(MARKET_CANDIDATES)):
        raise ValueError("Duplicate market feature name")
    if set(ACCOUNTING_CANDIDATES) & set(MARKET_CANDIDATES):
        raise ValueError("Accounting and market features overlap")

    missing_accounting = sorted(set(ACCOUNTING_CANDIDATES) - set(header))
    missing_market = sorted(set(MARKET_CANDIDATES) - set(header))
    accounting = [c for c in ACCOUNTING_CANDIDATES if c in header]
    market = [c for c in MARKET_CANDIDATES if c in header]
    if not accounting or not market:
        raise ValueError("At least one track has no source features")
    features = accounting + market
    available_meta = [c for c in OPTIONAL_METADATA if c in header]
    read_columns = set(REQUIRED + available_meta + features)
    all_characteristics = sorted(set(header) - set(REQUIRED) - set(OPTIONAL_METADATA))
    excluded_columns = sorted(set(all_characteristics) - set(features))

    if (OUTPUT / "DONE.json").exists():
        raise FileExistsError(f"Completed output exists; use a new version path: {OUTPUT}")
    monthly_dir = OUTPUT / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = {
        "schema_version": "sml-two-track-v1",
        "source": SOURCE.name,
        "source_size_bytes": SOURCE.stat().st_size,
        "accounting_table6_pure_fundamentals": accounting,
        "market_table8_pure_market": market,
        "accounting_count": len(accounting),
        "market_count": len(market),
        "missing_accounting_candidates": missing_accounting,
        "missing_market_candidates": missing_market,
        "mixed_or_unassigned_by_reason": MIXED_OR_UNASSIGNED,
        "excluded_source_columns": excluded_columns,
        "feature_transform": "Monthly cross-sectional average rank, 2*(rank-1)/(nonmissing_count-1)-1; nonfinite and missing -> 0; float32",
        "signal_month": "eom",
        "target": "next consecutive calendar month's raw ret_exc; missing if unavailable",
        "training": "Use only observed labels before forecast formation; train-only clipping, if any, belongs downstream",
    }
    save_json(OUTPUT / "feature_manifest.json", feature_manifest)
    print(f"Features: accounting={len(accounting)}, market={len(market)}; unavailable={len(missing_accounting)+len(missing_market)}", flush=True)

    # Keep every U.S. main observation; retain pre-filter rows so the next-month
    # return is aligned even when the next observation fails a predictor filter.
    pieces = []
    source_rows = us_rows = 0
    for chunk_number, chunk in enumerate(pd.read_csv(
        SOURCE, chunksize=CHUNK_SIZE, usecols=lambda c: c in read_columns,
        low_memory=False,
    ), start=1):
        source_rows += len(chunk)
        us = chunk.loc[(chunk["excntry"] == "USA") & (chunk["obs_main"] == 1)].copy()
        us_rows += len(us)
        if not us.empty:
            us[features] = us[features].apply(pd.to_numeric, errors="coerce").astype("float32")
            pieces.append(us)
        del chunk, us
        if chunk_number % 10 == 0:
            print(f"Read {source_rows:,} source rows; retained {us_rows:,} U.S. main rows", flush=True)
    if not pieces:
        raise ValueError("No U.S. main observations found")
    panel = pd.concat(pieces, ignore_index=True)
    del pieces

    panel["id"] = pd.to_numeric(panel["id"], errors="coerce")
    if panel["id"].isna().any():
        raise ValueError("Missing or invalid stock id")
    panel["id"] = panel["id"].astype("int64")
    panel["eom"] = pd.to_datetime(panel["eom"], format="%d%b%Y", errors="coerce")
    if panel["eom"].isna().any() or not panel["eom"].dt.is_month_end.all():
        raise ValueError("Invalid or non-month-end eom values")
    duplicate_count = int(panel.duplicated(["id", "eom"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate id/eom rows: {duplicate_count}; resolve upstream")
    panel = panel.sort_values(["id", "eom"], kind="mergesort").reset_index(drop=True)
    grouped = panel.groupby("id", sort=False)
    next_eom = grouped["eom"].shift(-1)
    consecutive = next_eom.eq(panel["eom"] + pd.offsets.MonthEnd(1))
    panel["ret_exc_lead1m"] = pd.to_numeric(grouped["ret_exc"].shift(-1), errors="coerce").where(consecutive)
    panel["ret_lead1m"] = pd.to_numeric(grouped["ret"].shift(-1), errors="coerce").where(consecutive)
    del grouped, next_eom, consecutive

    # Apply the plan's filters to predictor rows. Missing SIC is excluded since
    # financial/utility membership cannot otherwise be established. Missing or
    # nonpositive me is reported and retained for equal-weight analysis; the
    # value-weighted evaluator must require me > 0.
    filter_counts = {"us_obs_main": len(panel)}
    rules = [
        ("common_share_10_11", panel["crsp_shrcd"].isin([10, 11])),
        ("nyse_amex_nasdaq", panel["crsp_exchcd"].isin([1, 2, 3])),
        ("known_sic", pd.to_numeric(panel["sic"], errors="coerce").notna()),
        ("nonfinancial", ~pd.to_numeric(panel["sic"], errors="coerce").between(6000, 6999)),
        ("nonutility", ~pd.to_numeric(panel["sic"], errors="coerce").between(4900, 4999)),
        ("price_at_least_5", pd.to_numeric(panel["prc"], errors="coerce").abs().ge(5)),
    ]
    eligible = pd.Series(True, index=panel.index)
    for name, mask in rules:
        eligible &= mask.fillna(False)
        filter_counts[name] = int(eligible.sum())
        print(f"After {name}: {filter_counts[name]:,}", flush=True)
    panel = panel.loc[eligible].copy()
    del eligible, rules
    if panel.empty:
        raise ValueError("No rows remain after sample filters")

    stats = []
    processed_rows = 0
    months = []
    metadata = [c for c in OUTPUT_METADATA if c in panel.columns]
    for eom, month in panel.groupby("eom", sort=True):
        x = month[features].replace([np.inf, -np.inf], np.nan)
        counts = x.count()
        ranks = x.rank(method="average")
        scaled = (2 * (ranks - 1) / (counts - 1) - 1).fillna(0).astype("float32")
        if not np.isfinite(scaled.to_numpy()).all():
            raise ValueError(f"Nonfinite ranked feature in {eom}")
        if (scaled.to_numpy() < -1.000001).any() or (scaled.to_numpy() > 1.000001).any():
            raise ValueError(f"Rank outside [-1, 1] in {eom}")
        out = pd.concat([month[metadata].reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
        month_name = eom.strftime("%Y-%m")
        path = monthly_dir / f"{month_name}.parquet"
        out.to_parquet(path, index=False, compression="zstd")
        months.append(month_name)
        processed_rows += len(out)
        stats.append({
            "month": month_name,
            "rows": len(out),
            "stocks": int(out["id"].nunique()),
            "missing_excess_target": int(out["ret_exc_lead1m"].isna().sum()),
            "missing_raw_target": int(out["ret_lead1m"].isna().sum()),
            "invalid_value_weight": int((pd.to_numeric(out["me"], errors="coerce").fillna(0) <= 0).sum()),
            "accounting_missing_fraction": float(x[accounting].isna().to_numpy().mean()),
            "market_missing_fraction": float(x[market].isna().to_numpy().mean()),
            "file_bytes": path.stat().st_size,
        })
        if len(months) % 12 == 0:
            print(f"Saved {len(months)} months through {month_name}; {processed_rows:,} rows", flush=True)

    if processed_rows != len(panel) or len(months) != panel["eom"].nunique():
        raise ValueError("Output row/month count mismatch")
    files = sorted(monthly_dir.glob("????-??.parquet"))
    if len(files) != len(months):
        raise ValueError("Monthly output file count mismatch; check stale files")
    for path in [files[0], files[len(files) // 2], files[-1]]:
        check = pd.read_parquet(path)
        if check.duplicated(["id", "eom"]).any() or check[features].isna().any().any():
            raise ValueError(f"Parquet readback failed: {path}")
    qc = {
        "source_rows": source_rows,
        "us_main_rows": us_rows,
        "filter_counts": filter_counts,
        "processed_rows": processed_rows,
        "unique_stocks": int(panel["id"].nunique()),
        "month_count": len(months),
        "first_month": months[0],
        "last_month": months[-1],
        "missing_excess_targets": int(panel["ret_exc_lead1m"].isna().sum()),
        "missing_raw_targets": int(panel["ret_lead1m"].isna().sum()),
        "nonpositive_or_missing_me": int((pd.to_numeric(panel["me"], errors="coerce").fillna(0) <= 0).sum()),
        "elapsed_seconds": round(time.time() - started, 1),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "monthly": stats,
    }
    save_json(OUTPUT / "qc.json", qc)
    (OUTPUT / "README.txt").write_text(
        "SML full U.S. JKP 2000-2015 preprocessing (Task 2).\n"
        "Each monthly/YYYY-MM.parquet row is a signal-month stock. ret_exc_lead1m "
        "and ret_lead1m are realized returns in the immediately following calendar "
        "month, or missing. Accounting and market feature lists are in "
        "feature_manifest.json. Only selected pure-track features are monthly "
        "cross-sectional ranks in [-1,1]; missing/nonfinite feature values are 0. "
        "Raw me is the signal-month value-weighting variable; require me > 0 when "
        "computing value-weighted portfolios. Never use labels or future observations "
        "as model features. No target clipping is done here; if training needs "
        "winsorization, compute thresholds on each training window only. "
        "The first valid 60-month or 120-month forecast is determined by observed "
        "target dates, not a hard-coded year. Macro regimes and 2016-2025 are later "
        "joins/extensions. Licensed stock-level files stay outside GitHub.\n"
    )
    save_json(OUTPUT / "DONE.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": processed_rows,
        "months": len(months),
        "accounting_features": len(accounting),
        "market_features": len(market),
    })
    print(f"COMPLETE: {processed_rows:,} rows, {len(months)} months; output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
