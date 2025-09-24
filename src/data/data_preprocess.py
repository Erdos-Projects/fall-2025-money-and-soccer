#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preprocess.py

Build a merged FBref + Transfermarkt dataset tailored to:
"Do transfers from highly competitive leagues to emerging leagues lead to a decline in performance?"

Key improvements in this version:
- Robust column detection for transfers.csv (player/date/from/to ids or names)
- Clear diagnostics for league mapping
- Optional relaxation of Tier filters if they eliminate all rows
"""

import os
import sys
import re
import json
import pickle
import argparse
import datetime as dt
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Logging
# -----------------------------
REPORT_LINES: List[str] = []
def log(line: str):
    print(line)
    REPORT_LINES.append(line)

# -----------------------------
# Heuristics / constants
# -----------------------------
ID_LIKE_SUBSTRINGS = ["id", "uuid", "fbref_id", "player_id", "match_id"]
DEFAULT_CAT_GUESS = ["Position", "League", "Team", "Club", "Nation", "Season"]

# Defaults (can be overridden via CLI)
DEFAULT_TIER_A = {"GB1", "ES1", "IT1", "FR1", "L1", "DE1"}   # top-5 Europe (adjust to your codes)
DEFAULT_TIER_E = {"SA1", "CN1"}                              # Saudi, China (adjust/expand as needed)

# -----------------------------
# Generic cleaning helpers
# -----------------------------
def _infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_like = [c for c in df.columns if any(s in c.lower() for s in ID_LIKE_SUBSTRINGS)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for c in DEFAULT_CAT_GUESS:
        if c in df.columns and c not in cat_cols:
            cat_cols.append(c)
            if c in numeric_cols:
                numeric_cols.remove(c)
    for c in id_like:
        if c in numeric_cols:
            numeric_cols.remove(c)
        if c not in cat_cols:
            cat_cols.append(c)
    return numeric_cols, cat_cols

def _remove_obvious_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    before = df.shape[0]
    df2 = df.drop_duplicates(subset=subset, keep="first")
    return df2, before - df2.shape[0]

def _iqr_bounds(s: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = np.nanpercentile(s.astype(float), [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def _winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    return s.clip(lower=lower, upper=upper)

from scipy import stats
def _robust_z_outliers(s: pd.Series, z: float = 4.0) -> pd.Series:
    x = s.astype(float)
    med = np.nanmedian(x)
    mad = stats.median_abs_deviation(x, nan_policy="omit")
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=s.index)
    rz = (x - med) / (1.4826 * mad)
    return np.abs(rz) > z

def _correlation_prune(df_num: pd.DataFrame, threshold: float = 0.97) -> List[str]:
    if df_num.shape[1] < 2:
        return []
    corr = df_num.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

# -----------------------------
# Domain helpers
# -----------------------------
def _normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s\-']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_list_cell(cell) -> List[float]:
    if isinstance(cell, list):
        return [float(x) for x in cell]
    if pd.isna(cell):
        return []
    s = str(cell).strip().strip("[]")
    if not s:
        return []
    return [float(x) for x in re.split(r",\s*", s)]

def _ga90(df: pd.DataFrame) -> float:
    mins = df.get("minutes_played", pd.Series(dtype=float)).sum()
    if mins <= 0:
        return np.nan
    ga = df.get("goals", pd.Series(dtype=float)).sum() + df.get("assists", pd.Series(dtype=float)).sum()
    return float(ga) / (mins / 90.0)

def _compute_pre_post_perf(appear_df: pd.DataFrame,
                           player_name_norm: str,
                           center_date: pd.Timestamp,
                           window_days: int) -> Tuple[float, float, float]:
    """Returns (GA90_pre, GA90_post, pre_minutes)."""
    pdf = appear_df[appear_df["player_name_norm"] == player_name_norm].copy()
    if pdf.empty or "date" not in pdf.columns:
        return np.nan, np.nan, 0.0
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")

    start_pre = center_date - pd.Timedelta(days=window_days)
    end_pre = center_date
    start_post = center_date
    end_post = center_date + pd.Timedelta(days=window_days)

    pre = pdf[(pdf["date"] >= start_pre) & (pdf["date"] < end_pre)]
    post = pdf[(pdf["date"] > start_post) & (pdf["date"] <= end_post)]

    return _ga90(pre), _ga90(post), float(pre["minutes_played"].sum())

# -----------------------------
# FBref feature builder
# -----------------------------
def build_fbref_features(fbref_csv: str, name_db_csv: Optional[str] = None) -> pd.DataFrame:
    fb = pd.read_csv(fbref_csv)

    perc_col = "Percentiles" if "Percentiles" in fb.columns else None
    attr_col = "Attribute Vector" if "Attribute Vector" in fb.columns else None
    if perc_col is None or attr_col is None:
        raise KeyError(f"Expected 'Percentiles' and 'Attribute Vector' in {fbref_csv}. Found: {list(fb.columns)[:20]}")

    fb["Percentiles_parsed"] = fb[perc_col].apply(_parse_list_cell)
    fb["Attribute_Vector_parsed"] = fb[attr_col].apply(_parse_list_cell)

    fb["FBref_Percentile_Mean"] = fb["Percentiles_parsed"].apply(lambda xs: np.nan if len(xs)==0 else float(np.mean(xs)))
    fb["FBref_AttrVec_Mean"] = fb["Attribute_Vector_parsed"].apply(lambda xs: np.nan if len(xs)==0 else float(np.mean(xs)))

    agg = fb.groupby("Name", as_index=False).agg({
        "Position": "first",
        "FBref_Percentile_Mean": "mean",
        "FBref_AttrVec_Mean": "mean"
    }).rename(columns={"Name": "fbref_name"})

    agg["player_name_norm"] = agg["fbref_name"].map(_normalize_name)

    # optional explicit mapping
    if name_db_csv and os.path.exists(name_db_csv):
        nm = pd.read_csv(name_db_csv)
        fbref_col = None; tmk_col = None
        for c in nm.columns:
            cl = c.lower()
            if fbref_col is None and ("fbref" in cl or "source" in cl):
                fbref_col = c
            if tmk_col is None and ("transfer" in cl or "target" in cl or "tm" in cl):
                tmk_col = c
        if (fbref_col is None or tmk_col is None) and nm.shape[1] >= 3:
            fbref_col = nm.columns[1] if fbref_col is None else fbref_col
            tmk_col = nm.columns[2] if tmk_col is None else tmk_col
        if fbref_col and tmk_col and fbref_col in nm.columns and tmk_col in nm.columns:
            mapping = nm[[fbref_col, tmk_col]].dropna().rename(
                columns={fbref_col: "fbref_name", tmk_col: "player_name_mapped"}
            )
            mapping["player_name_norm_map"] = mapping["player_name_mapped"].astype(str).map(_normalize_name)
            agg = agg.merge(mapping[["fbref_name", "player_name_norm_map"]], on="fbref_name", how="left")
            agg["player_name_norm"] = np.where(
                agg["player_name_norm_map"].notna(),
                agg["player_name_norm_map"],
                agg["player_name_norm"]
            )
            agg = agg.drop(columns=["player_name_norm_map"], errors="ignore")

    return agg

# -----------------------------
# Transfermarkt target builder (robust)
# -----------------------------
def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Pick the first present column from candidates."""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def _try_map_club_names_to_ids(df: pd.DataFrame, name_col: str, clubs: pd.DataFrame) -> pd.Series:
    """Map club names to ids using clubs.csv (best-effort)."""
    # normalize in both tables
    clubs = clubs.copy()
    clubs["club_name_norm"] = clubs.get("name", pd.Series(dtype=str)).astype(str).map(_normalize_name)
    lut = dict(zip(clubs["club_name_norm"], clubs["club_id"]))
    return df[name_col].astype(str).map(_normalize_name).map(lut)


def derive_european_leagues(competitions_csv: str) -> List[str]:
    """
    Return list of league codes that are European domestic leagues
    (based on competitions.csv in Transfermarkt dump).
    """
    if not os.path.exists(competitions_csv):
        log(f"- WARNING: competitions.csv not found at {competitions_csv}. Cannot auto-derive European leagues.")
        return []
    comp = pd.read_csv(competitions_csv)
    # Be lenient with column names
    conf_col = _pick_col(list(comp.columns), ["confederation"])
    type_col = _pick_col(list(comp.columns), ["type", "competition_type"])
    code_col = _pick_col(list(comp.columns), ["domestic_league_code", "competition_id", "competition_code"])
    if code_col is None:
        log("- WARNING: Could not find a league code column in competitions.csv")
        return []
    if conf_col is None or type_col is None:
        log("- WARNING: competitions.csv missing 'confederation' or 'type'. Returning all codes.")
        return sorted(comp[code_col].dropna().astype(str).unique().tolist())

    # Keep domestic leagues in Europe
    comp_eu = comp[(comp[conf_col].astype(str).str.lower() == "europa") &
                   (comp[type_col].astype(str).str.contains("domestic_league", case=False, na=False))]
    leagues = sorted(comp_eu[code_col].dropna().astype(str).unique().tolist())
    log(f"- Auto-derived European domestic league codes: {leagues[:20]}{' ...' if len(leagues) > 20 else ''}")
    return leagues


def _compute_ga90_over_window(app_df, name_norm, start_date, end_date):
    """Compute GA/90 and total minutes for player within [start_date, end_date]."""
    sub = app_df[
        (app_df["player_name_norm"] == name_norm) &
        (app_df["date"] >= start_date) &
        (app_df["date"] <= end_date)
    ]
    if sub.empty:
        return np.nan, 0.0
    mins = sub["minutes_played"].fillna(0).sum()
    ga = (sub["goals"].fillna(0) + sub["assists"].fillna(0)).sum()
    ga90 = (ga / mins * 90.0) if mins > 0 else np.nan
    return ga90, float(mins)


def _compute_pre_post_perf_robust(
    app_df,
    name_norm,
    tdate,
    pre_days=365,
    post_days_seq=(365, 540, 720),
    min_pre_minutes=450,
    min_post_minutes=450,
    data_end_padding_days=30
):
    """
    Robust windowing for pre/post GA/90:
      * Pre: fixed 365-day window ending the day before transfer.
      * Post: try 365d, else 540d, else 720d until min_post_minutes reached or data end.
      * If transfer is too close to data end → mark InsufficientPostData.
    Returns a dict with GA90_pre, GA90_post, pre_minutes, post_minutes, flags, and UsedPostWindowDays.
    """
    if app_df["date"].dtype != "datetime64[ns]":
        # Be safe (idempotent if already datetime)
        app_df = app_df.copy()
        app_df["date"] = pd.to_datetime(app_df["date"], errors="coerce")

    max_date = app_df["date"].max()
    if pd.isna(tdate):
        return {
            "GA90_pre": np.nan, "GA90_post": np.nan,
            "pre_minutes": 0.0, "post_minutes": 0.0,
            "InsufficientPreData": 1, "InsufficientPostData": 1,
            "UsedPostWindowDays": np.nan
        }

    # If transfer is too recent, we won't have post data yet
    if tdate > (max_date - pd.Timedelta(days=data_end_padding_days)):
        ga90_pre, pre_min = _compute_ga90_over_window(
            app_df, name_norm,
            tdate - pd.Timedelta(days=pre_days),
            tdate - pd.Timedelta(days=1)
        )
        return {
            "GA90_pre": ga90_pre, "GA90_post": np.nan,
            "pre_minutes": pre_min, "post_minutes": 0.0,
            "InsufficientPreData": int(pre_min < min_pre_minutes),
            "InsufficientPostData": 1,
            "UsedPostWindowDays": np.nan
        }

    # Pre window (fixed)
    ga90_pre, pre_min = _compute_ga90_over_window(
        app_df, name_norm,
        tdate - pd.Timedelta(days=pre_days),
        tdate - pd.Timedelta(days=1)
    )

    # Post window (progressive)
    ga90_post, post_min, used_post_days = np.nan, 0.0, None
    for d in post_days_seq:
        end = min(tdate + pd.Timedelta(days=d), max_date)
        g, m = _compute_ga90_over_window(app_df, name_norm, tdate, end)
        ga90_post, post_min = g, m
        used_post_days = (end - tdate).days
        if post_min >= min_post_minutes:
            break

    return {
        "GA90_pre": ga90_pre, "GA90_post": ga90_post,
        "pre_minutes": pre_min, "post_minutes": post_min,
        "InsufficientPreData": int(pre_min < min_pre_minutes),
        "InsufficientPostData": int(post_min < min_post_minutes),
        "UsedPostWindowDays": used_post_days
    }


def build_targets_from_transfermarkt(
    tm_raw_dir: str,
    window_days: int,
    decline_pct: float,
    min_pre_minutes: int,
    from_tiers: Set[str],
    to_tiers: Set[str],
    no_tier_filter: bool = False,
    auto_emerging_europe: bool = False
) -> pd.DataFrame:
    """
    Returns one row per qualifying transfer:
      ['player_name','player_name_norm','player_id','transfer_date',
       'from_club_id','to_club_id','from_league','to_league',
       'GA90_pre','GA90_post','PerfChange_1y','DeclineFlag_1y']
    """
    transfers = pd.read_csv(os.path.join(tm_raw_dir, "transfers.csv"))
    players = pd.read_csv(os.path.join(tm_raw_dir, "players.csv"))
    clubs = pd.read_csv(os.path.join(tm_raw_dir, "clubs.csv"))
    competitions_csv = os.path.join(tm_raw_dir, "competitions.csv")
    appearances = pd.read_csv(os.path.join(tm_raw_dir, "appearances.csv"))
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["player_name_norm"] = appearances["player_name"].astype(str).map(_normalize_name)


    # Heuristically detect columns
    tcols = transfers.columns.tolist()
    player_col = _pick_col(tcols, ["player_name", "name", "player"])
    date_col   = _pick_col(tcols, ["date", "transfer_date"])
    from_id    = _pick_col(tcols, ["from_club_id", "from_team_id", "from_id"])
    to_id      = _pick_col(tcols, ["to_club_id", "to_team_id", "to_id"])
    from_name  = _pick_col(tcols, ["from_club_name", "from_club", "from_team_name"])
    to_name    = _pick_col(tcols, ["to_club_name", "to_club", "to_team_name"])
    pid_col    = _pick_col(tcols, ["player_id", "id"])

    log(f"- transfers.csv columns: {tcols}")
    log(f"- detected -> player: {player_col}, date: {date_col}, from_id: {from_id}, to_id: {to_id}, from_name: {from_name}, to_name: {to_name}, player_id: {pid_col}")

    if player_col is None or date_col is None:
        log("- ERROR: Could not detect essential columns in transfers.csv (need player & date).")
        return pd.DataFrame()

    tf = transfers.rename(columns={player_col: "player_name", date_col: "transfer_date"}).copy()
    tf["transfer_date"] = pd.to_datetime(tf["transfer_date"], errors="coerce")

    # Prefer club IDs; if missing, map from names
    if from_id not in tf.columns and from_name in tf.columns:
        tf["from_club_id"] = _try_map_club_names_to_ids(tf, from_name, clubs)
    else:
        tf["from_club_id"] = tf.get(from_id)
    if to_id not in tf.columns and to_name in tf.columns:
        tf["to_club_id"] = _try_map_club_names_to_ids(tf, to_name, clubs)
    else:
        tf["to_club_id"] = tf.get(to_id)

    if pid_col and pid_col in tf.columns:
        tf["player_id"] = tf[pid_col]
    else:
        tf["player_id"] = np.nan

    # Attach league codes from clubs
    clubs_small = clubs.rename(columns={"domestic_competition_id": "league_code"})[["club_id", "league_code"]].copy()
    tf = tf.merge(clubs_small.add_prefix("from_"), on="from_club_id", how="left")
    tf = tf.merge(clubs_small.add_prefix("to_"),   on="to_club_id",   how="left")

    # Normalize player name for appearances join
    tf["player_name_norm"] = tf["player_name"].astype(str).map(_normalize_name)

    # Diagnostics
    log(f"- sample from_league codes: {sorted(tf['from_league_code'].dropna().astype(str).unique()[:10]) if 'from_league_code' in tf.columns else 'N/A'}")
    log(f"- sample to_league codes: {sorted(tf['to_league_code'].dropna().astype(str).unique()[:10]) if 'to_league_code' in tf.columns else 'N/A'}")

    # ----- Europe-only emerging option -----
    if auto_emerging_europe:
        europe_codes = set(derive_european_leagues(competitions_csv))
        if europe_codes:
            # constrain to Europe for both sides
            tf = tf[
                tf["from_league_code"].isin(europe_codes) &
                tf["to_league_code"].isin(europe_codes)
            ].copy()
            # define emerging = Europe minus top-5
            to_tiers = set(code for code in europe_codes if code not in from_tiers)
            log(f"- Auto 'emerging' (Europe minus top-5) → to_tiers={sorted(to_tiers)}")
        else:
            log("- WARNING: Could not auto-derive Europe codes. Falling back to provided tiers.")

    # Tier filter (unless skipped)
    if not no_tier_filter and "from_league_code" in tf.columns and "to_league_code" in tf.columns:
        mask = tf["from_league_code"].isin(from_tiers) & tf["to_league_code"].isin(to_tiers)
        tfq = tf[mask].copy()
        log(f"- qualifying transfers after tier filter: {tfq.shape[0]}")
        if tfq.empty:
            log("- WARNING: Tier filter produced zero rows. Use --no_tier_filter or adjust tiers.")
            return pd.DataFrame()
    else:
        tfq = tf.copy()
        log("- no_tier_filter active or league codes missing; using ALL transfers for target build.")

    # Build appearances helper
    appearances = appearances.copy()
    appearances["player_name_norm"] = appearances["player_name"].astype(str).map(_normalize_name)

    # Compute pre/post GA90
    
    rows = []
    for _, r in tfq.iterrows():
        tdate = r["transfer_date"]
        pname = r["player_name_norm"]

        perf = _compute_pre_post_perf_robust(
            appearances, pname, tdate,
            pre_days=365,
            post_days_seq=(365, 540, 720),
            min_pre_minutes=min_pre_minutes,
            min_post_minutes=min_pre_minutes,  # same threshold both sides
            data_end_padding_days=30
        )

        ga_pre   = perf["GA90_pre"]
        ga_post  = perf["GA90_post"]
        pre_min  = perf["pre_minutes"]
        post_min = perf["post_minutes"]

        perf_change = np.nan if (np.isnan(ga_pre) or np.isnan(ga_post)) else (ga_post - ga_pre)

        decline = np.nan
        if (pre_min >= min_pre_minutes) and (post_min >= min_pre_minutes) and (not np.isnan(ga_pre)) and (not np.isnan(ga_post)):
            decline = 1 if ga_post <= (1 - decline_pct) * ga_pre else 0

        rows.append({
            "player_name": r["player_name"],
            "player_name_norm": pname,
            "player_id": r.get("player_id"),
            "transfer_date": tdate,
            "from_club_id": r.get("from_club_id"),
            "to_club_id": r.get("to_club_id"),
            "from_league": r.get("from_league_code"),
            "to_league": r.get("to_league_code"),
            "GA90_pre": ga_pre,
            "GA90_post": ga_post,
            "PerfChange_1y": perf_change,
            "DeclineFlag_1y": decline,
            "PreMinutes": pre_min,
            "PostMinutes": post_min,
            "InsufficientPreData": perf["InsufficientPreData"],
            "InsufficientPostData": perf["InsufficientPostData"],
            "UsedPostWindowDays": perf["UsedPostWindowDays"]
        })

    targets = pd.DataFrame(rows)
    log(f"- built targets: {targets.shape}")
    log(f"- rows with sufficient data: {int(((targets['InsufficientPreData']==0)&(targets['InsufficientPostData']==0)).sum())}")
    log(f"- % DeclineFlag_1y set: {targets['DeclineFlag_1y'].notna().mean():.2%}")


    return targets


# -----------------------------
# General cleaner (same idea)
# -----------------------------
def clean_dataset(
    df: pd.DataFrame,
    winsorize: bool = True,
    iqr_k: float = 1.5,
    z_thresh: float = 4.0,
    corr_thresh: float = 0.97
) -> Dict[str, any]:
    meta = {"start_rows": int(df.shape[0]), "start_cols": int(df.shape[1])}
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # --- infer types
    numeric_cols, cat_cols = _infer_column_types(df)
    log(f"- Inferred numeric columns: {len(numeric_cols)}")
    log(f"- Inferred categorical columns: {len(cat_cols)}")

    # --- drop duplicates
    df, dup_removed = _remove_obvious_duplicates(df)
    log(f"- Removed duplicate rows: {dup_removed}")

    # --- handle all-NaN columns up-front
    all_nan_num = [c for c in numeric_cols if df[c].isna().all()]
    if all_nan_num:
        log(f"- Dropping all-NaN numeric columns: {all_nan_num}")
        df = df.drop(columns=all_nan_num)
        numeric_cols = [c for c in numeric_cols if c not in all_nan_num]

    # For categoricals: keep them, but impute a constant so OHE can see a category
    # (Do NOT drop; otherwise ColumnTransformer will fit on fewer features than cat_cols length.)
    # We still note which were all NaN for transparency.
    all_nan_cat = [c for c in cat_cols if df[c].isna().all()]
    if all_nan_cat:
        log(f"- All-NaN categorical columns will be imputed with '__NA__': {all_nan_cat}")

    # --- outlier flags & optional winsorization (numeric only, skip if none)
    outlier_flags = {}
    if numeric_cols:
        for col in numeric_cols:
            s = df[col]
            # guard against all-NaN series already removed above
            lo, hi = _iqr_bounds(s, k=iqr_k)
            iqr_flag = (s < lo) | (s > hi)
            z_flag = _robust_z_outliers(s, z=z_thresh)
            combined = iqr_flag | z_flag
            outlier_flags[col] = combined
            df[f"OUT_{col}"] = combined.astype(int)
        n_out = int(pd.DataFrame(outlier_flags).sum().sum()) if outlier_flags else 0
        log(f"- Outlier flags created for {len(numeric_cols)} numeric columns (total flagged cells: {n_out})")

        if winsorize:
            capped = 0
            for col in numeric_cols:
                s = df[col]
                lo, hi = _iqr_bounds(s, k=iqr_k)
                capped += int(((s < lo) | (s > hi)).sum())
                df[col] = _winsorize_series(s, lo, hi)
            log(f"- Winsorized numeric columns using IQR k={iqr_k} (values capped: {capped})")
    else:
        log("- No numeric columns after all-NaN drop; skipping outlier steps.")

    # --- optional correlation pruning (on remaining numerics)
    drop_corr = _correlation_prune(df[numeric_cols], threshold=corr_thresh) if numeric_cols else []
    if drop_corr:
        log(f"- Dropping {len(drop_corr)} highly correlated features (>|{corr_thresh}|): {drop_corr}")
        numeric_cols = [c for c in numeric_cols if c not in drop_corr]
        df = df.drop(columns=drop_corr)

    # --- Build preprocessing pipeline
    # Numeric: median impute; if a column becomes all-NaN within a split, median imputer still works
    numeric_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True))
    ])

    # Categorical: CONSTANT impute so all-missing columns survive and OHE can encode them
    categorical_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Keep copies of the exact column lists we pass into the CT (for correct feature naming)
    num_used = list(numeric_cols)  # may be empty
    cat_used = list(cat_cols)      # may be empty

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, num_used),
            ("cat", categorical_transform, cat_used),
        ],
        remainder="drop"
    )

    # Fit & transform
    X = preprocessor.fit_transform(df)

    # --- Compose feature names from the EXACT lists used to fit
    feature_names = []
    if num_used:
        feature_names += [f"num__{c}" for c in num_used]
    if cat_used:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        # IMPORTANT: pass the same cat_used list to get_feature_names_out
        cat_feat_names = list(ohe.get_feature_names_out(cat_used))
        feature_names += cat_feat_names

    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)

    clean_df = df.copy()
    meta.update({
        "end_rows": int(clean_df.shape[0]),
        "end_cols_clean": int(clean_df.shape[1]),
        "end_cols_model": int(X_df.shape[1]),
        "dropped_high_corr": drop_corr,
        "numeric_cols_used": num_used,
        "categorical_cols_used": cat_used,
        "dropped_all_nan_numeric": all_nan_num,
        "all_nan_categorical_imputed": all_nan_cat
    })
    return {"clean_df": clean_df, "model_df": X_df, "pipeline": preprocessor, "meta": meta}


def _role_aware_fill(df, pct_drop=0.20):
        df = df.copy()
        # Defenders/GKs often have GA near zero; use utilization drop as proxy if GA90 label is missing
        is_def_gk = df["Position"].fillna("").str.contains("Back|CenterBack|FullBack|Defen|Goal|Keeper", case=False, regex=True)
        needs_fill = df["DeclineFlag_1y"].isna() & is_def_gk & (df["PreMinutes"] >= 450) & (df["PostMinutes"] >= 450)

        # Approximate minutes-per-90 over each window
        pre_mp90  = df["PreMinutes"] / (365 / 90.0)
        post_days = df["UsedPostWindowDays"].fillna(365).clip(lower=1)
        post_mp90 = df["PostMinutes"] / (post_days / 90.0)

        proxy_decline = post_mp90 <= (1 - pct_drop) * pre_mp90
        df.loc[needs_fill, "DeclineFlag_1y"] = proxy_decline[needs_fill].astype(int)
        return df




# -----------------------------
# CLI
# -----------------------------
def main():
    # Fixed config
    fbref_csv      = "data/processed/fbref/fbref.csv"
    fbref_name_db  = "data/raw/fbref/NAME_DB.csv"
    tm_raw_dir     = "data/raw/transfermarkt"
    merged_csv     = "data/processed/merged/players_transfer_outcomes.csv"
    clean_csv      = "data/processed/merged_clean.csv"
    model_csv      = "data/processed/merged_model_ready.csv"
    pipeline_pkl   = "artifacts/preprocessor.pkl"
    log_md         = "logs/cleaning_report.md"

    # Transfermarkt build params
    window_days    = 365
    decline_pct    = 0.20
    min_pre_minutes= 450
    from_tiers     = {"GB1", "ES1", "IT1", "FR1", "DE1", "L1"}  # top-5
    auto_emerging_europe = True

    # Cleaning params
    winsorize  = True
    iqr_k      = 1.5
    z_thresh   = 4.0
    corr_thresh= 0.97

    # Ensure dirs exist
    for path in [merged_csv, clean_csv, model_csv, pipeline_pkl, log_md]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    log("# Build & Clean Report")
    log(f"- Timestamp: {dt.datetime.now(dt.timezone.utc).isoformat()}")

    # === FBref features ===
    log(f"- Loading FBref features from: {fbref_csv}")
    fbref_feats = build_fbref_features(fbref_csv, name_db_csv=fbref_name_db)
    log(f"- FBref features shape: {fbref_feats.shape}")

    # === Transfermarkt targets ===
    log(f"- Building targets from Transfermarkt raw dir: {tm_raw_dir}")
    targets = build_targets_from_transfermarkt(
        tm_raw_dir=tm_raw_dir,
        window_days=window_days,
        decline_pct=decline_pct,
        min_pre_minutes=min_pre_minutes,
        from_tiers=from_tiers,
        to_tiers=set(),  # will be filled by auto_emerging_europe
        no_tier_filter=False,
        auto_emerging_europe=auto_emerging_europe
    )

    if targets.empty:
        log("- No targets built. Diagnostics were printed above.")
        pd.DataFrame().to_csv(merged_csv, index=False)
        log(f"- Wrote empty merged dataset to: {merged_csv}")
        with open(log_md, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT_LINES))
        return 0

    # Merge features with targets
    merged = targets.merge(
        fbref_feats[["player_name_norm", "fbref_name", "Position",
                     "FBref_Percentile_Mean", "FBref_AttrVec_Mean"]],
        on="player_name_norm", how="left"
    )


    

    merged = _role_aware_fill(merged, pct_drop=0.20)


    merged.to_csv(merged_csv, index=False)
    log(f"- Saved merged dataset: {merged_csv}  (rows={merged.shape[0]}, cols={merged.shape[1]})")

    # Clean & export
    log(f"- Cleaning merged dataset (winsorize={winsorize}, iqr_k={iqr_k}, corr_thresh={corr_thresh})")
    results = clean_dataset(
        merged,
        winsorize=winsorize,
        iqr_k=iqr_k,
        z_thresh=z_thresh,
        corr_thresh=corr_thresh
    )

    results["clean_df"].to_csv(clean_csv, index=False)
    results["model_df"].to_csv(model_csv, index=False)
    with open(pipeline_pkl, "wb") as f:
        pickle.dump(results["pipeline"], f)

    log(f"- Saved cleaned CSV: {clean_csv}")
    log(f"- Saved model-ready CSV: {model_csv}")
    log(f"- Saved pipeline: {pipeline_pkl}")

    log("\n## Summary")
    log("```json")
    log(json.dumps(results["meta"], indent=2))
    log("```")

    with open(log_md, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"- Wrote report: {log_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
