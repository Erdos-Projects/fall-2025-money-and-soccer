"""
data_loader.py

Loads raw CSVs from:
- data/raw/fbref/
- data/raw/transfermarkt/

Merges into a single dataset of transfers with:
- Transfermarkt features (clubs, leagues, appearances, GA90, DeclineFlag, player metadata)
- FBref features (positional performance stats)
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm


# =========================
# FBref DataBase
# =========================
class FBrefDataBase:
    """
    Loads FBref datasets from: data/raw/fbref/
    Combines positional CSVs into one table and saves to data/processed/fbref/fbref.csv
    """

    def __init__(self, base_path="data/raw/fbref"):
        self.base_path = base_path

        # raw dfs
        self.name_link = None
        self.midfielders = None
        self.forwards = None
        self.att_mid_wingers = None
        self.full_backs = None
        self.center_backs = None
        self.goal_keepers = None

        # combined df
        self.all_players = None

    def load_offline(self):
        """Load CSVs from local raw data directory."""
        self.name_link = pd.read_csv(os.path.join(self.base_path, "NAME_DB.csv"))

        self.midfielders = pd.read_csv(os.path.join(self.base_path, "Midfielders.csv"))
        self.forwards = pd.read_csv(os.path.join(self.base_path, "Forwards.csv"))
        self.att_mid_wingers = pd.read_csv(os.path.join(self.base_path, "AtMid_Wingers.csv"))
        self.full_backs = pd.read_csv(os.path.join(self.base_path, "FullBacks.csv"))
        self.center_backs = pd.read_csv(os.path.join(self.base_path, "CenterBacks.csv"))
        self.goal_keepers = pd.read_csv(os.path.join(self.base_path, "GoalKeepers.csv"))

    def combine(self):
        """Merge all positions into a single dataframe with a 'Position' column."""
        dfs = [
            (self.midfielders, "Midfielder"),
            (self.forwards, "Forward"),
            (self.att_mid_wingers, "AttMid_Winger"),
            (self.full_backs, "FullBack"),
            (self.center_backs, "CenterBack"),
            (self.goal_keepers, "GoalKeeper")
        ]
        combined = []
        for df, pos in dfs:
            if df is not None:
                df_copy = df.copy()
                df_copy["Position"] = pos
                combined.append(df_copy)
        self.all_players = pd.concat(combined, ignore_index=True)

    def save_processed(self, out_file="data/processed/fbref/fbref.csv"):
        """Save the combined dataframe into a single CSV."""
        if self.all_players is None:
            raise ValueError("No combined dataframe found. Call combine() first.")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.all_players.to_csv(out_file, index=False)


# =========================
# Transfermarkt DataBase
# =========================
class TransfermarktDataBase:
    """
    Loads and minimally cleans Transfermarkt CSVs from: data/raw/transfermarkt/
    Produces:
      - club_games_clean.csv
      - games_clean.csv
      - games_clubgames_merged.csv  (inner join on game_id)
    """

    def __init__(self, base_path="data/raw/transfermarkt"):
        self.base_path = base_path

        # raw dfs (load on demand)
        self.appearances = None
        self.club_games = None
        self.clubs = None
        self.competitions = None
        self.game_events = None
        self.game_lineups = None
        self.games = None
        self.player_valuations = None
        self.players = None
        self.transfers = None

        # cleaned / merged
        self.club_games_clean = None
        self.games_clean = None
        self.games_clubgames = None

    # ----------- helpers -----------
    @staticmethod
    def _map_hosting(series: pd.Series) -> pd.Series:
        mapping = {"Home": 1, "Away": 0}
        return series.map(mapping)

    @staticmethod
    def _map_formations(series: pd.Series) -> pd.Series:
        mapping = {
            np.nan: 0,
            '4-2-3-1': 1,
            '4-3-2-1': 2,
            '4-4-2 double 6': 3,
            '4-3-3 Defending': 4,
            '4-3-3 Attacking': 5,
            '4-5-1 flat': 6,
            '4-1-4-1': 7,
            '4-3-1-2': 8,
            '3-5-2 flat': 9,
            '4-4-2 Diamond': 10,
            '5-4-1': 11,
            '4-4-1-1': 12,
            '4-4-2': 13,
            '3-4-2-1': 14,
            '5-2-3': 15,
            '4-1-3-2': 16,
            '5-3-2': 17,
            '3-5-2': 18,
            '3-6-1': 19,
            '4-3-3': 20,
            '4-5-1': 21,
            '3-4-3': 22,
            '5-4-1 Diamond': 23,
            '2-5-3': 24,
            '3-4-1-2': 25,
            '3-5-2 Attacking': 26,
            '3-1-4-2': 27,
            '3-4-3 Diamond': 28,
            '3-3-3-1': 29,
            '4-2-4': 30,
            '4-6-0': 31,
            '6-1-3': 32,
            '6-2-2': 33,
            '2-4-4': 34,
            '2-7-1': 35,
            '3-3-4': 36
        }
        return series.map(mapping)

    @staticmethod
    def _encode_aggregate(series: pd.Series) -> pd.Series:
        # base mapping (tie scores + some common win/loss patterns)
        mapping = {
            # draws
            '0:0': 0, '1:1': 1, '2:2': 2, '3:3': 3, '4:4': 4, '5:5': 5,
            # win/loss (as in the user’s example)
            '0:1': 6, '0:2': 7, '0:3': 8, '0:4': 9, '0:5': 10, '0:6': 11,
            '0:7': 12, '0:8': 13, '0:9': 14, '0:13': 15,
            '1:2': 16, '1:3': 17, '1:4': 18, '1:5': 19, '1:6': 20, '1:7': 21, '1:8': 22,
            '2:3': 23, '2:4': 24, '2:5': 25, '2:6': 26, '2:7': 27, '2:8': 28,
            '3:4': 29, '3:5': 30, '3:6': 31, '3:7': 32,
            '4:5': 33, '4:6': 34,
            '9:1': 35, '9:2': 36, '10:0': 37, '10:1': 38, '10:2': 39,
        }
        # add mirrored keys automatically (e.g., 2:1 mirrors 1:2)
        extras = {}
        for k, v in list(mapping.items()):
            a, b = k.split(":")
            mirror = f"{b}:{a}"
            if mirror not in mapping:
                extras[mirror] = v
        mapping.update(extras)
        return series.map(mapping)

    # ----------- loaders -----------
    def load_offline(self):
        """Read all CSVs present in the transfermarkt folder that we care about."""
        def maybe_read(name):
            path = os.path.join(self.base_path, name)
            return pd.read_csv(path) if os.path.exists(path) else None

        self.appearances = maybe_read("appearances.csv")
        self.club_games = maybe_read("club_games.csv")
        self.clubs = maybe_read("clubs.csv")
        self.competitions = maybe_read("competitions.csv")
        self.game_events = maybe_read("game_events.csv")
        self.game_lineups = maybe_read("game_lineups.csv")
        self.games = maybe_read("games.csv")
        self.player_valuations = maybe_read("player_valuations.csv")
        self.players = maybe_read("players.csv")
        self.transfers = maybe_read("transfers.csv")

    # ----------- cleaning -----------
    def clean(self):
        """Apply the transformation steps you showed: club_games + games."""
        # --- club_games ---
        if self.club_games is None:
            raise FileNotFoundError("club_games.csv not found in transfermarkt raw folder.")
        cg = self.club_games.copy()

        # drop unneeded columns
        drop_cols_cg = ["own_manager_name", "opponent_manager_name"]
        cg = cg.drop(columns=[c for c in drop_cols_cg if c in cg.columns], errors="ignore")

        # map hosting to 1/0
        if "hosting" in cg.columns:
            cg["hosting"] = self._map_hosting(cg["hosting"])

        # remove rows with NA in own_position (mirrors your example)
        if "own_position" in cg.columns:
            cg = cg.dropna(subset=["own_position"])

        self.club_games_clean = cg

        # --- games ---
        if self.games is None:
            raise FileNotFoundError("games.csv not found in transfermarkt raw folder.")
        gm = self.games.copy()

        # drop unneeded columns
        drop_cols_gm = [
            "referee", "url", "home_club_name", "away_club_name",
            "home_club_manager_name", "away_club_manager_name", "stadium"
        ]
        gm = gm.drop(columns=[c for c in drop_cols_gm if c in gm.columns], errors="ignore")

        # filter to domestic_league
        if "competition_type" in gm.columns:
            gm = gm[gm["competition_type"] == "domestic_league"].copy()
            gm = gm.drop(columns=["competition_type"])

        # encode formations
        if "home_club_formation" in gm.columns:
            gm["home_club_formation"] = self._map_formations(gm["home_club_formation"])
        if "away_club_formation" in gm.columns:
            gm["away_club_formation"] = self._map_formations(gm["away_club_formation"])

        # normalize round -> digits only
        if "round" in gm.columns and gm["round"].dtype == object:
            gm["round"] = gm["round"].astype(str).str.extract(r"(\d+)")
        # encode aggregate to numeric bucket
        if "aggregate" in gm.columns:
            gm["aggregate"] = self._encode_aggregate(gm["aggregate"])

        self.games_clean = gm

        # --- merge club_games + games on game_id ---
        if "game_id" not in cg.columns or "game_id" not in gm.columns:
            raise KeyError("game_id must exist in both club_games and games to merge.")
        self.games_clubgames = pd.merge(
            cg, gm, on="game_id", how="inner", validate="m:1"  # club perspective → many cg to one game meta
        )

    # ----------- save -----------
    def save_processed(self, out_dir="data/processed/transfermarkt"):
        if self.club_games_clean is None or self.games_clean is None or self.games_clubgames is None:
            raise ValueError("Call clean() before save_processed().")
        os.makedirs(out_dir, exist_ok=True)
        self.club_games_clean.to_csv(os.path.join(out_dir, "club_games_clean.csv"), index=False)
        self.games_clean.to_csv(os.path.join(out_dir, "games_clean.csv"), index=False)
        self.games_clubgames.to_csv(os.path.join(out_dir, "games_clubgames_merged.csv"), index=False)



# -----------------------------
# Helpers
# -----------------------------
def _normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s\-']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _compute_ga90_over_window(app_df, name_norm, start_date, end_date):
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

def _compute_pre_post_perf(app_df, name_norm, tdate, pre_days=365, post_days=365):
    if app_df["date"].dtype != "datetime64[ns]":
        app_df = app_df.copy()
        app_df["date"] = pd.to_datetime(app_df["date"], errors="coerce")

    ga90_pre, pre_min = _compute_ga90_over_window(
        app_df, name_norm, tdate - timedelta(days=pre_days), tdate - timedelta(days=1)
    )
    ga90_post, post_min = _compute_ga90_over_window(
        app_df, name_norm, tdate, tdate + timedelta(days=post_days)
    )
    return ga90_pre, ga90_post, pre_min, post_min

# -----------------------------
# Build Transfermarkt targets
# -----------------------------
def build_targets_fast(tm: TransfermarktDataBase, decline_pct=0.20):
    transfers = tm.transfers.copy()
    players = tm.players.copy()
    clubs = tm.clubs.copy()
    comps = tm.competitions.copy()
    apps = tm.appearances.copy()

    # normalize names
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps["player_name_norm"] = apps["player_name"].astype(str).map(_normalize_name)
    transfers["player_name_norm"] = transfers["player_name"].astype(str).map(_normalize_name)
    transfers["transfer_date"] = pd.to_datetime(
        transfers.get("transfer_date", transfers.get("date")), errors="coerce"
    )

    # ---- join transfers to appearances ----
    merged = pd.merge(
        apps,
        transfers[["player_id", "player_name_norm", "transfer_date"]],
        on=["player_name_norm", "player_id"],
        how="inner"
    )

    # compute goals+assists and minutes
    merged["ga"] = merged[["goals", "assists"]].fillna(0).sum(axis=1)
    merged["minutes"] = merged["minutes_played"].fillna(0)

    # ----------------------------
    # Build 3 strategies
    # ----------------------------
    strategies = {
        # strict (original): 1y window, 450+ minutes required
        "strict": dict(pre_days=365, post_days=365, min_minutes=450),
        # relaxed: 1y window, only 180+ minutes required
        "relaxed": dict(pre_days=365, post_days=365, min_minutes=180),
        # extended: 2y window, 450+ minutes required
        "extended": dict(pre_days=730, post_days=730, min_minutes=450),
    }

    out = []

    for label, cfg in strategies.items():
        pre = merged[
            merged["date"].between(
                merged["transfer_date"] - pd.Timedelta(days=cfg["pre_days"]),
                merged["transfer_date"] - pd.Timedelta(days=1)
            )
        ]
        post = merged[
            merged["date"].between(
                merged["transfer_date"],
                merged["transfer_date"] + pd.Timedelta(days=cfg["post_days"])
            )
        ]

        agg_pre = pre.groupby(["player_id", "transfer_date"]).agg(
            PreMinutes=("minutes", "sum"),
            GA_pre=("ga", "sum")
        )
        agg_post = post.groupby(["player_id", "transfer_date"]).agg(
            PostMinutes=("minutes", "sum"),
            GA_post=("ga", "sum")
        )

        agg = agg_pre.join(agg_post, how="outer").reset_index()

        # compute GA90 + perf change
        agg[f"GA90_pre_{label}"] = np.where(agg["PreMinutes"] > 0, agg["GA_pre"] / agg["PreMinutes"] * 90, np.nan)
        agg[f"GA90_post_{label}"] = np.where(agg["PostMinutes"] > 0, agg["GA_post"] / agg["PostMinutes"] * 90, np.nan)
        agg[f"PerfChange_{label}"] = agg[f"GA90_post_{label}"] - agg[f"GA90_pre_{label}"]

        # decline flag
        agg[f"DeclineFlag_{label}"] = np.where(
            (agg["PreMinutes"] >= cfg["min_minutes"]) &
            (agg["PostMinutes"] >= cfg["min_minutes"]) &
            agg[f"GA90_pre_{label}"].notna() &
            agg[f"GA90_post_{label}"].notna(),
            (agg[f"GA90_post_{label}"] <= (1 - decline_pct) * agg[f"GA90_pre_{label}"]).astype(int),
            np.nan
        )

        out.append(agg)

    # merge all strategy outputs
    agg_all = out[0]
    for extra in out[1:]:
        agg_all = pd.merge(agg_all, extra, on=["player_id", "transfer_date"], how="outer")

    # join back to transfers
    result = pd.merge(transfers, agg_all, on=["player_id", "transfer_date"], how="left")

    # --- add club/league info ---
    clubs_small = clubs.rename(columns={"domestic_competition_id": "league_code"})[
        ["club_id", "name", "league_code"]
    ]
    club_lut = dict(zip(clubs_small["club_id"], clubs_small["name"]))
    league_lut = dict(zip(comps["competition_id"], comps["name"])) if "competition_id" in comps.columns else {}

    result["from_club_name"] = result["from_club_id"].map(club_lut)
    result["to_club_name"] = result["to_club_id"].map(club_lut)

    if "domestic_competition_id" in clubs.columns:
        club_league = clubs.set_index("club_id")["domestic_competition_id"]
        result["from_league"] = result["from_club_id"].map(club_league)
        result["to_league"] = result["to_club_id"].map(club_league)
    else:
        result["from_league"] = None
        result["to_league"] = None

    result["from_league_name"] = result["from_league"].map(league_lut)
    result["to_league_name"] = result["to_league"].map(league_lut)

    # --- add player metadata ---
    result = result.merge(players, on="player_id", how="left", suffixes=("", "_player"))

    return result


# -----------------------------
# Main
# -----------------------------
def main():
    merged_csv = "data/processed/merged/players_transfer_outcomes.csv"
    os.makedirs(os.path.dirname(merged_csv), exist_ok=True)

    # --- FBref ---
    fb = FBrefDataBase(base_path="data/raw/fbref")
    fb.load_offline()
    fb.combine()
    fbref_feats = fb.all_players.copy()
    fbref_feats["player_name_norm"] = fbref_feats["Name"].astype(str).map(_normalize_name)

    # --- Transfermarkt ---
    tm = TransfermarktDataBase(base_path="data/raw/transfermarkt")
    tm.load_offline()
    targets = build_targets_fast(tm)   # ✅ fast version

    if targets.empty:
        print("No transfer records found — merged dataset will not be created.")
        return

    # --- Merge ---
    print("Merging FBref and Transfermarkt...")

    # after merging
    merged = targets.merge(fbref_feats, on="player_name_norm", how="left")

    # ✅ keep only rows that actually matched FBref data
    merged = merged.dropna(subset=["Name", "Attribute Vector", "Percentiles", "Position"], how="any")

    # ======================================
    # Drop repetitive columns: keep EXTENDED only
    # ======================================
    drop_cols = [
        "PreMinutes_x", "GA_pre_x", "PostMinutes_x", "GA_post_x",
        "GA90_pre_strict", "GA90_post_strict", "PerfChange_strict", "DeclineFlag_strict",
        "PreMinutes_y", "GA_pre_y", "PostMinutes_y", "GA_post_y",
        "GA90_pre_relaxed", "GA90_post_relaxed", "PerfChange_relaxed", "DeclineFlag_relaxed"
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    
    # Rename extended columns to standard names
    rename_map = {
        "PreMinutes": "PreMinutes",
        "GA_pre": "GA_pre",
        "PostMinutes": "PostMinutes",
        "GA_post": "GA_post",
        "GA90_pre_extended": "GA90_pre",
        "GA90_post_extended": "GA90_post",
        "PerfChange_extended": "PerfChange",
        "DeclineFlag_extended": "DeclineFlag"
    }
    merged = merged.rename(columns=rename_map)


    # ======================================
    # Keep only rows with non-null target
    # ======================================
    merged = merged[merged["DeclineFlag"].notnull()].copy()

    # --- Save final cleaned dataset ---
    merged.to_csv(merged_csv, index=False)
    print(f"Saved merged dataset with target not null:\n {merged_csv} "
          f"(rows={merged.shape[0]}, cols={merged.shape[1]})")

    # ======================================
    # Checks: preview, columns, nulls
    # ======================================
    pd.set_option("display.max_columns", 100)
    print("\n=== Preview of merged dataset ===")
    print(merged.head())

    print("\n=== Column names ===")
    print(list(merged.columns))

    print("\n=== Null values per column ===")
    print(merged.isnull().sum())

    print("\n=== Percentage of nulls per column ===")
    print((merged.isnull().mean() * 100).round(2))



if __name__ == "__main__":
    main()
