# Build & Clean Report
- Timestamp: 2025-09-24T21:45:19.968047+00:00
- Loading FBref features from: data/processed/fbref/fbref.csv
- FBref features shape: (1980, 5)
- Building targets from Transfermarkt raw dir: data/raw/transfermarkt
- transfers.csv columns: ['player_id', 'transfer_date', 'transfer_season', 'from_club_id', 'to_club_id', 'from_club_name', 'to_club_name', 'transfer_fee', 'market_value_in_eur', 'player_name']
- detected -> player: player_name, date: transfer_date, from_id: from_club_id, to_id: to_club_id, from_name: from_club_name, to_name: to_club_name, player_id: player_id
- sample from_league codes: ['BE1', 'FR1', 'GR1', 'IT1', 'L1', 'PO1', 'RU1', 'SC1', 'TR1', 'UKR1']
- sample to_league codes: ['BE1', 'ES1', 'FR1', 'GB1', 'GR1', 'IT1', 'L1', 'PO1', 'TR1', 'UKR1']
- Auto-derived European domestic league codes: ['BE1', 'DK1', 'ES1', 'FR1', 'GB1', 'GR1', 'IT1', 'L1', 'NL1', 'PO1', 'RU1', 'SC1', 'TR1', 'UKR1']
- Auto 'emerging' (Europe minus top-5) → to_tiers=['BE1', 'DK1', 'GR1', 'NL1', 'PO1', 'RU1', 'SC1', 'TR1', 'UKR1']
- qualifying transfers after tier filter: 2056
- built targets: (2056, 17)
- rows with sufficient data: 886
- % DeclineFlag_1y set: 43.09%
- Saved merged dataset: data/processed/merged/players_transfer_outcomes.csv  (rows=2056, cols=21)
- Cleaning merged dataset (winsorize=True, iqr_k=1.5, corr_thresh=0.97)
- Inferred numeric columns: 11
- Inferred categorical columns: 10
- Removed duplicate rows: 0
- Dropping all-NaN numeric columns: ['FBref_Percentile_Mean', 'FBref_AttrVec_Mean']
- All-NaN categorical columns will be imputed with '__NA__': ['fbref_name', 'Position']
- Outlier flags created for 9 numeric columns (total flagged cells: 1309)
- Winsorized numeric columns using IQR k=1.5 (values capped: 1230)
- Saved cleaned CSV: data/processed/merged_clean.csv
- Saved model-ready CSV: data/processed/merged_model_ready.csv
- Saved pipeline: artifacts/preprocessor.pkl

## Summary
```json
{
  "start_rows": 2056,
  "start_cols": 21,
  "end_rows": 2056,
  "end_cols_clean": 28,
  "end_cols_model": 5385,
  "dropped_high_corr": [],
  "numeric_cols_used": [
    "GA90_pre",
    "GA90_post",
    "PerfChange_1y",
    "DeclineFlag_1y",
    "PreMinutes",
    "PostMinutes",
    "InsufficientPreData",
    "InsufficientPostData",
    "UsedPostWindowDays"
  ],
  "categorical_cols_used": [
    "player_name",
    "player_name_norm",
    "transfer_date",
    "from_league",
    "to_league",
    "fbref_name",
    "Position",
    "player_id",
    "from_club_id",
    "to_club_id"
  ],
  "dropped_all_nan_numeric": [
    "FBref_Percentile_Mean",
    "FBref_AttrVec_Mean"
  ],
  "all_nan_categorical_imputed": [
    "fbref_name",
    "Position"
  ]
}
```