import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_api.apisrc.core.database import get_db, SessionLocal
from fast_api.apisrc.utils.optimization_utils import load_models_from_db, forecast_48h
from fast_api.apisrc.utils.weather_utils import AH_gm3_from_T_RH, reconcile_environmental_data
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AH_FEATURES_TRAIN = [
    "AH", "AH_lag1", "AH_lag2", "AH_lag3",
    "Tin", "Tout", "AH_out",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season", "hvac_mode"
]

AH_FEATURES_DB = [
    "ah", "ah_lag1", "ah_lag2", "ah_lag3",
    "tin", "tout", "ah_out",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season", "hvac_mode"
]

def transform_ah(X, scaler):
    X = np.asarray(X)
    Xr = X[:, AH_INDEX_MAP]  # reorder to training order

    frozen = set(scaler.frozen_idx)
    non_frozen = [i for i in range(Xr.shape[1]) if i not in frozen]

    Xr[:, non_frozen] = scaler.transform(Xr[:, non_frozen])

    # map back to DB order
    inv = np.argsort(AH_INDEX_MAP)
    return Xr[:, inv]

# mapping: training index → db index
AH_INDEX_MAP = [AH_FEATURES_DB.index(c.lower()) for c in AH_FEATURES_TRAIN]

def build_features(full_timeline: pd.DataFrame) -> pd.DataFrame:
    df = full_timeline.copy()

    # ------------------
    # Absolute humidity
    # ------------------
    df["ah"] = AH_gm3_from_T_RH(df["tin"], df["rh"])
    df["ah_lag1"] = df["ah"].shift(1)
    df["ah_lag2"] = df["ah"].shift(2)
    df["ah_lag3"] = df["ah"].shift(3)

    #training had ah_out divided by 100 somewhere. This is to match that for now.
    df["ah_out"] = AH_gm3_from_T_RH(df["tout"], df["rh_out"]) / 100.0

    # ------------------
    # Time encodings
    # ------------------

    # df["season"] = np.where(df["month"].between(5, 10), 0.5, 1.0)

    # ------------------
    # Solar features
    # ------------------
    df["SW1h"] = df["sw_out"].rolling(window=2, min_periods=1).mean()
    df["SW3h"] = df["sw_out"].rolling(window=6, min_periods=1).mean()

    # ------------------
    # Diffs & rolling
    # ------------------
    df["tin_diff"] = df["tin"].diff()
    df["tout_diff"] = df["tout"].diff()

    df["tin_ma3"] = df["tin"].rolling(window=3, min_periods=1).mean()
    df["tout_ma3"] = df["tout"].rolling(window=3, min_periods=1).mean()

    return df


def build_df_from_db(db, site_id: int) -> pd.DataFrame:
    """
    Build the exact dataframe expected by forecast_48h.
    """
    reconcile_environmental_data(
        db=db, site_id=site_id, start_ts=datetime(2026, 1, 1), end_ts=datetime(2026, 1, 15), resolution="30min", fill_nulls_only=False, dry_run=False)

    rows = db.execute(
        text(
            """
            SELECT
                cd.timestamp,
                cd.tin        AS "tin",
                cd.rh         AS "rh",
                cd.hvac_mode  AS "hvac_mode",
                e.tout        AS "tout",
                e.rh_out      AS "rh_out",
                e.sw_out      AS "sw_out"
            FROM comfort_data cd
            JOIN environmental_data e
            ON e.site_id = cd.site_id
            AND e.timestamp = cd.timestamp
            WHERE cd.site_id = :site_id
            AND cd.timestamp >= :ts_start
            AND cd.timestamp <  :ts_end
            ORDER BY cd.timestamp
            """
        ),
        {
            "site_id": site_id,
            "ts_start": "2026-01-05",
            "ts_end": "2026-01-09",
        },
    ).mappings().all()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data returned from DB")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    # --- time features (must match notebook) ---
    idx = df.index
    df["hour"] = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month"] = idx.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # season encoding exactly like notebook
    df["season"] = ((df["month"] >= 11) | (df["month"] <= 4)).astype(int)
    df = build_features(df)


    return df


def main():
    site_id = 1

    db = SessionLocal()
    try:
        df = build_df_from_db(db, site_id)

        # choose a safe start index (enough history + future)
        start_idx = 100
        assert start_idx + 48 < len(df), "Not enough future data"

        # hardcoded HVAC schedule
        hvac = np.zeros(48, dtype=int)
        hvac[0:6] = 2   # force HIGH for first 3 hours
        print(df[["ah", "ah_out", "rh"]].iloc[start_idx-30:start_idx].describe())
        print(df[["ah", "ah_out", "rh"]].iloc[start_idx:start_idx+10])
        # load models
        models = load_models_from_db(db, site_id, DEVICE)
    

        # inject HVAC into future horizon
        df_temp = df.copy()
        for t in range(48):
            df_temp.iloc[start_idx + t, df_temp.columns.get_loc("hvac_mode")] = hvac[t]

        # run forecast
        out = forecast_48h(
            df=df_temp,
            start_idx=start_idx,
            models=models,
            horizon=48,
            device=DEVICE,
        )

        tin = out["Tin_pred"].values
        rh = out["RH_pred"].values

        print("=== ML ROLLOUT TEST ===")
        print("Initial Tin:", df["tin"].iloc[start_idx - 1])
        print("Tin first 10 steps:", np.round(tin[:10], 2))
        print("RH first 10 steps:", np.round(rh[:10], 1))
        print("Tin delta after HVAC ON:", tin[5] - tin[0])

        if tin[5] <= tin[0]:
            print("❌ Tin did NOT increase during HVAC ON")
        else:
            print("✅ Tin increases when HVAC ON")

    finally:
        db.close()


if __name__ == "__main__":
    main()