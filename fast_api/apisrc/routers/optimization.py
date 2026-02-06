import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import text, select
from fast_api.apisrc.core.database import get_db, SessionLocal
from fast_api.apisrc.routers import comfort
from fast_api.apisrc.utils.disaggregation_utils import run_disaggregation_for_site_util, run_hvac_disaggregation
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from fast_api.apisrc.core.models import OptimizationRun
from fast_api.apisrc.utils.weather_utils import reconcile_environmental_data, fetch_environmental_data, interpolate_value
from fast_api.apisrc.core.models import OptimizationData
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet,
    NonNegativeReals, Binary, Reals, SolverFactory, minimize, value
)
from fast_api.apisrc.utils.optimization_utils import (
    RCConfig,
    fit_rc_by_thermal_regime,
    optimize_schedule_with_rc,
    repair_to_feasible,
    harden_schedule_until_comfort,
    enforce_min_on_duration, compute_comfort_percent,
    rollout_48_steps_backend
)
import torch
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List
import warnings
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

class OptimizationRunRequest(BaseModel):
    manual_pv_48: Optional[
        Annotated[
            List[int],
            Field(min_length=48, max_length=48)
        ]
    ] = None

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
logger.setLevel(logging.INFO)

# ensure logs go to terminal even if uvicorn already configured logging
if not logger.handlers:
    h = logging.StreamHandler()  # stdout
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)

logger.propagate = False  # avoid duplicate logs if uvicorn root handlers also print

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )

router = APIRouter(prefix="/optimize", tags=["optimize"])


def run_hvac_disaggregation_for_site(
    site_id: int,
):
    # -----------------------------
    # 1. Load required data
    # -----------------------------
    db = SessionLocal()  # New session for background task
    try:
        sql = text("""
            SELECT
                cd.timestamp,
                cd.hvac_mode,
                c.value          AS energy_consumption,
                e.tout           AS tout
            FROM comfort_data cd
            JOIN consumption_data c
            ON c.site_id = cd.site_id
            AND c.timestamp = cd.timestamp
            JOIN environmental_data e
            ON e.site_id = cd.site_id
            AND e.timestamp = cd.timestamp
            WHERE cd.site_id = :site_id
            ORDER BY cd.timestamp
        """)

        rows = db.execute(sql, {"site_id": site_id}).mappings().all()

        if not rows:
            logger.warning(
                "HVAC disaggregation: no data available for site %s",
                site_id,
            )
            return

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        # if df.index.tz is None:
        #     raise RuntimeError("HVAC disaggregation received naive timestamps; expected UTC-aware")
        # -----------------------------
        # 2. Run disaggregation
        # -----------------------------
        site_sql = text("""
            SELECT
                temp_low_band,
                temp_high_band,
                temp_low_bin,
                temp_high_bin,
                is_residential,
                active_ratio,
                high_ratio,
                min_high,
                q,
                min_days_per_bin
            FROM site
            WHERE id = :site_id
        """)

        site_row = db.execute(site_sql, {"site_id": site_id}).mappings().first()


        results = run_hvac_disaggregation(
            df,
            load_col="energy_consumption",
            temp_col="tout",
            neutral_band=(site_row["temp_low_band"], site_row["temp_high_band"]),
            temp_bins=np.arange(site_row["temp_low_bin"], site_row["temp_high_bin"], 1),
            is_residential=site_row["is_residential"],
            active_ratio=site_row["active_ratio"],
            high_ratio=site_row["high_ratio"],
            min_high=site_row["min_high"],
            q=site_row["q"],
            min_days_per_bin=site_row["min_days_per_bin"],
        )

        df_out = results["df_with_hvac"]

        # -----------------------------
        # 3. Overwrite hvac_mode in DB
        # -----------------------------
        update_sql = text("""
            UPDATE comfort_data
            SET hvac_mode = :hvac_mode
            WHERE site_id = :site_id
            AND timestamp = :timestamp
        """)

        payload = [
            {
                "site_id": site_id,
                "timestamp": ts,
                "hvac_mode": int(mode),
            }
            for ts, mode in df_out["hvac_mode"].items()
        ]

        if payload:
            db.execute(update_sql, payload)
            db.commit()
        # -----------------------------
    # 4. Return summary
    # -----------------------------
    except Exception as exc:
        db.rollback()
        logger.exception(
            "HVAC disaggregation FAILED for site %s: %s",
            site_id,
            exc,
        )

    finally:
        db.close()

def _es_hPa_from_Tc(Tc: float) -> float:
    """Saturation vapor pressure in hPa at air temperature Tc (°C)."""
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

def AH_gm3_from_T_RH(Tc: float, RH_percent: float) -> float:
    """
    Absolute humidity [g/m³] from temperature (°C) and RH (%).
    AH = 216.7 * e / (T+273.15), with e = RH/100 * es(T).
    """
    T = np.asarray(Tc, dtype=np.float32)
    RHf = np.clip(np.asarray(RH_percent, dtype=np.float32) / 100.0, 0.0, 1.0)  # <-- vectorized clamp
    es = _es_hPa_from_Tc(T)  # assumes this already handles vector inputs (it does in your file)
    e = RHf * es
    AH = 216.7 * e / (T + 273.15)
    if isinstance(Tc, pd.Series):
        return pd.Series(AH, index=Tc.index, name="ah")
    return AH

def fit_rc_from_history(history_df: pd.DataFrame) -> dict:
    df = history_df.copy()
    req = {"tin", "tout", "hvac_mode"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"RC fit requires columns {sorted(req)}")

    df = df.dropna(subset=["tin", "tout", "hvac_mode"]).sort_index()
    if len(df) < 50:
        raise RuntimeError("Insufficient clean history for RC fit")

    Tin_t = df["tin"].values[:-1]
    Tin_next = df["tin"].values[1:]
    Tout_t = df["tout"].values[:-1]
    mode_t = df["hvac_mode"].values[:-1].astype(int)

    u_low = (mode_t == 1).astype(float)
    u_high = (mode_t == 2).astype(float)

    X = np.column_stack([Tin_t, Tout_t, u_low, u_high, np.ones_like(Tin_t)])
    theta, *_ = np.linalg.lstsq(X, Tin_next, rcond=None)
    a, b_tout, c_low, c_high, d = theta.tolist()

    a = float(np.clip(a, 0.7, 0.999))
    return {"a": float(a), "b_tout": float(b_tout), "c_low": float(c_low), "c_high": float(c_high), "d": float(d)}

def simulate_rc(*, rc: dict, tin0: float, tout_48: np.ndarray, hvac_mode_48: np.ndarray) -> np.ndarray:
    H = len(tout_48)
    tin = np.zeros(H + 1, dtype=float)
    tin[0] = float(tin0)
    for t in range(H):
        u_low = 1.0 if int(hvac_mode_48[t]) == 1 else 0.0
        u_high = 1.0 if int(hvac_mode_48[t]) == 2 else 0.0
        tin[t + 1] = (
            rc["a"] * tin[t]
            + rc["b_tout"] * float(tout_48[t])
            + rc["c_low"] * u_low
            + rc["c_high"] * u_high
            + rc["d"]
        )
    return tin[1:]

# def enforce_min_on_duration(hvac: np.ndarray, min_len: int = 2) -> np.ndarray:
#     h = np.asarray(hvac, dtype=int).copy()
#     if min_len <= 1:
#         return h
#     n = len(h)
#     i = 0
#     while i < n:
#         if h[i] == 0:
#             i += 1
#             continue
#         j = i
#         while j < n and h[j] == h[i]:
#             j += 1
#         if (j - i) < min_len:
#             h[i:j] = 0
#         i = j
#     return h


def solve_rc_milp_pyomo(
    *,
    tin0: float,
    tout: np.ndarray,          # shape (48,)
    pv: np.ndarray,            # shape (48,), 0/1
    season: float,             # 0.5 summer, 1.0 winter (as in notebook)
    params: dict,              # RC params
    Tmin: float,
    Tmax: float,
    time_limit_sec: int = 60,
) -> dict:

    H = len(tout)
    assert H == 48, "Expected 48-step horizon"

    m = ConcreteModel()

    m.T = RangeSet(0, H)
    m.K = RangeSet(0, H - 1)

    # ------------------
    # Decision variables
    # ------------------
    m.Tin = Var(m.T, domain=Reals)
    m.u_low = Var(m.K, domain=Binary)
    m.u_high = Var(m.K, domain=Binary)

    # ------------------
    # Initial condition
    # ------------------
    m.Tin[0].fix(tin0)

    # ------------------
    # RC dynamics
    # ------------------
    a = params["a"]
    b = params["b_tout"]
    c_low = params["c_low"]
    c_high = params["c_high"]
    d = params.get("d", 0.0)

    def rc_dyn(m, k):
        return (
            m.Tin[k + 1]
            == a * m.Tin[k]
            + b * tout[k]
            + c_low * m.u_low[k]
            + c_high * m.u_high[k]
            + d
        )

    m.rc_dyn = Constraint(m.K, rule=rc_dyn)

    # ------------------
    # HVAC mode exclusivity
    # ------------------
    def hvac_excl(m, k):
        return m.u_low[k] + m.u_high[k] <= 1

    m.hvac_excl = Constraint(m.K, rule=hvac_excl)

    # ------------------
    # Comfort bounds
    # ------------------
    def comfort_lo(m, k):
        return m.Tin[k + 1] >= Tmin

    def comfort_hi(m, k):
        return m.Tin[k + 1] <= Tmax

    m.comfort_lo = Constraint(m.K, rule=comfort_lo)
    m.comfort_hi = Constraint(m.K, rule=comfort_hi)

    # ------------------
    # Objective (PV utilization)
    # ------------------
    w_high = params.get("w_high", 1.0)
    w_low = params.get("w_low", 0.3)
    w_grid = params.get("w_grid", 1.0)

    def obj(m):
        cost = 0.0
        for k in m.K:
            hvac_power = w_low * m.u_low[k] + w_high * m.u_high[k]
            grid_penalty = (1 - pv[k]) * hvac_power
            cost += w_grid * grid_penalty
        return cost

    m.obj = Objective(rule=obj, sense=minimize)

    # ------------------
    # Solve
    # ------------------
    solver = SolverFactory("highs")
    solver.options["time_limit"] = time_limit_sec

    result = solver.solve(m, tee=False)

    if str(result.solver.termination_condition).lower() not in {"optimal", "feasible"}:
        raise RuntimeError(f"MILP failed: {result.solver.termination_condition}")

    # ------------------
    # Extract solution
    # ------------------
    hvac_mode = np.zeros(H, dtype=int)
    tin_rc = np.zeros(H, dtype=float)

    for k in range(H):
        if value(m.u_high[k]) > 0.5:
            hvac_mode[k] = 2
        elif value(m.u_low[k]) > 0.5:
            hvac_mode[k] = 1
        else:
            hvac_mode[k] = 0

        tin_rc[k] = value(m.Tin[k + 1])

    return {
        "hvac_mode": hvac_mode,
        "Tin_rc": tin_rc,
    }

def load_pv_indicator_48(*, db: Session, site_id: int, horizon_index: pd.DatetimeIndex) -> np.ndarray:
    ts0 = horizon_index[0].to_pydatetime()
    ts1 = (horizon_index[-1] + pd.Timedelta(minutes=30)).to_pydatetime()

    prod_rows = db.execute(
        text("""
            SELECT timestamp, value
            FROM forecasted_production_data
            WHERE site_id = :site_id
              AND timestamp >= :ts0
              AND timestamp < :ts1
        """),
        {"site_id": site_id, "ts0": ts0, "ts1": ts1},
    ).mappings().all()

    cons_rows = db.execute(
        text("""
            SELECT timestamp, value
            FROM forecasted_consumption_data
            WHERE site_id = :site_id
              AND timestamp >= :ts0
              AND timestamp < :ts1
        """),
        {"site_id": site_id, "ts0": ts0, "ts1": ts1},
    ).mappings().all()

    if not prod_rows or not cons_rows:
        return np.zeros(len(horizon_index), dtype=int)

    prod = {pd.to_datetime(r["timestamp"]): float(r["value"]) for r in prod_rows}
    cons = {pd.to_datetime(r["timestamp"]): float(r["value"]) for r in cons_rows}

    pv = np.zeros(len(horizon_index), dtype=int)
    for i, ts in enumerate(horizon_index):
        p = prod.get(ts)
        c = cons.get(ts)
        pv[i] = 1 if (p is not None and c is not None and p >= c) else 0
    return pv

#builds features for optimization model
def build_features(full_timeline: pd.DataFrame) -> pd.DataFrame:
    df = full_timeline.copy()
    # ------------------
    # Absolute humidity
    # ------------------
    df["ah"] = AH_gm3_from_T_RH(df["tin"], df["rh"])
    df["ah_lag1"] = df["ah"].shift(1)
    df["ah_lag2"] = df["ah"].shift(2)
    df["ah_lag3"] = df["ah"].shift(3)

    df["ah_out"] = AH_gm3_from_T_RH(df["tout"], df["rh_out"]) / 100.0


    # ------------------
    # Time encodings
    # ------------------
    idx = df.index
    idx = pd.to_datetime(full_timeline.index)


    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df["month"] = idx.month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    df["season"] = np.where(df["month"].between(5, 10), 0.5, 1.0)

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

def placeholder_optimize_next_24h(
    *,
    horizon_index: pd.DatetimeIndex,
    last_known_tin: float | None,
    last_known_rh: float | None,
    last_known_comfort: float | None,
) -> pd.DataFrame:
    
    tin = float(last_known_tin) if last_known_tin is not None else 22.0
    rh = float(last_known_rh) if last_known_rh is not None else 50.0
    comfort = float(last_known_comfort) if last_known_comfort is not None else 50.0

    df_out = pd.DataFrame(index=horizon_index)
    df_out["hvac_mode"] = 0
    df_out["tin"] = tin
    df_out["rh"] = rh
    df_out["comfort_index"] = comfort
    return df_out


@router.post("/{site_id}/disaggregation", status_code=202)
async def trigger_hvac_disaggregation(
    site_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Validate site exists
    exists = db.execute(
        text("SELECT 1 FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).scalar()

    if not exists:
        raise HTTPException(status_code=404, detail="Site not found")

    # IMPORTANT:
    # We pass a NEW session into the background task
    # Never reuse request-scoped session
    background_tasks.add_task(
        run_hvac_disaggregation_for_site,
        site_id,
    )

    return {
        "site_id": site_id,
        "status": "scheduled",
        "message": "HVAC disaggregation started in background",
    }

RES_MIN = 30
HORIZON_STEPS = 48
LSTM_MAX_WINDOW_STEPS = 24
WARMUP_BUFFER_STEPS = 24
SUMMER_MONTHS = {4, 5, 6, 7, 8, 9, 10}   # Apr–Oct
WINTER_MONTHS = {11, 12, 1, 2, 3} 


def reload_rc_history_df(db, site_id, index):
    rows = db.execute(
        text("""
            SELECT
                cd.timestamp,
                cd.tin AS tin,
                cd.hvac_mode,
                e.tout AS tout
            FROM comfort_data cd
            JOIN environmental_data e
              ON e.site_id = cd.site_id
             AND e.timestamp = cd.timestamp
            WHERE cd.site_id = :site_id
              AND cd.timestamp >= :start_ts
              AND cd.timestamp <= :end_ts
            ORDER BY cd.timestamp
        """),
        {
            "site_id": site_id,
            "start_ts": index.min(),
            "end_ts": index.max(),
        },
    ).mappings().all()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")

def run_optimization_for_site(run_id: int):
    """
    Background task:
    - loads run + marks running
    - loads required data (history + forecast env)
    - builds features
    - generates schedule
    - persists OptimizationData
    - marks succeeded/failed
    """
    db = SessionLocal()
    run: OptimizationRun | None = None

    try:
        # -----------------------------------
        # 1) Load run, validate, mark running
        # -----------------------------------
        run = db.get(OptimizationRun, run_id)
        if run is None:
            logger.error("OptimizationRun %s not found", run_id)
            return

        run.status = "running"
        run.error_message = None
        run.started_at = datetime.now()
        db.commit()

        site_id = run.site_id
        start_time = run.start_time
        end_time = run.end_time

        start_month = start_time.month

        if start_month in SUMMER_MONTHS:
            rc_months = SUMMER_MONTHS
            rc_season = "summer"
        else:
            rc_months = WINTER_MONTHS
            rc_season = "winter"

        # -----------------------------------
        # 2) Define horizon timestamps (30-min)
        # -----------------------------------
        horizon_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq="30min",
            inclusive="left",
            # tz="UTC",
        )

        if len(horizon_index) != 48:
            # Defensive: if start/end were not aligned, this catches it.
            raise RuntimeError(f"Expected 48 steps for 24h@30min, got {len(horizon_index)}")

        # -----------------------------------
        # 3) Load history window (last 12h or 24h)
        # -----------------------------------
        warmup_steps = LSTM_MAX_WINDOW_STEPS + WARMUP_BUFFER_STEPS

        history_start = start_time - timedelta(minutes=RES_MIN * warmup_steps)

        history_rows = db.execute(
            text(
                """
                SELECT
                    cd.timestamp,
                    cd.tin      AS tin,
                    cd.rh       AS rh,
                    cd.comfort_index AS comfort_index,
                    cd.hvac_mode AS hvac_mode,
                    e.tout      AS tout,
                    e.sw_out    AS sw_out,
                    e.rh_out    AS rh_out
                FROM comfort_data cd
                LEFT JOIN environmental_data e
                  ON e.site_id = cd.site_id
                 AND e.timestamp = cd.timestamp
                WHERE cd.site_id = :site_id
                  AND cd.timestamp >= :history_start
                  AND cd.timestamp < :start_time
                ORDER BY cd.timestamp
                """
            ),
            {"site_id": site_id, "history_start": history_start, "start_time": start_time},
        ).mappings().all()

        history_df = pd.DataFrame(history_rows)
        if not history_df.empty:
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df = history_df.set_index("timestamp")

        if history_df.empty:
            run.status = "failed"
            run.error_message = "No historical data available for optimization"
            db.commit()
            return
        
        if len(history_df) < LSTM_MAX_WINDOW_STEPS:
            run.status = "failed"
            run.error_message = (
                f"Insufficient history for LSTM warmup: "
                f"need {LSTM_MAX_WINDOW_STEPS} timesteps, got {len(history_df)}"
            )
            db.commit()
            return
        bad_hist_ts = history_df.index[history_df["tin"].isna()]
        if len(bad_hist_ts) > 0:
            print("HISTORY missing tin timestamps:", bad_hist_ts.tolist())
            print(history_df.loc[bad_hist_ts, ["tin", "rh", "hvac_mode", "comfort_index", "tout"]])
        # -----------------------------------
        # 4) Load environmental forecast for horizon
        # -----------------------------------
      
        site_row = db.execute(
            text(
                """
                SELECT latitude, longitude
                FROM sites
                WHERE id = :site_id
                """
            ),
            {"site_id": site_id},
        ).fetchone()

        if site_row is None:
            run.status = "failed"
            run.error_message = "Site not found while fetching weather"
            db.commit()
            return

        latitude = site_row.latitude
        longitude = site_row.longitude
        weather_hourly = fetch_environmental_data(
            latitude=latitude,
            longitude=longitude,
            start_ts=start_time - timedelta(hours=1),
            end_ts=end_time + timedelta(hours=1), 
            source="forecast",
        )
        if not weather_hourly:
            run.status = "failed"
            run.error_message = "No forecast weather data returned for optimization horizon"
            db.commit()
            return
        
        weather_df = (
            pd.DataFrame.from_dict(weather_hourly, orient="index")
            .sort_index()
        )

        weather_df.index = pd.to_datetime(weather_df.index)
        hourly_ts = sorted(weather_hourly.keys())

        if not hourly_ts:
            run.status = "failed"
            run.error_message = "No hourly weather timestamps available for interpolation"
            db.commit()
            return

        hourly_series = {
            "tout": {ts: weather_hourly[ts]["tout"] for ts in hourly_ts},
            "rh_out": {ts: weather_hourly[ts]["rh_out"] for ts in hourly_ts},
            "sw_out": {ts: weather_hourly[ts]["sw_out"] for ts in hourly_ts},
        }

        records = []
     
        for ts in horizon_index:
            records.append(
                {
                    "timestamp": ts,
                    "tout": interpolate_value(
                        target_ts=ts,
                        known_ts=hourly_ts,
                        values=hourly_series["tout"],
                    ),
                    "rh_out": interpolate_value(
                        target_ts=ts,
                        known_ts=hourly_ts,
                        values=hourly_series["rh_out"],
                    ),
                    "sw_out": interpolate_value(
                        target_ts=ts,
                        known_ts=hourly_ts,
                        values=hourly_series["sw_out"],
                    ),
                }
            )

        weather_30min = (
            pd.DataFrame(records)
            .set_index("timestamp")
        )
        if weather_30min["tout"].isna().any():
            run.status = "failed"
            run.error_message = (
                "Forecast Tout missing after interpolation "
                "(hourly weather coverage insufficient)"
            )
            db.commit()
            return
        if weather_30min["rh_out"].isna().any():
            raise RuntimeError(
                "Forecast RH missing after interpolation — cannot run AH model"
            )
            db.commit()
            return
        future_df = weather_30min.copy()

        # Explicitly add columns that will be predicted / decided later
        future_df["tin"] = np.nan
        future_df["rh"] = np.nan
        future_df["comfort_index"] = np.nan
        future_df["hvac_mode"] = np.nan
        future_df['ah_out'] = AH_gm3_from_T_RH(future_df["tout"], future_df["rh_out"])
        full_timeline = pd.concat(
            [history_df, future_df],
            axis=0,
        )            
        full_timeline = build_features(full_timeline)
        start_time = pd.Timestamp(start_time)

        if start_time not in full_timeline.index:
            # snap to nearest 30-min grid safely
            start_time = full_timeline.index[full_timeline.index.searchsorted(start_time)]
        start_idx = full_timeline.index.get_loc(start_time)

        if start_idx < 0 or start_idx + 48 > len(full_timeline):
            raise RuntimeError(
                f"Invalid start_idx={start_idx} for horizon=48 "
                f"(len(full_feat)={len(full_timeline)})"
            )

        for ts, row in full_timeline.iterrows():
            nan_cols = row[row.isna()].index.tolist()

        season_48 = (
            full_timeline["season"]
            .iloc[start_idx : start_idx + 48]
            .astype(float)
            .to_numpy()
        )
        if len(season_48) != 48:
            raise RuntimeError(
                f"season_48 length mismatch: expected 48, got {len(season_48)}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        warmup_feat = full_timeline.loc[full_timeline.index < start_time].tail(LSTM_MAX_WINDOW_STEPS)

        horizon_feat = full_timeline.loc[(full_timeline.index >= start_time) & (full_timeline.index < end_time)]
        tin0 = float(history_df["tin"].iloc[-1])
        rh0 = float(history_df["rh"].iloc[-1]) if pd.notna(history_df["rh"].iloc[-1]) else 50.0
        comfort0 = float(history_df["comfort_index"].iloc[-1]) if pd.notna(history_df["comfort_index"].iloc[-1]) else 50.0

        if run.manual_pv_48 is not None:
            pv_48 = np.asarray(run.manual_pv_48, dtype=int)

            if pv_48.shape != (48,):
                raise RuntimeError("manual_pv_48 must have length 48")

            if not set(np.unique(pv_48)).issubset({0, 1}):
                raise RuntimeError("manual_pv_48 must be binary (0/1)")

            logger.info("Using MANUAL PV override for run %s", run_id)

        else:
            pv_48 = load_pv_indicator_48(
                db=db,
                site_id=site_id,
                horizon_index=horizon_index,
            )
        tout_48 = horizon_feat["tout"].astype(float).values

        season = float(horizon_feat["season"].iloc[0])

        rc_rows = db.execute(
            text("""
                SELECT
                    cd.timestamp,
                    cd.tin,
                    cd.hvac_mode,
                    e.tout
                FROM comfort_data cd
                LEFT JOIN environmental_data e
                ON e.site_id = cd.site_id
                AND e.timestamp = cd.timestamp
                WHERE cd.site_id = :site_id
                AND EXTRACT(MONTH FROM cd.timestamp) = ANY(:rc_months)
                AND cd.timestamp < :start_time
                ORDER BY cd.timestamp
            """),
            {
                "site_id": site_id,
                "rc_months": list(rc_months),
                "start_time": start_time,
            },
        ).mappings().all()
        rc_history_df = pd.DataFrame(rc_rows)

        if not rc_history_df.empty:
            rc_history_df["timestamp"] = pd.to_datetime(
                rc_history_df["timestamp"]
            )
            rc_history_df = rc_history_df.set_index("timestamp")

        if rc_history_df["tout"].isna().any():
            reconcile_environmental_data(
                db=db,
                site_id=site_id,
                start_ts=rc_history_df.index.min(),
                end_ts=rc_history_df.index.max(),
                resolution="30min",
                fill_nulls_only=True,
                dry_run=False,
            )
            rc_history_df = reload_rc_history_df(db, site_id, rc_history_df.index)

        

        # reconcile_environmental_data(
        #     db=db,
        #     site_id=site_id,
        #     start_ts=rc_history_df.index.min(),
        #     end_ts=rc_history_df.index.max(),
        #     resolution=resolution,
        #     fill_nulls_only=True,
        #     dry_run=False,
        # )
        print('NULLS: ', rc_history_df.isna().sum())
        
        nan_tout = rc_history_df[rc_history_df["tout"].isna()]
        if not nan_tout.empty:
            missing_ts = list(nan_tout.index)
            logger.error("[RC DEBUG] Missing tout timestamps (%d): %s", len(missing_ts), missing_ts[:20])

            db_rows = db.execute(
                text("""
                    SELECT timestamp, tout, rh_out, sw_out
                    FROM environmental_data
                    WHERE site_id = :site_id
                    AND timestamp = ANY(:ts)
                    ORDER BY timestamp
                """),
                {"site_id": site_id, "ts": missing_ts},
            ).mappings().all()

            logger.error("[RC DEBUG] Environmental rows for missing timestamps: %s", db_rows[:20])
        nan_rows = rc_history_df[rc_history_df["tout"].isna()]
       
        missing_ts = list(nan_rows.index)

        rows = db.execute(
            text("""
                SELECT timestamp, tout, rh_out, sw_out
                FROM environmental_data
                WHERE site_id = :site_id
                AND timestamp = ANY(:ts)
                ORDER BY timestamp
            """),
            {
                "site_id": site_id,
                "ts": missing_ts,
            },
        ).mappings().all()

        env_df = pd.DataFrame(rows)
       # --- RC HISTORY SANITIZATION ---
        if rc_history_df["tout"].isna().any():
            nan_idx = rc_history_df[rc_history_df["tout"].isna()].index
            total_nans = len(nan_idx)

            if total_nans > 5:
                raise RuntimeError(
                    f"Too many Tout NaNs in RC history ({total_nans}), refusing to proceed"
                )

            logger.warning(
                "[RC FIX] Filling %d missing Tout values inside RC history window via interpolation",
                total_nans,
            )

            # Time interpolation first (best physical assumption)
            rc_history_df["tout"] = rc_history_df["tout"].interpolate(
                method="time",
                limit=5,
            )
            # rc_history_df["sw_out"] = rc_history_df["sw_out"].interpolate(
            #     method="time",
            #     limit=5,
            # )
            # rc_history_df["rh_out"] = rc_history_df["rh_out"].interpolate(
            #     method="time",
            #     limit=5,
            # )

            # If still NaNs (e.g. leading/trailing), forward/backward fill
            rc_history_df["tout"] = rc_history_df["tout"].ffill().bfill()
            # rc_history_df["sw_out"] = rc_history_df["sw_out"].ffill().bfill()
            # rc_history_df["rh_out"] = rc_history_df["rh_out"].ffill().bfill()


            # Final hard check
            if rc_history_df["tout"].isna().any():
                raise RuntimeError("Failed to sanitize Tout for RC fitting")
            
        rc_history_df = rc_history_df.dropna()
        print('NULLS: ', rc_history_df.isna().sum())

        rc_models = fit_rc_by_thermal_regime(rc_history_df)
        Tout_now = float(tout_48[0])

        regime_now = 'cooling' if season == 0.5 else 'heating'
        rc = rc_models[regime_now]

        Tmin = 22.0 if regime_now == 'cooling' else 20.0
        Tmax = 25.0 if regime_now == 'cooling' else 24.0

        cfg_rc = RCConfig(
            horizon=48,
            Tmin=Tmin,
            Tmax=Tmax,
            w_low=1.0,
            w_high=2.2,
            lambda_noPV=1.2,
            lambda_slack=80.0,
            lambda_switch=0.05,
            safety_buffer=0.0,
            solver="highs",
            time_limit_sec=60,
        )

        sol = optimize_schedule_with_rc(
            rc=rc,
            Tin0=tin0,
            Tout_forecast=tout_48,
            pv_forecast=pv_48,
            cfg=cfg_rc,
        )

        hvac_mode = enforce_min_on_duration(sol["hvac_mode"], min_len=2)
        def simulate_fn(schedule: np.ndarray):
            return rollout_48_steps_backend(
                df=full_timeline,
                start_idx=start_idx,
                hvac_mode_48=schedule,
                site_id=site_id,
                db=db,
                device=device,
            )
        
        
        Tin, RH = simulate_fn(hvac_mode)
        print("[COMFORT DEBUG] Tin range: min=%.3f max=%.3f" % (np.nanmin(Tin), np.nanmax(Tin)))
        print("[COMFORT DEBUG] RH  range: min=%.3f max=%.3f" % (np.nanmin(RH), np.nanmax(RH)))

        bad = np.where(~np.isfinite(Tin) | ~np.isfinite(RH))[0]
        if bad.size:
            i = int(bad[0])
            print("[COMFORT DEBUG] Non-finite at i=%d Tin=%r RH=%r" % (i, Tin[i], RH[i]))
            raise RuntimeError("Non-finite Tin/RH before PMV")

        comfort = compute_comfort_percent(Tin, RH, season_48)
        feasible = bool((comfort >= 80.0).all())
        if not feasible:
            hvac_mode, Tin, RH, feasible = repair_to_feasible(
                simulate_fn=simulate_fn,
                pv=pv_48,
                sched_in=hvac_mode,
                season_seq=season_48,
                comfort_min=80.0,
                max_iters=40,
            )
        if not feasible:
            hvac_mode, Tin, RH, feasible = harden_schedule_until_comfort(
                simulate_fn=simulate_fn,
                pv=pv_48,
                sched_in=hvac_mode,
                season_seq=season_48,
                comfort_min=80.0,
                min_on_steps=2,
                max_passes=6,
            )
        min_comfort = float(np.min(compute_comfort_percent(Tin, RH, season_48)))

        out_df = pd.DataFrame(index=horizon_index)

        out_df["hvac_mode"] = hvac_mode.astype(int)
        out_df["tin"] = Tin.astype(float)
        out_df["rh"] = RH.astype(float)
        out_df["comfort_index"] = compute_comfort_percent(
            Tin,
            RH,
            season_48,
        )

        db.execute(text("DELETE FROM optimization_data WHERE run_id = :run_id"), {"run_id": run_id})

        payload = [
            OptimizationData(
                run_id=run_id,
                timestamp=ts.to_pydatetime(),
                tin=float(out_df.at[ts, "tin"]),
                rh=float(out_df.at[ts, "rh"]),
                hvac_mode=int(out_df.at[ts, "hvac_mode"]),
                comfort_index=float(out_df.at[ts, "comfort_index"]),
            )
            for ts in out_df.index
        ]
        db.add_all(payload)

        run.status = "succeeded"
        run.error_message = None
        db.commit()
        return
            

    except Exception as exc:
        db.rollback()
        logger.exception("OptimizationRun %s FAILED: %s", run_id, exc)

        if run is not None:
            try:
                run.status = "failed"
                run.error_message = str(exc)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("Failed to mark OptimizationRun %s as failed", run_id)

    finally:
        db.close()

def ceil_to_half_hour(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minutes = dt.minute
    remainder = minutes % 30

    if remainder == 0:
        return dt

    return dt + timedelta(minutes=(30 - remainder))

@router.post("/{site_id}/run", status_code=202)
async def trigger_optimization_run(
    site_id: int,
    payload: OptimizationRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    
    existing = (
    db.query(OptimizationRun)
    .filter(
        OptimizationRun.site_id == site_id,
        OptimizationRun.status.in_(["queued", "running"]),
    )
    .first()
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Optimization already running for this site",
    )
    
    # Validate site exists
    exists = db.execute(
        text("SELECT 1 FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).scalar()

    if not exists:
        raise HTTPException(status_code=404, detail="Site not found")
    
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    history_start = now - timedelta(hours=48 * 2)  # or reuse your warmup logic
    end_time = now

    reconcile_environmental_data(
        db=db,
        site_id=site_id,
        start_ts=history_start,
        end_ts=end_time,
        resolution='30min',
        fill_nulls_only=True,
        dry_run=False,
    )

    run_disaggregation_for_site_util(db, site_id)

    # Horizon: next 24 hours, aligned to half-hour
    start_time = ceil_to_half_hour(datetime.now())

    end_time = start_time + timedelta(hours=24)
    run = OptimizationRun(
        site_id=site_id,
        start_time=start_time,
        end_time=end_time,
        status="queued",
        error_message=None,
        manual_pv_48=payload.manual_pv_48,
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(run_optimization_for_site, run.id)

    return {
        "run_id": run.id,
        "site_id": site_id,
        "status": run.status,
        "horizon": {"start_time": start_time, "end_time": end_time},
        "message": "Optimization scheduled (placeholder optimizer)",
    }


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT
                id,
                site_id,
                status,
                error_message,
                start_time,
                end_time,
                created_at
            FROM optimization_runs
            WHERE id = :run_id
        """),
        {"run_id": run_id},
    ).mappings().first()

    if row is None:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    return {
        "run_id": row["id"],
        "site_id": row["site_id"],
        "status": row["status"],          # queued | running | succeeded | failed
        "error_message": row["error_message"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "created_at": row["created_at"],
    }

@router.get("/runs/{run_id}/data")
def get_run_data(run_id: int, db: Session = Depends(get_db)):
    exists = db.execute(
        text("SELECT 1 FROM optimization_runs WHERE id = :run_id"),
        {"run_id": run_id},
    ).scalar()

    if not exists:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    rows = db.execute(
        text("""
            SELECT
                timestamp,
                tin,
                rh,
                hvac_mode,
                comfort_index
            FROM optimization_data
            WHERE run_id = :run_id
            ORDER BY timestamp
        """),
        {"run_id": run_id},
    ).mappings().all()

    return {
        "run_id": run_id,
        "count": len(rows),
        "data": rows,
    }

#get a recent successful optimization run for a site (useful for caching data)
@router.get("/{site_id}/latest")
def get_latest_valid_run(site_id: int, db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT
                id,
                site_id,
                status,
                created_at,
                start_time,
                end_time
            FROM optimization_runs
            WHERE site_id = :site_id
              AND status = 'succeeded'
              AND created_at >= NOW() - INTERVAL '6 hours'
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"site_id": site_id},
    ).mappings().first()

    if row is None:
        return {
            "has_recent": False,
            "run_id": None,
        }

    return {
        "has_recent": True,
        "run_id": row["id"],
        "created_at": row["created_at"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
    }

class ForecastRequest(BaseModel):
    start_time: datetime
    hvac_mode_48: List[int] = Field(..., min_lenght=48, max_length=48)

@router.post("/{site_id}/forecast")
def forecast_with_schedule(
    site_id: int,
    req: ForecastRequest,
    db: Session = Depends(get_db),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = pd.Timestamp(req.start_time)
    hvac_mode_48 = np.asarray(req.hvac_mode_48, dtype=int)

    if hvac_mode_48.shape != (48,):
        raise HTTPException(400, "hvac_mode_48 must have exactly 48 values")

    # ---------------------------------------------------
    # 1) Define horizon
    # ---------------------------------------------------
    end_time = start_time + timedelta(hours=24)

    horizon_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="30min",
        inclusive="left",
    )

    if len(horizon_index) != 48:
        raise HTTPException(500, "Horizon misaligned")

    # ---------------------------------------------------
    # 2) Load history window (same as optimization)
    # ---------------------------------------------------
    warmup_steps = LSTM_MAX_WINDOW_STEPS + WARMUP_BUFFER_STEPS
    history_start = start_time - timedelta(minutes=RES_MIN * warmup_steps)

    history_rows = db.execute(
        text(
            """
            SELECT
                cd.timestamp,
                cd.tin      AS tin,
                cd.rh       AS rh,
                cd.comfort_index AS comfort_index,
                cd.hvac_mode AS hvac_mode,
                e.tout      AS tout,
                e.sw_out    AS sw_out,
                e.rh_out    AS rh_out
            FROM comfort_data cd
            LEFT JOIN environmental_data e
              ON e.site_id = cd.site_id
             AND e.timestamp = cd.timestamp
            WHERE cd.site_id = :site_id
              AND cd.timestamp >= :history_start
              AND cd.timestamp < :start_time
            ORDER BY cd.timestamp
            """
        ),
        {"site_id": site_id, "history_start": history_start, "start_time": start_time},
    ).mappings().all()

    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        raise HTTPException(400, "No historical data available")

    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.set_index("timestamp")

    if len(history_df) < LSTM_MAX_WINDOW_STEPS:
        raise HTTPException(400, "Insufficient history for forecasting")

    # ---------------------------------------------------
    # 3) Load weather forecast 
    # ---------------------------------------------------
    site_row = db.execute(
        text("SELECT latitude, longitude FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).fetchone()

    if site_row is None:
        raise HTTPException(404, "Site not found")

    weather_hourly = fetch_environmental_data(
        latitude=site_row.latitude,
        longitude=site_row.longitude,
        start_ts=start_time - timedelta(hours=1),
        end_ts=end_time + timedelta(hours=1),
        source="forecast",
    )

    if not weather_hourly:
        raise HTTPException(500, "No weather data returned")

    hourly_ts = sorted(weather_hourly.keys())

    hourly_series = {
        "tout": {ts: weather_hourly[ts]["tout"] for ts in hourly_ts},
        "rh_out": {ts: weather_hourly[ts]["rh_out"] for ts in hourly_ts},
        "sw_out": {ts: weather_hourly[ts]["sw_out"] for ts in hourly_ts},
    }
 
    records = []
    for ts in horizon_index:
        records.append(
            {
                "timestamp": ts,
                "tout": interpolate_value(target_ts=ts, known_ts=hourly_ts, values=hourly_series["tout"]),
                "rh_out": interpolate_value(target_ts=ts, known_ts=hourly_ts, values=hourly_series["rh_out"]),
                "sw_out": interpolate_value(target_ts=ts, known_ts=hourly_ts, values=hourly_series["sw_out"]),
            }
        )

    weather_30min = pd.DataFrame(records).set_index("timestamp")

    if weather_30min.isna().any().any():
        raise HTTPException(500, "Weather interpolation failed")

    # ---------------------------------------------------
    # 4) Build future DF (same as optimization)
    # ---------------------------------------------------
    future_df = weather_30min.copy()
    future_df["tin"] = np.nan
    future_df["rh"] = np.nan
    future_df["comfort_index"] = np.nan
    future_df["hvac_mode"] = hvac_mode_48
    future_df["ah_out"] = AH_gm3_from_T_RH(future_df["tout"], future_df["rh_out"])

    # ---------------------------------------------------
    # 5) Build full timeline + features
    # ---------------------------------------------------
    full_timeline = pd.concat([history_df, future_df], axis=0)
    full_timeline = build_features(full_timeline)

    if start_time not in full_timeline.index:
        start_time = full_timeline.index[full_timeline.index.searchsorted(start_time)]

    start_idx = full_timeline.index.get_loc(start_time)

    if start_idx + 48 > len(full_timeline):
        raise HTTPException(500, "Invalid start index")

    # ---------------------------------------------------
    # 6) Run forecast (your existing engine)
    # ---------------------------------------------------
    tin_pred, rh_pred = rollout_48_steps_backend(
        df=full_timeline,
        start_idx=start_idx,
        hvac_mode_48=hvac_mode_48,
        site_id=site_id,
        db=db,
        device=device,
    )
    out_df = pd.DataFrame(index=horizon_index)

    out_df["hvac_mode"] = hvac_mode_48.astype(int)
    out_df["tin"] = tin_pred.astype(float)
    out_df["rh"] = rh_pred.astype(float)
    out_df["comfort_index"] = compute_comfort_percent(
        tin_pred,
        rh_pred,
        hvac_mode_48,
    )


    # ---------------------------------------------------
    # 7) Response
    # ---------------------------------------------------
    response = [
        {
            "timestamp": horizon_index[i].isoformat(),
            "tin_pred": float(tin_pred[i]),
            "rh_pred": float(rh_pred[i]),
            "hvac_mode": int(hvac_mode_48[i]),
            'comfort_index': float(out_df['comfort_index'][i])
        }
        for i in range(48)
    ]

    return {"site_id": site_id, "forecast": response}