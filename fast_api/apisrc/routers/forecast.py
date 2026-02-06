import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from fast_api.apisrc.core.database import SessionLocal
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List
from fast_api.apisrc.utils.weather_utils import AH_gm3_from_T_RH, RH_percent_from_AH_T
from sqlalchemy.orm import Session
from fast_api.apisrc.core.database import get_db
import random
class ForecastRequest(BaseModel):
    start: datetime
    hvac_mode_future: List[int]

 

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hour/24.0)

    month = ts.dt.month.astype(int)
    df["month"] = month
    df["month_sin"] = np.sin(2*np.pi*(month-1)/12.0)
    df["month_cos"] = np.cos(2*np.pi*(month-1)/12.0)

    # simple season encoding (adjust if you already have your own)
    # 0=winter(12-2), 1=spring(3-5), 2=summer(6-8), 3=fall(9-11)
    season = ((month % 12) // 3).astype(int)
    df["season"] = season
    return df

def add_solar_rollups(df: pd.DataFrame) -> pd.DataFrame:
    # assumes SW is instantaneous shortwave radiation at 30-min
    # SW1h = mean of last 2 steps, SW3h = mean of last 6 steps
    df["SW1h"] = df["SW"].rolling(2, min_periods=1).mean()
    df["SW3h"] = df["SW"].rolling(6, min_periods=1).mean()
    return df

def ensure_ah_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If AH_out missing but RH_out+Tout exist
    if "ah_out" not in df.columns and {"rh_out", "tout"} <= set(df.columns):
        df["ah_out"] = AH_gm3_from_T_RH(
            df["tout"].astype(float),
            df["rh_out"].astype(float),
        )

    # Indoor absolute humidity
    if "ah" not in df.columns and {"rh", "tin"} <= set(df.columns):
        df["ah"] = AH_gm3_from_T_RH(
            df["tin"].astype(float),
            df["rh"].astype(float),
        )

    return df

router = APIRouter(prefix="/forecast", tags=["forecast"])

# Next day forecast, if models dont exist return noisier last day consumption
@router.get("/{site_id}/timeseries/consumption")
def getConsumptionForecast(
    site_id,
    start_ts: datetime,
    use_last_day,
    db: Session = Depends(get_db),
):
    if use_last_day:
        prev_start = start_ts - timedelta(days=1)
        prev_end = start_ts - timedelta(minutes=30)
        
        sql = """
        SELECT cd.timestamp, cd.value,
        c.hvac_mode 
        FROM consumption_data cd
        LEFT JOIN comfort_data c
        ON c.site_id = cd.site_id
        AND c.timestamp = cd.timestamp
        WHERE cd.site_id = :site_id
          AND cd.timestamp <= :prev_end
          AND cd.timestamp >= :prev_start
        ORDER BY timestamp ASC
        """

        rows = db.execute(
            text(sql),
            {
                "site_id": site_id,
                "prev_start": prev_start,
                "prev_end": prev_end,
            },
        ).fetchall()

        
        if len(rows) < 48:
            raise ValueError(f"Not enough historical data: expected 48, got {len(rows)}")

        series = []
        for ts, value, hvac_mode in rows[:48]:
            noise_factor = 1 + random.uniform(-0.05, 0.05)  # ±5% noise
            noisy_value = value * noise_factor
            series.append(
                {
                    "timestamp": ts + timedelta(days=1),  # shift forward to forecast day
                    "value": noisy_value,
                    "hvac_mode": hvac_mode,
                }
            )

        return series
    else:
        return []