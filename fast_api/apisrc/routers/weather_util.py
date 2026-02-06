import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from datetime import date, timedelta, timezone, datetime
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import text
from fast_api.apisrc.core.database import get_db
from fast_api.apisrc.utils.weather_utils import reconcile_environmental_data

router = APIRouter(prefix="/weather", tags=["weather"])

@router.get("/forecast")
async def get_weather_forecast(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": 2,
        "temperature_unit": "celsius",
        "timezone": "auto" 
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            # Create DataFrame from hourly data
            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"]
            })
            df.set_index("time", inplace=True)
            now = pd.Timestamp.now(tz=df.index.tz)
            start_time = now.ceil("30min")
            df = df[df.index >= start_time]

            df_30min = df.resample("30min").interpolate(method="linear")
            df_30min = df_30min.iloc[:48]
            df_30min.index = df_30min.index.tz_localize(None)
            # Convert back to list of dicts
            hourly_data = [
                {
                    "time": timestamp.isoformat(),
                    "temperature_celsius": temp
                }
                for timestamp, temp in zip(df_30min.index, df_30min["temperature"])
            ]
            return {"hourly_forecast": hourly_data}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Weather API error: {str(e)}")


#Open Meteo has a 5 day delay so recent history needs the forecast api 
@router.get('/historical')
async def get_historical_weather(lat: float, lon: float, start_date: str, end_date: str):
    today = date.today()
    archive_cutoff = today - timedelta(days=5)
    if start_date > archive_cutoff:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,relativehumidity_2m,shortwave_radiation"
        }
    else:
        # Use archive API for older data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": min(end_date, archive_cutoff).isoformat(),
            "hourly": "temperature_2m,relativehumidity_2m,shortwave_radiation"
        }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "temperature_2m": data["hourly"]["temperature_2m"],
                "relativehumidity_2m": data["hourly"]["relativehumidity_2m"],
                "shortwave_radiation": data["hourly"]["shortwave_radiation"]
            })
            df.set_index("time", inplace=True)
            
            # Resample to 30-min with linear interpolation
            df_resampled = df.resample("30min").interpolate(method="linear")
            
            # Convert to records
            records = [
                {
                    "time": timestamp.isoformat(),
                    "temperature_2m": row["temperature_2m"],
                    "relativehumidity_2m": row["relativehumidity_2m"],
                    "shortwave_radiation": row["shortwave_radiation"]
                }
                for timestamp, row in df_resampled.iterrows()
            ]
            
            return {
                "latitude": data["latitude"],
                "longitude": data["longitude"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "data": records
            }
            
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Weather API error: {str(e)}")
        


WEATHER_COLS = [
    "outdoor_temp",
    "relative_humidity",
    "solar_irradiance",
]


async def enrich_weather_if_missing(
    df: pd.DataFrame,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Fill missing weather data in-memory ONLY.
    Never overwrites existing non-null values.
    """

    df = df.copy()

    # --- 1. Detect missing rows ---
    missing_mask = df[WEATHER_COLS].isna().any(axis=1)

    if not missing_mask.any():
        # Nothing to do
        return df

    missing_times = df.loc[missing_mask, "timestamp"]

    start = missing_times.min().date()
    end = missing_times.max().date()

    # --- 2. Decide which Open-Meteo endpoint to use ---
    today = date.today()
    archive_cutoff = today - timedelta(days=5)

    if start > archive_cutoff:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,relativehumidity_2m,shortwave_radiation",
            "timezone": "auto",
        }
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": min(end, archive_cutoff).isoformat(),
            "hourly": "temperature_2m,relativehumidity_2m,shortwave_radiation",
            "timezone": "auto",
        }

    # --- 3. Fetch weather ---
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    # --- 4. Build weather DataFrame ---
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "outdoor_temp": data["hourly"]["temperature_2m"],
        "relative_humidity": data["hourly"]["relativehumidity_2m"],
        "solar_irradiance": data["hourly"]["shortwave_radiation"],
    })
    weather_df["timestamp"] = weather_df["timestamp"].dt.tz_localize(None)

    # Resample to 30-min resolution
    weather_df = (
        weather_df
        .set_index("timestamp")
        .resample("30min")
        .interpolate(method="linear")
        .reset_index()
    )

    # Ensure naive timestamps to match df
    weather_df["timestamp"] = weather_df["timestamp"].dt.tz_localize(None)

    # --- 5. Merge WITHOUT overwriting ---
    df = df.merge(
        weather_df,
        on="timestamp",
        how="left",
        suffixes=("", "_weather"),
    )

    for col in WEATHER_COLS:
        df[col] = df[col].combine_first(df[f"{col}_weather"])
        df.drop(columns=f"{col}_weather", inplace=True)

    return df


from pydantic import BaseModel
from datetime import datetime

class EnvironmentalReconcileRequest(BaseModel):
    start_ts: datetime
    end_ts: datetime
    resolution: str = "30min"
    fill_nulls_only: bool = True
    dry_run: bool = False

@router.post("/sites/{site_id}/environmental-data/reconcile")
def reconcile_environmental_data_route(
    site_id: int,
    payload: EnvironmentalReconcileRequest,
    db: Session = Depends(get_db),
):
    if payload.start_ts >= payload.end_ts:
        raise HTTPException(
            status_code=400,
            detail="start_ts must be earlier than end_ts",
        )

    # --- pure SQL site existence check ---
    site_row = db.execute(
        text("SELECT id FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).fetchone()

    if site_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Site {site_id} not found",
        )

    result = reconcile_environmental_data(
        db=db,
        site_id=site_id,   # pass primitive, not ORM object
        start_ts=payload.start_ts,
        end_ts=payload.end_ts,
        resolution=payload.resolution,
        fill_nulls_only=payload.fill_nulls_only,
        dry_run=payload.dry_run,
    )

    return result
