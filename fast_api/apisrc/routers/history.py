from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta, timezone
import httpx
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from fast_api.apisrc.core.database import get_db
from typing import Dict, Tuple, List
from collections import defaultdict
from pythermalcomfort.models import pmv_ppd_iso
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/history", tags=["timeseries"])

METRIC_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Comfort
    "tin": ("comfort_data", "tin"),
    "rh": ("comfort_data", "rh"),
    "comfort_index": ("comfort_data", "comfort_index"),
    "hvac_mode": ("comfort_data", "hvac_mode"),

    # Environmental
    "tout": ("environmental_data", "tout"),
    "rh_out": ("environmental_data", "rh_out"),
    "sw_out": ("environmental_data", "sw_out"),

    # Energy
    "energy_consumption": ("consumption_data", "value"),
    "energy_production": ("production_data", "value"),

    # Forecasts
    "forecasted_consumption": ("forecasted_consumption_data", "value"),
    "forecasted_production": ("forecasted_production_data", "value"),
}

def iso(ts: datetime | None):
    return ts.isoformat() if ts else None

async def fetch_current_tout(lat: float, lon: float) -> float | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    return data.get("current", {}).get("temperature_2m")

@router.get("/{site_id}/timeseries")
def get_timeseries_window(
    site_id: str,
    metrics: str,
    start_ts: datetime,
    end_ts: datetime,
    db: Session = Depends(get_db),
):
    exists = db.execute(
        text("SELECT 1 FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).scalar()

    requested_metrics: List[str] = [m.strip() for m in metrics.split(",") if m.strip()]
    if not requested_metrics:
        raise HTTPException(status_code=400, detail="No metrics requested")

    # -----------------------------
    # Validate metrics
    # -----------------------------
    invalid = set(requested_metrics) - set(METRIC_REGISTRY.keys())
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metrics: {sorted(invalid)}",
        )

    # -----------------------------
    # Group metrics by table
    # -----------------------------
    table_groups: Dict[str, List[str]] = defaultdict(list)
    for metric in requested_metrics:
        table, _ = METRIC_REGISTRY[metric]
        table_groups[table].append(metric)

    # -----------------------------
    # Query each table
    # -----------------------------
    results_by_timestamp: Dict[datetime, Dict] = {}

    for table_name, table_metrics in table_groups.items():
        select_cols = []
        for metric in table_metrics:
            _, column = METRIC_REGISTRY[metric]
            select_cols.append(f"{column} AS {metric}")

        sql = text(f"""
            SELECT
                timestamp,
                {", ".join(select_cols)}
            FROM {table_name}
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
            ORDER BY timestamp
        """)

        rows = db.execute(
            sql,
            {
                "site_id": site_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        ).mappings().all()

        for row in rows:
            ts = row["timestamp"]
            if ts not in results_by_timestamp:
                results_by_timestamp[ts] = {"timestamp": ts}
            results_by_timestamp[ts].update(row)

    # -----------------------------
    # Return merged timeline
    # -----------------------------
    return sorted(results_by_timestamp.values(), key=lambda x: x["timestamp"])

def clo_from_timestamp(ts: datetime) -> float:
    return 0.5 if ts.month in (5, 6, 7, 8, 9, 10) else 1.0

@router.get("/{site_id}/metrics/latest")
async def get_latest_metrics(
    site_id: str,
    metrics: str,
    db: Session = Depends(get_db),
):
    # -----------------------------
    # Check site exists
    # -----------------------------
    site = db.execute(
        text("SELECT id, latitude, longitude FROM sites WHERE id = :site_id"),
        {"site_id": site_id},
    ).mappings().first()

    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    # -----------------------------
    # Parse + validate metrics
    # -----------------------------
    requested_metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    if not requested_metrics:
        raise HTTPException(status_code=400, detail="No metrics requested")

    invalid = set(requested_metrics) - set(METRIC_REGISTRY.keys())
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metrics: {sorted(invalid)}",
        )

    # -----------------------------
    # Group metrics by table
    # -----------------------------
    table_groups: Dict[str, List[str]] = defaultdict(list)
    for metric in requested_metrics:
        table, column = METRIC_REGISTRY[metric]
        table_groups[table].append((metric, column))

    # -----------------------------
    # Query latest row per table
    # -----------------------------
    response: Dict[str, Dict] = {}
    latest_tin = None
    latest_rh = None
    latest_comfort_ts = None

    for table_name, metric_specs in table_groups.items():
        select_cols = [
            f"{column} AS {metric}"
            for metric, column in metric_specs
        ]

        sql = text(f"""
            SELECT
                timestamp,
                {", ".join(select_cols)}
            FROM {table_name}
            WHERE site_id = :site_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = db.execute(
            sql,
            {"site_id": site_id},
        ).mappings().first()

        if not row:
            continue

        ts = row["timestamp"]

        for metric, _ in metric_specs:
            value = row[metric]

            response[metric] = {
                "value": value,
                "timestamp": iso(ts),
            }

            if metric == "tin":
                latest_tin = value
                latest_comfort_ts = ts

            if metric == "rh":
                latest_rh = value
                latest_comfort_ts = ts


    if not response:
        raise HTTPException(status_code=404, detail="No data found")
    
    if "comfort_index" in requested_metrics:
        if latest_tin is not None and latest_rh is not None and latest_comfort_ts:
            clo = clo_from_timestamp(latest_comfort_ts)
            result = pmv_ppd_iso(
                tdb=latest_tin,
                tr=latest_tin,
                vr=0.1,
                rh=latest_rh,
                met=1.1,
                clo=clo,
            )
            if(result):
                ppd = float(result["ppd"])
                comfort_index = 75.0 if ppd is None else float(100.0 - ppd)
            else:
                ppd = 0
            response["comfort_index"] = {
                "value": comfort_index,
                "timestamp": iso(latest_comfort_ts),
            }
        else:
            response["comfort_index"] = {
                "value": None,
                "timestamp": None,
            }
    # get outside temperature from openmeteo
    if "tout" in requested_metrics:
        try:
            tout = await fetch_current_tout(
                site["latitude"],
                site["longitude"],
            )
            response["tout"] = {
                "value": tout,
                "timestamp": iso(datetime.now(timezone.utc)),
            }
        except Exception:
            response["tout"] = {
                "value": None,
                "timestamp": None,
            }

    payload = {
        "site_id": site_id,
        "metrics": response,
    }

    res = JSONResponse(content=payload)

    res.headers["Cache-Control"] = "public, max-age=60"
    res.headers["Vary"] = "Accept-Encoding"

    return res


@router.get("/{site_id}/timeseries/last-24h")
def get_last_24h_timeseries(
    site_id: int,
    metrics: str,
    db: Session = Depends(get_db),
):
    # -----------------------------
    # Time window (server-defined)
    # -----------------------------
    end_ts = datetime.now()
    start_ts = end_ts - timedelta(hours=24)

    # -----------------------------
    # Parse & validate metrics
    # -----------------------------
    requested_metrics: List[str] = [m.strip() for m in metrics.split(",") if m.strip()]
    if not requested_metrics:
        raise HTTPException(status_code=400, detail="No metrics requested")

    invalid = set(requested_metrics) - set(METRIC_REGISTRY.keys())
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metrics: {sorted(invalid)}",
        )

    # -----------------------------
    # Group metrics by table
    # -----------------------------
    table_groups: Dict[str, List[str]] = defaultdict(list)
    for metric in requested_metrics:
        table, _ = METRIC_REGISTRY[metric]
        table_groups[table].append(metric)

    # -----------------------------
    # Execute queries
    # -----------------------------
    results_by_timestamp: Dict[datetime, Dict] = {}

    for table_name, table_metrics in table_groups.items():
        select_cols = []
        for metric in table_metrics:
            _, column = METRIC_REGISTRY[metric]
            select_cols.append(f"{column} AS {metric}")

        sql = text(f"""
            SELECT
                timestamp,
                {", ".join(select_cols)}
            FROM {table_name}
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
            ORDER BY timestamp
        """)

        rows = db.execute(
            sql,
            {
                "site_id": site_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        ).mappings().all()

        for row in rows:
            ts = row["timestamp"]
            if ts not in results_by_timestamp:
                results_by_timestamp[ts] = {"timestamp": ts}
            results_by_timestamp[ts].update(row)

    # -----------------------------
    # Return merged timeline
    # -----------------------------
    return sorted(results_by_timestamp.values(), key=lambda x: x["timestamp"])