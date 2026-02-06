from sqlalchemy import text
from datetime import datetime
from sqlalchemy.orm import Session
from bisect import bisect_left
import requests
from datetime import timezone, timedelta
import numpy as np
import pandas as pd

REQUIRED_FIELDS = ("tout", "rh_out", "sw_out")


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

    # Preserve pandas index if input is a Series
    if isinstance(Tc, pd.Series):
        return pd.Series(AH, index=Tc.index, name="ah")
    return AH

def RH_percent_from_AH_T(AH_gm3: float, Tc: float) -> float:
    """Relative humidity (%) from absolute humidity (g/m³) and temperature (°C)."""
    es = _es_hPa_from_Tc(Tc)
    # vapor pressure from AH
    e = max(0.0, AH_gm3) * (Tc + 273.15) / 216.7
    RH = 100.0 * (e / max(1e-6, es))
    return float(np.clip(RH, 0.0, 100.0))



def interpolate_value(
    *,
    target_ts,
    known_ts,
    values,
):
    idx = bisect_left(known_ts, target_ts)

    # exact match
    if idx < len(known_ts) and known_ts[idx] == target_ts:
        return values[target_ts]

    # out of bounds
    if idx == 0 or idx == len(known_ts):
        return None

    t0 = known_ts[idx - 1]
    t1 = known_ts[idx]

    v0 = values.get(t0)
    v1 = values.get(t1)

    if v0 is None or v1 is None:
        return None

    total = (t1 - t0).total_seconds()
    part = (target_ts - t0).total_seconds()
    return v0 + (v1 - v0) * (part / total)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_environmental_data(
    *,
    latitude: float,
    longitude: float,
    start_ts: datetime,
    end_ts: datetime,
    source: str,  # "archive" | "forecast"
):
    """
    Fetch hourly environmental data from Open-Meteo.

    Returns:
        {
            timestamp_utc: {
                "tout": float | None,
                "rh_out": float | None,
                "sw_out": float | None,
            }
        }
    """
    if start_ts.tzinfo is not None:
        start_ts = start_ts.replace(tzinfo=None)

    if end_ts.tzinfo is not None:
        end_ts = end_ts.replace(tzinfo=None)
    if source not in {"archive", "forecast"}:
        raise ValueError("source must be 'archive' or 'forecast'")

    url = ARCHIVE_URL if source == "archive" else FORECAST_URL

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "shortwave_radiation",
        ],
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly:
        return {}

    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rhs = hourly.get("relative_humidity_2m", [])
    solar = hourly.get("shortwave_radiation", [])

    result = {}

    for t, temp, rh, sol in zip(times, temps, rhs, solar):
        ts = datetime.fromisoformat(t)

        # Open-Meteo returns local wall-clock time when timezone=auto
        # Make it NAIVE to match DB
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)

        # Guard: restrict strictly to requested window
        if ts < start_ts or ts > end_ts:
            continue

        result[ts] = {
            "tout": float(temp) if temp is not None else None,
            "rh_out": float(rh) if rh is not None else None,
            "sw_out": float(sol) if sol is not None else None,
        }

    return result


def reconcile_environmental_data(
    *,
    db: Session,
    site_id: int,
    start_ts: datetime,
    end_ts: datetime,
    resolution: str,
    fill_nulls_only: bool,
    dry_run: bool,
) -> dict:
    """
    Reconcile EnvironmentalData rows for a site and time window.

    Responsibilities:
    - derive required timestamps from ComfortData
    - detect missing / incomplete EnvironmentalData
    - fetch Open-Meteo data (archive + forecast)
    - interpolate to target resolution
    - insert or fill-null rows
    - return structured summary
    """
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
        raise RuntimeError("Site disappeared during reconciliation")

    latitude = site_row.latitude
    longitude = site_row.longitude

    comfort_rows = db.execute(
        text(
            """
            SELECT DISTINCT timestamp
            FROM comfort_data
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
            ORDER BY timestamp
            """
        ),
        {
            "site_id": site_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).fetchall()

    required_ts = [r.timestamp.replace(tzinfo=None) for r in comfort_rows]


    if not required_ts:
        return {
            "site_id": site_id,
            "window": {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "resolution": resolution,
            },
            "reconciliation": {
                "timestamps_examined": 0,
                "rows_inserted": 0,
                "rows_updated": 0,
                "rows_unchanged": 0,
                "rows_still_incomplete": 0,
            },
            "fields": {
                "required": list(REQUIRED_FIELDS),
                "interpolation_applied": False,
            },
            "dry_run": dry_run,
            "notes": ["No ComfortData timestamps in window"],
        }
    
    T_required = set(required_ts)

    env_existing_rows = db.execute(
        text("""
            SELECT timestamp
            FROM environmental_data
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
        """),
        {
            "site_id": site_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).fetchall()

    T_existing = {r.timestamp.replace(tzinfo=None) for r in env_existing_rows}


    # -------------------------------------------------
    # 4. Incomplete environmental timestamps (T_incomplete)
    # -------------------------------------------------
    env_incomplete_rows = db.execute(
        text("""
            SELECT timestamp
            FROM environmental_data
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
              AND (
                  tout IS NULL
                  OR rh_out IS NULL
                  OR sw_out IS NULL
              )
        """),
        {
            "site_id": site_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).fetchall()

    T_incomplete = {r.timestamp.replace(tzinfo=None) for r in env_incomplete_rows}


    # -------------------------------------------------
    # 5. Derive action sets (EXPLICIT)
    # -------------------------------------------------
    T_insert = T_required - T_existing
    T_update = T_required & T_incomplete

    # Sanity invariants (safe to keep)
    assert T_insert.isdisjoint(T_existing)
    assert T_update.issubset(T_existing)

    timestamps_to_fix = sorted(T_insert | T_update)

    rows_unchanged = len(T_required) - len(timestamps_to_fix)

    # -------------------------------------------------
    # 6. Fetch Open-Meteo data only for needed timestamps
    # -------------------------------------------------
    if not timestamps_to_fix:
        return {
            "site_id": site_id,
            "window": {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "resolution": resolution,
            },
            "reconciliation": {
                "timestamps_examined": len(T_required),
                "rows_inserted": 0,
                "rows_updated": 0,
                "rows_unchanged": rows_unchanged,
                "rows_still_incomplete": 0,
            },
            "fields": {
                "required": list(REQUIRED_FIELDS),
                "interpolation_applied": False,
            },
            "dry_run": dry_run,
            "notes": ["No environmental reconciliation required"],
        }

    now_local = datetime.now()
    forecast_cutoff = now_local - timedelta(days=5)

    archive_ts = [ts for ts in timestamps_to_fix if ts < forecast_cutoff]
    forecast_ts = [ts for ts in timestamps_to_fix if ts >= forecast_cutoff]

    fetched_data = {}
    data_sources = {"archive_used": False, "forecast_used": False}

    if archive_ts:
        data_sources["archive_used"] = True
        fetched_data.update(
            fetch_environmental_data(
                latitude=latitude,
                longitude=longitude,
                start_ts=min(archive_ts),
                end_ts=max(archive_ts),
                source="archive",
            )
        )

    if forecast_ts:
        data_sources["forecast_used"] = True
        fetched_data.update(
            fetch_environmental_data(
                latitude=latitude,
                longitude=longitude,
                start_ts=min(forecast_ts),
                end_ts=max(forecast_ts),
                source="forecast",
            )
        )

    hourly_ts = sorted(fetched_data.keys())

    hourly_series = {
        field: {ts: fetched_data[ts][field] for ts in hourly_ts}
        for field in REQUIRED_FIELDS
    }

    # -------------------------------------------------
    # 7. Interpolate to target timestamps
    # -------------------------------------------------
    aligned = {}
    for ts in timestamps_to_fix:
        aligned[ts] = {
            field: interpolate_value(
                target_ts=ts,
                known_ts=hourly_ts,
                values=hourly_series[field],
            )
            for field in REQUIRED_FIELDS
        }

    # -------------------------------------------------
    # 8. Apply INSERT / UPDATE explicitly
    # -------------------------------------------------
    rows_inserted = 0
    rows_updated = 0
    rows_still_incomplete = 0

    for ts, values in aligned.items():
        if any(values[f] is None for f in REQUIRED_FIELDS):
            rows_still_incomplete += 1

        if ts in T_insert:
            rows_inserted += 1
            if not dry_run:
                db.execute(
                    text("""
                        INSERT INTO environmental_data (
                            site_id, timestamp, tout, rh_out, sw_out
                        )
                        VALUES (
                            :site_id, :timestamp, :tout, :rh_out, :sw_out
                        )
                    """),
                    {
                        "site_id": site_id,
                        "timestamp": ts,
                        "tout": values["tout"],
                        "rh_out": values["rh_out"],
                        "sw_out": values["sw_out"],
                    },
                )

        elif ts in T_update:
            rows_updated += 1
            if not dry_run:
                db.execute(
                    text("""
                        UPDATE environmental_data
                        SET
                            tout = COALESCE(tout, :tout),
                            rh_out = COALESCE(rh_out, :rh_out),
                            sw_out = COALESCE(sw_out, :sw_out)
                        WHERE site_id = :site_id
                          AND timestamp = :timestamp
                    """),
                    {
                        "site_id": site_id,
                        "timestamp": ts,
                        "tout": values["tout"],
                        "rh_out": values["rh_out"],
                        "sw_out": values["sw_out"],
                    },
                )

    if not dry_run:
        db.commit()

    return {
        "site_id": site_id,
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "resolution": resolution,
        },
        "data_sources": data_sources,
        "reconciliation": {
            "timestamps_examined": len(T_required),
            "rows_inserted": rows_inserted,
            "rows_updated": rows_updated,
            "rows_unchanged": rows_unchanged,
            "rows_still_incomplete": rows_still_incomplete,
        },
        "fields": {
            "required": list(REQUIRED_FIELDS),
            "interpolation_applied": True,
        },
        "dry_run": dry_run,
        "notes": [],
    }