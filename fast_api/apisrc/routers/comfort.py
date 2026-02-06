import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from datetime import date, timedelta, timezone, datetime
import httpx
from pythermalcomfort.models import pmv_ppd_iso
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List
from fast_api.apisrc.core.database import get_db

router = APIRouter(prefix="/comfort", tags=["comfort"])

@router.post("/{site_id}/update_comfort")
def recalculate_comfort_index(
    site_id: int,
    start_ts: datetime,
    end_ts: datetime,
    db: Session = Depends(get_db),
):
    # -----------------------------
    # 1. Fetch required data
    # -----------------------------
    rows = db.execute(
        text(
            """
            SELECT
                timestamp,
                tin,
                rh
            FROM comfort_data
            WHERE site_id = :site_id
              AND timestamp >= :start_ts
              AND timestamp <= :end_ts
              AND tin IS NOT NULL
              AND rh IS NOT NULL
            ORDER BY timestamp
            """
        ),
        {
            "site_id": site_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).mappings().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No comfort data available in the given range",
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -----------------------------
    # 2. Compute clo from timestamp
    # -----------------------------
    def clo_from_timestamp(ts: pd.Timestamp) -> float:
        return 0.5 if ts.month in (5, 6, 7, 8, 9, 10) else 1.0

    df["clo"] = df["timestamp"].apply(clo_from_timestamp)

    # -----------------------------
    # 3. Compute comfort index
    # -----------------------------
    comfort_values = []

    for _, row in df.iterrows():
        result = pmv_ppd_iso(
            tdb=row["tin"],
            tr=row["tin"],
            vr=0.1,
            rh=row["rh"],
            met=1.1,
            clo=row["clo"],
        )

        ppd = result.get("ppd")
        comfort_values.append(75.0 if ppd is None else float(ppd))

    df["comfort_index"] = comfort_values

    # -----------------------------
    # 4. Batch UPDATE (overwrite)
    # -----------------------------
    update_sql = text(
        """
        UPDATE comfort_data
        SET comfort_index = :comfort_index
        WHERE site_id = :site_id
          AND timestamp = :timestamp
        """
    )

    payload = [
        {
            "site_id": site_id,
            "timestamp": row["timestamp"],
            "comfort_index": row["comfort_index"],
        }
        for _, row in df.iterrows()
    ]

    db.execute(update_sql, payload)
    db.commit()

    return {
        "site_id": site_id,
        "updated_rows": len(payload),
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    }





