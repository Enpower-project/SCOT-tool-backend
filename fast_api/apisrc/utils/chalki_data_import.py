import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fast_api.apisrc.real_utils import collect_data, calculate_true_consumption, import_dataframes_to_db, collect_comfort_data, insert_to_db_comfort, get_historic_complete_data
from zoneinfo import ZoneInfo
from sqlalchemy.dialects.postgresql import insert
from fast_api.apisrc.core.database import SessionLocal
from fast_api.apisrc.core.models import Site, ConsumptionData, ComfortData
import traceback
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()  # loads .env into environment variables

from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")

def _chunked(items: List[T], size: int) -> Iterator[List[T]]:
    """Yield successive chunks (lists) of length <= size."""
    for i in range(0, len(items), size):
        yield items[i : i + size]

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env variable: {name}")
    return value

def _sanitize_records(records):
    out = []
    for r in records:
        rr = {}
        for k, v in r.items():
            # pandas Timestamp -> python datetime
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()

            # numpy scalar -> python scalar
            if isinstance(v, np.generic):
                v = v.item()

            # NaN/inf -> None  (IMPORTANT: DB NOT NULL columns will choke)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = None

            # tz-aware datetime -> naive (your DB columns are timezone=False)
            if isinstance(v, datetime) and v.tzinfo is not None:
                v = v.replace(tzinfo=None)

            rr[k] = v
        out.append(rr)
    return out

def execute_upsert_with_bisect(session, stmt_factory, records, chunk_size=1000):
    """
    Executes upserts in chunks. If a chunk fails, bisects to find the bad record.
    Uses SAVEPOINTs (begin_nested) so failures don't poison the transaction.
    """
    def run(chunk):
        # SAVEPOINT: if this fails, only this statement rolls back
        with session.begin_nested():
            session.execute(stmt_factory(chunk))

    def bisect(chunk):
        if len(chunk) == 1:
            print("❌ Bad record:", chunk[0])
            # run it again so you get the real DB error for THIS single record
            run(chunk)  # this will raise, now with a single-row statement
            return

        mid = len(chunk) // 2
        try:
            run(chunk[:mid])
        except SQLAlchemyError:
            bisect(chunk[:mid])

        try:
            run(chunk[mid:])
        except SQLAlchemyError:
            bisect(chunk[mid:])

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            run(chunk)
        except SQLAlchemyError:
            print(f"Chunk failed at offset {i}, size {len(chunk)}. Bisecting...")
            bisect(chunk)

def main():
    try:
        
        loc_tin = "Dimarxeio_sunedriaston_tin"
        loc = "Dimarxeio_sunedriaston"
        loc_energy = "Dimarxeio_sunedriaston_energy"
        loc_rh = "Dimarxeio_sundedriaston_rh"
        tin = get_env(f"{loc}_tin")
        rh = get_env(f"{loc}_rh")
        con_dimarxeio = get_env(f"{loc}_consumption")

        csv_path_tin = ROOT / "chalki_comfort.csv" 
        csv_path_rh = ROOT / f"chalki_{loc_rh}.csv" 
        csv_path_cons = ROOT / "chalki_energy.csv" 

        
        if csv_path_tin.exists():
            comfort_df = pd.read_csv(csv_path_tin)
            cons_df = pd.read_csv(csv_path_cons)
        else:
        
            df_tin = get_historic_complete_data(tin, loc_tin)
    
            df_rh = get_historic_complete_data(rh, loc_rh)
            
            df_cons = get_historic_complete_data(con_dimarxeio, loc_energy)

            local_tz = "Europe/Athens"

            for df in (df_cons, df_tin, df_rh):
                # parse as UTC, convert to Athens (tz-aware)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise").dt.tz_convert(local_tz)

                df.sort_values("timestamp", inplace=True)

                # IMPORTANT: drop raw-source duplicate timestamps BEFORE set_index
                df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

                df.set_index("timestamp", inplace=True)
            
            df_cons["energy_kwh"] = df_cons["value"].interpolate(
                method="time", limit=24, limit_direction="both"
            )
            df_tin["tin"] = df_tin["value"].interpolate(
                method="time", limit=24, limit_direction="both"
            )
            df_rh['rh'] = df_rh["value"].interpolate(
                method="time", limit=24, limit_direction="both"
            )

            e = df_cons["energy_kwh"].clip(lower=0)

            q1, q3 = e.quantile(0.25), e.quantile(0.75)
            iqr = q3 - q1
            upper_iqr = q3 + 10.0 * iqr if pd.notna(iqr) else np.nan
            upper_q = e.quantile(0.999) if 0 < 0.999 < 1 else np.nan

            candidates = [v for v in (upper_iqr, upper_q) if pd.notna(v) and np.isfinite(v)]
            upper_cap = min(candidates) if candidates else float(e.max())

            clipped_mask = e > upper_cap
            n_clipped = int(clipped_mask.sum())
            pct_clipped = float(clipped_mask.mean() * 100)

            df_cons["energy_kwh"] = e.clip(upper=upper_cap)

            dfs = [df_cons, df_tin, df_rh]
            # for df in dfs:
            #     local_tz = "Europe/Athens"

            #     df.index = df.index.tz_convert(local_tz).tz_localize(None)
            SITE_NAME = "Dimarxeio_sunedriaston"

            latitude = 36.2296
            longitude = 27.5672

            df_cons_db = df_cons[["energy_kwh"]].copy()


            df_comfort_db = pd.concat(
                [df_tin[["tin"]], df_rh[["rh"]]],
                axis=1
            ).copy()

            cons_df = df_cons_db.reset_index().rename(columns={"energy_kwh": "value"})
            # cons_df['timestamp'] = pd.to_datetime(cons_df["timestamp"], errors="raise")
            # cons_df["timestamp"] = cons_df["timestamp"].dt.to_pydatetime()
            cons_df["timestamp"] = cons_df["timestamp"].dt.tz_localize(None)
            cons_df.sort_values("timestamp", inplace=True)
            cons_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            

            comfort_df = df_comfort_db.reset_index()
            comfort_df["timestamp"] = comfort_df["timestamp"].dt.tz_localize(None)

            # Collapse duplicates created by DST fall-back (choose a policy; keep last is fine)
            comfort_df.sort_values("timestamp", inplace=True)
            comfort_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

            # Convert NaN -> None so DB gets NULLs
            comfort_df = comfort_df.astype(object).where(pd.notna(comfort_df), None)

            comfort_df["hvac_mode"] = None
            comfort_df["comfort_index"] = None
            # comfort_df['timestamp'] = pd.to_datetime(comfort_df["timestamp"], errors="raise")
            # comfort_df["timestamp"] = comfort_df["timestamp"].dt.to_pydatetime()
            comfort_df = comfort_df.drop_duplicates(subset=["timestamp"], keep="last")
            
            # comfort_df.to_csv(f"chalki_comfort.csv")
            # cons_df.to_csv(f"chalki_energy.csv")

            print('Historic data done saved')
        
        cons_payload = cons_df[["timestamp", "value"]].to_dict("records")
        cons_payload = _sanitize_records(cons_payload)

        comfort_payload = comfort_df[["timestamp", "tin", "rh", "hvac_mode", "comfort_index"]].to_dict("records")
        comfort_payload = _sanitize_records(comfort_payload)
        with SessionLocal() as session:
            site = session.query(Site).filter_by(name=SITE_NAME).one_or_none()
            if site is None:
                site = Site(
                    name=SITE_NAME,
                    latitude=latitude,
                    longitude=longitude,
                    temp_low_band=25.0,
                    temp_high_band=70.0,
                    temp_low_bin=9.0,
                    temp_high_bin=32.0,
                    temp_bins_step=1.0,
                    is_residential=True,
                    active_ratio=0.01,
                    high_ratio=0.40,
                    min_high=2.0,
                    q=0.25,
                    min_days_per_bin=2,
                )
                session.add(site)
                session.flush()  # ensures site.id exists

            # attach site_id to payloads
            for d in cons_payload:
                d["site_id"] = site.id
            for d in comfort_payload:
                d["site_id"] = site.id

            def cons_stmt_factory(chunk):
                stmt = insert(ConsumptionData).values(chunk)
                return stmt.on_conflict_do_update(
                    index_elements=["site_id", "timestamp"],
                    set_={"value": stmt.excluded.value},
                )

            def comfort_stmt_factory(chunk):
                stmt = insert(ComfortData).values(chunk)
                return stmt.on_conflict_do_update(
                    index_elements=["site_id", "timestamp"],
                    set_={
                        "tin": stmt.excluded.tin,
                        "rh": stmt.excluded.rh,
                        "hvac_mode": stmt.excluded.hvac_mode,
                        "comfort_index": stmt.excluded.comfort_index,
                    },
                )

            # Execute with bisection to find bad row(s)
            execute_upsert_with_bisect(session, cons_stmt_factory, cons_payload, chunk_size=1000)
            execute_upsert_with_bisect(session, comfort_stmt_factory, comfort_payload, chunk_size=1000)

            session.commit()

    except Exception as e:
        print("SQLAlchemy:", type(e).__name__)
        if getattr(e, "orig", None) is not None:
            orig = e.orig
            print("DBAPI:", type(orig).__name__)
            print("pgcode:", getattr(orig, "pgcode", None))
            print("message:", str(orig))
        raise


if __name__ == "__main__":
    main()
