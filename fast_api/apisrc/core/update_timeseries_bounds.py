import pandas as pd
from fast_api.apisrc.core.database import SessionLocal
from fast_api.apisrc.core.models import TimeseriesData


def update_timeseries_bounds() -> None:
    session = SessionLocal()

    try:
        ts_rows = session.query(TimeseriesData).all()

        for ts in ts_rows:
            uri = ts.storage_uri

            # Read only the timestamp column
            df = pd.read_parquet(uri, columns=["timestamp"])

            if df.empty:
                print(f"⚠️ Empty dataset at {uri}, skipping")
                continue

            start_ts = df["timestamp"].min()
            end_ts = df["timestamp"].max()

            ts.start_ts = start_ts
            ts.end_ts = end_ts

            print(
                f"Updated TimeseriesData {ts.timeseries_id}: "
                f"{start_ts} → {end_ts}"
            )

        session.commit()
        print("✅ Timeseries bounds updated successfully")

    except Exception as e:
        session.rollback()
        print("❌ Failed to update timeseries bounds")
        raise e

    finally:
        session.close()


if __name__ == "__main__":
    update_timeseries_bounds()
