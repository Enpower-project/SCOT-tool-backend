# from fast_api.apisrc.core.models import Building, Dataset, ForecastData, OptimizationData, TimeseriesData
from models import Site, ConsumptionData, EnvironmentalData, ComfortData
from database import SessionLocal
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from zoneinfo import ZoneInfo
from datetime import timezone
from sqlalchemy.dialects.postgresql import insert
from database import Base
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parents[1]  # api_src
DATA_DIR = BASE_DIR / "data"
import os
print("DATABASE_URL =", os.getenv("DATABASE_URL"))

# DATABASE_URL = "postgresql://user:password@localhost:5432/your_db"
CSV_PATH = DATA_DIR / "House_01.csv" 
SITE_NAME = "House_01"
DATABASE_URL = "postgresql+psycopg2://postgres:Mistborn2002@localhost:5432/enpowerdb"
SITE_TIMEZONE = ZoneInfo("Europe/Athens")  # change if needed
latitude = 37.9838
longitude = 23.7275

print("KNOWN TABLES:", Base.metadata.tables.keys())
def seed_sensor_metrics():
    print("DATABASE_URL =", DATABASE_URL)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        session.execute(
            text(
                "TRUNCATE TABLE "
                "consumption_data, "
                "comfort_data, "
                "environmental_data, "
                "optimization_data "
                "RESTART IDENTITY CASCADE"
            )
        )
        session.commit()

    Base.metadata.create_all(bind=engine)

    df = pd.read_csv(CSV_PATH)

    # keep only required columns
    df = df[
        [
            "timestamp",
            "energy_consumption",
            "Tin",
            "Tout",
            "RH",
            "RH_out",
            "hvac_mode",
        ]
    ]
    table_specs = {
        ConsumptionData: {
            "columns": ["energy_consumption"],
            "builder": lambda site_id, ts, r: ConsumptionData(
                site_id=site_id,
                timestamp=ts,
                value=float(r["energy_consumption"]),
            ),
        },
        ComfortData: {
            "columns": ["Tin", "RH", "hvac_mode"],
            "builder": lambda site_id, ts, r: ComfortData(
                site_id=site_id,
                timestamp=ts,
                tin=None if pd.isna(r["Tin"]) else float(r["Tin"]),
                rh=None if pd.isna(r["RH"]) else float(r["RH"]),
                hvac_mode=None if pd.isna(r["hvac_mode"]) else int(r["hvac_mode"]),
            ),
        },
        EnvironmentalData: {
            "columns": ["Tout", "RH_out", "SW_out"],
            "builder": lambda site_id, ts, r: EnvironmentalData(
                site_id=site_id,
                timestamp=ts,
                tout=None if pd.isna(r["Tout"]) else float(r["Tout"]),
                rh_out=None if pd.isna(r["RH_out"]) else float(r["RH_out"]),
                sw_out=None if pd.isna(r["SW_out"]) else float(r["SW_out"]),
            ),
        },
    }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

# Attach site timezone ONLY to resolve DST
    df["timestamp"] = df["timestamp"].dt.tz_localize(
        SITE_TIMEZONE,
        ambiguous=False,          # choose first occurrence on fall-back
        nonexistent="shift_forward"
    )

    # Shift years in LOCAL TIME
    df["timestamp"] = df["timestamp"].apply(
        lambda ts: ts + relativedelta(years=3)
    )

    # DROP timezone info before storing to DB
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df.sort_values("timestamp")

    # Drop duplicate timestamps, keep the first occurrence
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
        
    rows = []

    with SessionLocal() as session:
        # get or create site
        # site = session.query(Site).filter_by(name=SITE_NAME).one_or_none()
        # if site is None:
    
        # upsert or create site
        site = session.query(Site).filter_by(name=SITE_NAME).one_or_none()
        if site is None:
            site = Site(
                name=SITE_NAME,
                latitude=latitude,
                longitude=longitude,
                temp_low_band=20.0,
                temp_high_band=40.0,
                temp_low_bin=0,
                temp_high_bin=40.0,
                temp_bins_step=1.0,
                is_residential=True,
                active_ratio=0.05,
                high_ratio=0.50,
                min_high=500.0,
                q=0.25,
                min_days_per_bin=2)
   
            session.add(site)
            session.flush()

        for _, r in df.iterrows():
            ts = r["timestamp"]

            for table, spec in table_specs.items():
                if all(pd.isna(r[col]) for col in spec["columns"]):
                    continue

                if table == EnvironmentalData:  
                    continue;

                rows.append(
                    spec["builder"](site.id, ts, r)
                )
                    

        session.bulk_save_objects(rows)
        session.commit()

# def seed_database() -> None:
#     session = SessionLocal()

#     try:
#         print("Seeding database...")

#         # --- Building ---
#         building = Building(
#             name="Plegma",
#             latitude=37.9838,
#             longitude=23.7275
#         )
#         session.add(building)
#         session.flush()  


#         dataset = Dataset(
#             name="Plegma House 01",
#             resolution=30,
#         )
#         building.datasets.append(dataset)
#         session.flush()  # get dataset_id

#         measured_uri = (
#             DATA_DIR
#             / "timeseries"
#             / f"building_{building.building_id}"
#             / f"dataset_{dataset.dataset_id}"
#             / "measured"
#         )


#         dataset_ts = TimeseriesData(
#             storage_uri=str(measured_uri),
#             kind="measured",
#             schema_version=1,
#             start_ts=None,  
#             end_ts=None,
#             dataset_id=dataset.dataset_id,
#         )

#         # --- Persist ---
#         session.add(dataset_ts)

#         # --- Persist ---
#         session.commit()

#         print("Database seeded successfully.")


#     except Exception as e:
#         session.rollback()
#         print("❌ Database seeding failed")

#         raise e
#     finally:
#         session.close()


if __name__ == "__main__":
    seed_sensor_metrics()