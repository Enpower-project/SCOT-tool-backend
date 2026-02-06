# from models import Site, ConsumptionData, ComfortData, EnvironmentalData

import os

from sqlalchemy import text
import inspect
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_api.apisrc.core.database import Base, engine
from fast_api.apisrc.core import models


with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database(), current_schema()"))
    print("CONNECTED TO:", result.fetchone())

def reset_database() -> None:
    print("Dropping all tables...")
    print("METADATA TABLES AT STARTUP:", list(Base.metadata.tables.keys()))

    Base.metadata.drop_all(bind=engine)
    print("TABLES SEEN BY SQLALCHEMY:", Base.metadata.tables.keys())

    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)

    print("Database schema reset complete.")


if __name__ == "__main__":
    reset_database()