from sqlalchemy import create_engine
from pathlib import Path
import sys

# Add project root to path (same as in models.py)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_api.apisrc.core.database import Base
from fast_api.apisrc.core.models import (  # adjust import path as needed
    Site, ConsumptionData, ComfortData, EnvironmentalData,
    SiteModel, OptimizationRun, OptimizationData,
    ProductionData, ForecastedConsumptionData, ForecastedProductionData
)

# Database connection
DATABASE_URL = "postgresql://enpower:enpower@147.102.6.183:5432/scot_final"

def setup_database():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("Database schema created successfully")

if __name__ == "__main__":
    setup_database()