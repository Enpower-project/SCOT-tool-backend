# models.py
from datetime import datetime, timezone
from sqlalchemy import (
    SmallInteger, create_engine, Column, Integer, String, Float, DateTime, ForeignKey,
    Index, CheckConstraint, Enum, UniqueConstraint, Boolean, text, JSON
)
from sqlalchemy.orm import relationship, sessionmaker, Mapped, mapped_column
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_api.apisrc.core.database import Base
from enum import IntEnum
class Site(Base):
    __tablename__ = 'sites'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temp_low_band = Column(Float, nullable=True)
    temp_high_band = Column(Float, nullable=True)
    temp_low_bin = Column(Float, nullable=True)
    temp_high_bin = Column(Float, nullable=True)
    temp_bins_step = Column(Float, nullable=True)
    is_residential = Column(Boolean, nullable=True)
    active_ratio = Column(Float, nullable=True)
    high_ratio = Column(Float, nullable=True)
    min_high_abs = Column(Float, nullable=True)
    min_act_abs = Column(Float, nullable=True)
    q = Column(Float, nullable=True)
    min_days_per_bin = Column(Float, nullable=True)

    # timezone = Column(String, nullable=False)
    # Relationships
    consumption_data = relationship("ConsumptionData", back_populates="site", cascade="all, delete-orphan")
    comfort_data = relationship("ComfortData", back_populates="site", cascade="all, delete-orphan")
    environmental_data = relationship("EnvironmentalData", back_populates="site", cascade="all, delete-orphan")
    optimization_runs = relationship("OptimizationRun", back_populates="site", cascade="all, delete-orphan")
    production_data = relationship("ProductionData", back_populates="site", cascade="all, delete-orphan")
    forecasted_consumption_data = relationship(
            "ForecastedConsumptionData",
            back_populates="site",
            cascade="all, delete-orphan"
        )
    forecasted_production_data = relationship(
        "ForecastedProductionData",
        back_populates="site",
        cascade="all, delete-orphan"
    )

    site_models = relationship("SiteModel", back_populates="site", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            "<Site("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"latitude={self.latitude}, "
            f"longitude={self.longitude}, "
            f"temp_low_band={self.temp_low_band}, "
            f"temp_high_band={self.temp_high_band}, "
            f"temp_low_bin={self.temp_low_bin}, "
            f"temp_high_bin={self.temp_high_bin}, "
            f"temp_bins_step={self.temp_bins_step}, "
            f"is_residential={self.is_residential}, "
            f"active_ratio={self.active_ratio}, "
            f"high_ratio={self.high_ratio}, "
            f"min_high={self.min_high}, "
            f"q={self.q}, "
            f"min_days_per_bin={self.min_days_per_bin}"
            ")>"
        )
class ConsumptionData(Base):
    __tablename__ = "consumption_data"

    site_id = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)

    value = Column(Float, nullable=False)

    site = relationship("Site", back_populates="consumption_data")

    __table_args__ = (
        Index(
            "ix_consumption_site_ts_desc",
            "site_id",
            text("timestamp DESC"),
        ),
    )

    def __repr__(self):
        return f"<ConsumptionData(site_id={self.site_id}, timestamp='{self.timestamp}', value={self.value})>"
    
class ComfortData(Base):
    __tablename__ = "comfort_data"

    site_id = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)

    tin = Column(Float, nullable=True)
    rh = Column(Float, nullable=True)
    hvac_mode = Column(SmallInteger, nullable=True)
    comfort_index = Column(Float, nullable=True)

    site = relationship("Site", back_populates="comfort_data")

    __table_args__ = (
        Index(
        "ix_comfort_site_ts_desc",
        "site_id",
        text("timestamp DESC"),
        ),
        CheckConstraint("hvac_mode IN (0, 1, 2)", name="ck_hvac_mode_valid"),
    )

    def __repr__(self):
        return (
            f"<ComfortData(site_id={self.site_id}, timestamp='{self.timestamp}', "
            f"tin={self.tin}, rh={self.rh}, "
            f"comfort_index={self.comfort_index})>"
        )

class EnvironmentalData(Base):
    __tablename__ = "environmental_data"

    site_id = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)

    tout = Column(Float, nullable=True)
    rh_out = Column(Float, nullable=True)
    sw_out = Column(Float, nullable=True)

    site = relationship("Site", back_populates="environmental_data")

    __table_args__ = (
        Index(
            "ix_environmental_site_ts_desc",
            "site_id",
            text("timestamp DESC"),
        ),
    )

    def __repr__(self):
        return (
            f"<EnvironmentalData(site_id={self.site_id}, timestamp='{self.timestamp}', "
            f"tout={self.tout}, rh_out={self.rh_out}, sw_out={self.sw_out})>"
        )

    
class SiteModel(Base):
    __tablename__ = "site_models"

    id = Column(Integer, primary_key=True)

    site_id = Column(
        Integer,
        ForeignKey("sites.id", ondelete="CASCADE"),
        nullable=False,
    )

    model_type = Column(String(50), nullable=False)
    model_version = Column(String(50), nullable=False)

    framework = Column(String(50), nullable=True)
    artifact_path = Column(String(255), nullable=True)

    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(
        DateTime(timezone=False),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    site = relationship("Site", back_populates="site_models")

    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "model_type",
            "model_version",
            name="uq_site_model_version",
        ),
    )

    def __repr__(self):
        return (
            f"<SiteModel(site_id={self.site_id}, "
            f"model_type='{self.model_type}', "
            f"model_version='{self.model_version}', "
            f"is_active={self.is_active})>"
        )
    
class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True)
    site_id = Column(
        Integer,
        ForeignKey("sites.id", ondelete="CASCADE"),
        nullable=False,
    )
    start_time = Column(DateTime(timezone=False), nullable=False)
    end_time = Column(DateTime(timezone=False), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=False), nullable=False, server_default=text("NOW()"))
    error_message = Column(String, nullable=True)
    manual_pv_48 = Column(JSON, nullable=True)

    site = relationship("Site", back_populates="optimization_runs")
    optimization_data = relationship("OptimizationData", back_populates="optimization_runs", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_optimization_run_site_created", "site_id", "created_at"),
    )


    def __repr__(self):
        return (
            f"<OptimizationRun(id={self.id}, site_id={self.site_id}, "
            f"start_time='{self.start_time}', end_time='{self.end_time}', "
            f"status='{self.status}')>"
        )
    
class OptimizationData(Base):
    __tablename__ = 'optimization_data'

    run_id = Column(Integer, ForeignKey('optimization_runs.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)
    tin = Column(Float, nullable=False)
    rh = Column(Float, nullable=False)
    hvac_mode = Column(SmallInteger, nullable=False)
    comfort_index = Column(Float, nullable=False)
    

    optimization_runs = relationship("OptimizationRun", back_populates="optimization_data")

    __table_args__ = (
        Index("ix_optimization_data_run_ts", "run_id", "timestamp"),
    )

    def __repr__(self):
        return f"<OptimizationData(run_id={self.run_id}, timestamp='{self.timestamp}', tin={self.tin}, rh={self.rh}, hvac_mode={self.hvac_mode}, comfort_index={self.comfort_index})>"

class ProductionData(Base):
    __tablename__ = 'production_data'

    site_id = Column(Integer, ForeignKey('sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)
    value = Column(Float, nullable=False)

    site = relationship("Site", back_populates="production_data")

    __table_args__ = (
        Index(
            "ix_production_site_ts_desc",
            "site_id",
            text("timestamp DESC"),
        ),
    )

    def __repr__(self):
        return f"<ProductionData(site_id={self.site_id}, timestamp='{self.timestamp}', value={self.value})>"

class ForecastedConsumptionData(Base):
    __tablename__ = 'forecasted_consumption_data'

    site_id = Column(Integer, ForeignKey('sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)
    value = Column(Float, nullable=False)

    site = relationship("Site", back_populates="forecasted_consumption_data")

    __table_args__ = (
        Index(
            "ix_forecasted_consumption_site_ts_desc",
            "site_id",
            text("timestamp DESC"),
        ),
    )

    def __repr__(self):
        return f"<ForecastedConsumptionData(site_id={self.site_id}, timestamp='{self.timestamp}', value={self.value})>"

class ForecastedProductionData(Base):
    __tablename__ = 'forecasted_production_data'

    site_id = Column(Integer, ForeignKey('sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime(timezone=False), primary_key=True)
    value = Column(Float, nullable=False)

    site = relationship("Site", back_populates="forecasted_production_data")

    __table_args__ = (
        Index(
            "ix_forecasted_production_site_ts_desc",
            "site_id",
            text("timestamp DESC"),
        ),
    )

    def __repr__(self):
        return f"<ForecastedProductionData(site_id={self.site_id}, timestamp='{self.timestamp}', value={self.value})>"
    

# Uncomment and modify these functions as needed
# DATABASE_URL = "postgresql://enpower:enpower@147.102.6.183:5432"

# def get_engine():
#     return create_engine(DATABASE_URL)

# def get_session():
#     engine = get_engine()
#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     return SessionLocal()
