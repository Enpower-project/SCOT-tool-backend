# schemas.py - Pydantic models for API request/response validation
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date


class CurrentWeatherMetrics(BaseModel):
    """Relevant current weather metrics for the dashboard."""
    timestamp_utc: datetime = Field(...,
                                    description="Time of the weather reading in UTC")
    temperature_celsius: Optional[float] = Field(
        None, description="Air temperature at 2 meters (°C)")
    humidity_percent: Optional[float] = Field(
        None, description="Relative humidity at 2 meters (%)")
    wind_speed_kmh: Optional[float] = Field(
        None, description="Wind speed at 10 meters (km/h)")
    solar_radiation_ghi_instant: Optional[float] = Field(
        None, description="Global Horizontal Irradiance (W/m²), instantaneous")

    class Config:
        from_attributes = True


class LifecycleEmissionReductionResponse(BaseModel):
    production_site_id: int
    total_energy_produced_kwh: Optional[float] = None
    total_energy_produced_mwh: Optional[float] = None
    co2_emissions_saved_tons: Optional[float] = None
    lignite_saved_tons: Optional[float] = None
    equivalent_trees_planted: Optional[float] = None

    class Config:
        from_attributes = True  # For Pydantic V2


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float

    # Allow conversion from ORM objects using Pydantic V2 syntax
    class Config:
        from_attributes = True  # Changed from orm_mode = True


class ConsumptionDailyData(BaseModel):
    actuals: List[TimeSeriesPoint]
    forecasts: List[TimeSeriesPoint]


class ProductionDailyData(BaseModel):
    actuals: List[TimeSeriesPoint]
    forecasts: List[TimeSeriesPoint]


class SiteDailyEnergyData(BaseModel):
    site_id: int  # This will represent the CONSUMPTION site ID requested
    target_date: date
    consumption: Optional[ConsumptionDailyData] = None
    # Note: Production data is always from site_id=1
    production: Optional[ProductionDailyData] = None


class SiteConsumptionResponse(BaseModel):
    site_id: int
    target_date: date
    # Contains actuals and forecasts
    data: Optional[ConsumptionDailyData] = None


class ProductionResponse(BaseModel):
    target_date: date
    production_site_id: int = 1
    # Contains actuals and forecasts
    data: Optional[ProductionDailyData] = None