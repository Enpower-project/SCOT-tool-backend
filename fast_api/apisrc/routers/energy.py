from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, date as date_type, time
from typing import List, Optional
from enum import Enum
import logging
from zoneinfo import ZoneInfo

try:
    from core.models import ConsumptionData, ProductionData, ForecastedConsumptionData, ForecastedProductionData
    from core.database import get_db
except ImportError as e:
    print(f"ERROR: Failed to import dependencies - {e}")
    raise

from pydantic import BaseModel, Field

class DataType(str, Enum):
    historical = "historical"
    forecast = "forecast"

class EnergyDataPoint(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: float = Field(..., description="Energy value")

class EnergyDataResponse(BaseModel):
    data: List[EnergyDataPoint]
    site_id: int = Field(..., description="Site identifier")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    data_type: str = Field(..., description="Type of data: historical or forecast")

class ErrorResponse(BaseModel):
    error: str
    code: str

router = APIRouter(prefix="/api/energy", tags=["Energy Data"])

@router.get(
    "/production",
    response_model=EnergyDataResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "No data found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get Production Data"
)
async def get_production_data(
    data_type: DataType = Query(..., description="Type of data: historical or forecast"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    timezone: Optional[str] = Query(None, description="Timezone (e.g., 'Europe/Athens'). If not provided, treats date as UTC."),
    db: Session = Depends(get_db)
):
    """
    Retrieves production data for the specified date.
    Production data is always from site_id=1.
    
    If timezone is provided, the date is interpreted in that timezone and the API returns
    UTC data covering the full day (00:00-24:00) in the specified timezone.
    If no timezone is provided, maintains backward compatibility by treating date as UTC.
    """
    try:
        target_date = date_type.fromisoformat(date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid date format. Expected YYYY-MM-DD", "code": "INVALID_DATE_FORMAT"}
        )

    # Production data is always from site_id=1
    production_site_id = 1
    
    # Calculate UTC time range based on timezone
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Invalid timezone: {timezone}", "code": "INVALID_TIMEZONE"}
            )
        
        # Create datetime in the specified timezone for start and end of day
        start_local = datetime.combine(target_date, time.min).replace(tzinfo=tz)
        end_local = datetime.combine(target_date, time.max).replace(tzinfo=tz)
        
        # Convert to UTC
        start_dt_utc = start_local.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
        end_dt_utc = end_local.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
    else:
        # Backward compatibility: treat date as UTC
        start_dt_utc = datetime.combine(target_date, time.min)
        end_dt_utc = datetime.combine(target_date, time.max)

    logging.info(f"Fetching {data_type} production data for site {production_site_id} for date {date}")

    try:
        if data_type == DataType.historical:
            results = db.query(ProductionData).filter(
                ProductionData.site_id == production_site_id,
                ProductionData.timestamp >= start_dt_utc,
                ProductionData.timestamp <= end_dt_utc
            ).order_by(ProductionData.timestamp).all()
        else:  # forecast
            results = db.query(ForecastedProductionData).filter(
                ForecastedProductionData.site_id == production_site_id,
                ForecastedProductionData.timestamp >= start_dt_utc,
                ForecastedProductionData.timestamp <= end_dt_utc
            ).order_by(ForecastedProductionData.timestamp).all()

        logging.info(f"Found {len(results)} {data_type} production records")

        if not results:
            raise HTTPException(
                status_code=404,
                detail={"error": f"No {data_type} production data found for date {date}", "code": "NO_DATA_FOUND"}
            )

        # Convert to response format with timestamp-value pairs
        data_points = [
            EnergyDataPoint(
                timestamp=rec.timestamp.isoformat() + 'Z',
                value=rec.value
            ) for rec in results
        ]

        return EnergyDataResponse(
            data=data_points,
            site_id=production_site_id,
            date=date,
            data_type=data_type.value
        )

    except SQLAlchemyError as e:
        logging.error(f"DB error fetching {data_type} production data for date {date}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Database error occurred", "code": "DATABASE_ERROR"}
        )
    except Exception as e:
        logging.error(f"Unexpected error fetching {data_type} production data for date {date}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "An unexpected error occurred", "code": "INTERNAL_ERROR"}
        )

@router.get(
    "/consumption",
    response_model=EnergyDataResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "No data found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get Consumption Data"
)
async def get_consumption_data(
    data_type: DataType = Query(..., description="Type of data: historical or forecast"),
    site_id: int = Query(..., description="Site identifier"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    timezone: Optional[str] = Query(None, description="Timezone (e.g., 'Europe/Athens'). If not provided, treats date as UTC."),
    db: Session = Depends(get_db)
):
    """
    Retrieves consumption data for the specified date and site.
    
    If timezone is provided, the date is interpreted in that timezone and the API returns
    UTC data covering the full day (00:00-24:00) in the specified timezone.
    If no timezone is provided, maintains backward compatibility by treating date as UTC.
    """
    try:
        target_date = date_type.fromisoformat(date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid date format. Expected YYYY-MM-DD", "code": "INVALID_DATE_FORMAT"}
        )

    # Calculate UTC time range based on timezone
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Invalid timezone: {timezone}", "code": "INVALID_TIMEZONE"}
            )
        
        # Create datetime in the specified timezone for start and end of day
        start_local = datetime.combine(target_date, time.min).replace(tzinfo=tz)
        end_local = datetime.combine(target_date, time.max).replace(tzinfo=tz)
        
        # Convert to UTC
        start_dt_utc = start_local.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
        end_dt_utc = end_local.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
    else:
        # Backward compatibility: treat date as UTC
        start_dt_utc = datetime.combine(target_date, time.min)
        end_dt_utc = datetime.combine(target_date, time.max)

    logging.info(f"Fetching {data_type} consumption data for site {site_id} for date {date}")

    try:
        if data_type == DataType.historical:
            results = db.query(ConsumptionData).filter(
                ConsumptionData.site_id == site_id,
                ConsumptionData.timestamp >= start_dt_utc,
                ConsumptionData.timestamp <= end_dt_utc
            ).order_by(ConsumptionData.timestamp).all()
        else:  # forecast
            results = db.query(ForecastedConsumptionData).filter(
                ForecastedConsumptionData.site_id == site_id,
                ForecastedConsumptionData.timestamp >= start_dt_utc,
                ForecastedConsumptionData.timestamp <= end_dt_utc
            ).order_by(ForecastedConsumptionData.timestamp).all()

        logging.info(f"Found {len(results)} {data_type} consumption records")

        if not results:
            raise HTTPException(
                status_code=404,
                detail={"error": f"No {data_type} consumption data found for site {site_id} on date {date}", "code": "NO_DATA_FOUND"}
            )

        # Convert to response format with timestamp-value pairs
        data_points = [
            EnergyDataPoint(
                timestamp=rec.timestamp.isoformat() + 'Z',
                value=rec.value
            ) for rec in results
        ]

        return EnergyDataResponse(
            data=data_points,
            site_id=site_id,
            date=date,
            data_type=data_type.value
        )

    except SQLAlchemyError as e:
        logging.error(f"DB error fetching {data_type} consumption data for site {site_id}, date {date}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Database error occurred", "code": "DATABASE_ERROR"}
        )
    except Exception as e:
        logging.error(f"Unexpected error fetching {data_type} consumption data for site {site_id}, date {date}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "An unexpected error occurred", "code": "INTERNAL_ERROR"}
        )