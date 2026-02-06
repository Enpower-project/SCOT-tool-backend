from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, date, time
import logging

try:
    from core.models import ConsumptionData, ProductionData, ForecastedConsumptionData, ForecastedProductionData
    from core.schemas import (
        LifecycleEmissionReductionResponse, SiteDailyEnergyData,
        ConsumptionDailyData, ProductionDailyData, TimeSeriesPoint
    )
    from core.database import get_db
except ImportError as e:
    print(f"ERROR: Failed to import dependencies - {e}")
    raise

router = APIRouter(tags=["Analytics", "Energy Data"])

# Define constants for factors
CO2_EMISSION_FACTOR_TONS_PER_MWH = 0.312
LIGNITE_SAVED_FACTOR_TONS_PER_MWH = 1.5
TREES_ABSORPTION_TONS_CO2_PER_TREE = 0.039  # Factor for tree equivalency
MWH_TO_KWH_FACTOR = 1000.0

# Define the fixed Production Site ID
PRODUCTION_SITE_ID_FOR_PV = 1


@router.get(
    "/production/emission-reduction/total",
    response_model=LifecycleEmissionReductionResponse,
    summary="Calculate Total Lifecycle Emission Reductions for PV Production"
)
async def get_total_lifecycle_emission_reduction(
    db: Session = Depends(get_db)
):
    """
    Calculates the estimated total emission reductions based on the entire
    history of actual energy produced by the PV site (site_id=1).

    - Uses all energy data from `production_data` table for `site_id=1`.
    - Assumes energy data in the table is in kWh.
    - Converts total kWh to MWh before applying emission factors.
    """
    logging.info(
        f"Calculating total lifecycle emission reductions for site {PRODUCTION_SITE_ID_FOR_PV}")

    try:
        # --- Query to sum energy for the specific site across ALL time ---
        total_energy_kwh_result = db.query(func.sum(ProductionData.value)).filter(
            ProductionData.site_id == PRODUCTION_SITE_ID_FOR_PV
        ).scalar()  # scalar() returns the single sum value or None

        # Handle case where no data exists for the site at all
        total_energy_kwh = total_energy_kwh_result if total_energy_kwh_result is not None else 0.0

        logging.info(
            f"Total lifecycle energy produced (kWh): {total_energy_kwh}")

        # --- Calculate Metrics ---
        if total_energy_kwh > 0:
            total_energy_mwh = total_energy_kwh / MWH_TO_KWH_FACTOR

            co2_saved = total_energy_mwh * CO2_EMISSION_FACTOR_TONS_PER_MWH

            lignite_saved = total_energy_mwh * LIGNITE_SAVED_FACTOR_TONS_PER_MWH

            trees_planted = co2_saved / \
                TREES_ABSORPTION_TONS_CO2_PER_TREE if TREES_ABSORPTION_TONS_CO2_PER_TREE > 0 else 0

            # Prepare response values with rounding
            response_data = {
                "total_energy_produced_kwh": round(total_energy_kwh, 3),
                "total_energy_produced_mwh": round(total_energy_mwh, 3),
                "co2_emissions_saved_tons": round(co2_saved, 3),
                "lignite_saved_tons": round(lignite_saved, 3),
                # Round trees to whole number
                "equivalent_trees_planted": round(trees_planted, 0)
            }
        else:
            # No production recorded ever for this site
            response_data = {
                "total_energy_produced_kwh": 0.0,
                "total_energy_produced_mwh": 0.0,
                "co2_emissions_saved_tons": 0.0,
                "lignite_saved_tons": 0.0,
                "equivalent_trees_planted": 0.0
            }

        return LifecycleEmissionReductionResponse(
            production_site_id=PRODUCTION_SITE_ID_FOR_PV,
            **response_data  # Unpack the calculated values
        )

    except SQLAlchemyError as e:
        logging.error(
            f"Database error calculating total lifecycle emission reduction for site {PRODUCTION_SITE_ID_FOR_PV}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred.")
    except Exception as e:
        logging.error(
            f"Unexpected error calculating total lifecycle emission reduction for site {PRODUCTION_SITE_ID_FOR_PV}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred.")


@router.get(
    "/sites/{site_id}/energy_data/{target_date_str}",
    response_model=SiteDailyEnergyData,
    summary="Get Actual and Forecasted Energy Data for a Consumption Site"
)
async def get_site_daily_energy_data(
    site_id: int,  # This ID is for the CONSUMPTION site
    target_date_str: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves the actual and forecasted consumption data for the specified `site_id`,
    and the actual and forecasted production data (always from site_id=1),
    for the given target date.

    - **site_id**: The integer ID of the CONSUMPTION site.
    - **target_date_str**: The target date in YYYY-MM-DD format.
    """
    # --- Fixed Production Site ID ---
    PRODUCTION_SITE_ID = 1

    try:
        target_date = date.fromisoformat(target_date_str)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    start_dt_utc = datetime.combine(target_date, time.min)
    end_dt_utc = datetime.combine(target_date, time.max)

    logging.info(
        f"Fetching data for consumption site {site_id} and production site {PRODUCTION_SITE_ID} on {target_date_str}")

    try:
        # --- Fetch Consumption Data  ---
        actual_con_results = db.query(ConsumptionData).filter(
            ConsumptionData.site_id == site_id,
            ConsumptionData.timestamp >= start_dt_utc,
            ConsumptionData.timestamp <= end_dt_utc
        ).order_by(ConsumptionData.timestamp).all()

        forecast_con_results = db.query(ForecastedConsumptionData).filter(
            ForecastedConsumptionData.site_id == site_id,
            ForecastedConsumptionData.timestamp >= start_dt_utc,
            ForecastedConsumptionData.timestamp <= end_dt_utc
        ).order_by(ForecastedConsumptionData.timestamp).all()

        # --- Fetch Production Data  ---
        actual_prod_results = db.query(ProductionData).filter(
            ProductionData.site_id == PRODUCTION_SITE_ID,
            ProductionData.timestamp >= start_dt_utc,
            ProductionData.timestamp <= end_dt_utc
        ).order_by(ProductionData.timestamp).all()

        forecast_prod_results = db.query(ForecastedProductionData).filter(
            ForecastedProductionData.site_id == PRODUCTION_SITE_ID,
            ForecastedProductionData.timestamp >= start_dt_utc,
            ForecastedProductionData.timestamp <= end_dt_utc
        ).order_by(ForecastedProductionData.timestamp).all()

        # --- Structure the Response ---
        logging.info(
            f"Found {len(actual_con_results)} actual consumption records for site {site_id}.")
        logging.info(
            f"Found {len(forecast_con_results)} forecast consumption records for site {site_id}.")
        logging.info(
            f"Found {len(actual_prod_results)} actual production records for site {PRODUCTION_SITE_ID}.")
        logging.info(
            f"Found {len(forecast_prod_results)} forecast production records for site {PRODUCTION_SITE_ID}.")

        consumption_data = None
        if actual_con_results or forecast_con_results:
            consumption_data = ConsumptionDailyData(
                actuals=[TimeSeriesPoint(
                    timestamp=rec.timestamp, value=rec.value) for rec in actual_con_results],
                forecasts=[TimeSeriesPoint(
                    timestamp=rec.timestamp, value=rec.value) for rec in forecast_con_results]
            )

        production_data = None
        if actual_prod_results or forecast_prod_results:
            production_data = ProductionDailyData(
                actuals=[TimeSeriesPoint(
                    timestamp=rec.timestamp, value=rec.value) for rec in actual_prod_results],
                forecasts=[TimeSeriesPoint(
                    timestamp=rec.timestamp, value=rec.value) for rec in forecast_prod_results]
            )

        # Return the combined data, reporting the requested consumption site ID
        return SiteDailyEnergyData(
            site_id=site_id,
            target_date=target_date,
            consumption=consumption_data,
            production=production_data
        )

    except SQLAlchemyError as e:
        logging.error(
            f"Database error fetching energy data for cons site {site_id}, prod site {PRODUCTION_SITE_ID}, date {target_date_str}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred.")
    except Exception as e:
        logging.error(
            f"Unexpected error fetching energy data for cons site {site_id}, prod site {PRODUCTION_SITE_ID}, date {target_date_str}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred.")