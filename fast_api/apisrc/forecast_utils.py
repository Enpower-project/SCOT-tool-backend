"""
Complete Forecast System
========================
This module contains all forecasting functionalities:
- PV production forecasting from API
- Consumption forecasting using hybrid logic
- Database operations for both forecast types
- Plotting and visualization functions

Requirements:
- pandas, matplotlib, pytz, requests
- sqlalchemy with PostgreSQL
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, ForeignKey, Index, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
import sys

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./forecast_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()


class ForecastedProductionData(Base):
    """Model for forecasted production data table"""
    __tablename__ = 'forecasted_production_data'
    site_id = Column(Integer, ForeignKey(
        'sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    value = Column(Float, nullable=False)
    __table_args__ = (
        Index('ix_forecasted_production_data_timestamp', 'timestamp'),)


class ForecastedConsumptionData(Base):
    """Model for forecasted consumption data table"""
    __tablename__ = 'forecasted_consumption_data'
    site_id = Column(Integer, ForeignKey(
        'sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    value = Column(Float, nullable=False)
    __table_args__ = (
        Index('ix_forecasted_consumption_data_timestamp', 'timestamp'),)


class ConsumptionData(Base):
    """Model for actual consumption data table"""
    __tablename__ = 'consumption_data'
    site_id = Column(Integer, ForeignKey(
        'sites.id', ondelete='CASCADE'), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    value = Column(Float, nullable=False)
    __table_args__ = (Index('ix_consumption_data_timestamp', 'timestamp'),)

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


class ForecastDatabase:
    """Handles all database operations for forecast data"""

    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database connection"""
        try:
            self.engine = create_engine(database_url, echo=False)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")

    def upsert_production_forecast(self, df: pd.DataFrame, site_id: int = 1) -> int:
        """Upsert PV production forecast data"""
        if df.empty:
            logger.info("No production forecast data to upsert")
            return 0

        try:
            df_copy = df.copy()
            df_copy['site_id'] = site_id
            records = df_copy.rename(columns={'energy': 'value'})[
                ['site_id', 'timestamp', 'value']].to_dict('records')

            stmt = insert(ForecastedProductionData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['site_id', 'timestamp'],
                set_=dict(value=stmt.excluded.value)
            )

            result = self.session.execute(stmt)
            self.session.commit()

            rows_affected = result.rowcount
            logger.info(
                f"Production forecast upsert completed: {rows_affected} rows affected")
            return rows_affected

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Production forecast upsert failed: {e}")
            raise

    def upsert_consumption_forecast(self, df: pd.DataFrame, site_id: int = 2) -> int:
        """Upsert consumption forecast data"""
        if df.empty:
            logger.info("No consumption forecast data to upsert")
            return 0

        try:
            df_copy = df.copy()
            df_copy['site_id'] = site_id
            records = df_copy.rename(columns={'energy': 'value'})[
                ['site_id', 'timestamp', 'value']].to_dict('records')

            stmt = insert(ForecastedConsumptionData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['site_id', 'timestamp'],
                set_=dict(value=stmt.excluded.value)
            )

            result = self.session.execute(stmt)
            self.session.commit()

            rows_affected = result.rowcount
            logger.info(
                f"Consumption forecast upsert completed: {rows_affected} rows affected")
            return rows_affected

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Consumption forecast upsert failed: {e}")
            raise

    def get_consumption_data_for_day(self, target_date_gr: datetime, timezone_str: str = 'Europe/Athens') -> pd.DataFrame:
        """Fetch consumption data for a specific day with timezone handling"""
        gr_tz = pytz.timezone(timezone_str)

        day_start_gr = target_date_gr.replace(
            hour=0, minute=0, second=0, microsecond=0)
        day_end_gr = day_start_gr + timedelta(days=1) - timedelta.resolution

        day_start_utc = day_start_gr.astimezone(pytz.utc)
        day_end_utc = day_end_gr.astimezone(pytz.utc)

        logger.info(
            f"Querying consumption data for {target_date_gr.date()} (Greece). UTC range: {day_start_utc} to {day_end_utc}")

        query = self.session.query(ConsumptionData).filter(
            and_(
                ConsumptionData.site_id == 2,
                ConsumptionData.timestamp >= day_start_utc,
                ConsumptionData.timestamp <= day_end_utc
            )
        ).order_by(ConsumptionData.timestamp)

        results = query.all()

        if not results:
            logger.warning(
                f"No consumption data found for {target_date_gr.date()}")
            return pd.DataFrame(columns=['timestamp', 'value'])

        source_data = [(row.timestamp, row.value) for row in results]
        df = pd.DataFrame(source_data, columns=['timestamp', 'value'])

        df['timestamp'] = pd.to_datetime(
            df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(gr_tz)
        df['time'] = df['timestamp'].dt.time

        return df

# =============================================================================
# PV PRODUCTION FORECASTING
# =============================================================================


def get_day_ahead_forecast():
    """
    Get day-ahead PV production forecast for tomorrow with timezone handling.
    Returns DataFrame with columns: timestamp (UTC), energy
    """
    TIMEZONE = 'Europe/Athens'
    gr_tz = pytz.timezone(TIMEZONE)
    run_time_gr = datetime.now(gr_tz)

    # API configuration
    url = "https://www.meteogen.com/api/"
    username = "vmichalakopoulos@epu.ntua.gr"
    password = "5=M/70z697?u"
    asset_id = 75605  # E.K. Chalki

    try:
        # Authentication
        auth_payload = {"username": username, "password": password}
        auth_response = requests.post(
            url + "Authenticate/Login", json=auth_payload)

        if auth_response.status_code != 200:
            raise Exception(
                f"Authentication failed with status code {auth_response.status_code}")

        token = auth_response.json().get("token")
        if not token:
            raise Exception("Token not found in authentication response")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Define forecast day in Greece timezone
        # removed the +1 to get today's forecast
        forecast_day_gr = run_time_gr
        forecast_start_gr = forecast_day_gr.replace(
            hour=0, minute=0, second=0, microsecond=0)
        forecast_end_gr = forecast_start_gr + timedelta(days=1)

        logger.info(
            f"Generating PV forecast for: {forecast_day_gr.date()} (Timezone: {TIMEZONE})")

        # Convert to UTC for API request
        forecast_start_utc = forecast_start_gr.astimezone(pytz.utc)
        forecast_end_utc = forecast_end_gr.astimezone(pytz.utc)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        forecast_payload = {
            "from": forecast_start_utc.strftime(date_format),
            "to": forecast_end_utc.strftime(date_format),
            "interval": "FifteenMinutes"
        }

        # Get forecast
        forecast_response = requests.post(
            url + f"Assets/{asset_id}/Forecast",
            headers=headers,
            json=forecast_payload
        )

        if forecast_response.status_code != 200:
            raise Exception(
                f"Forecast request failed with status code {forecast_response.status_code}")

        forecasts = forecast_response.json()

        if not forecasts:
            logger.warning("No forecast data returned from API")
            return pd.DataFrame(columns=["timestamp", "energy"])

        # Process data with timezone handling
        data = []
        for forecast in forecasts:
            timestamp_utc = datetime.strptime(
                forecast["validDate"], date_format)
            timestamp_utc_aware = pytz.utc.localize(timestamp_utc)
            timestamp_gr = timestamp_utc_aware.astimezone(gr_tz)

            energy = forecast.get("energy", 0)
            data.append([timestamp_gr, energy])

        df = pd.DataFrame(data, columns=["timestamp", "energy"])

        # Convert back to UTC for database storage
        df['timestamp'] = df['timestamp'].dt.tz_convert(
            'UTC').dt.tz_localize(None)

        logger.info(
            f"Generated PV production forecast for {len(df)} intervals")
        return df

    except Exception as e:
        logger.error(f"Failed to generate PV production forecast: {e}")
        raise

# =============================================================================
# CONSUMPTION FORECASTING
# =============================================================================


def generate_hybrid_forecast_for_today():
    """
    Generate hybrid consumption forecast for today using yesterday's and day-before-yesterday's data.
    Returns DataFrame with columns: timestamp (UTC), energy
    """
    TIMEZONE = 'Europe/Athens'
    gr_tz = pytz.timezone(TIMEZONE)
    run_time_gr = datetime.now(gr_tz)

    try:
        with ForecastDatabase() as db:
            # Define relevant days in Greece timezone
            forecast_day_gr = run_time_gr
            source_day_primary_gr = run_time_gr - timedelta(days=1)
            source_day_fallback_gr = run_time_gr - timedelta(days=2)

            logger.info(
                f"Generating consumption forecast for: {forecast_day_gr.date()} (Timezone: {TIMEZONE})")
            logger.info(
                f"Primary source: {source_day_primary_gr.date()}, Fallback source: {source_day_fallback_gr.date()}")

            # Fetch historical data
            primary_df = db.get_consumption_data_for_day(
                source_day_primary_gr, TIMEZONE)
            fallback_df = db.get_consumption_data_for_day(
                source_day_fallback_gr, TIMEZONE)

            if fallback_df.empty:
                raise Exception(
                    f"Fallback data from {source_day_fallback_gr.date()} is missing. Cannot generate forecast.")

            # Create forecast scaffold for today
            forecast_start_gr = forecast_day_gr.replace(
                hour=0, minute=0, second=0, microsecond=0)
            forecast_timestamps_gr = [
                forecast_start_gr + timedelta(minutes=15*i) for i in range(96)]
            forecast_df = pd.DataFrame(
                forecast_timestamps_gr, columns=['timestamp'])
            forecast_df['time'] = forecast_df['timestamp'].dt.time

            # Apply hybrid logic
            forecast_df = pd.merge(
                forecast_df, primary_df[['time', 'value']], on='time', how='left')
            forecast_df.rename(
                columns={'value': 'value_primary'}, inplace=True)

            forecast_df = pd.merge(
                forecast_df, fallback_df[['time', 'value']], on='time', how='left')
            forecast_df.rename(
                columns={'value': 'value_fallback'}, inplace=True)

            run_time_of_day = run_time_gr.time()
            forecast_df['energy'] = forecast_df.apply(
                lambda row: row['value_primary'] if row['time'] < run_time_of_day else row['value_fallback'],
                axis=1
            )

            # Handle missing values
            if forecast_df['energy'].isnull().any():
                logger.warning(
                    "Missing values detected. Using fallback data to fill.")
                forecast_df['energy'].fillna(
                    forecast_df['value_fallback'], inplace=True)
                forecast_df['energy'].fillna(method='ffill', inplace=True)
                forecast_df['energy'].fillna(method='bfill', inplace=True)

            if forecast_df['energy'].isnull().any():
                raise Exception(
                    "Could not resolve missing values in the final forecast.")

            # Finalize for database
            final_forecast_df = forecast_df[['timestamp', 'energy']].copy()
            final_forecast_df['timestamp'] = final_forecast_df['timestamp'].dt.tz_convert(
                'UTC').dt.tz_localize(None)

            logger.info(
                f"Generated consumption forecast for {len(final_forecast_df)} intervals")
            return final_forecast_df

    except Exception as e:
        logger.error(f"Failed to generate consumption forecast: {e}")
        raise

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_forecast(df, forecast_type="Forecast", timezone="Europe/Athens", save_path=None):
    """Plot forecast data with proper timezone display"""
    if df is None or df.empty:
        print("No data to plot")
        return

    plot_df = df.copy()
    plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
    plot_df['timestamp'] = plot_df['timestamp'].dt.tz_localize('UTC')

    local_tz = pytz.timezone(timezone)
    plot_df['timestamp'] = plot_df['timestamp'].dt.tz_convert(local_tz)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['timestamp'], plot_df['energy'],
             linewidth=2, color='blue', marker='o', markersize=3)

    plt.title(
        f'{forecast_type} - {plot_df["timestamp"].dt.date.iloc[0]}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Time ({timezone})', fontsize=12)
    plt.ylabel('Energy (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add statistics
    avg_energy = plot_df['energy'].mean()
    max_energy = plot_df['energy'].max()
    total_energy = plot_df['energy'].sum()

    stats_text = f'Avg: {avg_energy:.2f} kWh\nMax: {max_energy:.2f} kWh\nTotal: {total_energy:.2f} kWh'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    print(f"\n{forecast_type} Summary:")
    print(f"Date: {plot_df['timestamp'].dt.date.iloc[0]}")
    print(f"Data points: {len(plot_df)}")
    print(f"Average energy: {avg_energy:.2f} kWh")
    print(f"Maximum energy: {max_energy:.2f} kWh")
    print(f"Total energy: {total_energy:.2f} kWh")


def plot_multiple_forecasts(pv_df=None, consumption_df=None, timezone="Europe/Athens", save_path=None):
    """Plot both PV and consumption forecasts for comparison"""
    if (pv_df is None or pv_df.empty) and (consumption_df is None or consumption_df.empty):
        print("No data to plot")
        return

    plt.figure(figsize=(14, 8))
    local_tz = pytz.timezone(timezone)

    if pv_df is not None and not pv_df.empty:
        pv_plot = pv_df.copy()
        pv_plot['timestamp'] = pd.to_datetime(
            pv_plot['timestamp']).dt.tz_localize('UTC').dt.tz_convert(local_tz)
        plt.plot(pv_plot['timestamp'], pv_plot['energy'],
                 linewidth=2, color='orange', marker='o', markersize=3, label='PV Production')

    if consumption_df is not None and not consumption_df.empty:
        cons_plot = consumption_df.copy()
        cons_plot['timestamp'] = pd.to_datetime(
            cons_plot['timestamp']).dt.tz_localize('UTC').dt.tz_convert(local_tz)
        plt.plot(cons_plot['timestamp'], cons_plot['energy'],
                 linewidth=2, color='red', marker='s', markersize=3, label='Consumption')

    forecast_date = pv_plot['timestamp'].dt.date.iloc[0] if 'pv_plot' in locals(
    ) else cons_plot['timestamp'].dt.date.iloc[0]
    plt.title(f'Energy Forecasts - {forecast_date}',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'Time ({timezone})', fontsize=12)
    plt.ylabel('Energy (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================


def run_pv_forecast():
    """Generate and store PV production forecast"""
    try:
        logger.info("Starting PV production forecast process")
        forecast_df = get_day_ahead_forecast()

        if forecast_df.empty:
            logger.warning("No PV forecast data generated")
            return 0

        with ForecastDatabase() as db:
            rows_affected = db.upsert_production_forecast(forecast_df)

        logger.info(f"PV forecast completed: {rows_affected} rows affected")
        return rows_affected

    except Exception as e:
        logger.error(f"PV forecast process failed: {e}")
        raise


def run_consumption_forecast():
    """Generate and store consumption forecast"""
    try:
        logger.info("Starting consumption forecast process")
        forecast_df = generate_hybrid_forecast_for_today()

        if forecast_df.empty:
            logger.warning("No consumption forecast data generated")
            return 0

        with ForecastDatabase() as db:
            rows_affected = db.upsert_consumption_forecast(forecast_df)

        logger.info(
            f"Consumption forecast completed: {rows_affected} rows affected")
        return rows_affected

    except Exception as e:
        logger.error(f"Consumption forecast process failed: {e}")
        raise


def run_all_forecasts():
    """Run both PV and consumption forecasts"""
    logger.info("Starting complete forecast system")

    pv_rows = run_pv_forecast()
    consumption_rows = run_consumption_forecast()

    logger.info(
        f"All forecasts completed - PV: {pv_rows} rows, Consumption: {consumption_rows} rows")
    return pv_rows, consumption_rows
