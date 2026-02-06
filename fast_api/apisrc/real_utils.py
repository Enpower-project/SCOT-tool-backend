import sys
from typing import Optional, Tuple
from datetime import datetime
import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, ForeignKey, Index, func
import requests
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

# real_utils.py lives in fast_api/apisrc, so project root is two levels up.
ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_api.apisrc.core.models import Site, ComfortData, ProductionData, ConsumptionData
from zoneinfo import ZoneInfo


ATHENS = ZoneInfo("Europe/Athens")
UTC = ZoneInfo("UTC")

def to_utc_ts(dt_like, assume_tz=ATHENS) -> pd.Timestamp:
    """
    Convert dt_like to a tz-aware pandas.Timestamp in UTC.
    - If dt_like is naive (no tz), assume Europe/Athens.
    - If dt_like already has tz/offset (e.g. '...Z' or '+02:00'), respect it.
    """
    ts = pd.to_datetime(dt_like, errors="raise")

    # Pandas Timestamp has tzinfo in .tzinfo (None if naive)
    if ts.tzinfo is None:
        ts = ts.tz_localize(assume_tz)

    return ts.tz_convert(UTC)

def fmt_rfc3339_z(dt_like, assume_tz=ATHENS) -> str:
    """
    Format as RFC3339 with milliseconds and trailing Z (UTC).
    Example: 2025-03-01T00:00:00.000Z
    """
    ts_utc = to_utc_ts(dt_like, assume_tz=assume_tz)
    return ts_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Database configuration
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class SiemensApiConfig:
    """
    Holds all static configuration for the Siemens API.
    """
    CLIENT_ID = "TJKUZT5LnNKzUtpI70Ggi3f9vOXArmFg"
    CLIENT_SECRET = "u8JrB8PzhP4M2Ij5FsTwajvtmT-lqmZHSha8IMVE6TUM1-38a5ByOQhG2hjA6Uk1"
    AUTH_URL = "https://siemens-bt-015.eu.auth0.com/oauth/token"
    AUDIENCE = "https://horizon.siemens.com"
    BASE_URL = "https://api.bpcloud.siemens.com/operations/partitions/59dee194-e8ec-4392-b808-cfc1ece60b12"

    # Data point IDs with unique identifiers
    DATA_POINTS = {
        "pv_prod": "c68108e0-e977-5912-8aa2-0661e7ed3404",
        "pv_delivery": "f547efd8-1eee-5a9e-ab91-8298d36a1433",
        "mesi_tasi_import": "81da4407-5b27-5005-9a9a-1e67eb1bd95f",
        "mesi_tasi_export": "78e42ba5-cd51-57e7-a335-a40e48fc3968",
    }


class SimpleSiemensClient:
    """Simplified client for periodic data collection"""

    def __init__(self):
        self.config = SiemensApiConfig()
        self._token = None
        self._token_expires = 0

    def _get_token(self):
        """Get or refresh access token"""
        if self._token and time.time() < self._token_expires:
            return self._token

        payload = {
            "client_id": self.config.CLIENT_ID,
            "client_secret": self.config.CLIENT_SECRET,
            "audience": self.config.AUDIENCE,
            "grant_type": "client_credentials"
        }

        response = requests.post(self.config.AUTH_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        self._token = data['access_token']
        self._token_expires = time.time() + data['expires_in'] - 60

        return self._token

    def _get_headers(self):
        """Get auth headers"""
        return {
            'Authorization': f'Bearer {self._get_token()}',
            'Accept': 'application/json'
        }

    def get_point_data(self, point_id: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch data for a single point"""
        url = f"{self.config.BASE_URL}/points/{point_id}/values"
        params = {
            "filter[timestamp][from]": start_time,
            "filter[timestamp][to]": end_time
        }

        all_data = []
        current_url = url

        while current_url:
            response = requests.get(
                current_url, headers=self._get_headers(), params=params)
            response.raise_for_status()

            json_data = response.json()
            all_data.extend(json_data.get('data', []))
            current_url = json_data.get('links', {}).get('next')
            params = {}  # Clear params for next page

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame
        records = [item.get('attributes', {}) for item in all_data]
        df = pd.DataFrame(records)

        if df.empty or 'timestamp' not in df.columns or 'value' not in df.columns:
            return pd.DataFrame()

        # Clean and format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Replace NaN values with 0 instead of dropping them
        df['value'] = df['value'].fillna(0)
        # Only drop if timestamp is missing
        df = df.dropna(subset=['timestamp'])

        if df.empty:
            return pd.DataFrame()

        df = df.set_index('timestamp').sort_index()
        return df[['value']]

    def get_data_point(self, data_point_id: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Get data for a single data point by its unique ID"""
        if data_point_id not in self.config.DATA_POINTS:
            raise ValueError(f"Unknown data point ID: {data_point_id}")

        point_id = self.config.DATA_POINTS[data_point_id]
        df = self.get_point_data(point_id, start_time, end_time)

        if not df.empty:
            df = df.rename(columns={'value': data_point_id})

        return df

    def get_data_points(self, data_point_ids: list, start_time: str, end_time: str) -> pd.DataFrame:
        """Get data for multiple data points by their unique IDs"""
        dataframes = []

        for data_point_id in data_point_ids:
            df = self.get_data_point(data_point_id, start_time, end_time)
            if not df.empty:
                dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        # Concatenate all dataframes and fill NaN values with 0
        result = pd.concat(dataframes, axis=1, join='outer').sort_index()
        result = result.fillna(0)  # Replace NaN values with 0

        return result

    def get_location_data(self, location: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Get all metrics for a location (backward compatibility)"""
        location_mapping = {
            "pv_park": ["pv_prod", "pv_delivery"],
            "mesi_tasi": ["mesi_tasi_import", "mesi_tasi_export"]
        }

        if location not in location_mapping:
            raise ValueError(
                f"Unknown location: {location}. Available locations: {list(location_mapping.keys())}")

        data_point_ids = location_mapping[location]
        return self.get_data_points(data_point_ids, start_time, end_time)


def get_time_range_minutes(minutes_back: int = 15) -> tuple[str, str]:
    """Get time range for last N minutes (default 15)"""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes_back)

    def format_time(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return format_time(start_time), format_time(end_time)


def get_time_range_days(days_back: int = 1, hours_back: int = 0) -> tuple[str, str]:
    """
    Get time range for last N days and hours

    Args:
        days_back: Number of days back from now
        hours_back: Additional hours back from now

    Returns:
        Tuple of (start_time, end_time) in RFC3339 format

    Examples:
        # Last 7 days
        start, end = get_time_range_days(7)

        # Last 2 days and 6 hours
        start, end = get_time_range_days(2, 6)

        # Last 12 hours
        start, end = get_time_range_days(0, 12)
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back, hours=hours_back)

    def format_time(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return format_time(start_time), format_time(end_time)


def resample_data(df: pd.DataFrame, interval: str = '15min', method: str = 'mean') -> pd.DataFrame:
    """
    Resample data to specified interval

    Args:
        df: DataFrame with datetime index
        interval: Resampling interval ('15min', '30min', '1h', etc.)
        method: Aggregation method ('mean', 'sum', 'last')

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    if method == 'mean':
        return df.resample(interval).mean()
    elif method == 'sum':
        return df.resample(interval).sum()
    elif method == 'last':
        return df.resample(interval).last()
    else:
        return df.resample(interval).mean()


def calculate_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cumulative data to interval values.
    Calculates the difference between consecutive readings.

    Args:
        df: DataFrame with cumulative data.

    Returns:
        A new DataFrame with interval values.
    """
    if df.empty:
        return df

    interval_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        if df[col].isna().all():
            continue

        intervals = df[col].diff()

        # Handle meter resets (negative values) by setting them to 0
        intervals[intervals < 0] = 0

        interval_df[col] = intervals

    # The first row will be NaN. With the extended time range, this row is before
    # our period of interest, so we can safely drop it.
    return interval_df.dropna()

#get environmental data tin,rh
def collect_comfort_data(location_name: str, data_point_id: str, start_time: str, end_time: str,  resample_interval: str = '30min'):
    client = SimpleSiemensClient()
    local_tz = "Europe/Athens"

    try:
        # Extend the start_time to fetch the last data point before the requested window
        start_time_dt = pd.to_datetime(start_time)
        extended_start_time = (start_time_dt - pd.Timedelta(minutes=30)
                               ).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        # df = client.get_data_point(
        #     data_point_id, extended_start_time, end_time)
        df = client.get_point_data(data_point_id, extended_start_time, end_time)
        # print(df)
        if not df.empty:
            df = df.rename(columns={'value': data_point_id})

        if df.empty or len(df) < 2:
            print(
                f"✗ Not enough data for {data_point_id}, {location_name} to calculate intervals.")
            return pd.DataFrame()

        print(
            f"  📊 Raw data (with lookbehind): {len(df)} records from {df.index.min()} to {df.index.max()}")

        df = df[df.index >= start_time_dt]

        # Resample the data
        print(f"  🔧 Resampling to {resample_interval}")
        df_resampled = resample_data(
            df, interval=resample_interval, method='mean')

        if df_resampled.empty:
            print(f"✗ No data after resampling for {data_point_id}")
            return pd.DataFrame()

        print(
            f"  📊 After resampling: {len(df_resampled)} records from {df_resampled.index.min()} to {df_resampled.index.max()}")

        # Format numbers
        for col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].apply(
                lambda x: round(float(x), 2) if pd.notna(x) else x)

        print(
            f"✓ Collected {len(df_resampled)} records for {data_point_id}")
        
        df_resampled = df_resampled.tz_convert(local_tz)
        return df_resampled.reset_index()
        
    except Exception as e:
        print(f"✗ Error collecting {data_point_id}: {e}")
        return pd.DataFrame()
    

#get environmental data tin,rh
def collect_energy_data(location_name: str, data_point_id: str, start_time: str, end_time: str,  resample_interval: str = '30min'):
    client = SimpleSiemensClient()
    local_tz = "Europe/Athens"

    try:
        # Extend the start_time to fetch the last data point before the requested window
        start_time_dt = pd.to_datetime(start_time)
        extended_start_time = (start_time_dt - pd.Timedelta(minutes=30)
                               ).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        # df = client.get_data_point(
        #     data_point_id, extended_start_time, end_time)
        df = client.get_point_data(data_point_id, extended_start_time, end_time)
        # print(df)
        if not df.empty:
            df = df.rename(columns={'value': data_point_id})

        if df.empty or len(df) < 2:
            print(
                f"✗ Not enough data for {data_point_id}, {location_name} to calculate intervals.")
            return pd.DataFrame()

        print(
            f"  📊 Raw data (with lookbehind): {len(df)} records from {df.index.min()} to {df.index.max()}")

        df = df[df.index >= start_time_dt]

        # Resample the data
        print(f"  🔧 Resampling to {resample_interval}")
        df = calculate_intervals(df)
        df_resampled = resample_data(
            df, interval=resample_interval, method='sum')

        if df_resampled.empty:
            print(f"✗ No data after resampling for {data_point_id}")
            return pd.DataFrame()

        print(
            f"  📊 After resampling: {len(df_resampled)} records from {df_resampled.index.min()} to {df_resampled.index.max()}")

        # Format numbers
        for col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].apply(
                lambda x: round(float(x), 2) if pd.notna(x) else x)

        print(
            f"✓ Collected {len(df_resampled)} records for {data_point_id}")
        
        df_resampled = df_resampled.tz_convert(local_tz)
        return df_resampled.reset_index()
        
    except Exception as e:
        print(f"✗ Error collecting {data_point_id}: {e}")
        return pd.DataFrame()

def get_historic_complete_data(id, loc):
    client = SimpleSiemensClient()
    try:
        start_time = fmt_rfc3339_z("2025-03-01", assume_tz=ATHENS)
        end_time   = fmt_rfc3339_z(datetime.now(ATHENS), assume_tz=ATHENS)
        df = client.get_point_data(id, start_time, end_time)
        if loc == 'Dimarxeio_sunedriaston_energy':
            df = df.ffill().bfill()
            df = calculate_intervals(df)
            df_resampled = resample_data(df, interval='30min', method='sum')

            return df_resampled.reset_index()  

        df_resampled = resample_data(df, interval='30min', method='mean')

        return df_resampled.reset_index()  
    except Exception as e:
        print(f"✗ Error collecting {id}: {e}")
        return pd.DataFrame()


def collect_data(data_point_id: str, start_time: str, end_time: str,
                 convert_to_intervals: bool = True, resample_interval: str = '15min'):
    """
    Main function for data collection for a single data point.
    It fetches an extra data point before the start_time for accurate interval calculation.
    """
    client = SimpleSiemensClient()
    try:
        # Extend the start_time to fetch the last data point before the requested window
        start_time_dt = pd.to_datetime(start_time)
        extended_start_time = (start_time_dt - pd.Timedelta(minutes=30)
                               ).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        df = client.get_data_point(
            data_point_id, extended_start_time, end_time)

        if df.empty or len(df) < 2:
            print(
                f"✗ Not enough data for {data_point_id} to calculate intervals.")
            return pd.DataFrame()

        print(
            f"  📊 Raw data (with lookbehind): {len(df)} records from {df.index.min()} to {df.index.max()}")

        # Convert pv_prod from Wh to kWh by dividing by 1000
        if data_point_id == 'pv_prod':
            df[data_point_id] = df[data_point_id] / 1000

        # It's better to calculate intervals on the raw data before resampling
        if convert_to_intervals:
            print(f"  🔄 Converting to intervals...")
            df = calculate_intervals(df)  # Use a new, safer function

            if df.empty:
                print(
                    f"✗ No data after interval calculation for {data_point_id}")
                return pd.DataFrame()

            print(f"  🔄 After intervals: {len(df)} records")

        # Filter the DataFrame to the original requested time range
        df = df[df.index >= start_time_dt]

        # Resample the data
        if convert_to_intervals:
            print(f"  🔧 Resampling to {resample_interval}")
            df_resampled = resample_data(
                df, interval=resample_interval, method='sum')
        else:
            df_resampled = resample_data(
                df, interval=resample_interval, method='mean')

        if df_resampled.empty:
            print(f"✗ No data after resampling for {data_point_id}")
            return pd.DataFrame()

        print(
            f"  📊 After resampling: {len(df_resampled)} records from {df_resampled.index.min()} to {df_resampled.index.max()}")

        # Format numbers
        for col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].apply(
                lambda x: round(float(x), 2) if pd.notna(x) else x)

        print(
            f"✓ Collected {len(df_resampled)} records for {data_point_id}")
        
        return df_resampled.reset_index()  

    except Exception as e:
        print(f"✗ Error collecting {data_point_id}: {e}")
        return pd.DataFrame()


def calculate_true_consumption(start_time: str, end_time: str, resample_interval: str = '15min'):
    """
    Calculate true consumption using the formula:
    true_consumption = pv_prod - mesi_tasi_export + mesi_tasi_import

    Args:
        start_time: Start time in RFC3339 format
        end_time: End time in RFC3339 format
        resample_interval: Resampling interval ('15min', '30min', '1h')

    Returns:
        DataFrame with timestamp as index and true_consumption column

    Example:
        start, end = get_time_range_days(1)
        df = calculate_true_consumption(start, end, resample_interval='1h')
    """
    try:
        # Collect all required data points
        pv_prod = collect_data("pv_prod", start_time,
                               end_time, resample_interval=resample_interval)
        mesi_export = collect_data(
            "mesi_tasi_export", start_time, end_time, resample_interval=resample_interval)
        mesi_import = collect_data(
            "mesi_tasi_import", start_time, end_time, resample_interval=resample_interval)

        # Check if we have at least MESI data (the essential components)
        if mesi_export.empty and mesi_import.empty:
            print("✗ No MESI data available (both export and import are empty)")
            return pd.DataFrame()

        # Create a list of non-empty dataframes to determine the time range
        dataframes = []
        if not pv_prod.empty:
            dataframes.append(pv_prod)
        if not mesi_export.empty:
            dataframes.append(mesi_export)
        if not mesi_import.empty:
            dataframes.append(mesi_import)

        if not dataframes:
            print("✗ No data available from any source")
            return pd.DataFrame()

          # Set timestamp as index for proper alignment, then concatenate
        for i, df in enumerate(dataframes):
            if 'timestamp' in df.columns:
                dataframes[i] = df.set_index('timestamp')

        # Combine all available dataframes on timestamp index
        combined_df = pd.concat(dataframes, axis=1, join='outer')

        # Ensure all required columns exist (fill missing columns with 0)
        if 'pv_prod' not in combined_df.columns:
            combined_df['pv_prod'] = 0
            print("⚠ No PV production data - treating as 0")
        if 'mesi_tasi_export' not in combined_df.columns:
            combined_df['mesi_tasi_export'] = 0
            print("⚠ No MESI export data - treating as 0")
        if 'mesi_tasi_import' not in combined_df.columns:
            combined_df['mesi_tasi_import'] = 0
            print("⚠ No MESI import data - treating as 0")

        # Fill any remaining NaN values with 0
        combined_df = combined_df.fillna(0)

        # Calculate true consumption: pv_prod - mesi_export + mesi_import
        combined_df['true_consumption'] = (combined_df['pv_prod'] -
                                           combined_df['mesi_tasi_export'] +
                                           combined_df['mesi_tasi_import'])

        # Return only the true consumption column
        result = combined_df[['true_consumption']]

        print(f"✓ Calculated true consumption for {len(result)} time periods")
        return result.reset_index()

    except Exception as e:
        print(f"✗ Error calculating true consumption: {e}")
        return pd.DataFrame()


#!/usr/bin/env python3
"""
SQLAlchemy Database Import Script for PV Production and True Consumption Data
===========================================================================

This script imports pandas dataframes into PostgreSQL using SQLAlchemy with:
- Data validation and synthetic data generation for empty pv_prod
- Upsert functionality (INSERT ... ON CONFLICT DO UPDATE)
- Proper error handling and logging for cronjob execution
- Database session management

"""


# Configure logging for cronjob monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./data_import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# SQLAlchemy setup
Base = declarative_base()

# class ConsumptionData(Base):
#     """Model for consumption data table"""
#     __tablename__ = 'consumption_data'
#     site_id = Column(Integer, ForeignKey(
#         'sites.id', ondelete='CASCADE'), primary_key=True)
#     timestamp = Column(DateTime, primary_key=True)
#     value = Column(Float, nullable=False)
#     # site = relationship("Site", back_populates="consumption_data")
#     __table_args__ = (Index('ix_consumption_data_timestamp', 'timestamp'),)


# class ProductionData(Base):
#     """Model for production data table"""
#     __tablename__ = 'production_data'
#     site_id = Column(Integer, ForeignKey(
#         'sites.id', ondelete='CASCADE'), primary_key=True)
#     timestamp = Column(DateTime, primary_key=True)
#     value = Column(Float, nullable=False)
#     # site = relationship("Site", back_populates="production_data")
#     __table_args__ = (Index('ix_production_data_timestamp', 'timestamp'),)


class DataImporter:
    """Handles database operations for importing PV production and consumption data"""

    def __init__(self, database_url: str):
        """Initialize database connection and session"""
        try:
            self.engine = create_engine(database_url, echo=False)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")

    def generate_synthetic_pv_data(self, tc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic PV production data when pv_prod dataframe is empty.
        Uses timestamps from tc dataframe with values set to 0.

        Args:
            tc_df: DataFrame with columns ['timestamp', 'true_consumption']

        Returns:
            DataFrame with columns ['timestamp', 'pv_prod'] where pv_prod values are 0
        """
        if tc_df.empty:
            logger.warning("Both pv_prod and tc dataframes are empty")
            return pd.DataFrame(columns=['timestamp', 'pv_prod'])

        synthetic_df = pd.DataFrame({
            'timestamp': tc_df['timestamp'].copy(),
            'pv_prod': 0.0
        })

        logger.info(
            f"Generated synthetic PV data for {len(synthetic_df)} timestamps")
        return synthetic_df

    def validate_dataframe(self, df: pd.DataFrame, expected_columns: list, df_name: str) -> bool:
        """
        Validate dataframe structure and data types

        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            df_name: Name of dataframe for logging

        Returns:
            bool: True if valid, False otherwise
        """
        if df is None:
            logger.error(f"{df_name} is None")
            return False

        if df.empty:
            logger.warning(f"{df_name} is empty")
            return True  # Empty is valid, will be handled separately

        # Check columns
        if not all(col in df.columns for col in expected_columns):
            logger.error(
                f"{df_name} missing required columns. Expected: {expected_columns}, Got: {list(df.columns)}")
            return False

        # Check timestamp column
        if 'timestamp' in expected_columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.error(f"{df_name} timestamp conversion failed: {e}")
                return False

        # Check for null values in critical columns
        if df[expected_columns].isnull().any().any():
            logger.warning(f"{df_name} contains null values")
            # Remove rows with null values
            df.dropna(subset=expected_columns, inplace=True)
            logger.info(f"Removed null rows. {df_name} now has {len(df)} rows")

        logger.info(f"{df_name} validation passed: {len(df)} rows")
        return True

    def upsert_production_data(self, df: pd.DataFrame) -> int:
        """
        Upsert production data using PostgreSQL's ON CONFLICT DO UPDATE

        Args:
            df: DataFrame with columns ['timestamp', 'pv_prod']

        Returns:
            int: Number of rows affected
        """
        if df.empty:
            logger.info("No production data to upsert")
            return 0

        try:
            # Prepare data for upsert
            df['site_id'] = 1  # site_id = 1 for pv_prod
            records = df.rename(columns={'pv_prod': 'value'})[
                ['site_id', 'timestamp', 'value']].to_dict('records')

            # Create upsert statement
            stmt = insert(ProductionData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['site_id', 'timestamp'],
                set_=dict(
                    value=stmt.excluded.value,
                    # Could add updated_at timestamp here if column exists
                )
            )

            # Execute upsert
            result = self.session.execute(stmt)
            self.session.commit()

            rows_affected = result.rowcount
            logger.info(
                f"Production data upsert completed: {rows_affected} rows affected")
            return rows_affected

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Production data upsert failed: {e}")
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Unexpected error during production data upsert: {e}")
            raise

    def upsert_consumption_data(self, df: pd.DataFrame) -> int:
        """
        Upsert consumption data using PostgreSQL's ON CONFLICT DO UPDATE

        Args:
            df: DataFrame with columns ['timestamp', 'true_consumption']

        Returns:
            int: Number of rows affected
        """
        if df.empty:
            logger.info("No consumption data to upsert")
            return 0

        try:
            # Prepare data for upsert
            df['site_id'] = 2  # site_id = 2 for tc
            records = df.rename(columns={'true_consumption': 'value'})[
                ['site_id', 'timestamp', 'value']].to_dict('records')

            # Create upsert statement
            stmt = insert(ConsumptionData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['site_id', 'timestamp'],
                set_=dict(
                    value=stmt.excluded.value,
                    # Could add updated_at timestamp here if column exists
                )
            )

            # Execute upsert
            result = self.session.execute(stmt)
            self.session.commit()

            rows_affected = result.rowcount
            logger.info(
                f"Consumption data upsert completed: {rows_affected} rows affected")
            return rows_affected

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Consumption data upsert failed: {e}")
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Unexpected error during consumption data upsert: {e}")
            raise

    def upsert_comfort_data(self, df_tin: pd.DataFrame, df_rh: pd.DataFrame, loc: str) -> int:
        """
        Upsert environmental data using PostgreSQL's ON CONFLICT DO UPDATE

        Returns:
            int: Number of rows affected
        """
        if (df_tin is None or df_tin.empty) and (df_rh is None or df_rh.empty):
            return 0

        try:
            # 1) Resolve site_id from location name
            site_id = self.session.query(Site.id).filter(Site.name == loc).scalar()
            if site_id is None:
                raise ValueError(f"No site found with name='{loc}'")

            
            def _normalize(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame(columns=["timestamp", out_col])

                d = df.copy()

                if "timestamp" not in d.columns:
                    raise ValueError(f"Expected column 'timestamp' in df for {out_col}, got {list(d.columns)}")

                # If already has the desired output column, keep it
                if out_col in d.columns:
                    value_col = out_col
                # If generic shape, use "value"
                elif "value" in d.columns:
                    value_col = "value"
                else:
                    # Otherwise: accept Siemens-style column name (data_point_id)
                    candidates = [c for c in d.columns if c != "timestamp"]
                    if len(candidates) != 1:
                        raise ValueError(
                            f"Expected exactly one value column besides 'timestamp' for {out_col}, "
                            f"got {candidates} from {list(d.columns)}"
                        )
                    value_col = candidates[0]

                if value_col != out_col:
                    d = d.rename(columns={value_col: out_col})

                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
                # DB column is timezone=False; strip tz if present
                if getattr(d["timestamp"].dt, "tz", None) is not None:
                    d["timestamp"] = d["timestamp"].dt.tz_convert(None)

                d = d[["timestamp", out_col]].dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
                return d

            tin_df = _normalize(df_tin, "tin")
            rh_df  = _normalize(df_rh, "rh")
            merged = pd.merge(tin_df, rh_df, on="timestamp", how="outer")

            if merged.empty:
                return 0

            merged["site_id"] = site_id
            # Build records for insert
            records = merged[["site_id", "timestamp", "tin", "rh"]].to_dict("records")

            # 4) Upsert into ComfortData
            stmt = insert(ComfortData).values(records)

            # IMPORTANT: COALESCE prevents overwriting existing non-null values with NULL
            stmt = stmt.on_conflict_do_update(
                index_elements=["site_id", "timestamp"],
                set_={
                    "tin": func.coalesce(stmt.excluded.tin, ComfortData.tin),
                    "rh":  func.coalesce(stmt.excluded.rh,  ComfortData.rh),
                },
            )

            result = self.session.execute(stmt)
            self.session.commit()

            rows_affected = result.rowcount or 0
            logger.info(f"tin/rh upsert completed for site='{loc}' (site_id={site_id}): {rows_affected} rows affected")
            return rows_affected

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"tin/rh upsert failed for site='{loc}': {e}")
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error during tin/rh upsert for site='{loc}': {e}")
            raise

    def upsert_energy_data(self, df_energy: pd.DataFrame, loc: str) -> int:
        """
        Upsert environmental data using PostgreSQL's ON CONFLICT DO UPDATE

        Returns:
            int: Number of rows affected
        """
        if (df_energy is None or df_energy.empty):
            return 0

        try:
            # 1) Resolve site_id from location name
            site_id = self.session.query(Site.id).filter(Site.name == loc).scalar()
            if site_id is None:
                raise ValueError(f"No site found with name='{loc}'")

            
            def _normalize(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame(columns=["timestamp", out_col])

                d = df.copy()

                if "timestamp" not in d.columns:
                    raise ValueError(f"Expected column 'timestamp' in df for {out_col}, got {list(d.columns)}")

                # If already has the desired output column, keep it
                if out_col in d.columns:
                    value_col = out_col
                # If generic shape, use "value"
                elif "value" in d.columns:
                    value_col = "value"
                else:
                    # Otherwise: accept Siemens-style column name (data_point_id)
                    candidates = [c for c in d.columns if c != "timestamp"]
                    if len(candidates) != 1:
                        raise ValueError(
                            f"Expected exactly one value column besides 'timestamp' for {out_col}, "
                            f"got {candidates} from {list(d.columns)}"
                        )
                    value_col = candidates[0]

                if value_col != out_col:
                    d = d.rename(columns={value_col: out_col})

                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
                # DB column is timezone=False; strip tz if present
                if getattr(d["timestamp"].dt, "tz", None) is not None:
                    d["timestamp"] = d["timestamp"].dt.tz_convert(None)

                d = d[["timestamp", out_col]].dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
                return d

            energy_df = _normalize(df_energy, "energy_consumption")

            energy_df["site_id"] = site_id
            # Build records for insert
            records = energy_df[["site_id", "timestamp", "value"]].to_dict("records")

            # 4) Upsert into ComfortData
            stmt = insert(ConsumptionData).values(records)

            # IMPORTANT: COALESCE prevents overwriting existing non-null values with NULL
            stmt = insert(ConsumptionData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["site_id", "timestamp"],
                set_={
                    "value": func.coalesce(stmt.excluded.value, ConsumptionData.value),
                },
            )

            result = self.session.execute(stmt)
            self.session.commit()


            rows_affected = result.rowcount or 0
            logger.info(
                f"consumption upsert completed for site='{loc}' (site_id={site_id}): {rows_affected} rows affected"
            )
            return rows_affected
        
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"energy upsert failed for site='{loc}': {e}")
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error during energy upsert for site='{loc}': {e}")
            raise


def import_dataframes_to_db(pv_prod_df: Optional[pd.DataFrame],
                            tc_df: Optional[pd.DataFrame]) -> Tuple[int, int]:
    """
    Main function to import dataframes into PostgreSQL database

    Args:
        pv_prod_df: DataFrame with columns ['timestamp', 'pv_prod']
        tc_df: DataFrame with columns ['timestamp', 'true_consumption']

    Returns:
        Tuple[int, int]: (production_rows_affected, consumption_rows_affected)
    """
    start_time = datetime.now()
    logger.info(f"Starting data import process at {start_time}")

    try:
        with DataImporter(DATABASE_URL) as importer:
            production_rows = 0
            consumption_rows = 0

            # Validate consumption data first (needed for synthetic generation)
            if tc_df is not None and importer.validate_dataframe(tc_df, ['timestamp', 'true_consumption'], 'tc'):
                if not tc_df.empty:
                    consumption_rows = importer.upsert_consumption_data(
                        tc_df.copy())

            # Handle production data
            if pv_prod_df is not None and importer.validate_dataframe(pv_prod_df, ['timestamp', 'pv_prod'], 'pv_prod'):
                # Check if pv_prod is empty and generate synthetic data if needed
                if pv_prod_df.empty and tc_df is not None and not tc_df.empty:
                    logger.info(
                        "pv_prod is empty, generating synthetic data from tc timestamps")
                    pv_prod_df = importer.generate_synthetic_pv_data(tc_df)

                if not pv_prod_df.empty:
                    production_rows = importer.upsert_production_data(
                        pv_prod_df.copy())

            duration = datetime.now() - start_time
            logger.info(
                f"Data import completed successfully in {duration.total_seconds():.2f} seconds")
            logger.info(
                f"Summary: {production_rows} production rows, {consumption_rows} consumption rows affected")

            return production_rows, consumption_rows

    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise



def insert_to_db_comfort(df_tin, df_rh, loc):
    start_time = datetime.now()
    logger.info(f"Starting data import process at {start_time}")
    try:
        with DataImporter(DATABASE_URL) as importer:
            comfort_rows = 0
            comfort_rows = importer.upsert_comfort_data(df_tin.copy(), df_rh.copy(), loc)
            duration = datetime.now() - start_time
            logger.info(
                f"Data import completed successfully in {duration.total_seconds():.2f} seconds")
            
            return comfort_rows
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise


def insert_to_db_energy(df_energy, loc):
    start_time = datetime.now()
    logger.info(f"Starting data import process at {start_time}")
    try:
        with DataImporter(DATABASE_URL) as importer:
            energy_rows = 0
            energy_rows = importer.upsert_energy_data(df_energy.copy(), loc)
            duration = datetime.now() - start_time
            logger.info(
                f"Data import completed successfully in {duration.total_seconds():.2f} seconds")
           
            return energy_rows
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise
