from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import logging
import httpx

try:
    from core.schemas import CurrentWeatherMetrics
except ImportError as e:
    print(f"ERROR: Failed to import schemas from core.schemas - {e}")
    raise

router = APIRouter(prefix="/weather", tags=["Weather"])

ISLAND_LATITUDE = 36.2333
ISLAND_LONGITUDE = 27.5667
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@router.get(
    "/current",
    response_model=CurrentWeatherMetrics,
    summary="Get current weather metrics for the island"
)
async def get_current_weather():
    """
    Retrieves the latest available ('current') weather metrics from Open-Meteo
    for the predefined island coordinates (36.23°N, 27.57°E).

    Includes: Temperature, Humidity, Wind Speed, and Instantaneous GHI Solar Radiation.
    """
    current_params = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation_instant",  # GHI Instant
        "weather_code",
    ]

    params = {
        "latitude": ISLAND_LATITUDE,
        "longitude": ISLAND_LONGITUDE,
        "current": ",".join(current_params),
        "timezone": "Europe/Athens",  # Correct timezone for the location

    }

    logging.info(
        f"Requesting current weather from Open-Meteo with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()  # Raise HTTP errors
            data = response.json()

            # --- Data Extraction and Validation ---
            if "current" not in data:
                logging.error(
                    "Open-Meteo response missing 'current' data", extra={"data": data})
                raise HTTPException(
                    status_code=502, detail="Invalid response format from weather service (missing 'current').")

            current_weather = data.get("current", {})

            # Check if all requested keys are present
            required_keys = ["time", "temperature_2m", "relative_humidity_2m",
                             "wind_speed_10m", "shortwave_radiation_instant"]
            if not all(key in current_weather for key in required_keys):
                logging.error(
                    f"Open-Meteo 'current' data missing expected keys. Got: {current_weather.keys()}", extra={"data": data})
                raise HTTPException(
                    status_code=502, detail="Invalid response format from weather service (missing current keys).")

            # --- Create Response Object ---
            try:
                # Open-Meteo 'current.time' is typically seconds since epoch or ISO string depending on API/version
                # Let's assume ISO string based on recent API behavior
                timestamp_dt = datetime.fromisoformat(current_weather["time"])
                if timestamp_dt.tzinfo is None:
                    # If no timezone info, treat as UTC
                    timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)

            except (ValueError, KeyError) as e:
                logging.error(f"Error parsing timestamp from Open-Meteo: {e}", extra={
                              "time_value": current_weather.get("time")})
                raise HTTPException(
                    status_code=502, detail="Could not parse timestamp from weather service.")

            metrics = CurrentWeatherMetrics(
                timestamp_utc=timestamp_dt,
                temperature_celsius=current_weather.get("temperature_2m"),
                humidity_percent=current_weather.get("relative_humidity_2m"),
                wind_speed_kmh=current_weather.get("wind_speed_10m"),
                solar_radiation_ghi_instant=current_weather.get(
                    "shortwave_radiation_instant"),

            )

            logging.info(
                f"Successfully retrieved and parsed current weather: {metrics}")
            return metrics

    except httpx.RequestError as e:
        logging.error(
            f"Error requesting current weather from Open-Meteo: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Could not connect to weather service: {e}")
    except httpx.HTTPStatusError as e:
        logging.error(
            f"Open-Meteo API returned an error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code,
                            # Pass status code through
                            detail=f"Weather service error: {e.response.text}")
    except Exception as e:
        logging.error(
            f"Unexpected error fetching current weather: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while fetching weather data.")