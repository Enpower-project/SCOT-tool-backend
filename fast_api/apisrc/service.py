# --- Existing imports ---
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
# from fast_api.apisrc.core.reset_db import reset_database
# from fast_api.apisrc.core.seeding import seed_sensor_metrics

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("RESET_DB_ON_STARTUP") == "true":
        print("⚠️  RESET_DB_ON_STARTUP enabled — DEV MODE")

        # reset_database()
        # seed_sensor_metrics()

        print("✅ Database reset and seeded")

    yield  # Application runs here
from fast_api.apisrc.routers import weather_util, history, optimization, comfort, metadata, forecast

# --- Import routers ---
# try:
#     from routers import weather, analytics, energy
# except ImportError as e:
#     print(f"ERROR: Failed to import routers - {e}")
#     print("Ensure router files are in the routers/ directory.")
#     raise

# --- FastAPI setup ---
tags_metadata = [
    {"name": "Energy Data",
        "description": "Endpoints for retrieving energy consumption and production data"},
    {"name": "Hello World", "description": "REST API for hello word"},
    {"name": "Analytics",
     "description": "Endpoints for derived calculations like emission reductions"},
    {"name": "Weather",
     "description": "Endpoints for retrieving weather forecast and current data"},
]

app = FastAPI(
    title="FLEXIBILITY TOOL API v2",
    description="Collection of REST APIs for Serving Execution of Self Consumption Optimization tool for ENPOWER EU",
    version="0.0.1",
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,

)

# --- CORS setup ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # --- Include routers ---
# app.include_router(weather.router)
# app.include_router(analytics.router)
# app.include_router(energy.router)
app.include_router(weather_util.router)
app.include_router(comfort.router)
app.include_router(metadata.router)
app.include_router(history.router)
app.include_router(optimization.router)
app.include_router(forecast.router)
# --- Root endpoint ---


@app.get("/", tags=['Hello World'])
async def read_root():
    return {"message": "Hello, World!"}
