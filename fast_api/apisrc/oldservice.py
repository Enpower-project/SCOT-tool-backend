from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, Mapped
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI
from sqlalchemy import Column, Integer, String, Float, Index
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import os
from dotenv import load_dotenv

import numpy as np
import joblib

load_dotenv()

tags_metadata = [
    {"name": "Hello World", "description": "REST API for hello word"},
]

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


#engine1 = create_engine(my_database_connection, pool_pre_ping = True)
#Base1 = automap_base()
#base = Base1.prepare(engine1, reflect = True)
#basesession = sessionmaker(bind=engine1)
#session = basesession()

#xt = Base1.classes.energy_efficiency_measures


# engine = create_engine(my_database_connection, pool_pre_ping = True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=True, bind=engine)
Base = declarative_base()

series_list = []



Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="FLEXIBILITY TOOL API",
    description="Collection of REST APIs for Serving Execution of Self Consumption Optimization tool for ENPOWER EU",
    version="0.0.1",
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()





"""
@app.get("/buildings/", tags=['Building Information'])
async def read_buildings(db:Session = Depends(get_db)):
    buildings = db.query(Building).all()
    #series_list.clear()
    #all_u_columns.clear()
    #all_area_columns.clear()
    #all_remaining_columns.clear()
    #all_inputs.clear()
    for index, i in enumerate(buildings):
        i.building_id = index
    return buildings

"""

# """
# @app.get("/production-analytics/emission-reduction/", tags=["Database"])
# def fetch_data_from_database(cel_id: str, db: Session = Depends(get_db)):
#     """

#     Return the weather features (MIN, MAX AVG) for a specific date (day)
    
#     """

#     cel_mapping = {
#         "cel3-pv": "CN506 - Minoan Energy - Sarafali Mandra"
#     }

#     cel = cel_mapping.get(cel_id)

#     if not cel:
#         return {"error": "Invalid cel_id provided"}

#     query = text("""
#             WITH total_power AS (
#                 SELECT 
#                     collect_time, SUM(active_power) AS total_active_power
#                 FROM inverter_data
#                 WHERE plant_code = :cel
#                 GROUP BY collect_time
#             )
#             SELECT SUM(total_active_power) FROM total_power 
#     """)

#     result = db.execute(query, {"cel": cel}).fetchone()

#     energy_kW = result[0]

#     energy = (energy_kW * 0.0833)/1000 #transform energy to MWh

#     emissions = energy * 0.312  #calculate the tons of CO2 emissions saved 

#     #hard_coal_saved = energy * 0.52  #calculate the tons of hard coal saved

#     lignite_saved = energy * 1.5  #calculate the tons of lignite saved (Greece context)

#     trees_planted = emissions/0.039  #calculate the equivalent of number of trees planted


#     try:

#         final = {
#             "co2_emissions_saved": round(emissions, 3),
#             "lignite_saved": round(lignite_saved, 3),
#             "equivalent_tree": round(trees_planted, 3)
#         }

#         return final

#     except Exception as e:
#         return {"error": str(e)}

# """
@app.get("/", tags=['Hello World'])
async def read_root():
    return {"message": "Hello, World!"}
