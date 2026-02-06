from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from fast_api.apisrc.core.database import SessionLocal

router = APIRouter(prefix="/metadata", tags=["metadata"])

@router.get("/get_id")
def resolve_dataset(
    building_name: str,
    dataset_name: str,
):
    session = SessionLocal()

    try:
        sql = text("""
        SELECT
            b.building_id,
            d.dataset_id
        FROM building b
        JOIN dataset d
            ON d.building_id = b.building_id
        WHERE b.name = :building_name
          AND d.name = :dataset_name
        """)

        row = (
            session.execute(
                sql,
                {
                    "building_name": building_name,
                    "dataset_name": dataset_name,
                },
            )
            .mappings()
            .first()
        )

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Building or dataset not found",
            )

        return {
            "building_id": row["building_id"],
            "dataset_id": row["dataset_id"],
        }

    finally:
        session.close()

@router.get("/get_all_buildings")
def get_all_buildings():
    session = SessionLocal()

    try:
        sql = text("""
        SELECT
            id,
            name
        FROM sites 
        """)

        rows = session.execute(sql).mappings().all()

        buildings = {}
        for row in rows:
            b_id = row["id"]

            if b_id not in buildings:
                buildings[b_id] = {
                    "site_id": b_id,
                    "site_name": row["name"],
                }

        return list(buildings.values())

    finally:
        session.close()