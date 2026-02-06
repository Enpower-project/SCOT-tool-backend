from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    *,
    read_csv_kwargs: Mapping[str, Any] | None = None,
    to_parquet_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    csv_path = Path(csv_path)
    csv_only = csv_path / "House_01.csv"
    if not csv_only.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_only}")

    # if parquet_path is None:
    #     parquet_path = csv_only.with_suffix(".parquet")
    # else:
    #     parquet_path = Path(parquet_path)

    read_kwargs = dict(read_csv_kwargs or {})
    to_kwargs = {"index": False}
    to_kwargs.update(to_parquet_kwargs or {})

    df = pd.read_csv(csv_only, **read_kwargs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    BASE_DIR = Path(csv_path)
    OUTPUT_DIR = BASE_DIR / "timeseries" / "building_1" / "dataset_1" / "measured"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR,
        engine="pyarrow",
        partition_cols=["year", "month"],
        index=False
    )

    return "parquet_path"
    