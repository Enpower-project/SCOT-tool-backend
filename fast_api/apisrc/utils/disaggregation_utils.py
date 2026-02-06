import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)
# ======================================================
# === 1. DAILY PROFILES & TEMPERATURE CLASSIFICATION ===
# ======================================================

def make_daily_profiles(df, load_col='total_energy', slots_per_day=48):
    """Pivot time series into daily profiles with user-defined slot count (default 48 = 30-min)."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df['timestamps'])

    df[load_col] = df[load_col].ffill()
    df['day'] = df.index.floor('D')
    freq_minutes = 24 * 60 // slots_per_day
    df['slot'] = ((df.index - df['day']) / pd.Timedelta(f'{freq_minutes}min')).astype(int)
    daily = df.pivot(index='day', columns='slot', values=load_col)
    return daily.dropna(how="all")


def classify_days_by_temp(df, temp_col='Tout', neutral_band=(45, 55)):
    """Split days into mild (non-HVAC) and HVAC-active based on daily mean temperature."""
    daily_temp = df[temp_col].resample('D').mean()
    t_low, t_high = np.percentile(daily_temp.dropna(), neutral_band)
    mild_days = daily_temp[(daily_temp >= t_low) & (daily_temp <= t_high)].index
    hvac_days = daily_temp.index.difference(mild_days)
    return list(mild_days), list(hvac_days), float(t_low), float(t_high)

# ===============================================
# === 2. BASELINE & RESIDUAL EXTRACTION UTILS ===
# ===============================================

def _medfilt3(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x
    y = x.copy()
    y[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def quantile_baseline_per_slot(daily_profiles, days, q=0.20, smooth=3):
    """Compute per-slot low quantile baseline over given mild days."""
    days = [d for d in days if d in daily_profiles.index]
    if len(days) == 0:
        base = daily_profiles.quantile(q=q, axis=0)
    else:
        base = daily_profiles.loc[days].quantile(q=q, axis=0)
    base = base.to_numpy()
    if smooth >= 3:
        base = _medfilt3(base)
    return base


def build_temp_conditioned_baselines(
    df, daily_profiles, mild_days,
    temp_col="Tout", bins=np.arange(-10, 36, 5),
    q=0.20, smooth=3, min_days_per_bin=6
):
    """Compute quantile baselines within temperature bins for mild days."""
    daily_tout = df[temp_col].resample("D").mean()
    mild_days = [pd.Timestamp(d).normalize() for d in mild_days if d in daily_profiles.index]
    global_base = quantile_baseline_per_slot(daily_profiles, mild_days, q=q, smooth=smooth)

    centers = 0.5 * (bins[:-1] + bins[1:])
    bases = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bin_days = [d for d in mild_days if lo <= float(daily_tout.get(d, np.nan)) < hi]
        base = global_base if len(bin_days) < min_days_per_bin else \
               quantile_baseline_per_slot(daily_profiles, bin_days, q=q, smooth=smooth)
        bases.append(base)
    return {"edges": bins, "centers": centers, "baselines": np.vstack(bases)}


def _interp_baseline_for_day(day_mean_tout: float, tb: dict) -> np.ndarray:
    """Linear interpolation of temperature-conditioned baselines."""
    edges, centers, bases = tb["edges"], tb["centers"], tb["baselines"]
    if day_mean_tout <= edges[0]:  return bases[0]
    if day_mean_tout >= edges[-1]: return bases[-1]
    i = np.searchsorted(edges, day_mean_tout) - 1
    i = np.clip(i, 0, len(centers) - 1)
    c_lo, c_hi = centers[i], centers[min(i + 1, len(centers) - 1)]
    w = (day_mean_tout - c_lo) / (c_hi - c_lo)
    return (1 - w) * bases[i] + w * bases[min(i + 1, len(centers) - 1)]


def extract_residuals_with_baseline(
    df, daily_profiles, mild_days, hvac_days, t_low, t_high,
    temp_col="Tout", time_col="timestamps", slots_per_day=48,
    method="quantile", q=0.20, smooth=3,
    temp_bins=np.arange(-10, 36, 5), min_days_per_bin=6, margin=0.5,
    is_residential=True
):
    """
    Compute HVAC residuals as (profile - mild baseline), gated by active temperature.

    Parameters
    ----------
    slots_per_day : int, default=48
        Number of timesteps per day (e.g. 48 for 30min, 96 for 15min).
    is_residential : bool, default=True
        If False, HVAC cannot be active outside 9:00–17:00 hours.
    """
    freq_minutes = 24 * 60 // slots_per_day

    ts = pd.to_datetime(df[time_col], errors="coerce") if time_col in df.columns else pd.to_datetime(df.index)
    ts = ts.dropna()
    if not ts.empty:
        last_day = ts.iloc[-1].normalize()

        if (ts.dt.normalize() == last_day).sum() < slots_per_day:
            mild_days = [d for d in mild_days if pd.Timestamp(d).normalize() != last_day]
            hvac_days = [d for d in hvac_days if pd.Timestamp(d).normalize() != last_day]

    tout_series = (df.set_index(pd.to_datetime(df[time_col]))[temp_col]
                   if time_col in df.columns else df[temp_col])
    tout_series.index = pd.to_datetime(tout_series.index)

    mean_consumption = daily_profiles.values.flatten()
    mean_consumption = mean_consumption[mean_consumption > 0]  # exclude zeros
    consumption_threshold = 0.60 * np.mean(mean_consumption)  # 60% of mean

    if method == "temp_conditioned":
        tb = build_temp_conditioned_baselines(df, daily_profiles, mild_days,
                                              temp_col=temp_col, bins=temp_bins,
                                              q=q, smooth=smooth, min_days_per_bin=min_days_per_bin)
        use_temp_conditioned = True
    else:
        base_global = quantile_baseline_per_slot(daily_profiles, mild_days, q=q, smooth=smooth)
        use_temp_conditioned = False

    daily_tout = df[temp_col].resample("D").mean()
    residuals, kept = [], []

    for day in [d for d in hvac_days if d in daily_profiles.index]:
        prof = daily_profiles.loc[day].to_numpy()

        base = (_interp_baseline_for_day(float(daily_tout.get(day, np.nan)), tb)
                if use_temp_conditioned and np.isfinite(float(daily_tout.get(day, np.nan)))
                else base_global)
        resid_full = np.maximum(prof - base, 0.0)

        times = pd.date_range(day, periods=slots_per_day, freq=f"{freq_minutes}min")
        Tout = tout_series.reindex(times).to_numpy()
        mask_active = ((Tout < (t_low - margin)) | (Tout > (t_high + margin)))
        mask_active = np.nan_to_num(mask_active, nan=False).astype(bool)

        # Check sufficient load in BOTH profile AND residual
        mask_sufficient_load = (prof > consumption_threshold) & (resid_full > consumption_threshold * 0.1)
        mask_sufficient_load = np.nan_to_num(mask_sufficient_load, nan=False).astype(bool)

        # Combine masks
        mask_active = mask_active & mask_sufficient_load

        # Enforce working-hour constraint for non-residential sites
        if not is_residential:
            hours = np.arange(slots_per_day) * (24 / slots_per_day)
            work_mask = (hours >= 9) & (hours < 17)
            mask_active &= work_mask

        if not mask_active.any():
            continue

        residuals.append(np.where(mask_active, resid_full, 0.0))
        kept.append(day)

    return (np.vstack(residuals) if residuals else np.empty((0, slots_per_day))), kept

# ================================================
# === 3. TERNARY CLASSIFICATION OF RESIDUALS ====
# ================================================

def classify_residuals_ternary_rescaled(
    resids,
    active_ratio=0.02,
    high_ratio=0.5,
    min_high=1.0,
    dtype=int
):
    """Classify residuals into OFF(0), LOW(1), HIGH(2) using per-day rescaled thresholds."""
    resids = np.asarray(resids, dtype=float)
    if resids.ndim == 1:
        resids = resids.reshape(1, -1)
   
    row_max = np.nanpercentile(resids, 100, axis=1, keepdims=True)

    row_max_safe = np.where(row_max == 0, np.nan, row_max)

    active_mask = resids > (active_ratio * row_max_safe)
    high_mask_rel = resids >= (high_ratio * row_max_safe)
    high_mask_abs = resids >= min_high
    high_mask = active_mask & (high_mask_rel | high_mask_abs)
    low_mask = active_mask & ~high_mask

    ternary = np.zeros_like(resids, dtype=dtype)
    ternary[low_mask] = 1
    ternary[high_mask] = 2
    return ternary


def liul_filter(daily_profiles,
                power_thresh=1000,
                min_days_active=5,
                fill_with='baseline'):
    baseline = daily_profiles.median(axis=0)
    resid = (daily_profiles - baseline).clip(lower=0)
    active = resid.gt(power_thresh)
    days_used = active.sum(axis=0)
    liul_slots = days_used[days_used < min_days_active].index

    if fill_with == 'baseline':
        for slot in liul_slots:
            for day in daily_profiles.index:
                if resid.at[day, slot] > power_thresh:
                    # ts      = day + pd.Timedelta(minutes=15*slot)
                    # orig_dp = daily_profiles.at[day, slot]
                    # orig_df = df['total_energy'].at[ts]
                    # print(f"  → Replacing spike {orig_dp:.2f} W @ {ts} (raw {orig_df:.2f} W)")
                    if fill_with == 'baseline':
                        daily_profiles.at[day, slot] = baseline[slot]
                    else:
                        daily_profiles.at[day, slot] = 0
    else:
        daily_profiles.loc[:, liul_slots] = 0

    print(f"Total LIUL slots replaced: {len(liul_slots)}")

    return daily_profiles


@dataclass(frozen=True)
class DisaggParams:
    neutral_band: tuple[float, float] = (20, 40)
    is_residential: bool = True
    active_ratio: float = 0.05
    high_ratio: float = 0.50
    min_high: float = 500
    temp_bins: Optional[np.ndarray] = None
    q: float = 0.25
    min_days_per_bin: int = 2


def run_hvac_disaggregation(df, load_col="total_consumption", temp_col="Tout",
                            neutral_band=(0, 2), temp_bins=None,
                            is_residential=False, active_ratio=0.0, high_ratio=0.5,
                            min_high=1000, q=0.20, smooth=3, min_days_per_bin=3, slots_per_day=48):
    """Run full HVAC disaggregation pipeline on input DataFrame."""
    df = df.copy()

    # Ensure timestamps column exists
    if 'timestamps' not in df.columns:
        df['timestamps'] = pd.to_datetime(df.index)

    df.index = pd.to_datetime(df.index)

    # Set default temperature bins based on building type
    if temp_bins is None:
        temp_bins = np.arange(0, 40, 4)
       
    # Step 1: Make daily profiles
    daily_profiles = make_daily_profiles(df, load_col=load_col)

    # Step 2: Classify days by temperature
    mild_days, hvac_days, t_low, t_high = classify_days_by_temp(
        df, temp_col=temp_col, neutral_band=neutral_band
    )

    # Step 3: Extract residuals with baseline
    residual_matrix, kept_days = extract_residuals_with_baseline(
        df, daily_profiles, mild_days, hvac_days, t_low, t_high,
        temp_col=temp_col, time_col="timestamps",
        method="quantile", q=q, smooth=smooth,
        temp_bins=temp_bins,
        min_days_per_bin=min_days_per_bin, margin=0.5,
        is_residential=is_residential
    )

    # Step 4: Classify residuals into ternary (OFF/LOW/HIGH)
    hvac_ternary = classify_residuals_ternary_rescaled(
        residual_matrix,
        active_ratio=active_ratio,
        high_ratio=high_ratio,
        min_high=min_high
    )
    
    hvac_mode = pd.Series(
        0,
        index=df.index,
        dtype=int,
        name="hvac_mode",
    )

    freq_minutes = 24 * 60 // slots_per_day

    for day_idx, day in enumerate(kept_days):
        start = pd.Timestamp(day)
        times = pd.date_range(
            start,
            periods=slots_per_day,
            freq=f"{freq_minutes}min",
        )
        values = hvac_ternary[day_idx]
        day_series = pd.Series(
            values,
            index=times,
            dtype=hvac_mode.dtype,
        )

        # Align on actual existing timestamps
        common_idx = hvac_mode.index.intersection(day_series.index)
        if common_idx.empty:
            continue

        hvac_mode.loc[common_idx] = day_series.loc[common_idx]

    # -----------------------------
    # Output dataframe
    # -----------------------------
    df_with_hvac = df.copy()
    df_with_hvac["hvac_mode"] = hvac_mode

    return {
        "df_with_hvac": df_with_hvac,
        "daily_profiles": daily_profiles,
        "residual_matrix": residual_matrix,
        "kept_days": kept_days,
        "mild_days": mild_days,
        "hvac_days": hvac_days,
        "t_low": float(t_low),
        "t_high": float(t_high),
    }


def run_disaggregation_for_site_util(db: Session, site_id: int):
    rows = db.execute(
        text("""
        SELECT
            c.timestamp,
            c.tin,
            c.rh,
            c.comfort_index,
            e.tout AS tout,
            cons.value AS energy_consumption
        FROM comfort_data c
        JOIN consumption_data cons
            ON c.site_id = cons.site_id
           AND c.timestamp = cons.timestamp
        LEFT JOIN environmental_data e
            ON c.site_id = e.site_id
           AND c.timestamp = e.timestamp
        WHERE c.site_id = :site_id
        ORDER BY c.timestamp
        """),
        {"site_id": site_id},
    ).mappings().all()

    if not rows:
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.set_index("timestamp")
    # df.to_csv("debug_hvac_disagg_input.csv")
    site_sql = text("""
            SELECT
                temp_low_band,
                temp_high_band,
                temp_low_bin,
                temp_high_bin,
                is_residential,
                active_ratio,
                high_ratio,
                min_high,
                q,
                min_days_per_bin
            FROM sites
            WHERE id = :site_id
        """)

    site_row = db.execute(site_sql, {"site_id": site_id}).mappings().first()

    hvac_out = run_hvac_disaggregation(
        df,
        load_col="energy_consumption",
        temp_col="tout",
        neutral_band=(site_row["temp_low_band"], site_row["temp_high_band"]),
        temp_bins=np.arange(site_row["temp_low_bin"], site_row["temp_high_bin"], 1),
        is_residential=site_row["is_residential"],
        active_ratio=site_row["active_ratio"],
        high_ratio=site_row["high_ratio"],
        min_high=site_row["min_high"],
        q=site_row["q"],
        min_days_per_bin=site_row["min_days_per_bin"],
    )
    hvac_mode_series = hvac_out["df_with_hvac"]["hvac_mode"]
    updates = [
        {
            "site_id": site_id,
            "timestamp": ts.to_pydatetime(),   # ts is already a Timestamp from the index
            "hvac_mode": int(mode),
        }
        for ts, mode in hvac_mode_series.items()
    ]

    if not updates:
        logger.warning(f"[disaggregation] No valid HVAC rows to update for site {site_id}")
        return

    db.execute(
        text("""
            UPDATE comfort_data
            SET hvac_mode = :hvac_mode
            WHERE site_id = :site_id
            AND timestamp = :timestamp
        """),
        updates,
    )
    db.commit()