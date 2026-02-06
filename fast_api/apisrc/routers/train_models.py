from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import List
import re
import optuna
import lightgbm as lgb
from datetime import datetime
from fastapi import Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import timedelta, time, timezone
from torch.utils.data import DataLoader
from fast_api.apisrc.core.database import get_db, SessionLocal
from fast_api.apisrc.utils.weather_utils import reconcile_environmental_data, fetch_environmental_data, interpolate_value
from fast_api.apisrc.routers.optimization import build_features
from fast_api.apisrc.utils.disaggregation_utils import run_disaggregation_for_site_util
import logging
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
try:
    from sklearn.metrics import root_mean_squared_error
    _HAS_RMSE = True
except ImportError:
    from sklearn.metrics import mean_squared_error
    _HAS_RMSE = False
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================
import random
def make_file_logger(site_id: int, root_dir: str | Path) -> tuple[logging.Logger, Path]:
    root_dir = Path(root_dir)
    logs_dir = root_dir / "fast_api" / "apisrc" / "logs" / "training"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"site_{site_id}_train_{ts}.log"

    logger = logging.getLogger(f"site_{site_id}_hvac_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # remove existing handlers (important if called multiple times)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)

    return logger, log_path

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

fix_seed(42)

def _transform_with_freeze(X, scaler):
    X = np.asarray(X)
    orig_shape = X.shape

    if X.ndim == 2:
        X_flat = X.copy()
    elif X.ndim == 3:
        # Flatten time dimension
        X_flat = X.reshape(-1, X.shape[-1]).copy()
    else:
        raise ValueError("X must be 2D or 3D")

    D = X_flat.shape[1]
    frozen = set(getattr(scaler, "frozen_idx", []))
    non_frozen = [i for i in range(D) if i not in frozen]

    if non_frozen:
        X_flat[:, non_frozen] = scaler.transform(X_flat[:, non_frozen])

    if len(orig_shape) == 2:
        return X_flat
    else:
        return X_flat.reshape(orig_shape)


FEATURES_TIN_ON = [
    'tin',
    'tout',
    'tin_diff',
    'tout_diff', 'tin_ma3', 'tout_ma3',
    'hour_sin', 'hour_cos',
    'SW1h', 'SW3h',
    'month', 
    'hvac_mode'
]


FEATURES_TIN_OFF_V3 = [
    "tin",
    "tout",
    "hour_sin",
    "hour_cos",
    "month", "season",
    "SW1h", "SW3h"
]
# Features for AH prediction
FEATURES_AH = [
    "ah", "ah_lag1", "ah_lag2", "ah_lag3",
    "tin", "tout", "ah_out",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season",  'hvac_mode'
]

heating_months = [11, 12, 1, 2, 3, 4]
cooling_months = [5, 6, 7, 8, 9, 10]

def _es_hPa_from_Tc(Tc: float) -> float:
    """Saturation vapor pressure in hPa at air temperature Tc (°C)."""
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

def AH_gm3_from_T_RH(Tc: float, RH_percent: float) -> float:
    """
    Absolute humidity [g/m³] from temperature (°C) and RH (%).
    AH = 216.7 * e / (T+273.15), with e = RH/100 * es(T).
    """
    T = np.asarray(Tc, dtype=np.float32)
    RHf = np.clip(np.asarray(RH_percent, dtype=np.float32) / 100.0, 0.0, 1.0) 
    es = _es_hPa_from_Tc(T)  # assumes this already handles vector inputs
    e = RHf * es
    AH = 216.7 * e / (T + 273.15)
    if isinstance(Tc, pd.Series):
        return pd.Series(AH, index=Tc.index, name="ah")
    return AH

def AH_to_RH(Tc, AH):
    """RH[%] back from AH[g/m^3] and T[°C]."""
    Tc = np.asarray(Tc, dtype=float)
    AH = np.asarray(AH, dtype=float)
    es = _es_hPa_from_Tc(Tc)
    RH = 100.0 * (AH * (Tc + 273.15) / 216.7) / es
    return np.clip(RH, 0.0, 100.0)

def build_features(full_timeline: pd.DataFrame) -> pd.DataFrame:
    df = full_timeline.copy()

    # ------------------
    # Absolute humidity
    # ------------------
    df["tin_target"] = df["tin"].shift(-1)
    # df['tin_target'] = df["tin_target"].bfill().ffill()

    df["ah"] = AH_gm3_from_T_RH(df["tin"], df["rh"])
    df["ah_lag1"] = df["ah"].shift(1)
    df["ah_lag2"] = df["ah"].shift(2)
    df["ah_lag3"] = df["ah"].shift(3)

    df["ah_out"] = AH_gm3_from_T_RH(df["tout"], df["rh_out"]) / 100.0
    df['ah_target'] = df['ah'].shift(-1)

    # ------------------
    # Time encodings
    # ------------------
    idx = df.index
    idx = pd.to_datetime(full_timeline.index)


    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df["month"] = idx.month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    df["season"] = np.where(df["month"].between(5, 10), 0.5, 1.0)

    # ------------------
    # Solar features
    # ------------------
    df["SW1h"] = df["sw_out"].rolling(window=2, min_periods=1).mean()
    df["SW3h"] = df["sw_out"].rolling(window=6, min_periods=1).mean()

    # ------------------
    # Diffs & rolling
    # ------------------
    df["tin_diff"] = df["tin"].diff()
    df["tout_diff"] = df["tout"].diff()

    df["tin_ma3"] = df["tin"].rolling(window=3, min_periods=1).mean()
    df["tout_ma3"] = df["tout"].rolling(window=3, min_periods=1).mean()
    return df

def fetch_environmental_data_auto(
    *,
    latitude: float,
    longitude: float,
    start_ts: datetime,
    end_ts: datetime,
    archive_delay_days: int = 5,  # match Open-Meteo docs note about recent-days delay
):
    """
    Fetch hourly environmental data across a window that may span both:
      - Historical archive (older data)
      - Forecast API (most recent days / future)

    Returns same shape as fetch_environmental_data():
        { timestamp_naive_local: { "tout": ..., "rh_out": ..., "sw_out": ... } }
    """
    # Normalize to naive (matches your existing function’s behavior)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.replace(tzinfo=None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.replace(tzinfo=None)

    if start_ts > end_ts:
        raise ValueError("start_ts must be <= end_ts")

    # Conservative cutoff: treat the most recent N days as forecast territory.
    # Open-Meteo states historical data has a delay and suggests using Forecast API + past_days for recent days. :contentReference[oaicite:1]{index=1}
    now_utc = datetime.now(timezone.utc)      # preferred, works everywhere
    last_archive_date = (datetime.now(timezone.utc) - timedelta(days=archive_delay_days)).date()
    cutoff_ts = datetime.combine(last_archive_date, time.max)

    out: dict = {}

    # 1) Archive segment (everything up to cutoff_ts)
    if start_ts <= cutoff_ts:
        archive_end = min(end_ts, cutoff_ts)
        out.update(
            fetch_environmental_data(
                latitude=latitude,
                longitude=longitude,
                start_ts=start_ts,
                end_ts=archive_end,
                source="archive",
            )
        )

    # 2) Forecast segment (everything after cutoff_ts)
    if end_ts > cutoff_ts:
        # Start forecast at next day midnight to avoid overlap with archive day
        forecast_start_floor = datetime.combine(last_archive_date + timedelta(days=1), time.min)
        forecast_start = max(start_ts, forecast_start_floor)

        out.update(
            fetch_environmental_data(
                latitude=latitude,
                longitude=longitude,
                start_ts=forecast_start,
                end_ts=end_ts,
                source="forecast",
            )
        )

    return out

def register_site_model(
    db,
    *,
    site_id: int,
    model_type: str,          # e.g. "tin_on_heating" / "tin_on_cooling"
    model_version: str,       # e.g. "20260129_213000" or git hash
    framework: str,           # e.g. "lightgbm"
    artifact_path: str,       # absolute path to .pkl
    activate: bool = True,
):
    if activate:
        db.execute(
            text("""
                UPDATE site_models
                SET is_active = FALSE
                WHERE site_id = :site_id
                  AND model_type = :model_type
                  AND is_active = TRUE
            """),
            {"site_id": site_id, "model_type": model_type},
        )

    db.execute(
        text("""
            INSERT INTO site_models (
                site_id, model_type, model_version, framework, artifact_path, is_active, created_at
            )
            VALUES (
                :site_id, :model_type, :model_version, :framework, :artifact_path, :is_active, :created_at
            )
        """),
        {
            "site_id": site_id,
            "model_type": model_type,
            "model_version": model_version,
            "framework": framework,
            "artifact_path": artifact_path,
            "is_active": activate,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),  # your DB is timezone=False
        },
    )
    db.commit()

def train_regime_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list,
    hvac_col: str,
    regime_name: str,
    logger,
    n_trials: int = 30,
):
    """
    Train a LightGBM model for a specific thermodynamic regime.

    regime_filter: boolean mask selecting appropriate rows (heating or cooling)
    """
    def _xy(frame: pd.DataFrame):
        X = frame[feature_cols]
        y = frame['tin_target']
        m = (~X.isna().any(axis=1)) & y.notna()
        return X.loc[m], y.loc[m]


    tr = df_train[df_train[hvac_col].isin([1,2])]
    vl = df_val[df_val[hvac_col].isin([1,2])]

    if len(tr) < 150:
        logger.warning(f"[{regime_name}] Insufficient training samples ({len(tr)}). Skipping.")
        return None

    X_train, y_train = _xy(tr)
    X_val,   y_val   = _xy(vl)

    logger.info(f"[{regime_name}] Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Inner time-series CV
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'device_type': 'gpu',
            'verbosity': -1,
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'max_depth': -1,
        }

        model = lgb.LGBMRegressor(**params)
        maes = []

        for train_idx_cv, val_idx_cv in tscv.split(X_train):
            X_tr_cv, y_tr_cv = X_train.iloc[train_idx_cv], y_train.iloc[train_idx_cv]
            X_vl_cv, y_vl_cv = X_train.iloc[val_idx_cv], y_train.iloc[val_idx_cv]

            model.fit(
                X_tr_cv, y_tr_cv,
                eval_set=[(X_vl_cv, y_vl_cv)],
                callbacks=[
                    lgb.early_stopping(40),
                    lgb.log_evaluation(period=0),
                ],
            )
            pred = model.predict(X_vl_cv)
            maes.append(mean_absolute_error(y_vl_cv, pred))

        return np.mean(maes)

    # Run Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"[{regime_name}] Best CV MAE: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'device_type': 'gpu',
        'verbosity': -1,
        'random_state': 42,
        'max_depth': -1,
    })

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(period=0),
        ],
    )

    # Validation diagnostics
    if len(X_val) > 0:
        yvp = final_model.predict(X_val)
        logger.info(
            f"[{regime_name}] Val: "
            f"MAE={mean_absolute_error(y_val, yvp):.3f} "
            f"RMSE={root_mean_squared_error(y_val, yvp):.3f} "
            f"R2={r2_score(y_val, yvp):.3f}"
        )

    return final_model

@dataclass
class MaskedSeqDataset(Dataset):
    """Simple masked sequence dataset for pre-scaled data."""
    X: np.ndarray  # (N, window, D)
    y: np.ndarray  # (N,)
    mask: np.ndarray  # (N,) float in {0,1}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.mask[i], dtype=torch.float32),
        )

def save_scaler_with_features(
    scaler,
    input_cols: List[str],
    save_path: str,
    logger,
):
    frozen_idx = tuple(_indices_to_freeze(input_cols))  # your existing helper

    obj = {
        "scaler": scaler,
        "feature_order": list(input_cols),
        "frozen_idx": frozen_idx,
    }

    joblib.dump(obj, save_path)
    logger.info(f"Saved scaler: features={len(input_cols)} frozen_idx={frozen_idx} path={save_path}")


def train_masked_model(
    logger,
    model: nn.Module,
    train_dl: DataLoader,
    input_cols: List[str],
    val_dl: Optional[DataLoader] = None,
    *,
    max_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 40,
    grad_clip: float = 1.0,
    loss_type: str = "huber",
    huber_beta: float = 0.5,
    save_dir: Optional[str] = None,
    metric_label: str = "RMSE",
) -> Tuple[nn.Module, StandardScaler]:
    """
    Train a PyTorch model with masking, early stopping, and freeze-aware scaling.

    Returns:
        Tuple of (trained_model, fitted_scaler)
    """
    # Fit scaler from training data
    frozen_idx = _indices_to_freeze(input_cols) if input_cols is not None else []
    X_all = []
    for Xb, _, _ in train_dl:
        X_all.append(Xb.numpy())
    X_all = np.concatenate(X_all, axis=0)
    Ntot, T, D = X_all.shape

    scaler = StandardScaler()
    if frozen_idx:
        non_frozen = [j for j in range(D) if j not in set(frozen_idx)]
        if non_frozen:
            X_flat = X_all.reshape(Ntot*T, D)
            scaler.fit(X_flat[:, non_frozen])
        else:
            scaler.fit(np.zeros((1, 1)))
    else:
        scaler.fit(X_all.reshape(Ntot*T, D))

    scaler.frozen_idx = tuple(frozen_idx)

    # Rebuild loaders with freeze-aware scaling
    def scaled_loader(dloader: DataLoader, shuffle: bool) -> DataLoader:
        Xs, ys, ms = [], [], []
        for Xb, yb, mb in dloader:
            N, T, D = Xb.shape
            X2 = Xb.numpy().reshape(N*T, D)
            X2s = _transform_with_freeze(X2, scaler).reshape(N, T, D)
            Xs.append(torch.from_numpy(X2s).float())
            ys.append(yb)
            ms.append(mb)
        Xs = torch.cat(Xs, dim=0)
        ys = torch.cat(ys, dim=0)
        ms = torch.cat(ms, dim=0)
        ds = MaskedSeqDataset(Xs.numpy(), ys.numpy(), ms.numpy())
        return DataLoader(ds, batch_size=dloader.batch_size, shuffle=shuffle)

    train_dl = scaled_loader(train_dl, shuffle=True)
    if val_dl is not None:
        val_dl = scaled_loader(val_dl, shuffle=False)

    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, cooldown=0
    )

    def _loss(pred, y, w):
        if loss_type == "huber":
            per = F.smooth_l1_loss(pred, y, reduction="none", beta=huber_beta)
        elif loss_type == "mse":
            per = (pred - y) ** 2
        elif loss_type == "mae":
            per = (pred - y).abs()
        else:
            raise ValueError(f"Unknown loss_type={loss_type}")
        denom = w.sum()
        return (per * w).sum() / torch.clamp(denom, min=1.0)

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        tr_sq_sum, tr_w_sum = 0.0, 0.0
        for Xb, yb, mb in train_dl:
            Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
            if mb.sum().item() < 1:
                continue
            pred = model(Xb)
            loss = _loss(pred, yb, mb)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            tr_sq_sum += ((pred - yb) ** 2 * mb).sum().item()
            tr_w_sum += mb.sum().item()

        tr_rmse = (tr_sq_sum / max(tr_w_sum, 1.0)) ** 0.5 if tr_w_sum > 0 else float("nan")

        # Validate
        val_rmse = float("nan")
        if val_dl is not None:
            model.eval()
            vsq_sum, vw_sum = 0.0, 0.0
            with torch.no_grad():
                for Xb, yb, mb in val_dl:
                    Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
                    if mb.sum().item() < 1:
                        continue
                    pred = model(Xb)
                    vsq_sum += ((pred - yb) ** 2 * mb).sum().item()
                    vw_sum += mb.sum().item()
            val_rmse = (vsq_sum / max(vw_sum, 1.0)) ** 0.5 if vw_sum > 0 else float("nan")

            metric = val_rmse if np.isfinite(val_rmse) else float("inf")
            scheduler.step(metric)

            if metric < best_val - 1e-6:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0

                # Save best model
                if save_dir is not None:
                    out_dir = Path(save_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    model_path = os.path.join(out_dir, 'best_ah.pt')
                    torch.save(best_state, model_path)
                    

                    scaler_dir = os.path.join(out_dir, "scalers")
                    os.makedirs(scaler_dir, exist_ok=True)
                    scaler_path = os.path.join(scaler_dir, 'scaler_ah.pkl')
                    save_scaler_with_features(scaler, FEATURES_AH, str(scaler_path), logger)
                    # joblib.dump(scaler, scaler_path)
  
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
 
    logger.info(f"[DONE] Training complete. Best val {metric_label} = {best_val:.3f}")
    model.eval()
    with torch.no_grad():
        Xb, yb, mb = next(iter(train_dl))
        Xb, yb = Xb.to(device), yb.to(device)
        pred_dbg = model(Xb).cpu().numpy().ravel()
        y_dbg = yb.cpu().numpy().ravel()

    logger.info(f"  y_true (first 10): {y_dbg[:10]}")
    logger.info(f"  y_pred (first 10): {pred_dbg[:10]}")
    logger.info(f"  y_true range: {y_dbg.min()} → {y_dbg.max()}")
    logger.info(f"  y_pred range: {pred_dbg.min()} → {pred_dbg.max()}")
    return model, scaler

def evaluate_ah_split(
    logger,
    df: pd.DataFrame,
    indices: np.ndarray,
    model: torch.nn.Module,
    scaler,
    input_cols: List[str],
    window: int = 24,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hvac_col: str = "hvac_mode",
    title_note: str = "Validation"
) -> Tuple[Dict, pd.DataFrame]:
    """
    Evaluate AH model on a data split and convert to RH predictions.

    Returns:
        Tuple of (metrics_dict, predictions_dataframe)
    """
    sub = df.loc[df.index[indices]]
    sub = sub.sort_index()
    # sub = df.iloc[indices].copy()


    # Build sequences
    X_all = sub[input_cols].values.astype(np.float32)
    N = len(sub)
    seqs, target_rows = [], []

    for i in range(window-1, N-1):
        seq = X_all[i-window+1:i+1]
        seqs.append(seq)
        target_rows.append(i)

    if not seqs:
        raise ValueError("Not enough points for the given window")

    X = np.asarray(seqs, dtype=np.float32)
    M, T, D = X.shape
    Xs = X.reshape(M*T, D)

    X_scaled = _transform_with_freeze(X, scaler)

    # Predict
    with torch.no_grad():
        x_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        model = model.to(device).eval()
        AH_pred = model(x_t).detach().cpu().numpy().astype(float).ravel()

    # Gather ground truth
    times, AH_true, RH_true, Tin_true = [], [], [], []
    Tout_true, hvac_list, hour_list, month_list = [], [], [], []

    for j, i in enumerate(target_rows):
        t1 = sub.index[i+1]
        times.append(t1)
        AH_true.append(float(sub["ah"].iloc[i+1]))
        Tin_true.append(float(sub["tin"].iloc[i+1]) if "tin" in sub.columns else np.nan)
        Tout_true.append(float(sub["tout"].iloc[i+1]) if "tout" in sub.columns else np.nan)
        hvac_list.append(int(sub[hvac_col].iloc[i+1]) if hvac_col in sub.columns else 0)
        hour_list.append(int(pd.Timestamp(t1).hour))
        month_list.append(int(pd.Timestamp(t1).month))
        RH_true.append(float(sub["rh"].iloc[i+1]) if "rh" in sub.columns else np.nan)

    AH_true = np.array(AH_true)
    Tin_arr = np.array(Tin_true)
    RH_true = np.array(RH_true)


    RH_pred = AH_to_RH(Tin_arr, AH_pred)

    # Calculate metrics
    def _mk(y, yhat):
        return dict(
            rmse=_rmse(y, yhat),
            mae=float(mean_absolute_error(y, yhat)),
            r2=float(r2_score(y, yhat)),
            smape=float(smape(y, yhat)),
        )

    m_ah = _mk(AH_true, AH_pred)
    mask = np.isfinite(RH_true) & np.isfinite(RH_pred)
    m_rh = _mk(RH_true[mask], np.asarray(RH_pred)[mask]) if mask.any() else {
        k: np.nan for k in ("rmse", "mae", "r2", "smape")
    }

    metrics = {f"AH_{k}": v for k, v in m_ah.items()} | {f"RH_{k}": v for k, v in m_rh.items()}

    dfp = pd.DataFrame({
        "ah_true": AH_true,
        "ah_pred": AH_pred,
        "rh_true": RH_true,
        "rh_pred": RH_pred,
        "tin_true": Tin_arr,
        "tout": np.array(Tout_true),
        hvac_col: np.array(hvac_list, dtype=int),
        "hour": np.array(hour_list, dtype=int),
        "month": np.array(month_list, dtype=int),
    }, index=pd.to_datetime(times)).sort_index()

    logger.info(f"==== {title_note} — AH (g/m³) ====")
    logger.info(
        f"RMSE={metrics['AH_rmse']:.3f}  MAE={metrics['AH_mae']:.3f}  "
        f"R2={metrics['AH_r2']:.3f}  sMAPE={metrics['AH_smape']:.2f}%"
    )
    logger.info(f"==== {title_note} — RH (%) ====")
    logger.info(
        f"RMSE={metrics['RH_rmse']:.3f}  MAE={metrics['RH_mae']:.3f}  "
        f"R2={metrics['RH_r2']:.3f}  sMAPE={metrics['RH_smape']:.2f}%"
    )
    return metrics, dfp


def detect_good_hvac_days(df, min_active_hh=4, hvac_col="hvac_mode"):
    hvac_active = (df[hvac_col].isin([1, 2])).astype(int)
    daily_counts = hvac_active.resample("D").sum()

    good_days = daily_counts[daily_counts >= min_active_hh].index
    return set(good_days)

def chronological_season_split(df, good_days, train_ratio=0.7, val_ratio=0.15):
    """Performs chronological train/val/test splits on HVAC-active days only."""
    # restrict to good days only
    df_g = df[df.index.normalize().isin(good_days)]

    if len(df_g) == 0:
        return [], [], []

    # unique sorted days
    days = sorted(df_g.index.normalize().unique())
    n = len(days)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_days = days[:n_train]
    val_days   = days[n_train:n_train+n_val]
    test_days  = days[n_train+n_val:]

    return train_days, val_days, test_days

def make_hvac_splits(df, min_active_hh=4, hvac_col="hvac_mode"):
    """Creates seasonal HVAC splits by identifying active days and splitting them chronologically."""
    df = df.copy()
    df["season_t"] = df.index.map(lambda t: "winter" if t.month in [11,12,1,2,3,4] else "summer")
    df["split"] = "none"

    for season in ["winter", "summer"]:
        df_s = df[df["season_t"] == season]

        # find good HVAC days
        good_days = detect_good_hvac_days(df_s, min_active_hh=min_active_hh, hvac_col=hvac_col)

    
        if len(good_days) < 10:
            # print("  → SKIPPED: not enough HVAC-active days for a meaningful split.")
            continue

        train_days, val_days, test_days = chronological_season_split(df_s, good_days)


        df.loc[df.index.normalize().isin(train_days), "split"] = "train"
        df.loc[df.index.normalize().isin(val_days),   "split"] = "val"
        df.loc[df.index.normalize().isin(test_days),   "split"] = "test"

    return df

def get_summer_test_range(df_on):
    """Extracts the start and end timestamps of the summer test period from HVAC-ON data."""
    df_summer = df_on[(df_on["season_t"] == "summer") & (df_on["split"] == "test")]
    if df_summer.empty:
        raise ValueError("No summer test data found in HVAC-ON splits.")
    return df_summer.index.min(), df_summer.index.max()

def make_global_off_split(df, summer_test_start, summer_test_end, val_ratio=0.15):
    """Creates global train/val/test splits aligned with the summer HVAC-ON test period."""
    df = df.copy()
    df["global_split"] = "none"

    # Split cutoff
    before_test = df[df.index < summer_test_start]
    n_before = len(before_test)
    n_val = int(n_before * val_ratio)

    # Determine validation region
    if n_val > 0:
        val_start = before_test.index[-n_val]
    else:
        val_start = summer_test_start

    # ---- assign TEST ----
    df.loc[summer_test_start:summer_test_end, "global_split"] = "test"

    # ---- assign VAL ----
    df.loc[val_start: summer_test_start - pd.Timedelta(minutes=30), "global_split"] = "val"

    # ---- assign TRAIN ----
    df.loc[:val_start, "global_split"] = "train"

    # ---- AFTER TEST = NONE (not train!) ----
    df.loc[summer_test_end:, "global_split"] = "none"

    return df

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0

def _rmse(y_true, y_pred):
    """Calculate RMSE with sklearn version compatibility."""
    if _HAS_RMSE:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_model(
    logger,
    df_test,
    model,
    feature_cols,
    hvac_col,
    regime_name,
    _save_dir_ignored=None,   # keep signature compatible with your existing calls
):
    # filter to HVAC ON samples (same as your old code)
    test_on = df_test[df_test[hvac_col].isin([1, 2])]
    if len(test_on) == 0:
        logger.warning(f"[{regime_name}] No test samples")
        return {}

    X = test_on[feature_cols]
    y = test_on["tin_target"]
    m = (~X.isna().any(axis=1)) & y.notna()
    X, y = X.loc[m], y.loc[m]

    if len(X) == 0:
        logger.warning(f"[{regime_name}] No valid test samples after filtering")
        return {}

    y_pred = model.predict(X)

    rmse = _rmse(y.values, y_pred)
    mae  = mean_absolute_error(y.values, y_pred)
    r2   = r2_score(y.values, y_pred)
    smp  = smape(y.values, y_pred)

    logger.info(
        f"[{regime_name}] TEST METRICS | RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  sMAPE={smp:.3f}"
    )

    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "smape": float(smp)}




def _indices_to_freeze(input_cols):
    FROZEN = {
        "hour_sin", "hour_cos",
        "season", "month", "month_sin", "month_cos",
        "is_trans_off",
        "off_runtime_1h",
    }

    return [j for j, c in enumerate(input_cols) if c in FROZEN]


def transition_weighted_loss(pred, target, mask, is_transition, base_loss_fn=F.huber_loss):
    """
    Weighted loss that emphasizes transition periods.

    Args:
        pred: Predictions (B,)
        target: Targets (B,)
        mask: Sample mask (B,)
        is_transition: Transition indicator (B,) - 1.0 for transitions, 0.0 otherwise
        base_loss_fn: Base loss function (default: Huber loss)

    Returns:
        Scalar loss value
    """
    # Compute base loss per sample
    loss_per_sample = base_loss_fn(pred, target, reduction='none')

    # Weight: 2.5x for transitions, 1.0x for steady-state
    weights = torch.where(is_transition > 0.5, 2.5, 1.0)

    # Apply weights and mask
    weighted_loss = (loss_per_sample * weights * mask).sum() / (mask.sum() + 1e-8)

    return weighted_loss

class CausalBlock(nn.Module):
    """Causal dilated convolution block with residual connection (same as original)."""

    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.15):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, dilation=dilation, padding=pad)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)[:, :, :x.size(2)]  # Causal: trim future
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.drop(out)

        res = self.residual(x)
        return out + res


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on recent timesteps during transitions.

    During transitions, recent history is more important than distant past.
    This attention layer learns to weight different parts of the sequence.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 4)
        self.key = nn.Linear(hidden_dim, hidden_dim // 4)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // 4) ** -0.5

    def forward(self, x):
        """
        Args:
            x: (B, C, T) - batch, channels, time

        Returns:
            Attended features: (B, C)
        """
        # Transpose to (B, T, C) for attention
        x = x.transpose(1, 2)  # (B, T, C)

        Q = self.query(x)  # (B, T, C/4)
        K = self.key(x)    # (B, T, C/4)
        V = self.value(x)  # (B, T, C)

        # Attention scores: focus on recent timesteps
        # Use last timestep as query
        q_last = Q[:, -1:, :]  # (B, 1, C/4)
        scores = torch.matmul(q_last, K.transpose(1, 2)) * self.scale  # (B, 1, T)
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, T)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, V).squeeze(1)  # (B, C)

        return attended

class ImprovedTCNForOFF(nn.Module):
    """
    Enhanced Residual TCN with transition-aware attention mechanism.

    IMPROVEMENTS:
    1. Temporal attention layer to focus on recent history during transitions
    2. Skip connections from input features for better gradient flow
    3. Separate pathway for transition-specific features
    """

    def __init__(self, in_dim, hidden=256, levels=6, kernel_size=7, dropout=0.15, use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        # TCN backbone (same as original ResTCNRegressor)
        ch = in_dim
        blocks = []
        for l in range(levels):
            dil = 2 ** l
            blocks.append(CausalBlock(ch, hidden, k=kernel_size, dilation=dil, dropout=dropout))
            ch = hidden

        self.tcn_net = nn.Sequential(*blocks)

        # Optional attention mechanism
        if use_attention:
            self.attention = TemporalAttention(hidden)
            final_dim = hidden
        else:
            final_dim = hidden

        # Output head
        self.head = nn.Sequential(
            nn.Linear(final_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, T, D) - batch, time, features

        Returns:
            predictions: (B,) - one value per batch
        """
        # Transpose for Conv1d: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)  # (B, D, T)

        # TCN processing
        features = self.tcn_net(x)  # (B, hidden, T)

        # Apply attention if enabled
        if self.use_attention:
            pooled = self.attention(features)  # (B, hidden)
        else:
            pooled = features[:, :, -1]  # Just take last timestep (B, hidden)

        # Final prediction
        out = self.head(pooled).squeeze(-1)  # (B,)

        return out
    
class AHSeqDataset(Dataset):
    """
    PyTorch Dataset for Absolute Humidity sequence prediction.

    Creates sliding windows from time series data.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        window: int = 24,
        target_col: str = "ah_target",
        indices: Optional[np.ndarray] = None
    ):
        if indices is None:
            indices = np.arange(len(df))

        sub = df.iloc[indices]
        X_all = sub[input_cols].values.astype(np.float32)
        y_all = sub[target_col].values.astype(np.float32)

        seqs, ys, msks = [], [], []
        for i in range(window-1, len(sub)-1):
            y = y_all[i]
            if not np.isfinite(y):
                continue
            # Extract sequence
            seq = X_all[i-window+1:i+1]
            # Skip sequences with NaN values in any feature
            if not np.isfinite(seq).all():
                continue
            seqs.append(seq)
            ys.append(y)
            msks.append(1.0)

        self.X = np.asarray(seqs, np.float32)
        self.y = np.asarray(ys, np.float32)
        self.mask = np.asarray(msks, np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i]),
            torch.tensor(self.y[i]),
            torch.tensor(self.mask[i])
        )


class CNNLSTMWithFuture(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, cnn_out_channels=32, kernel_size=3):
        super().__init__()
        # self.norm = nn.LayerNorm(input_size) # Add normalization layer before Conv1D
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.15,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        # x_seq = self.norm(x_seq)
        x_seq = x_seq.permute(0, 2, 1)
        x_seq = self.relu(self.conv1d(x_seq))
        x_seq = x_seq.permute(0, 2, 1)

        # LSTM + output
        _, (hn, _) = self.lstm(x_seq)
        return self.fc(hn[-1]).squeeze(-1)


def _scale_3d_with_freeze(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, feature_names):
    scaler = StandardScaler()
    scaler.frozen_idx = tuple(_indices_to_freeze(feature_names))

    N_train, W, D = X_train.shape
    non_frozen = [j for j in range(D) if j not in scaler.frozen_idx]

    X_train_flat = X_train.reshape(-1, D)
    scaler.fit(X_train_flat[:, non_frozen])

    X_train_scaled = _transform_with_freeze(X_train_flat, scaler).reshape(N_train, W, D)
    X_val_scaled = _transform_with_freeze(X_val.reshape(-1, D), scaler).reshape(len(X_val), W, D)
    X_test_scaled = _transform_with_freeze(X_test.reshape(-1, D), scaler).reshape(len(X_test), W, D)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled

def build_v3_dataset(
    df: pd.DataFrame,
    input_cols: list,
    window: int = 48,
    indices: np.ndarray = None,
    hvac_col: str = "hvac_mode",
    target_col: str = "tin_target",
    oversample_transitions: bool = True,
):

    if indices is None:
        indices = np.arange(len(df))
    indices = np.array(sorted(indices))


    # Check all features exist
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    sub = df.iloc[indices].copy()

    X_all = sub[input_cols].values.astype(np.float32)
    y_all = sub[target_col].values.astype(np.float32)
    # hvac_all = pd.to_numeric(sub[hvac_col], errors="coerce").fillna(0).astype(int).values
    hvac_all = (
        pd.to_numeric(sub[hvac_col], errors="coerce")
        .fillna(-1)                 # unknown
        .astype(np.int16)
        .to_numpy()
    )


    # Use is_trans_off flag to identify transitions
    if 'is_trans_off' in sub.columns:
        trans_flag = sub['is_trans_off'].values.astype(int)
    else:
        trans_flag = np.zeros(len(sub), dtype=int)

    seqs, ys, msks, is_trans = [], [], [], []
    ts_out = []

    step = pd.Timedelta(minutes=30)
    expected_span = window * step

    for i in range(window-1, len(sub)-1):
        y = y_all[i]
        hvac_state = hvac_all[i]

        # Only OFF periods
        if not np.isfinite(y) or hvac_state != 0:
            continue

        # Sequence from i-window to i (exclusive)
        seq = X_all[i-window+1:i+1]

        # Skip NaN
        if not np.isfinite(seq).all():
            continue

        # Check if this is a transition period
        # Look at recent history: if ANY of last 12 steps had is_trans_off=1
        lookback = 2  # 6 hours
        start_check = max(0, i - lookback)
        is_transition_period = trans_flag[start_check:i+1].sum() > 0
        if (sub.index[i] - sub.index[i-window+1]) != (window-1) * step:
            continue
        # Add sample
        seqs.append(seq)
        ys.append(y)
        msks.append(1.0)
        is_trans.append(float(is_transition_period))
        ts_out.append(sub.index[i])

        # Oversample transitions (add 2 more copies)
        if oversample_transitions and is_transition_period:
            for _ in range(2):
                seqs.append(seq)
                ys.append(y)
                msks.append(1.0)
                is_trans.append(1.0)
                ts_out.append(sub.index[i])

    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    mask = np.asarray(msks, dtype=np.float32)
    is_transition = np.asarray(is_trans, dtype=np.float32)

    # print(f"  Built dataset: {len(y)} samples ({is_transition.sum():.0f} transition)")
    # print(f"  Shape: {X.shape} (samples, window={window}, features={len(input_cols)})")

    return X, y, mask, is_transition, np.array(ts_out)

class TransitionAwareDataset(Dataset):
    """PyTorch Dataset with transition indicators for weighted loss."""

    def __init__(self, X, y, mask, is_transition):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.is_transition = torch.tensor(is_transition, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx], self.is_transition[idx]

def evaluate_tcn_model(
    *,
    model,
    X: np.ndarray,
    y: np.ndarray,
    device="cuda",
    logger=None,
    prefix: str = "TCN",
) -> dict:
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_pred = model(X_t).detach().cpu().numpy().ravel()

    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(_rmse(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    smp = float(smape(y, y_pred))

    if logger:
        logger.info(f"{prefix} | RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} sMAPE={smp:.3f}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "smape": smp}

def train_tcn_with_transitions(
    *,
    model,
    train_dataset,
    val_dataset=None,
    batch_size: int = 256,
    max_epochs: int = 200,
    lr: float = 1e-3,
    device="cuda",
    patience: int = 20,
    save_path: str | None = None,
    logger=None,
) -> tuple:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0) if val_dataset else None

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for X, y, mask, is_trans in train_loader:
            X, y, mask, is_trans = X.to(device), y.to(device), mask.to(device), is_trans.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = transition_weighted_loss(pred, y, mask, is_trans)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        history["train_loss"].append(avg_train_loss)

        if val_loader is not None:
            model.eval()
            val_losses = []
            val_preds, val_targets = [], []

            with torch.no_grad():
                for X, y, mask, is_trans in val_loader:
                    X, y, mask, is_trans = X.to(device), y.to(device), mask.to(device), is_trans.to(device)
                    pred = model(X)
                    loss = transition_weighted_loss(pred, y, mask, is_trans)

                    val_losses.append(loss.item())
                    val_preds.extend(pred.detach().cpu().numpy().ravel())
                    val_targets.extend(y.detach().cpu().numpy().ravel())

            avg_val_loss = float(np.mean(val_losses))
            val_mae = float(mean_absolute_error(val_targets, val_preds))

            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(val_mae)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if logger and (epoch + 1) % 10 == 0:
                logger.info(
                    f"TCN epoch {epoch+1}/{max_epochs} | train_loss={avg_train_loss:.4f} "
                    f"| val_loss={avg_val_loss:.4f} | val_mae={val_mae:.4f}"
                )

            if patience_counter >= patience:
                if logger:
                    logger.info(f"TCN early stopping at epoch {epoch+1}")
                break
        else:
            if logger and (epoch + 1) % 10 == 0:
                logger.info(f"TCN epoch {epoch+1}/{max_epochs} | train_loss={avg_train_loss:.4f}")

    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, history


WIN = 12
BATCH_SIZE = 128
MAX_EPOCHS = 300
LEARNING_RATE = 1e-3
PATIENCE = 200
win_ah = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate_time_limited(
    df: pd.DataFrame,
    cols,
    max_gap: int = 20,
    *,
    method: str = "time",
    limit_direction: str = "both",
    drop_remaining_nans: bool = True,
) -> pd.DataFrame:
    """Interpolate selected columns with a maximum consecutive-gap limit.

    Requires a DatetimeIndex.

    Args:
        df: input df (DatetimeIndex)
        cols: columns to interpolate (must exist)
        max_gap: maximum consecutive NaNs to fill
        method: interpolation method
        limit_direction: interpolation direction
        drop_remaining_nans: if True, drop rows where any of cols remains NaN

    Returns:
        df copy with interpolated cols
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("interpolate_time_limited requires a DatetimeIndex")

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    out = df.copy()
    out[cols] = out[cols].interpolate(
        method=method,
        limit=max_gap,
        limit_direction=limit_direction,
    )

    if drop_remaining_nans:
        out = out.loc[~out[cols].isna().any(axis=1)].copy()

    return out

def train(
    site_id: int,
    logger,
    save_root: str | Path = r"C:\EPU\scot-backend\fast_api\apisrc\data\models\dimarxeio",
    # db: Session = Depends(get_db)
):
    db = SessionLocal()  # New session for background task

    try:
        exists = db.execute(
            text("SELECT 1 FROM sites WHERE id = :site_id"),
            {"site_id": site_id},
        ).scalar()

        if not exists:
            raise HTTPException(status_code=404, detail=f"Site {site_id} not found")

        sql = text("""
            SELECT
                MIN(timestamp) AS start_ts,
                MAX(timestamp) AS end_ts
            FROM consumption_data
            WHERE site_id = :site_id
        """)

        row = db.execute(sql, {"site_id": site_id}).mappings().one()
        start_ts = row["start_ts"]
        end_ts = row["end_ts"]

        # If there is no data for that site, MIN/MAX will both be NULL
        if start_ts is None or end_ts is None:
            raise HTTPException(
                status_code=404,
                detail=f"No consumption_data found for site {site_id}",
            )
        
        site_row = db.execute(
            text(
                """
                SELECT latitude, longitude
                FROM sites
                WHERE id = :site_id
                """
            ),
            {"site_id": site_id},
        ).fetchone()

        latitude = site_row.latitude
        longitude = site_row.longitude

        reconcile_environmental_data(
            db=db,
            site_id=1,
            start_ts=start_ts,
            end_ts=end_ts,
            resolution="30min",
            fill_nulls_only=False,
            dry_run=False,
        )
        run_disaggregation_for_site_util(db, site_id)

        sql = text("""
                SELECT
                    cons.timestamp AS timestamp,
                    c.tin          AS tin,
                    e.tout         AS tout,
                    cons.value     AS energy_consumption,
                    c.rh           AS rh,
                    e.rh_out       AS rh_out,
                    e.sw_out       AS sw_out,
                    c.hvac_mode    AS hvac_mode
                FROM consumption_data cons
                LEFT JOIN comfort_data c
                    ON c.site_id = cons.site_id
                    AND c.timestamp = cons.timestamp
                LEFT JOIN environmental_data e
                    ON e.site_id = cons.site_id
                    AND e.timestamp = cons.timestamp
                WHERE cons.site_id = :site_id
                AND cons.timestamp >= :start_ts
                AND cons.timestamp <= :end_ts
                ORDER BY cons.timestamp
            """)

        rows = db.execute(
            sql,
            {"site_id": site_id, "start_ts": start_ts, "end_ts": end_ts},
        ).mappings().all()

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Force a regular grid so gaps become NaN rows (instead of disappearing)
        df = df.asfreq("30min")
        CONT_COLS = ["tin", "tout", "rh", "rh_out", "sw_out", 'energy_consumption']  # pick what you trust to interpolate

        df = interpolate_time_limited(
            df,
            cols=CONT_COLS,
            max_gap=20,                 # 20 * 30min = 10 hours (adjust)
            drop_remaining_nans=False,  # keep big gaps as NaN
        )
        df["hvac_mode"] = pd.to_numeric(df["hvac_mode"], errors="coerce").fillna(-1).astype("Int16")

        df = build_features(df)

        df = make_hvac_splits(df, hvac_col='hvac_mode')
        # df['season'] = df['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)
        df_train_c = df[(df["season_t"] == "summer") & (df["split"] == "train")]
        df_val_c = df[(df["season_t"] == "summer") & (df["split"] == "val")]
        df_test_c = df[(df["season_t"] == "summer") & (df["split"] == "test")]
        df_train_h = df[(df["season_t"] == "winter") & (df["split"] == "train")]
        df_val_h = df[(df["season_t"] == "winter") & (df["split"] == "val")]
        df_test_h = df[(df["season_t"] == "winter") & (df["split"] == "test")]
        
        feature_cols = FEATURES_TIN_ON
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required features: {missing}")
            raise ValueError(f"Missing required features: {missing}")

        model_heating = None
        model_cooling = None
        if df_train_c.empty:
            logger.info("[ERROR] Insufficient data for train/val/test split summer")
        else:
            model_cooling = train_regime_model(
                df_train_c, df_val_c, FEATURES_TIN_ON, 'hvac_mode',
                "Cooling", logger, n_trials=30
            )

        if df_train_h.empty:
            logger.info("[ERROR] Insufficient data for train/val/test split summer")
        else:
            model_heating = train_regime_model(
                df_train_h, df_val_h, FEATURES_TIN_ON, 'hvac_mode',
                "Heating", logger, n_trials=30
            )
        save_dir = Path(save_root)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        framework = "lightgbm"


        if model_heating is not None:
            p = save_dir / "lgbm_heating.pkl"
            joblib.dump(model_heating, p)
            logger.info(f"Saved heating model: {p}")

            register_site_model(
                db,
                site_id=site_id,
                model_type="lgbm_heating",
                model_version=model_version,
                framework=framework,
                artifact_path=str(p),
                activate=True,
            )
            logger.info("Registered heating model in site_models.")

        if model_cooling is not None:
            p = save_dir / "lgbm_cooling.pkl"
            joblib.dump(model_cooling, p)
            logger.info(f"Saved cooling model: {p}")

            register_site_model(
                db,
                site_id=site_id,
                model_type="lgbm_cooling",
                model_version=model_version,
                framework=framework,
                artifact_path=str(p),
                activate=True,
            )
            logger.info("Registered cooling model in site_models.")

        if model_cooling is not None and df_test_c is not None:
            evaluate_model(logger, df_test_c, model_cooling, FEATURES_TIN_ON, 'hvac_mode', 'Cooling', './metrics')
        if model_heating is not None and df_test_h is not None:
            evaluate_model(logger, df_test_h, model_heating, FEATURES_TIN_ON, 'hvac_mode', 'Heating', './metrics')

        #MODEL NUMBER 2 - TIN OFF

        summer_test_start, summer_test_end = get_summer_test_range(df)
        df = make_global_off_split(df, summer_test_start, summer_test_end)
        df["split"] = df["global_split"]

        train_idx = np.where(df["split"].values == "train")[0]
        val_idx   = np.where(df["split"].values == "val")[0]
        test_idx  = np.where(df["split"].values == "test")[0]

        missing = [c for c in FEATURES_TIN_OFF_V3 if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        if "tin_target" not in df.columns:
            raise ValueError(f"Missing target column tin_target")

        # --- build datasets  ---
        X_train, y_train, mask_train, is_trans_train, _ = build_v3_dataset(
            df, FEATURES_TIN_OFF_V3, window=WIN, indices=train_idx,
            hvac_col="hvac_mode", oversample_transitions=False
        )
        X_val, y_val, mask_val, is_trans_val, _ = build_v3_dataset(
            df, FEATURES_TIN_OFF_V3, window=WIN, indices=val_idx,
            hvac_col="hvac_mode", oversample_transitions=False
        )
        X_test, y_test, mask_test, is_trans_test, ts_test = build_v3_dataset(
            df, FEATURES_TIN_OFF_V3, window=WIN, indices=test_idx,
            hvac_col="hvac_mode", oversample_transitions=False
        )
        def _ds_stats(name, X, y, ts):
            dy = y - X[:, -1, 0]  # assuming feature 0 is 'tin' in FEATURES_TIN_OFF_V3
            print(f"\n[{name}] N={len(y)}")
            print("  y range:", float(np.min(y)), "→", float(np.max(y)))
            print("  last Tin in window range:", float(np.min(X[:, -1, 0])), "→", float(np.max(X[:, -1, 0])))
            print("  delta (y - Tin_last) range:", float(np.min(dy)), "→", float(np.max(dy)))
            print("  delta percentiles:", np.percentile(dy, [1,5,50,95,99]).tolist())
            if ts is not None and len(ts) > 2:
                dts = pd.to_datetime(ts[1:]) - pd.to_datetime(ts[:-1])
                print("  ts step counts:", pd.Series(dts).value_counts().head(5).to_dict())

        _ds_stats("OFF train", X_train, y_train, None)
        _ds_stats("OFF val",   X_val,   y_val,   None)
        _ds_stats("OFF test",  X_test,  y_test,  ts_test)
        # --- scaling ---
        scaler, X_train_scaled, X_val_scaled, X_test_scaled = _scale_3d_with_freeze(
            X_train, X_val, X_test, FEATURES_TIN_OFF_V3
        )

        train_dataset = TransitionAwareDataset(X_train_scaled, y_train, mask_train, is_trans_train)
        val_dataset   = TransitionAwareDataset(X_val_scaled, y_val, mask_val, is_trans_val)

        # --- model init ---
        model = ImprovedTCNForOFF(
            in_dim=len(FEATURES_TIN_OFF_V3),
            hidden=128,
            levels=2,
            kernel_size=7,
            dropout=0.15,
            use_attention=True
        ).to(DEVICE)

        save_dir = Path(save_root)
        model_path = save_dir / "best_model_v3.pt"

        model, history = train_tcn_with_transitions(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                lr=LEARNING_RATE,
                device=DEVICE,
                patience=PATIENCE,
                save_path=str(model_path),
            )
        
        test_metrics = evaluate_tcn_model(model=model, X=X_test_scaled, y=y_test, device=DEVICE, logger=logger, prefix="TCN test")
        Tin_last = X_test_scaled[:, -1, 0]  # careful: this is SCALED tin if you scaled it
        # better: use unscaled X_test for baseline BEFORE scaling
        Tin_last_raw = X_test[:, -1, 0]
        yhat_persist = Tin_last_raw
        print("[BASELINE persist] MAE=", mean_absolute_error(y_test, yhat_persist),
            "RMSE=", _rmse(y_test, yhat_persist), "R2=", r2_score(y_test, yhat_persist))
        # transition vs steady (same reporting)
        trans_mask = is_trans_test > 0.5
        steady_mask = ~trans_mask

        trans_metrics = None
        steady_metrics = None
        if trans_mask.sum() > 0:
            trans_metrics = evaluate_tcn_model(model=model, X=X_test_scaled[trans_mask], y=y_test[trans_mask], device=DEVICE, logger=logger, prefix='TCN transition mask')
        if steady_mask.sum() > 0:
            steady_metrics = evaluate_tcn_model(model=model, X=X_test_scaled[steady_mask], y=y_test[steady_mask], device=DEVICE, logger=logger, prefix='TCN steady mask')


        best_val = float(np.min(history["val_mae"])) if "val_mae" in history else None

        scaler_path = save_dir / "scalers" / "scaler_v3.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        save_scaler_with_features(scaler, FEATURES_TIN_OFF_V3, str(scaler_path), logger)
        # model_path = save_root / "best_model_v3.pt"
        # # save
        # torch.save(model.state_dict(), model_path)

        # register
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        register_site_model(
            db,
            site_id=site_id,
            model_type="tcn_off",
            model_version=model_version,
            framework="pytorch",
            artifact_path=str(model_path),
            activate=True,
        )

        register_site_model(
            db,
            site_id=site_id,
            model_type="tin_scaler",
            model_version=model_version,
            framework="sklearn",
            artifact_path=str(scaler_path),
            activate=True,
        )

        #MODEL 3 TRAIN - RH
        required = FEATURES_AH
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        train_ds = AHSeqDataset(df, required, window=win_ah, indices=train_idx)
        val_ds = AHSeqDataset(df, required, window=win_ah, indices=val_idx)

        ah_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        ah_val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

        # Initialize model
        ah_model = CNNLSTMWithFuture(
            input_size=len(required),
            hidden_size=128,
            num_layers=2,
            cnn_out_channels=32,
            kernel_size=3
        ).to(DEVICE)

        ah_model, ah_scaler = train_masked_model(
            logger,
            model=ah_model,
            train_dl=ah_train_loader,
            input_cols=required,
            val_dl=ah_val_loader,
            max_epochs=200,
            lr=1e-3,
            device=DEVICE,
            loss_type="huber",
            huber_beta=0.2,
            metric_label="RMSE (AH)",
            save_dir=save_root,
        )
        test_metrics, test_dfp = evaluate_ah_split(
            logger,
            df=df,
            indices=test_idx,
            model=ah_model,
            scaler=ah_scaler,
            input_cols=required,
            window=win_ah,
            device=DEVICE,
            title_note="Test"
        )
        
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_path = save_dir / "scalers" / "scaler_ah.pkl"
        model_path = save_dir / "best_ah.pt"

        register_site_model(
            db,
            site_id=site_id,
            model_type="ah_rh",
            model_version=model_version,
            framework="pytorch",
            artifact_path=str(model_path),
            activate=True,
        )

        register_site_model(
            db,
            site_id=site_id,
            model_type="ah_scaler",
            model_version=model_version,
            framework="sklearn",
            artifact_path=str(scaler_path),
            activate=True,
        )


        return df, model_cooling, model_heating
    finally:
        db.close()

logger, log_path = make_file_logger(1, ROOT)
train(1, logger)






# def run_tcn_v3_for_csv(
#     file_path: Path,
#     df_index: int,
#     hvac_source_col: str,
#     save_root: Path,
# ) -> dict:


#     # --- feature checks (unchanged) ---
    

#     # --- save dirs ---
#     save_dir = save_root / f"df_{df_index}"
#     save_dir.mkdir(parents=True, exist_ok=True)
#     model_path = save_dir / "best_model_v3.pt"

#     # --- train (unchanged) ---
    


#     # --- evaluate (unchanged) ---
   
#     return {
#         "house": df_index,
#         "file": file_path.name,
#         "hvac_source_col": hvac_source_col,
#         "save_root": str(save_root),
#         "best_val_mae": best_val,
#         "test_metrics": test_metrics,
#         "trans_metrics": trans_metrics,
#         "steady_metrics": steady_metrics,
#         "model_path": str(model_path),
#         "scaler_path": str(scaler_path),
#     }

