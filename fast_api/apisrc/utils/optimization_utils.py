from dataclasses import dataclass
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso
import torch
from sqlalchemy import text
from pathlib import Path
import joblib 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from fast_api.apisrc.utils.weather_utils import RH_percent_from_AH_T, AH_gm3_from_T_RH
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet,
    NonNegativeReals, Param, Binary, Reals, SolverFactory, minimize, value
)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

OFF_DRIFT = -0.2          # °C per 30-min
OFF_NOISE_STD = 0.03      # °C (small noise)
TIN_MIN, TIN_MAX = 10.0, 30.0   # keep your safety bounds
rng = np.random.default_rng(42)
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

def enforce_min_on_duration(hvac: np.ndarray, min_len: int = 2) -> np.ndarray:
    """
    Enforce minimum HVAC ON duration (default 2 steps = 1 hour).
    Removes short runs and extends to multiples of min_len.
    """
    h = hvac.copy().astype(int)
    n = len(h)
    if min_len <= 1:
        return h
    
    # Pass 1: Remove ON runs shorter than min_len
    i = 0
    while i < n:
        if h[i] == 0:
            i += 1
            continue
        j = i
        while j < n and h[j] > 0:
            j += 1
        if (j - i) < min_len:
            h[i:j] = 0
        i = j + 1
    
    # Pass 2: Snap to multiples of min_len
    i = 0
    while i < n:
        if h[i] == 0:
            i += 1
            continue
        j = i
        while j < n and h[j] > 0:
            j += 1
        rem = (j - i) % min_len
        if rem != 0:
            need = min_len - rem
            end = min(j + need, n)
            h[i:end] = 1
        i = j + 1
    
    return h

def calculate_overall_comfort(list_tin, list_rh, list_season=None):
    """
    Calculate comfort (PPD) for given temperature and humidity sequences.

    Args:
        list_tin: List of indoor temperatures
        list_rh: List of relative humidity values
        list_season: Optional list of season values (0=winter, 1=spring, 2=summer, 3=fall)
                     If None, defaults to summer (clo=0.5)

    Returns:
        List of PPD values
    """
    ppd_values = []

    for i, (tin, rh) in enumerate(zip(list_tin, list_rh)):
        try:
            # assert np.isfinite(tin) and np.isfinite(rh) and (0.0 <= rh <= 100.0), f"[BAD INPUT] i={i} tin={tin} rh={rh}"
            comfort = pmv_ppd_iso(
                tdb=tin,
                tr=tin,
                vr=0.1,
                rh=rh,
                met=1.1,
                clo=1.0
            )
            if comfort['ppd'] is None:
                ppd_values.append(75)
            else:
                ppd_values.append(comfort['ppd'])
        except Exception as e:
            print(f"[PMV CRASH] i={i} tin={tin!r} rh={rh!r} type(tin)={type(tin)} type(rh)={type(rh)} exc={e!r}")
            raise    
    return ppd_values
        

def compute_comfort_percent(Tin_seq: np.ndarray, RH_seq: np.ndarray, season_seq: np.ndarray = None) -> np.ndarray:
    """Calculate comfort percentage from temperature and humidity."""
    vals = calculate_overall_comfort(list(Tin_seq), list(RH_seq), None)
    vals = np.asarray(vals, dtype=float)
    return 100.0 - vals


def fetch_active_site_models(db, site_id: int) -> dict:
    rows = db.execute(
        text(
            """
            SELECT
                model_type,
                framework,
                artifact_path
            FROM site_models
            WHERE site_id = :site_id
              AND is_active = TRUE
            """
        ),
        {"site_id": site_id},
    ).mappings().all()

    if not rows:
        raise RuntimeError(f"No active models found for site_id={site_id}")

    models = {row["model_type"]: row for row in rows}

    required = {
        "tcn_off",
        "lgbm_heating",
        "ah_rh",
        "tin_scaler",
        "ah_scaler",
    }
    missing = required - models.keys()
    if missing:
        raise RuntimeError(
            f"Missing site models for site_id={site_id}: {sorted(missing)}"
        )

    return models

AH_SCALER_COLS = [
    "ah",
    "ah_lag1",
    "ah_lag2",
    "ah_lag3",
    "tin",
    "tout",
    "ah_out",
    "hvac_mode",
]

def indices_from_names(all_features, scaler_features):
    return [all_features.index(c) for c in scaler_features]

def _indices_to_freeze(input_cols):

    FROZEN = {
        # "Tin",
        # "Tout",
        "hour_sin", "hour_cos",
        "season", "month", "month_sin", "month_cos",
        "is_trans_off",
        "off_runtime_1h", 
    }

    return [j for j, c in enumerate(input_cols) if c in FROZEN]

def load_models_from_db(
    db,
    house_num: int,
    device: str = "cpu",
):
    
    print(f"Loading models for Site {house_num} from DB...")

    site_models = fetch_active_site_models(db, house_num)

    # ---------------------------------------------------------
    # TCN OFF
    # ---------------------------------------------------------
    tcn_path = Path(site_models["tcn_off"]["artifact_path"])
    tin_scaler_path = Path(site_models["tin_scaler"]["artifact_path"])

    # features_path = tcn_path.with_name("features_v3.json")
    # with open(features_path, "r") as f:
    #     tin_off_features = json.load(f)["features"]
    tin_off_features = [
        "tin",
        "tout",
        "hour_sin",
        "hour_cos",
        "month",
        "season",
        "SW1h",
        "SW3h"
    ]

    tin_scaler_obj = joblib.load(tin_scaler_path)
    if isinstance(tin_scaler_obj, dict):
        tin_scaler = tin_scaler_obj["scaler"]
        tin_scaler.frozen_idx = tuple(tin_scaler_obj["frozen_idx"])
    else:
        tin_scaler = tin_scaler_obj
        if not hasattr(tin_scaler, "frozen_idx"):
            tin_scaler.frozen_idx = tuple(_indices_to_freeze(tin_off_features))

    tcn_model = ImprovedTCNForOFF(
        in_dim=len(tin_off_features),
        hidden=128,
        levels=2,
        kernel_size=7,
        dropout=0.15,
        use_attention=True,
    ).to(device)

    tcn_state = torch.load(tcn_path, map_location=device)
    tcn_model.load_state_dict(tcn_state)
    tcn_model.eval()

    print(f"  ✓ TCN-OFF loaded ({len(tin_off_features)} features)")

    # ---------------------------------------------------------
    # LGBM ON (exact logic preserved)
    # ---------------------------------------------------------
    lgbm_features = [
        'tin', 'tout',
        'tin_diff', 'tout_diff',
        'tin_ma3', 'tout_ma3',
        'hour_sin', 'hour_cos',
        'SW1h', 'SW3h',
        'month',
        'hvac_mode' 
    ]

    lgbm_path = Path(site_models["lgbm_heating"]["artifact_path"])
    lgbm_model = joblib.load(lgbm_path)

    print(f"  ✓ LGBM-ON loaded ({len(lgbm_features)} features)")

    # ---------------------------------------------------------
    # AH / RH
    # ---------------------------------------------------------
    ah_path = Path(site_models["ah_rh"]["artifact_path"])
    ah_scaler_path = Path(site_models["ah_scaler"]["artifact_path"])

    ah_features = [
        "ah",
        "ah_lag1",
        "ah_lag2",
        "ah_lag3",
        "tin",
        "tout",
        "ah_out",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "season",
        "hvac_mode"
    ]
        
    ah_scaler_obj = joblib.load(ah_scaler_path)

    if isinstance(ah_scaler_obj, dict):
        ah_scaler = ah_scaler_obj["scaler"]
        ah_scaler.frozen_idx = tuple(ah_scaler_obj["frozen_idx"])
    else:
        # Legacy artifact: bare StandardScaler
        ah_scaler = ah_scaler_obj

    #     # IMPORTANT: frozen_idx must match TRAINING, not DB names
    #     AH_FEATURES_TRAIN = [
    #         "AH", "AH_lag1", "AH_lag2", "AH_lag3",
    #         "Tin", "Tout", "AH_out",
    #         "hour_sin", "hour_cos",
    #         "month_sin", "month_cos",
    #         "season", "hvac_mode"
    #     ]

        # ah_scaler.frozen_idx = tuple(_indices_to_freeze(AH_FEATURES_TRAIN))

    # if not hasattr(ah_scaler, "frozen_idx") or ah_scaler.frozen_idx is None:
    ah_scaler.frozen_idx = tuple(_indices_to_freeze(ah_features))

    ah_model = CNNLSTMWithFuture(
        input_size=len(ah_features),
        hidden_size=128,
        num_layers=2,
        cnn_out_channels=32,
        kernel_size=3,
    ).to(device)

    ah_state = torch.load(ah_path, map_location=device)
    ah_model.load_state_dict(ah_state)
    ah_model.eval()

    print(f"  ✓ RH model loaded ({len(ah_features)} features)")

    return {
        "tcn_off": tcn_model,
        "tin_scaler": tin_scaler,
        "tin_off_features": tin_off_features,
        "lgbm_model": lgbm_model,
        "lgbm_features": lgbm_features,
        "ah_model": ah_model,
        "ah_scaler": ah_scaler,
        "ah_features": ah_features,
    }

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

def transform_with_known_indices(X, scaler, scale_idx):
    X = np.asarray(X, dtype=np.float32)
    orig_shape = X.shape

    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])

    X_out = X.copy()

    X_out[:, scale_idx] = (
        X[:, scale_idx] - scaler.mean_
    ) / scaler.scale_

    return X_out.reshape(orig_shape)
    
@dataclass
class RCConfig:
    """Configuration for RC MILP optimization."""
    horizon: int = 48
    Tmin: float = 20.0
    Tmax: float = 24.5
    w_low: float = 1.0
    w_high: float = 2.2
    lambda_noPV: float = 1.2
    lambda_slack: float = 40.0
    lambda_switch: float = 0.05
    safety_buffer: float = 0.0
    solver: str = "highs"
    time_limit_sec: int = 60
@dataclass
class RCParams:
    """RC thermal model parameters."""
    a: float  # Thermal inertia
    b_tout: float  # Outdoor temp influence
    c_low: float  # LOW mode cooling effect
    c_high: float  # HIGH mode cooling effect
    d: float  # Bias

def fit_rc_from_df(df):
    """
    Fit simple RC thermal model via least squares.

    Model: Tin[t+1] = a·Tin[t] + b·Tout[t] + c_low·u_low + c_high·u_high + d
    """

    df = df.copy().reset_index(drop=True)
    u_low = (df['hvac_mode'] == 1).astype(float).values[:-1]
    u_high = (df['hvac_mode'] == 2).astype(float).values[:-1]
    Tin_t = df['tin'].values[:-1]
    Tout_t = df['tout'].values[:-1]
    Tin_next = df['tin'].values[1:]
    mode_t = df["hvac_mode"].values[:-1].astype(int)

    X = np.column_stack([Tin_t, Tout_t, u_low, u_high, np.ones_like(Tin_t)])

    
    theta, *_ = np.linalg.lstsq(X, Tin_next, rcond=None)
    a, b_tout, c_low, c_high, d = theta.tolist()
    a = float(np.clip(a, 0.7, 0.999))  # Stabilize

    return RCParams(
        a=float(a),
        b_tout=float(b_tout),
        c_low=float(c_low),
        c_high=float(c_high),
        d=float(d)
    )

def fit_rc_by_thermal_regime(df: pd.DataFrame, min_samples: int = 200) -> dict:
    """
    Fit RC model(s) dynamically based on thermal regime detection.

    Returns dict:
        {
            "cooling": RCParams,
            "heating": RCParams
        }
    If the dataset contains only one regime, both map to same RC model.
    """

    # Determine thermal regime thresholds
    tout = df["tout"].values

    # Heuristic: Fahrenheit equivalent clusters
    cold_threshold = np.percentile(tout, 30)   # lower third
    hot_threshold = np.percentile(tout, 70)    # upper third

    has_heating = np.any(tout < cold_threshold)
    has_cooling = np.any(tout > hot_threshold)

    # One-regime case (your new dataset)
    if not (has_heating and has_cooling):
        # print("✓ Single thermal regime detected → fitting one RC model")
        global_model = fit_rc_from_df(df)

        return {
            "heating": global_model,
            "cooling": global_model
        }

    # Two-regime case
    # print("✓ Dual thermal regimes detected → fitting heating & cooling RC models")

    # Split by outdoor temperature
    cooling_df = df[df["tout"] >= hot_threshold]
    heating_df = df[df["tout"] <= cold_threshold]

    if len(cooling_df) < min_samples or len(heating_df) < min_samples:
        global_model = fit_rc_from_df(df)
        return {
            "heating": global_model,
            "cooling": global_model
        }

    rc_cooling = fit_rc_from_df(cooling_df)
    rc_heating = fit_rc_from_df(heating_df)

    return {
        "heating": rc_heating,
        "cooling": rc_cooling
    }

def optimize_schedule_with_rc(
        rc,
        Tin0,
        Tout_forecast,
        pv_forecast,
        cfg
):
    """
    Optimize HVAC schedule using RC model and MILP.

    Returns dict with keys: hvac_mode, Tin_rc, u_low, u_high
    """
    H = cfg.horizon
    assert len(Tout_forecast) == H and len(pv_forecast) == H, f"Forecasts must be length {H}"

    m = ConcreteModel()
    m.T = RangeSet(0, H)
    m.K = RangeSet(1, H)

    # Parameters
    m.a = Param(initialize=rc.a)
    m.b_tout = Param(initialize=rc.b_tout)
    m.c_low = Param(initialize=rc.c_low)
    m.c_high = Param(initialize=rc.c_high)
    m.d = Param(initialize=rc.d)
    m.Tout = Param(m.K, initialize={t: float(Tout_forecast[t - 1]) for t in m.K})
    m.PV = Param(m.K, initialize={t: float(pv_forecast[t - 1]) for t in m.K})

    # Variables
    m.u_low = Var(m.K, domain=Binary)
    m.u_high = Var(m.K, domain=Binary)
    m.Tin = Var(m.T, domain=Reals)
    m.s = Var(m.K, domain=NonNegativeReals)  # Slack for comfort
    m.dlow = Var(RangeSet(2, H), domain=NonNegativeReals)  # Switch tracking
    m.dhigh = Var(RangeSet(2, H), domain=NonNegativeReals)

    # Initial condition
    m.init_con = Constraint(expr=m.Tin[0] == Tin0)

    # Thermal dynamics
    def dyn_con(m, t):
        return (m.Tin[t] * m.a + m.b_tout * m.Tout[t + 1] +
                m.c_low * m.u_low[t + 1] + m.c_high * m.u_high[t + 1] + m.d == m.Tin[t + 1])

    m.dyn_con = Constraint(RangeSet(0, H - 1), rule=dyn_con)

    # Mode exclusivity
    m.excl = Constraint(m.K, rule=lambda m, t: m.u_low[t] + m.u_high[t] <= 1)

    # Comfort bounds with slack
    Tmin_eff = cfg.Tmin + cfg.safety_buffer
    Tmax_eff = cfg.Tmax - cfg.safety_buffer
    m.comfort_lo = Constraint(m.K, rule=lambda m, t: m.Tin[t] >= Tmin_eff - m.s[t])
    m.comfort_hi = Constraint(m.K, rule=lambda m, t: m.Tin[t] <= Tmax_eff + m.s[t])

    # Switch tracking
    m.switch_low_pos = Constraint(RangeSet(2, H), rule=lambda m, t: m.dlow[t] >= m.u_low[t] - m.u_low[t - 1])
    m.switch_low_neg = Constraint(RangeSet(2, H), rule=lambda m, t: m.dlow[t] >= m.u_low[t - 1] - m.u_low[t])
    m.switch_high_pos = Constraint(RangeSet(2, H), rule=lambda m, t: m.dhigh[t] >= m.u_high[t] - m.u_high[t - 1])
    m.switch_high_neg = Constraint(RangeSet(2, H), rule=lambda m, t: m.dhigh[t] >= m.u_high[t - 1] - m.u_high[t])

    # Objective: maximize PV overlap, minimize grid usage and switches
    def obj_rule(m):
        pv_reward = sum(m.PV[t] * (cfg.w_low * m.u_low[t] + cfg.w_high * m.u_high[t]) for t in m.K)
        nopv_pen = cfg.lambda_noPV * sum((1.0 - m.PV[t]) * (m.u_low[t] + m.u_high[t]) for t in m.K)
        slack_pen = cfg.lambda_slack * sum(m.s[t] for t in m.K)
        switch_pen = cfg.lambda_switch * (
                    sum(m.dlow[t] for t in range(2, H + 1)) + sum(m.dhigh[t] for t in range(2, H + 1)))
        return (nopv_pen + slack_pen + switch_pen) - pv_reward

    m.obj = Objective(rule=obj_rule, sense=minimize)

    # Solve
    opt = SolverFactory(cfg.solver)
    if cfg.time_limit_sec:
        if cfg.solver == "cbc":
            opt.options["sec"] = cfg.time_limit_sec
        elif cfg.solver == "highs":
            opt.options["time_limit"] = cfg.time_limit_sec

    res = opt.solve(m, tee=False)

    # Extract solution
    u_low = np.array([value(m.u_low[t]) for t in range(1, H + 1)], dtype=float)
    u_high = np.array([value(m.u_high[t]) for t in range(1, H + 1)], dtype=float)
    Tin_rc = np.array([value(m.Tin[t]) for t in range(0, H + 1)], dtype=float)

    hvac_mode = np.zeros(H, dtype=int)
    hvac_mode[u_low > 0.5] = 1
    hvac_mode[u_high > 0.5] = 2

    return {"hvac_mode": hvac_mode, "Tin_rc": Tin_rc, "u_low": u_low, "u_high": u_high}



def forecast_48h(
    df: pd.DataFrame,
    start_idx,
    models,
    horizon = 48,
    win_tin = 12,
    win_ah = 24,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:

    # Extract models
    tcn_off = models['tcn_off']
    tin_scaler = models['tin_scaler']
    tin_off_features = models['tin_off_features']
    lgbm_model = models['lgbm_model']
    lgbm_features = models['lgbm_features']
    ah_model = models['ah_model']
    ah_scaler = models['ah_scaler']
    ah_features = models['ah_features']
    ah_scale_idx = indices_from_names(ah_features, AH_SCALER_COLS)
    
    FILL_COLS = ["tout", "ah_out", "rh_out"]

    df = df.copy()

    # Ensure time order
    df = df.sort_index()

    # Forward-fill only the known-bad exogenous columns
    df[FILL_COLS] = df[FILL_COLS].ffill()

    # Optional: backward fill first row if needed
    df[FILL_COLS] = df[FILL_COLS].bfill()

    # Final safety check (keep this during debugging)
    if df[FILL_COLS].isna().any().any():
        bad = df[FILL_COLS].isna().sum()
        raise ValueError(f"NaNs remain after ffill: {bad}")
    # Safety checks
    need_hist = max(win_tin, win_ah, 3)
    # feature_map = {
    #     'tin': Tin_now,
    #     'tout': Tout,
    #     'hour_sin': float(row['hour_sin']),
    #     'hour_cos': float(row['hour_cos']),
    #     'month_sin': float(row['month_sin']),
    #     'month_cos': float(row['month_cos']),
    #     'season': float(row['season']),
    #     'SW1h': float(row.get('SW1h', 0.0)),
    #     'SW3h': float(row.get('SW3h', 0.0)),
    # }
     
    # Initialize Tin-OFF sequence (scaled)
    tin_seq = []
    tin_seq_ts = [] 
    for i in range(start_idx - win_tin, start_idx):
        row = df.iloc[i]
        init_vals = [float(row[c]) for c in tin_off_features]     
        v = np.array(init_vals, dtype=np.float32).reshape(1, -1)
        v_scaled = _transform_with_freeze(v, tin_scaler)
        tin_seq_ts.append(df.index[i])
        tin_seq.append(v_scaled[0].tolist())

    hist_start = start_idx - win_ah
    hist_end = start_idx  # exclusive


    # Initialize RH sequence (scaled)
    ah_seq = []
    for i in range(start_idx - win_ah, start_idx):
        row = df.iloc[i]
        vals = [float(row[c]) for c in ah_features]
        v = np.array(vals, dtype=np.float32).reshape(1, -1)
          
        v_scaled = transform_with_known_indices(
            v,
            ah_scaler,
            ah_scale_idx,
        )
        # v_scaled = _transform_with_freeze(v, ah_scaler)
        ah_seq.append(v_scaled[0].tolist())
        # ah_seq.append(v[0].tolist())

    ah_seq_arr = np.array(ah_seq, dtype=np.float32)
    if ah_seq_arr.shape != (win_ah, len(ah_features)):
        raise ValueError(f"AH seq shape wrong: {ah_seq_arr.shape}, expected {(win_ah, len(ah_features))}")

    if not np.isfinite(ah_seq_arr).all():
        bad = np.argwhere(~np.isfinite(ah_seq_arr))
        t0, d0 = bad[0]
       
        raise ValueError("AH history contains NaN/Inf")
    # Initialize buffers for autoregressive features
    tin_buf = [
        float(df['tin'].iloc[start_idx - 3]),
        float(df['tin'].iloc[start_idx - 2]),
        float(df['tin'].iloc[start_idx-1]),
    ]

    tout_buf = [
        float(df['tout'].iloc[start_idx - 2]),
        float(df['tout'].iloc[start_idx - 1]),
    ]

    tin_prev = float(df['tin'].iloc[start_idx - 1])

    # RH autoregressive lags
    ah_prev = float(df['ah'].iloc[start_idx - 1])
    ah_l1 = float(df['ah_lag1'].iloc[start_idx - 1])
    ah_l2 = float(df['ah_lag2'].iloc[start_idx - 1])
    ah_l3 = float(df['ah_lag3'].iloc[start_idx - 1])
   

    # Transition tracking
    prev_mode = None
    steps_since_transition = 999

    results = []
    # Recursive forecasting loop
    with torch.no_grad():
        for step in range(horizon):
            i = start_idx + step
            row = df.iloc[i]              
            
            # Current HVAC mode
            hvac_mode = int(row['hvac_mode'])

            # Track transitions
            if prev_mode is None:
                prev_mode = hvac_mode
                steps_since_transition = 999
            elif prev_mode != hvac_mode:
                steps_since_transition = 0
                prev_mode = hvac_mode
            else:
                steps_since_transition += 1

            # Current exogenous variables
            Tout = float(row['tout'])
            Tin_now = tin_buf[-1]
            Tin_prev1 = tin_buf[-2]
            Tin_prev2 = tin_buf[-3]

            Tout_prev1 = tout_buf[-1]
            Tout_prev2 = tout_buf[-2]
            sw1 = float(row.get("SW1h", 0.0))
            sw3 = float(row.get("SW3h", 0.0))
            sw_out = float(row.get("sw_out", 0.0))

            # if raw solar is basically zero, force derived features to zero
            if sw_out < 1e-3:
                sw1 = 0.0
                sw3 = 0.0
            vals_seq = [Tin_now, Tout,
                float(row["hour_sin"]), float(row["hour_cos"]),
                float(row["month"]), float(row["season"]),
                sw1, sw3]
            v_seq = np.array(vals_seq, np.float32).reshape(1, -1)
            v_seq_scaled = _transform_with_freeze(v_seq, tin_scaler)
            tin_seq = tin_seq[1:] + [v_seq_scaled[0].tolist()]
            tin_seq_ts = tin_seq_ts[1:] + [df.index[i]]
            # ===== Predict Tin =====
            if hvac_mode == 0:
                # Use TCN for OFF periods
                # # Tin_now = tin_buf[-1]
                # Tout_prev1 = tout_buf[-1]
                # Tout_prev2 = tout_buf[-2]

                Tout_diff = Tout - Tout_prev1
                Tout_ma3 = (Tout + Tout_prev1 + Tout_prev2) / 3.0

                # Build feature row for TCN
                

                # # Fill any missing features from dataframe
                # for c in tin_off_features:
                #     if c not in feature_map:
                #         feature_map[c] = float(row.get(c, 0.0))

                # vals = [feature_map[c] for c in tin_off_features]
                # v = np.array(vals, dtype=np.float32).reshape(1, -1)
                # v_scaled = _transform_with_freeze(v, tin_scaler)

                # # Update sequence (sliding window)
                # tin_seq = tin_seq[1:] + [v_scaled[0].tolist()]
                # tin_seq_ts = tin_seq_ts[1:] + [df.index[i]]


                # Predict
                x_tin = torch.tensor([tin_seq], dtype=torch.float32, device=device)
                # tin_pred = float(tcn_off(x_tin).detach().cpu().item())
                tin_pred = Tin_now + OFF_DRIFT + float(rng.normal(0.0, OFF_NOISE_STD))

                # Optional: keep within plausible bounds (prevents runaway)
                tin_pred = float(np.clip(tin_pred, TIN_MIN, TIN_MAX))
                
            else:
                # Use LGBM for ON periods
                # Tin_now = tin_buf[-1]


                feature_map = {
                    'tin': Tin_now,
                    'tout': Tout,
                    'tin_diff': Tin_now - Tin_prev1,
                    'tout_diff': Tout - Tout_prev1,
                    'tin_ma3': (Tin_now + Tin_prev1 + Tin_prev2) / 3.0,
                    'tout_ma3': (Tout + Tout_prev1 + Tout_prev2) / 3.0,
                    'hour_sin': float(row['hour_sin']),
                    'hour_cos': float(row['hour_cos']),
                    'SW1h': sw1,
                    'SW3h': sw3,
                    'month': float(row['month']),
                    'hvac_mode': float(row['hvac_mode']),
                }

                # Fill missing features
                for c in lgbm_features:
                    if c not in feature_map:
                        feature_map[c] = float(row.get(c, 0.0))

                x_lgbm = [feature_map[c] for c in lgbm_features]

                # Select model based on mode
                model = lgbm_model

                tin_pred = float(model.predict([x_lgbm])[0])

            # Apply constraints and blending
            # tin_pred = constrain_prediction(tin_pred, tin_prev, hvac_mode)
            # tin_pred = blend_on_transition(
            #     tin_pred, tin_prev,
            #     prev_mode if step > 0 else hvac_mode,
            #     hvac_mode,
            #     steps_since_transition
            # )
            if tin_pred > 30.0 or tin_pred < 10.0 or (not np.isfinite(tin_pred)):
                src = "TCN_OFF" if hvac_mode == 0 else "LGBM_ON"
                print("\n[TIN FIRST BAD]")
                print("[DEBUG] df sw_out=", float(row.get("sw_out", np.nan)),
                "SW1h=", float(row.get("SW1h", np.nan)),
                "SW3h=", float(row.get("SW3h", np.nan)))
                print(" step=", step, "i=", i, "ts=", df.index[i], "src=", src, "hvac_mode=", hvac_mode)
                print(" Tin_now=", float(Tin_now), "Tin_pred=", float(tin_pred), "Tout=", float(Tout))
                print(" tin_buf(before)=", [float(x) for x in tin_buf], "tout_buf(before)=", [float(x) for x in tout_buf])

                if hvac_mode == 0:
                    # show exactly what TCN saw (already computed in your code)
                    print(" raw_feats=", dict(zip(tin_off_features, vals_seq)))
                    print(" scaled_vec=", v_seq_scaled[0].tolist())
                    print(" tin_seq_ts_tail=", tin_seq_ts[-5:])
                else:
                    print(" x_lgbm=", x_lgbm)

                raise RuntimeError("Tin went out of range")
            tin_prev = tin_pred

            # Update temperature buffers
            tin_buf = [tin_buf[-2], tin_buf[-1], tin_pred]
            tout_buf = [tout_buf[-1], Tout]

            # ===== Predict RH =====
            ah_feature_map = {
                'ah': ah_prev,
                'ah_lag1': ah_l1,
                'ah_lag2': ah_l2,
                'ah_lag3': ah_l3,
                'tin': tin_pred,
                'tout': Tout,
                'ah_out': float(row['ah_out']),
                'hour_sin': float(row['hour_sin']),
                'hour_cos': float(row['hour_cos']),
                'month_sin': float(row.get('month_sin', 0.0)),
                'month_cos': float(row.get('month_cos', 0.0)),
                'season': float(row.get('season', 0.0)),
                'hvac_mode': float(hvac_mode),
            }
            
            for c in ah_features:
                if c not in ah_feature_map:
                    ah_feature_map[c] = float(row.get(c, 0.0))


            ah_vals = [ah_feature_map[c] for c in ah_features]
           
            v_ah = np.array(ah_vals, dtype=np.float32).reshape(1, -1)
           
            ah_scale_idx = indices_from_names(ah_features, AH_SCALER_COLS)

            v_ah_scaled = transform_with_known_indices(
                v_ah,
                ah_scaler,
                ah_scale_idx,
            )
            
            ah_seq = ah_seq[1:] + [v_ah_scaled[0].tolist()]
          
            # ah_seq = ah_seq[1:] + [v_ah[0].tolist()]


            x_ah = torch.tensor([ah_seq], dtype=torch.float32, device=device)
           

            ah_pred = float(ah_model(x_ah).detach().cpu().item())
           
            # Update AH lags
            ah_l3, ah_l2, ah_l1, ah_prev = ah_l2, ah_l1, ah_prev, ah_pred

            # Convert AH to RH percentage
            rh_pred = RH_percent_from_AH_T(ah_pred, tin_pred)

            if step == 0:
                rh_direct = RH_percent_from_AH_T(ah_pred, tin_pred)
                print(
                    "[RH DIRECT CHECK]",
                    "AH=", ah_pred,
                    "Tin=", tin_pred,
                    "RH=", rh_direct
                )

            if step == 0:
                print(
                    "[RH FINAL]",
                    rh_pred
                )
            # Store results
            results.append({
                'timestamp': df.index[i],
                'Tin_pred': tin_pred,
                'Tin_true': float(row['tin']),
                'RH_pred': rh_pred,
                'RH_true': float(row['rh']) if 'rh' in row else np.nan,
                'hvac_mode': hvac_mode,
                'Tin_target': float(row['Tin_target']) if 'Tin_target' in row else np.nan,
            })

    return pd.DataFrame(results).set_index('timestamp')

def repair_to_feasible(
        simulate_fn,
        pv,
        sched_in,
        season_seq,
        comfort_min = 80.0,
        lead_seq = (1, 2, 3, 4, 6, 8),
        escalate_inside_pv_first = True,
        max_iters = 40,
):
    """
    Greedy PV-friendly repair: escalate HVAC at low-comfort timesteps.
    Prioritizes HIGH mode during PV periods.
    """
    H = len(sched_in)
    pv = np.asarray(pv, int)
    s = np.asarray(sched_in, int).copy()

    for _ in range(max_iters):
        Tin, RH = simulate_fn(s)
        comfort = compute_comfort_percent(Tin, RH, season_seq)
        if comfort.min() >= comfort_min:
            return s, Tin, RH, True

        t = int(np.argmin(comfort))

        # If in PV period, escalate to HIGH first
        if escalate_inside_pv_first and pv[t] == 1:
            if s[t] < 2:
                s[t] = 2
                continue

        # Try lookahead windows
        fixed = False
        for lead in lead_seq:
            i0 = max(0, t - lead)
            i1 = min(H, t + (lead // 2) + 1)

            cand = s.copy()
            cand[i0:i1] = np.maximum(cand[i0:i1], 1)
            Tin_c, RH_c = simulate_fn(cand)
            if compute_comfort_percent(Tin_c, RH_c, season_seq).min() >= comfort_min:
                s = cand
                fixed = True
                break

        if fixed:
            continue

        # Last resort: force HIGH in window
        span = 4
        i0 = max(0, t - span // 2)
        i1 = min(H, t + (span - span // 2))
        s[i0:i1] = np.maximum(s[i0:i1], 2)

    Tin, RH = simulate_fn(s)
    return s, Tin, RH, False

def harden_schedule_until_comfort(
        simulate_fn,
        pv,
        sched_in,
        season_seq,
        comfort_min = 80.0,
        min_on_steps = 2,
        max_passes = 6,
):
    """
    Aggressive fallback repair: escalate all violation periods.
    """
    H = len(sched_in)
    s = np.asarray(sched_in, int).copy()
    pv = np.asarray(pv, int)

    for _ in range(max_passes):
        Tin, RH = simulate_fn(s)
        comfort = compute_comfort_percent(Tin, RH, season_seq)
        bad = np.where(comfort < comfort_min)[0]
        if bad.size == 0:
            return s, Tin, RH, True

        for t in bad:
            i0 = max(0, t - 2)
            i1 = min(H, t + 1)
            s[i0:i1] = np.maximum(s[i0:i1], 2 if pv[t] == 1 else 1)

        s = enforce_min_on_duration(s, min_len=min_on_steps)

    Tin, RH = simulate_fn(s)
    ok = (compute_comfort_percent(Tin, RH, season_seq).min() >= comfort_min)
    return s, Tin, RH, ok

@torch.no_grad()
def rollout_48_steps_backend(
    df,
    start_idx,
    hvac_mode_48,
    site_id,
    db,
    device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:

    models = load_models_from_db(db, site_id, device)

    df_temp = df.copy()
    for t in range(48):
        df_temp.loc[df_temp.index[start_idx + t], "hvac_mode"] = int(hvac_mode_48[t])

    result = forecast_48h(
        df=df_temp,
        start_idx=start_idx,
        models=models,
        horizon=48,
        device=device,
    )
    print("\n[FORECAST_48H HEAD]")
    print(result[["Tin_pred", "RH_pred", "hvac_mode"]].head(8).to_string())

    print("\n[FORECAST_48H TAIL]")
    print(result[["Tin_pred", "RH_pred", "hvac_mode"]].tail(8).to_string())

    print("\n[FORECAST_48H STATS]")
    print("Tin_pred:", "min=", float(result["Tin_pred"].min()),
        "max=", float(result["Tin_pred"].max()))
    print("RH_pred :", "min=", float(result["RH_pred"].min()),
        "max=", float(result["RH_pred"].max()))

    # Print first offending timestep (if any)
    bad = result.index[(~np.isfinite(result["Tin_pred"])) | (result["Tin_pred"] > 60) | (result["Tin_pred"] < 0)]
    if len(bad):
        ts0 = bad[0]
        row = result.loc[ts0]
        print("\n[FORECAST_48H BAD FIRST]")
        print("ts=", ts0, "Tin_pred=", float(row["Tin_pred"]), "RH_pred=", float(row["RH_pred"]), "hvac_mode=", int(row["hvac_mode"]))


    return result["Tin_pred"].values, result["RH_pred"].values

