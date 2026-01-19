"""
Multi-Step Forecast Utilities for Trading Forecast v2

This module provides utilities for multi-step ahead forecasting with strict
data leakage prevention. All functions are designed to work with the
train_and_evaluate() method in data_science.py.

Key Features:
- Multi-step target construction (y = [Close_{t+1}, ..., Close_{t+horizon}])
- Per-step and aggregated evaluation metrics
- Multi-step baselines (naive, drift, rolling mean)
- Feature validation to prevent leakage
- Target transforms (price, pct_change, log_return) with roundtrip support

Usage:
    from forecast_utils import (
        build_multistep_targets,
        evaluate_multistep,
        compute_multistep_baselines,
        validate_no_leakage,
        roll_multistep_predictions,
        aggregate_rolling_predictions,
        compute_rolling_baselines,
        evaluate_rolling_forecast,
        create_10day_trend_labels,
        # Target transforms
        get_target_transform,
        to_target,
        from_target,
        from_target_cumulative,
        evaluate_in_both_spaces,
        VALID_TARGET_TRANSFORMS
    )
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def build_multistep_targets(close_series: np.ndarray, horizon: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build multi-step target matrix from Close price series.
    
    For each sample t, creates target vector y[t] = [Close_{t+1}, ..., Close_{t+horizon}].
    This is used for direct multi-output regression.
    
    IMPORTANT: Last `horizon` samples will have incomplete targets and are excluded.
    The returned valid_mask indicates which indices have complete targets.
    
    Args:
        close_series: 1D array of Close prices (length N)
        horizon: Number of steps to forecast (default: 10)
    
    Returns:
        y_multistep: 2D array of shape (N - horizon, horizon) containing future prices
        valid_indices: 1D array of valid sample indices (0 to N - horizon - 1)
    
    Example:
        >>> close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        >>> y, valid_idx = build_multistep_targets(close, horizon=3)
        >>> y.shape
        (8, 3)  # 11 - 3 = 8 valid samples
        >>> y[0]  # Target for t=0: [Close_1, Close_2, Close_3]
        array([101, 102, 103])
    """
    n = len(close_series)
    
    if n <= horizon:
        raise ValueError(f"Series length ({n}) must be greater than horizon ({horizon})")
    
    # Number of valid samples (those with complete horizon-step targets)
    n_valid = n - horizon
    
    # Pre-allocate target matrix
    y_multistep = np.zeros((n_valid, horizon), dtype=np.float64)
    
    # Fill target matrix
    for t in range(n_valid):
        # y[t] = [Close_{t+1}, Close_{t+2}, ..., Close_{t+horizon}]
        y_multistep[t, :] = close_series[t + 1 : t + 1 + horizon]
    
    valid_indices = np.arange(n_valid)
    
    logger.debug(f"Built multi-step targets: {n_valid} samples, {horizon} steps ahead")
    
    return y_multistep, valid_indices


def evaluate_multistep(y_true: np.ndarray, y_pred: np.ndarray, 
                       last_known_prices: np.ndarray = None) -> Dict:
    """
    Evaluate multi-step forecast predictions.
    
    Computes:
    - Aggregated metrics (RMSE_avg, MAE_avg, R²_avg across all steps)
    - Per-step metrics (RMSE_step_1, ..., RMSE_step_horizon)
    - MAPE if prices are provided
    
    Args:
        y_true: 2D array of actual values, shape (n_samples, horizon)
        y_pred: 2D array of predicted values, shape (n_samples, horizon)
        last_known_prices: 1D array of last known prices for MAPE calculation (optional)
    
    Returns:
        Dict with keys:
            - rmse_avg: Average RMSE across all steps
            - mae_avg: Average MAE across all steps
            - r2_avg: Average R² across all steps
            - rmse_step_k: RMSE for step k (k = 1 to horizon)
            - mae_step_k: MAE for step k
            - r2_step_k: R² for step k
            - mape_avg: Average MAPE (if last_known_prices provided)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    n_samples, horizon = y_true.shape
    
    metrics = {}
    
    # Per-step metrics
    rmse_per_step = []
    mae_per_step = []
    r2_per_step = []
    
    for step in range(horizon):
        y_t = y_true[:, step]
        y_p = y_pred[:, step]
        
        # Handle any NaN values
        valid_mask = ~(np.isnan(y_t) | np.isnan(y_p))
        if np.sum(valid_mask) < 2:
            rmse_per_step.append(np.nan)
            mae_per_step.append(np.nan)
            r2_per_step.append(np.nan)
            continue
        
        y_t_valid = y_t[valid_mask]
        y_p_valid = y_p[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_t_valid, y_p_valid))
        mae = mean_absolute_error(y_t_valid, y_p_valid)
        r2 = r2_score(y_t_valid, y_p_valid)
        
        rmse_per_step.append(float(rmse))
        mae_per_step.append(float(mae))
        r2_per_step.append(float(r2))
        
        # Store per-step metrics (1-indexed for human readability)
        metrics[f'rmse_step_{step + 1}'] = float(rmse)
        metrics[f'mae_step_{step + 1}'] = float(mae)
        metrics[f'r2_step_{step + 1}'] = float(r2)
    
    # Aggregated metrics (nanmean to handle any missing steps)
    metrics['rmse_avg'] = float(np.nanmean(rmse_per_step))
    metrics['mae_avg'] = float(np.nanmean(mae_per_step))
    metrics['r2_avg'] = float(np.nanmean(r2_per_step))
    
    # Also compute overall metrics on flattened arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    valid_flat = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    
    if np.sum(valid_flat) > 1:
        metrics['rmse_overall'] = float(np.sqrt(mean_squared_error(
            y_true_flat[valid_flat], y_pred_flat[valid_flat])))
        metrics['mae_overall'] = float(mean_absolute_error(
            y_true_flat[valid_flat], y_pred_flat[valid_flat]))
        metrics['r2_overall'] = float(r2_score(
            y_true_flat[valid_flat], y_pred_flat[valid_flat]))
    
    # MAPE calculation (if prices provided)
    if last_known_prices is not None and len(last_known_prices) == n_samples:
        # MAPE = mean(|actual - predicted| / |actual|) * 100
        mape_per_step = []
        for step in range(horizon):
            y_t = y_true[:, step]
            y_p = y_pred[:, step]
            valid = (y_t != 0) & ~np.isnan(y_t) & ~np.isnan(y_p)
            if np.sum(valid) > 0:
                mape = np.mean(np.abs((y_t[valid] - y_p[valid]) / y_t[valid])) * 100
                mape_per_step.append(float(mape))
                metrics[f'mape_step_{step + 1}'] = float(mape)
        
        if mape_per_step:
            metrics['mape_avg'] = float(np.mean(mape_per_step))
    
    metrics['horizon'] = horizon
    metrics['n_samples'] = n_samples
    
    return metrics


def compute_multistep_baselines(train_close: np.ndarray, 
                                 holdout_close: np.ndarray,
                                 horizon: int = 10) -> Dict[str, Dict]:
    """
    Compute multi-step baseline forecasts for holdout period.
    
    Baselines:
    1. Naive: All future predictions = last training price
    2. Drift: Linear extrapolation based on recent trend
    3. Rolling Mean: All predictions = mean of last N training prices
    
    Args:
        train_close: 1D array of training Close prices
        holdout_close: 1D array of holdout Close prices (for evaluation)
        horizon: Number of steps to forecast
    
    Returns:
        Dict with baseline results, each containing:
            - predictions: 2D array (n_holdout_samples, horizon)
            - metrics: Dict with RMSE, MAE, R² (aggregated and per-step)
    """
    last_train_price = train_close[-1]
    n_holdout = len(holdout_close) - horizon  # Valid holdout samples
    
    if n_holdout <= 0:
        logger.warning(f"Holdout period ({len(holdout_close)}) too short for horizon ({horizon})")
        return {}
    
    # Build holdout target matrix
    y_holdout, _ = build_multistep_targets(holdout_close, horizon)
    
    baselines = {}
    
    # === NAIVE BASELINE ===
    # Predict last known price for all steps
    naive_pred = np.full((n_holdout, horizon), last_train_price)
    naive_metrics = evaluate_multistep(y_holdout, naive_pred)
    baselines['naive'] = {
        'predictions': naive_pred,
        'metrics': naive_metrics,
        'description': 'Last training price for all steps'
    }
    
    # === DRIFT BASELINE ===
    # Linear extrapolation based on recent trend
    n_trend = min(20, len(train_close) // 4)
    if n_trend > 1:
        recent_prices = train_close[-n_trend:]
        daily_drift = (recent_prices[-1] - recent_prices[0]) / (n_trend - 1)
    else:
        daily_drift = 0
    
    # For each holdout sample, drift continues from last train price
    drift_pred = np.zeros((n_holdout, horizon))
    for step in range(horizon):
        drift_pred[:, step] = last_train_price + daily_drift * (step + 1)
    
    drift_metrics = evaluate_multistep(y_holdout, drift_pred)
    baselines['drift'] = {
        'predictions': drift_pred,
        'metrics': drift_metrics,
        'description': f'Linear drift based on last {n_trend} training prices'
    }
    
    # === ROLLING MEAN BASELINE ===
    window = min(20, len(train_close))
    rolling_mean = np.mean(train_close[-window:])
    rolling_pred = np.full((n_holdout, horizon), rolling_mean)
    rolling_metrics = evaluate_multistep(y_holdout, rolling_pred)
    baselines['rolling_mean_20'] = {
        'predictions': rolling_pred,
        'metrics': rolling_metrics,
        'description': f'Rolling mean of last {window} training prices'
    }
    
    logger.info(f"Multi-step baselines computed for {n_holdout} holdout samples, {horizon} steps:")
    logger.info(f"  Naive RMSE_avg: ${baselines['naive']['metrics']['rmse_avg']:.2f}")
    logger.info(f"  Drift RMSE_avg: ${baselines['drift']['metrics']['rmse_avg']:.2f}")
    logger.info(f"  Rolling Mean RMSE_avg: ${baselines['rolling_mean_20']['metrics']['rmse_avg']:.2f}")
    
    return baselines


def validate_no_leakage(feature_columns: List[str], 
                        data_index: np.ndarray = None,
                        target_index: np.ndarray = None) -> Tuple[bool, List[str]]:
    """
    Validate feature columns and indices for data leakage.
    
    Checks:
    1. No unshifted rolling/return features
    2. Feature column names follow expected pattern
    3. Target indices are strictly > feature indices (if provided)
    
    Args:
        feature_columns: List of feature column names
        data_index: Array of data sample indices (optional)
        target_index: Array of target sample indices (optional)
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True
    
    # Check 1: Suspicious column names
    leakage_patterns = [
        'Close',  # Raw Close should not be in features
        '_t+',    # Future indicators
        '_future',
        '_next',
        '_forward',
        '_ahead'
    ]
    
    for col in feature_columns:
        # Raw Close (not Close_lag, Close_return, etc.)
        if col == 'Close':
            warnings.append(f"CRITICAL: Raw 'Close' in features - DATA LEAKAGE!")
            is_valid = False
        
        # Future-looking patterns
        for pattern in leakage_patterns[1:]:
            if pattern in col.lower():
                warnings.append(f"CRITICAL: Future pattern '{pattern}' in '{col}' - DATA LEAKAGE!")
                is_valid = False
    
    # Check 2: Verify all derived features have proper lag
    expected_patterns = ['_lag_', '_return_', '_rolling_', '_ma_ratio_', '_ROC_', '_momentum_', '_acceleration']
    
    close_derived = [c for c in feature_columns if c.startswith('Close_')]
    for col in close_derived:
        if not any(p in col for p in expected_patterns):
            warnings.append(f"WARNING: '{col}' may not be properly lagged")
    
    # Check 3: Index validation (if provided)
    if target_index is not None and data_index is not None:
        if np.any(target_index <= data_index):
            warnings.append("CRITICAL: Target indices overlap with feature indices - DATA LEAKAGE!")
            is_valid = False
    
    # Info about feature set
    if len(close_derived) == 0:
        warnings.append("INFO: No Close-derived features found")
    else:
        logger.debug(f"Found {len(close_derived)} Close-derived features")
    
    return is_valid, warnings


def recursive_multistep_forecast(model, 
                                  close_history: np.ndarray,
                                  feature_scaler,
                                  y_scaler,
                                  horizon: int = 10,
                                  lags: List[int] = None,
                                  windows: List[int] = None,
                                  model_type: str = 'sklearn',
                                  lstm_config: dict = None) -> np.ndarray:
    """
    Generate recursive multi-step forecast for FUTURE prediction (beyond known data).
    
    For Multi-Output models: Uses direct prediction (all steps at once).
    For Single-Output models: Uses true recursive prediction.
    
    Args:
        model: Trained regression model
        close_history: Array of historical Close prices (last ~50 values)
        feature_scaler: Fitted StandardScaler for X features
        y_scaler: Fitted StandardScaler for y (Close prices) - can be for single or multi-output
        horizon: Number of steps to forecast
        lags: Lag periods used in feature engineering (default: [1, 5, 10])
        windows: Rolling window sizes (default: [5, 20])
        model_type: 'sklearn', 'keras', 'lstm', or 'multioutput'
        lstm_config: Optional dict with 'timesteps' and 'features_per_step' for enhanced LSTM
    
    Returns:
        Array of predicted prices for next `horizon` steps
    """
    if lags is None:
        lags = [1, 5, 10]
    if windows is None:
        windows = [5, 20]
    
    # Build features from the last known prices using _build_features_from_prices
    prices = list(close_history[-50:].copy())
    features = _build_features_from_prices(np.array(prices), lags=lags, windows=windows)
    
    # Check if feature_scaler expects different number of features
    # If so, we need to use the actual DataFrame features instead
    expected_n_features = feature_scaler.n_features_in_ if hasattr(feature_scaler, 'n_features_in_') else len(features)
    
    if len(features) != expected_n_features:
        logger.warning(f"Feature mismatch: generated {len(features)}, scaler expects {expected_n_features}")
        logger.warning("Falling back to direct multi-output prediction from model")
        # Cannot generate recursive - return last price repeated
        return np.full(horizon, close_history[-1])
    
    # Scale features
    try:
        features_scaled = feature_scaler.transform(features.reshape(1, -1))
    except ValueError as e:
        logger.warning(f"Feature scaling failed: {e}")
        return np.full(horizon, close_history[-1])
    
    # Check for LSTM config - either from parameter or model attribute
    if lstm_config is None and model_type == 'lstm':
        if hasattr(model, '_lstm_timesteps'):
            lstm_config = {
                'timesteps': model._lstm_timesteps,
                'features_per_step': model._lstm_features_per_step
            }
    
    # Make prediction based on model type
    try:
        if model_type == 'lstm':
            # Handle enhanced LSTM with multi-timestep input
            if lstm_config is not None:
                timesteps = lstm_config['timesteps']
                features_per_step = lstm_config['features_per_step']
                total_needed = timesteps * features_per_step
                n_features = features_scaled.shape[1]
                
                # Pad or truncate features to match expected input
                if n_features < total_needed:
                    features_padded = np.pad(features_scaled, ((0, 0), (0, total_needed - n_features)), mode='constant')
                else:
                    features_padded = features_scaled[:, :total_needed]
                
                features_3d = features_padded.reshape(1, timesteps, features_per_step)
            else:
                # Fallback for simple LSTM
                features_3d = features_scaled.reshape(1, 1, -1)
            pred_scaled = model.predict(features_3d, verbose=0)
        elif model_type in ['keras', 'dnn']:
            pred_scaled = model.predict(features_scaled, verbose=0)
        else:
            pred_scaled = model.predict(features_scaled)
        
        # Flatten predictions
        pred_scaled = np.array(pred_scaled).flatten()
        
        # Check if multi-output (horizon predictions) or single output
        if len(pred_scaled) >= horizon:
            # Multi-output model - all horizon steps predicted at once
            pred_scaled = pred_scaled[:horizon]
            
            # Inverse transform - handle multi-output y_scaler
            if hasattr(y_scaler, 'n_features_in_') and y_scaler.n_features_in_ == horizon:
                # y_scaler is for multi-output
                pred_original = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            else:
                # y_scaler is for single output - apply to each step
                pred_original = np.array([
                    y_scaler.inverse_transform([[p]])[0, 0] for p in pred_scaled
                ])
            
            return pred_original
        
        else:
            # Single output - need to do true recursive
            predictions = []
            prices_extended = list(close_history[-50:].copy())
            
            for step in range(horizon):
                features_step = _build_features_from_prices(np.array(prices_extended), lags=lags, windows=windows)
                features_step_scaled = feature_scaler.transform(features_step.reshape(1, -1))
                
                if model_type == 'lstm':
                    pred = model.predict(features_step_scaled.reshape(1, 1, -1), verbose=0).flatten()[0]
                elif model_type in ['keras', 'dnn']:
                    pred = model.predict(features_step_scaled, verbose=0).flatten()[0]
                else:
                    pred = model.predict(features_step_scaled).flatten()[0]
                
                # Inverse transform single value
                pred_original = y_scaler.inverse_transform([[pred]])[0, 0]
                
                # Sanity check
                last_price = prices_extended[-1]
                if pred_original < 0:
                    pred_original = last_price * 0.5
                elif pred_original > last_price * 2:
                    pred_original = last_price * 1.2
                
                predictions.append(pred_original)
                prices_extended.append(pred_original)
            
            return np.array(predictions)
    
    except Exception as e:
        logger.warning(f"Recursive forecast failed: {e}")
        return np.full(horizon, close_history[-1])


def _build_features_from_prices(prices: np.ndarray, 
                                 lags: List[int] = None,
                                 windows: List[int] = None) -> np.ndarray:
    """
    Build feature vector from price history (same structure as training).
    
    All features are LAGGED by 1 day to prevent leakage.
    
    Features (16 total for lags=[1,5,10], windows=[5,20]):
    1. Close_lag_1, Close_lag_5, Close_lag_10
    2. Close_return_1, Close_return_5, Close_return_10
    3. Close_log_return
    4. Close_rolling_5_mean, Close_rolling_5_std, Close_ma_ratio_5
    5. Close_rolling_20_mean, Close_rolling_20_std, Close_ma_ratio_20
    6. Close_ROC_10, Close_momentum_10, Close_acceleration
    
    Args:
        prices: Array of price history (most recent is last)
        lags: Lag periods
        windows: Rolling window sizes
    
    Returns:
        Feature vector (1D array)
    """
    if lags is None:
        lags = [1, 5, 10]
    if windows is None:
        windows = [5, 20]
    
    features = []
    
    ref_price = prices[-2] if len(prices) > 1 else prices[-1]  # t-1 price
    
    # 1. LAG FEATURES
    for lag in lags:
        if len(prices) > lag:
            features.append(prices[-1 - lag])
        else:
            features.append(prices[0])
    
    # 2. RETURN FEATURES (lagged by 1)
    for lag in [1, 5, 10]:
        idx_start = -1 - lag - 1
        idx_end = -2
        if len(prices) > lag + 1 and prices[idx_start] != 0:
            ret = (prices[idx_end] - prices[idx_start]) / prices[idx_start]
        else:
            ret = 0.0
        features.append(ret)
    
    # 3. LOG RETURN (lagged by 1)
    if len(prices) > 2 and prices[-3] > 0:
        log_ret = np.log(prices[-2] / prices[-3])
    else:
        log_ret = 0.0
    features.append(log_ret)
    
    # 4. ROLLING FEATURES (lagged by 1)
    for window in windows:
        start_idx = max(0, len(prices) - 1 - window)
        window_prices = prices[start_idx:-1] if len(prices) > 1 else prices
        
        rolling_mean = np.mean(window_prices) if len(window_prices) > 0 else ref_price
        features.append(rolling_mean)
        
        rolling_std = np.std(window_prices) if len(window_prices) > 1 else 0.0
        features.append(rolling_std)
        
        if rolling_mean != 0:
            ma_ratio = (ref_price - rolling_mean) / rolling_mean
        else:
            ma_ratio = 0.0
        features.append(ma_ratio)
    
    # 5. TECHNICAL INDICATORS (lagged by 1)
    if len(prices) > 12 and prices[-13] != 0:
        roc = ((prices[-3] - prices[-13]) / prices[-13]) * 100
    else:
        roc = 0.0
    features.append(roc)
    
    if len(prices) > 12:
        momentum = prices[-3] - prices[-13]
    else:
        momentum = 0.0
    features.append(momentum)
    
    if len(prices) > 3:
        ret_t1 = (prices[-2] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
        ret_t2 = (prices[-3] - prices[-4]) / prices[-4] if prices[-4] != 0 else 0
        acceleration = ret_t1 - ret_t2
    else:
        acceleration = 0.0
    features.append(acceleration)
    
    return np.array(features)


def postprocess_forecasts(forecasts: np.ndarray, 
                          last_price: float, 
                          max_change_pct: float = 15.0) -> np.ndarray:
    """
    Postprocess predictions to ensure they're realistic.
    
    Applies reasonable bounds to predictions to prevent extreme values.
    
    Args:
        forecasts: Array of predicted prices
        last_price: Last observed price
        max_change_pct: Maximum allowed daily change (%)
    
    Returns:
        Array of adjusted forecasts
    """
    if not isinstance(forecasts, np.ndarray):
        forecasts = np.array(forecasts)
    
    adjusted = []
    current = last_price
    max_change = last_price * (max_change_pct / 100)
    
    for pred in forecasts.flatten():
        if np.isnan(pred) or np.isinf(pred):
            adjusted.append(current)
        else:
            # Limit change
            adjusted_pred = np.clip(pred, current - max_change, current + max_change)
            adjusted_pred = max(adjusted_pred, last_price * 0.5)  # Floor
            adjusted.append(float(adjusted_pred))
            current = adjusted_pred
    
    return np.array(adjusted)


# ============================================================================
# TREND CLASSIFICATION FOR MULTI-STEP
# ============================================================================

def create_multistep_trend_labels(close_series: np.ndarray, 
                                   horizon: int = 10,
                                   threshold: float = 0.02) -> np.ndarray:
    """
    Create trend labels based on price at t+horizon vs t.
    
    Label definition:
    - UP (2): Close_{t+horizon} > Close_t * (1 + threshold)
    - DOWN (0): Close_{t+horizon} < Close_t * (1 - threshold)
    - SIDEWAYS (1): Otherwise
    
    Args:
        close_series: Array of Close prices
        horizon: Steps ahead to evaluate trend
        threshold: Percent change threshold for UP/DOWN (default: 2%)
    
    Returns:
        Array of trend labels (last `horizon` entries are SIDEWAYS as placeholder)
    """
    n = len(close_series)
    labels = np.ones(n, dtype=np.int32)  # Default SIDEWAYS
    
    for t in range(n - horizon):
        current = close_series[t]
        future = close_series[t + horizon]
        
        if current == 0:
            continue
        
        pct_change = (future - current) / current
        
        if pct_change > threshold:
            labels[t] = 2  # UP
        elif pct_change < -threshold:
            labels[t] = 0  # DOWN
        # else: SIDEWAYS (already set)
    
    return labels


# ============================================================================
# ROLLING HOLDOUT FUNCTIONS (NEW)
# ============================================================================

def roll_multistep_predictions(
    model,
    X_holdout: np.ndarray,
    y_scaler,
    horizon: int = 10,
    stride: int = 1,
    model_type: str = 'sklearn',
    lstm_config: dict = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate rolling multi-step predictions over holdout period.
    
    For each origin t in holdout (with stride), predicts [t+1, ..., t+horizon].
    Returns a matrix of predictions.
    
    Args:
        model: Trained multi-output model
        X_holdout: Features for holdout period (n_samples, n_features)
        y_scaler: Fitted scaler for inverse transform of predictions
        horizon: Forecast horizon (default: 10)
        stride: Step size between origins (default: 1)
        model_type: 'sklearn', 'keras', or 'lstm'
        lstm_config: Optional dict with 'timesteps' and 'features_per_step' for enhanced LSTM
    
    Returns:
        pred_matrix: Shape (n_origins, horizon) - predictions in original scale
        origin_indices: Indices of origin points in holdout
    
    NO DATA LEAKAGE: Each prediction uses only features available at origin t.
    """
    n_samples = len(X_holdout)
    n_features = X_holdout.shape[1]
    
    # Calculate number of valid origins (must have room for full horizon)
    max_origin = n_samples - horizon
    if max_origin <= 0:
        logger.warning(f"Holdout too short ({n_samples}) for horizon ({horizon})")
        return np.array([]), np.array([])
    
    origin_indices = np.arange(0, max_origin, stride)
    n_origins = len(origin_indices)
    
    pred_matrix = np.zeros((n_origins, horizon), dtype=np.float64)
    
    # Check for LSTM config - either from parameter or model attribute
    if lstm_config is None:
        lstm_config = getattr(model, '_lstm_timesteps', None)
        if lstm_config is not None:
            # Build config from model attributes
            lstm_config = {
                'timesteps': model._lstm_timesteps,
                'features_per_step': model._lstm_features_per_step
            }
    
    for i, origin_idx in enumerate(origin_indices):
        X_origin = X_holdout[origin_idx:origin_idx+1]  # Shape (1, n_features)
        
        try:
            if model_type == 'lstm':
                # Handle enhanced LSTM with multi-timestep input
                if lstm_config is not None:
                    timesteps = lstm_config['timesteps']
                    features_per_step = lstm_config['features_per_step']
                    total_needed = timesteps * features_per_step
                    
                    # Pad or truncate features to match expected input
                    if n_features < total_needed:
                        X_padded = np.pad(X_origin, ((0, 0), (0, total_needed - n_features)), mode='constant')
                    else:
                        X_padded = X_origin[:, :total_needed]
                    
                    X_3d = X_padded.reshape((1, timesteps, features_per_step))
                else:
                    # Fallback for simple LSTM: single timestep
                    X_3d = X_origin.reshape((1, 1, -1))
                    
                pred_scaled = model.predict(X_3d, verbose=0)
            
            elif model_type == 'xlstm':
                # Handle xLSTM with PyTorch
                import torch
                # Get xlstm_config from lstm_config parameter (reused)
                if lstm_config is not None:
                    timesteps = lstm_config.get('timesteps', 4)
                    features_per_step = lstm_config.get('features_per_step', n_features)
                    total_needed = timesteps * features_per_step
                    
                    if n_features < total_needed:
                        X_padded = np.pad(X_origin, ((0, 0), (0, total_needed - n_features)), mode='constant')
                    else:
                        X_padded = X_origin[:, :total_needed]
                    
                    X_3d = X_padded.reshape((1, timesteps, features_per_step))
                else:
                    X_3d = X_origin.reshape((1, 1, -1))
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_3d).to(device)
                    pred_scaled = model(X_t).cpu().numpy()
                
            elif model_type in ['keras', 'dnn']:
                pred_scaled = model.predict(X_origin, verbose=0)
            elif model_type in ['tcn', 'nbeats', 'moe', 'multitask']:
                # PyTorch Model Zoo models
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_origin).to(device)
                    if model_type == 'multitask':
                        pred_scaled = model(X_t)[0].cpu().numpy()  # First output is regression
                    else:
                        pred_scaled = model(X_t).cpu().numpy()
            else:
                # sklearn models (linear, krr, lightgbm)
                pred_scaled = model.predict(X_origin)
            
            # Inverse transform to original scale
            if y_scaler is not None:
                pred_original = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))
            else:
                pred_original = pred_scaled
            
            pred_matrix[i, :] = pred_original.flatten()[:horizon]
            
        except Exception as e:
            logger.warning(f"Prediction failed at origin {origin_idx}: {e}")
            pred_matrix[i, :] = np.nan
    
    logger.debug(f"Rolling predictions: {n_origins} origins, {horizon} steps, stride={stride}")
    
    return pred_matrix, origin_indices


def aggregate_rolling_predictions(
    pred_matrix: np.ndarray,
    origin_indices: np.ndarray,
    holdout_length: int,
    horizon: int = 10,
    method: str = 'mean'
) -> Dict[str, np.ndarray]:
    """
    Aggregate rolling multi-step predictions to daily forecasts.
    
    For each date d in holdout, collects all predictions that target d,
    then aggregates (mean or median).
    
    Args:
        pred_matrix: Shape (n_origins, horizon) - predictions per origin
        origin_indices: Indices of origin points
        holdout_length: Total length of holdout period
        horizon: Forecast horizon
        method: 'mean' or 'median'
    
    Returns:
        Dictionary with:
            - 'pred_agg': Aggregated prediction per holdout date
            - 'pred_std': Standard deviation per date (uncertainty)
            - 'counts': Number of predictions contributing to each date
            - 'target_indices': Valid holdout date indices
    
    Example:
        If origin t=0 predicts [t+1..t+10], then date d=5 receives contributions
        from origins t=0 (step 5), t=1 (step 4), t=2 (step 3), etc.
    """
    n_origins = len(origin_indices)
    
    # Build mapping: for each target date, collect all predictions
    date_predictions = {d: [] for d in range(holdout_length)}
    
    for i, origin_idx in enumerate(origin_indices):
        for step in range(horizon):
            target_date = origin_idx + step + 1  # +1 because step 0 = t+1
            if target_date < holdout_length:
                pred_val = pred_matrix[i, step]
                if not np.isnan(pred_val):
                    date_predictions[target_date].append(pred_val)
    
    # Aggregate
    pred_agg = np.full(holdout_length, np.nan)
    pred_std = np.full(holdout_length, np.nan)
    counts = np.zeros(holdout_length, dtype=int)
    
    for d in range(holdout_length):
        preds = date_predictions[d]
        counts[d] = len(preds)
        if len(preds) > 0:
            if method == 'median':
                pred_agg[d] = np.median(preds)
            else:
                pred_agg[d] = np.mean(preds)
            pred_std[d] = np.std(preds) if len(preds) > 1 else 0.0
    
    # Find valid dates (those with at least one prediction)
    valid_mask = counts > 0
    target_indices = np.where(valid_mask)[0]
    
    logger.debug(f"Aggregated {holdout_length} dates, {np.sum(counts)} total contributions")
    
    return {
        'pred_agg': pred_agg,
        'pred_std': pred_std,
        'counts': counts,
        'target_indices': target_indices,
        'valid_mask': valid_mask
    }


def compute_rolling_baselines(
    close_train: np.ndarray,
    close_holdout: np.ndarray,
    horizon: int = 10,
    stride: int = 1,
    agg_method: str = 'mean'
) -> Dict[str, Dict]:
    """
    Compute rolling multi-step baselines for comparison.
    
    Implements:
    - Naive: Last known price for all future steps
    - Drift: Linear extrapolation from last known prices
    - Rolling Mean: Use rolling mean as forecast
    
    Returns aggregated predictions same format as model predictions.
    """
    holdout_length = len(close_holdout)
    max_origin = holdout_length - horizon
    
    if max_origin <= 0:
        return {}
    
    origin_indices = np.arange(0, max_origin, stride)
    n_origins = len(origin_indices)
    
    # Naive baseline: last known price
    naive_matrix = np.zeros((n_origins, horizon))
    for i, origin_idx in enumerate(origin_indices):
        last_price = close_holdout[origin_idx]
        naive_matrix[i, :] = last_price
    
    naive_agg = aggregate_rolling_predictions(
        naive_matrix, origin_indices, holdout_length, horizon, agg_method
    )
    
    # Drift baseline: linear trend
    drift_matrix = np.zeros((n_origins, horizon))
    for i, origin_idx in enumerate(origin_indices):
        # Use last 5 points to estimate drift
        lookback = min(5, origin_idx + 1)
        if lookback > 1:
            recent = close_holdout[origin_idx - lookback + 1:origin_idx + 1]
            drift = (recent[-1] - recent[0]) / (lookback - 1)
        else:
            drift = 0
        last_price = close_holdout[origin_idx]
        for step in range(horizon):
            drift_matrix[i, step] = last_price + drift * (step + 1)
    
    drift_agg = aggregate_rolling_predictions(
        drift_matrix, origin_indices, holdout_length, horizon, agg_method
    )
    
    # Rolling mean baseline
    rolling_matrix = np.zeros((n_origins, horizon))
    window = 20
    for i, origin_idx in enumerate(origin_indices):
        lookback = min(window, origin_idx + 1)
        rolling_mean = np.mean(close_holdout[max(0, origin_idx - lookback + 1):origin_idx + 1])
        rolling_matrix[i, :] = rolling_mean
    
    rolling_agg = aggregate_rolling_predictions(
        rolling_matrix, origin_indices, holdout_length, horizon, agg_method
    )
    
    return {
        'naive': {
            'pred_matrix': naive_matrix,
            'aggregated': naive_agg,
            'description': 'Last known price for all steps'
        },
        'drift': {
            'pred_matrix': drift_matrix,
            'aggregated': drift_agg,
            'description': 'Linear trend extrapolation'
        },
        'rolling_mean_20': {
            'pred_matrix': rolling_matrix,
            'aggregated': rolling_agg,
            'description': '20-day rolling mean'
        }
    }


def evaluate_rolling_forecast(
    actual: np.ndarray,
    aggregated: Dict,
    baseline_agg: Dict = None
) -> Dict:
    """
    Evaluate aggregated rolling forecast against actual values.
    
    Args:
        actual: Actual close prices for holdout period
        aggregated: Output from aggregate_rolling_predictions()
        baseline_agg: Optional baseline for comparison
    
    Returns:
        Dictionary with RMSE, MAE, R², and comparison metrics
    """
    pred_agg = aggregated['pred_agg']
    valid_mask = aggregated['valid_mask']
    
    # Only evaluate where we have predictions
    y_true = actual[valid_mask]
    y_pred = pred_agg[valid_mask]
    
    # Remove any remaining NaN
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) < 2:
        return {'error': 'Insufficient valid predictions'}
    
    metrics = {
        'rmse_agg': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae_agg': float(mean_absolute_error(y_true, y_pred)),
        'r2_agg': float(r2_score(y_true, y_pred)),
        'n_samples': int(len(y_true)),
        'avg_contributions': float(np.mean(aggregated['counts'][valid_mask]))
    }
    
    # Compare to baseline if provided
    if baseline_agg is not None:
        baseline_pred = baseline_agg['pred_agg'][valid_mask][valid]
        baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
        metrics['baseline_rmse'] = baseline_rmse
        metrics['beats_baseline'] = metrics['rmse_agg'] < baseline_rmse
        metrics['improvement_pct'] = float((baseline_rmse - metrics['rmse_agg']) / baseline_rmse * 100)
    
    return metrics


def create_10day_trend_labels(
    close_series: np.ndarray,
    threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create trend labels based on 10-day ahead price change.
    
    Args:
        close_series: Close prices
        threshold: Percentage threshold for UP/DOWN (default: 2%)
    
    Returns:
        labels: 0=DOWN, 1=SIDEWAYS, 2=UP
        pct_changes: Actual percentage changes (for regression target)
    """
    n = len(close_series)
    horizon = 10
    
    if n <= horizon:
        return np.array([]), np.array([])
    
    n_valid = n - horizon
    labels = np.ones(n_valid, dtype=int)  # Default: SIDEWAYS
    pct_changes = np.zeros(n_valid, dtype=np.float64)
    
    for t in range(n_valid):
        current = close_series[t]
        future = close_series[t + horizon]
        pct_change = (future - current) / current
        pct_changes[t] = pct_change
        
        if pct_change >= threshold:
            labels[t] = 2  # UP
        elif pct_change <= -threshold:
            labels[t] = 0  # DOWN
        # else: SIDEWAYS (already set)
    
    return labels, pct_changes


# ============================================================================
# TARGET TRANSFORM FUNCTIONS
# ============================================================================

# Valid target transforms
VALID_TARGET_TRANSFORMS = ['price', 'pct_change', 'log_return']


def get_target_transform(config: dict) -> str:
    """
    Determine target transform from config with backward compatibility.
    
    Priority:
    1. data_science.target_transform (if set)
    2. target_transform at top level (if config IS the data_science section)
    3. data_science.use_percentage_returns (legacy, maps to pct_change/price)
    4. Default: 'price'
    
    Args:
        config: Configuration dictionary (can be full config or just data_science section)
    
    Returns:
        Target transform: 'price', 'pct_change', or 'log_return'
    """
    # Handle both cases: full config with 'data_science' key, or just the data_science section
    if 'data_science' in config:
        ds_config = config.get('data_science', {})
    else:
        # Config is already the data_science section
        ds_config = config
    
    target_transform = ds_config.get('target_transform', None)
    use_pct = ds_config.get('use_percentage_returns', None)
    
    # Check for conflicts
    if target_transform is not None and use_pct is not None:
        logger.warning("Both 'target_transform' and 'use_percentage_returns' set. "
                      f"Using target_transform='{target_transform}' (ignoring use_percentage_returns)")
    
    if target_transform is not None:
        if target_transform not in VALID_TARGET_TRANSFORMS:
            raise ValueError(f"Invalid target_transform '{target_transform}'. "
                           f"Must be one of: {VALID_TARGET_TRANSFORMS}")
        return target_transform
    
    # Legacy fallback
    if use_pct is True:
        return 'pct_change'
    
    return 'price'


def to_target(y_prices: np.ndarray, transform: str, 
              context_prices: np.ndarray = None) -> np.ndarray:
    """
    Transform price targets to specified target space.
    
    Args:
        y_prices: Price targets, shape (n_samples,) or (n_samples, horizon)
        transform: 'price', 'pct_change', or 'log_return'
        context_prices: For pct_change/log_return, the reference prices (t=0).
                       Shape (n_samples,) - the last known price for each sample.
                       Required for pct_change and log_return transforms.
    
    Returns:
        Transformed targets, same shape as y_prices
    
    Math:
        pct_change:  y_transformed = (y_price - context) / context
        log_return:  y_transformed = log(y_price / context)
    """
    if transform == 'price':
        return y_prices.copy()
    
    if context_prices is None:
        raise ValueError(f"context_prices required for transform='{transform}'")
    
    # Handle multi-step targets (n_samples, horizon)
    is_multistep = y_prices.ndim == 2
    
    # Avoid division by zero
    safe_context = context_prices.copy()
    safe_context[safe_context == 0] = 1e-8
    
    if is_multistep:
        # Broadcast context to (n_samples, 1) for division
        safe_context = safe_context[:, None]
    
    if transform == 'pct_change':
        return (y_prices - safe_context) / safe_context
    
    elif transform == 'log_return':
        # Avoid log of non-positive
        y_safe = np.maximum(y_prices, 1e-8)
        return np.log(y_safe / safe_context)
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


def from_target(y_transformed: np.ndarray, transform: str,
                context_prices: np.ndarray) -> np.ndarray:
    """
    Inverse transform targets back to price space.
    
    Args:
        y_transformed: Transformed targets, shape (n_samples,) or (n_samples, horizon)
        transform: 'price', 'pct_change', or 'log_return'
        context_prices: Reference prices for reconstruction.
                       Shape (n_samples,) - the last known price for each sample.
    
    Returns:
        Price targets, same shape as y_transformed
    
    Math:
        pct_change:  y_price = context * (1 + y_transformed)
        log_return:  y_price = context * exp(y_transformed)
    
    For multi-step horizons:
        pct_change[step k]: price[k] = context * (1 + pct_change[k])
            Each step is relative to the SAME context (last known price)
        log_return[step k]: price[k] = context * exp(log_return[k])
            Each step is relative to the SAME context (last known price)
    
    NOTE: For cumulative/chained returns (where each step is relative to 
          the previous step), use from_target_cumulative() instead.
    """
    if transform == 'price':
        return y_transformed.copy()
    
    # Handle multi-step targets
    is_multistep = y_transformed.ndim == 2
    
    if is_multistep:
        context = context_prices[:, None]  # (n_samples, 1)
    else:
        context = context_prices
    
    if transform == 'pct_change':
        return context * (1 + y_transformed)
    
    elif transform == 'log_return':
        return context * np.exp(y_transformed)
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


def from_target_cumulative(y_transformed: np.ndarray, transform: str,
                           context_prices: np.ndarray) -> np.ndarray:
    """
    Inverse transform cumulative/chained targets to price path.
    
    Use this when each step's return is relative to the PREVIOUS step's price
    (e.g., for recursive forecasting).
    
    Args:
        y_transformed: Transformed targets, shape (n_samples, horizon)
                      Each y[i, k] is the return from step k-1 to step k
        transform: 'price', 'pct_change', or 'log_return'
        context_prices: Last known prices, shape (n_samples,)
    
    Returns:
        Price paths, shape (n_samples, horizon)
    
    Math:
        pct_change:  price[k] = price[k-1] * (1 + return[k])
                     => price[k] = context * prod_{j=1}^{k}(1 + return[j])
        log_return:  price[k] = context * exp(sum_{j=1}^{k} log_return[j])
    """
    if transform == 'price':
        return y_transformed.copy()
    
    n_samples, horizon = y_transformed.shape
    prices = np.zeros_like(y_transformed)
    
    if transform == 'pct_change':
        # Cumulative product of (1 + return)
        cum_factor = np.cumprod(1 + y_transformed, axis=1)
        prices = context_prices[:, None] * cum_factor
    
    elif transform == 'log_return':
        # Cumulative sum of log returns, then exp
        cum_log = np.cumsum(y_transformed, axis=1)
        prices = context_prices[:, None] * np.exp(cum_log)
    
    else:
        raise ValueError(f"Unknown transform: {transform}")
    
    return prices


def evaluate_in_both_spaces(y_true_prices: np.ndarray, 
                            y_pred_transformed: np.ndarray,
                            transform: str,
                            context_prices: np.ndarray,
                            y_true_transformed: np.ndarray = None) -> dict:
    """
    Evaluate predictions in both transformed space and price space.
    
    Args:
        y_true_prices: True prices, shape (n_samples, horizon)
        y_pred_transformed: Predicted values in transformed space
        transform: Target transform used
        context_prices: Last known prices for reconstruction
        y_true_transformed: True values in transformed space (optional, computed if not provided)
    
    Returns:
        Dict with:
            - transformed_space: metrics in training space (return/log-return)
            - price_space: metrics in price space (after inverse transform)
    """
    results = {'transform': transform}
    
    # Compute true transformed if not provided
    if y_true_transformed is None and transform != 'price':
        y_true_transformed = to_target(y_true_prices, transform, context_prices)
    
    # Metrics in transformed space
    if transform != 'price' and y_true_transformed is not None:
        results['transformed_space'] = evaluate_multistep(y_true_transformed, y_pred_transformed)
    
    # Reconstruct prices from predictions
    y_pred_prices = from_target(y_pred_transformed, transform, context_prices)
    
    # Metrics in price space
    results['price_space'] = evaluate_multistep(y_true_prices, y_pred_prices, context_prices)
    results['y_pred_prices'] = y_pred_prices
    
    return results


# ============================================================================
# RECURSIVE HOLDOUT FORECAST (TRUE AUTOREGRESSIVE)
# ============================================================================

def recursive_holdout_forecast(
    model,
    X_start: np.ndarray,
    y_scaler,
    holdout_length: int,
    model_horizon: int = 5,
    model_type: str = 'sklearn',
    lstm_config: dict = None,
    target_transform: str = 'price',
    last_known_price: float = None
) -> Dict[str, np.ndarray]:
    """
    Generate TRUE RECURSIVE forecasts over the entire holdout period.
    
    This simulates a real-world scenario where:
    1. Model predicts its native horizon (e.g., 5 steps)
    2. Predictions are fed back to predict further into the future
    3. Process continues until entire holdout period is covered
    
    Unlike rolling predictions (which have access to true features at each origin),
    this method only uses the initial feature vector and its own predictions.
    
    Args:
        model: Trained multi-output model
        X_start: Initial feature vector at holdout start, shape (1, n_features)
        y_scaler: Fitted scaler for inverse transform of predictions
        holdout_length: Total number of periods to forecast
        model_horizon: Model's native forecast horizon (default: 5)
        model_type: 'sklearn', 'keras', 'lstm', 'xlstm', 'tcn', 'nbeats', 'moe'
        lstm_config: Optional dict with 'timesteps' and 'features_per_step' for LSTM
        target_transform: 'price', 'pct_change', or 'log_return'
        last_known_price: Last known price before holdout (required for transforms)
    
    Returns:
        Dictionary with:
            - 'predictions': Full recursive forecast, shape (holdout_length,)
            - 'predictions_transformed': Predictions in training space
            - 'chunk_boundaries': Indices where new predictions were made
            - 'method': 'RECURSIVE_AUTOREGRESSIVE'
    
    NO DATA LEAKAGE: All predictions use only initial features + prior predictions.
    """
    import torch
    
    n_features = X_start.shape[1]
    predictions_transformed = []  # In training space (log_return, pct_change, etc.)
    chunk_boundaries = [0]
    
    # Track cumulative price for transforms
    current_price = last_known_price if last_known_price is not None else 1.0
    all_predicted_prices = [current_price]
    
    # Get lstm_config from model if not provided
    if lstm_config is None:
        lstm_config = getattr(model, '_lstm_timesteps', None)
        if lstm_config is not None:
            lstm_config = {
                'timesteps': model._lstm_timesteps,
                'features_per_step': model._lstm_features_per_step
            }
    
    # Calculate number of prediction chunks needed
    n_chunks = int(np.ceil(holdout_length / model_horizon))
    
    # Current feature vector (will be updated with predictions)
    X_current = X_start.copy()
    
    logger.info(f"  Recursive forecast: {holdout_length} periods, {n_chunks} chunks of {model_horizon}")
    
    for chunk_idx in range(n_chunks):
        steps_remaining = holdout_length - len(predictions_transformed)
        steps_this_chunk = min(model_horizon, steps_remaining)
        
        if steps_this_chunk <= 0:
            break
        
        try:
            # Make prediction based on model type
            if model_type == 'lstm':
                if lstm_config is not None:
                    timesteps = lstm_config['timesteps']
                    features_per_step = lstm_config['features_per_step']
                    total_needed = timesteps * features_per_step
                    
                    if n_features < total_needed:
                        X_padded = np.pad(X_current, ((0, 0), (0, total_needed - n_features)), mode='constant')
                    else:
                        X_padded = X_current[:, :total_needed]
                    
                    X_3d = X_padded.reshape((1, timesteps, features_per_step))
                else:
                    X_3d = X_current.reshape((1, 1, -1))
                pred_scaled = model.predict(X_3d, verbose=0)
                
            elif model_type == 'xlstm':
                if lstm_config is not None:
                    timesteps = lstm_config.get('timesteps', 4)
                    features_per_step = lstm_config.get('features_per_step', n_features)
                    total_needed = timesteps * features_per_step
                    
                    if n_features < total_needed:
                        X_padded = np.pad(X_current, ((0, 0), (0, total_needed - n_features)), mode='constant')
                    else:
                        X_padded = X_current[:, :total_needed]
                    
                    X_3d = X_padded.reshape((1, timesteps, features_per_step))
                else:
                    X_3d = X_current.reshape((1, 1, -1))
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_3d).to(device)
                    pred_scaled = model(X_t).cpu().numpy()
                    
            elif model_type in ['keras', 'dnn']:
                pred_scaled = model.predict(X_current, verbose=0)
                
            elif model_type in ['tcn', 'nbeats', 'moe', 'multitask']:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_current).to(device)
                    if model_type == 'multitask':
                        pred_scaled = model(X_t)[0].cpu().numpy()
                    else:
                        pred_scaled = model(X_t).cpu().numpy()
            else:
                # sklearn models
                pred_scaled = model.predict(X_current)
            
            # Inverse transform to target space
            if y_scaler is not None:
                pred_target = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            else:
                pred_target = pred_scaled.flatten()
            
            # Take only steps needed for this chunk
            pred_chunk = pred_target[:steps_this_chunk]
            predictions_transformed.extend(pred_chunk.tolist())
            
            # Convert to prices for next iteration
            if target_transform == 'log_return':
                # Each step: price_t+1 = price_t * exp(log_return_t+1)
                for lr in pred_chunk:
                    current_price = current_price * np.exp(lr)
                    all_predicted_prices.append(current_price)
            elif target_transform == 'pct_change':
                # Each step: price_t+1 = price_t * (1 + pct_change_t+1)
                for pc in pred_chunk:
                    current_price = current_price * (1 + pc)
                    all_predicted_prices.append(current_price)
            else:
                # Direct price predictions
                for p in pred_chunk:
                    current_price = p
                    all_predicted_prices.append(current_price)
            
            # Update chunk boundary
            chunk_boundaries.append(len(predictions_transformed))
            
            # Update feature vector with predictions for next chunk
            # Shift features and inject predicted returns/prices
            # This is a simplified update - in reality, we'd need to recompute
            # all features based on the new predicted prices
            if target_transform != 'price':
                # For log_return/pct_change: update lag features with predicted values
                n_lags = min(len(pred_chunk), 6)  # Assume up to 6 lag features
                if n_features > n_lags:
                    X_current = np.roll(X_current, -n_lags, axis=1)
                    X_current[0, -n_lags:] = pred_chunk[-n_lags:]
            
        except Exception as e:
            logger.warning(f"  Chunk {chunk_idx} failed: {e}")
            # Fill remaining with last prediction or NaN
            fill_value = predictions_transformed[-1] if predictions_transformed else 0.0
            predictions_transformed.extend([fill_value] * steps_this_chunk)
            chunk_boundaries.append(len(predictions_transformed))
    
    # Convert predictions_transformed to prices
    predictions_transformed = np.array(predictions_transformed[:holdout_length])
    predicted_prices = np.array(all_predicted_prices[1:holdout_length+1])  # Skip initial price
    
    logger.info(f"  Recursive forecast complete: {len(predicted_prices)} periods")
    
    return {
        'predictions': predicted_prices,
        'predictions_transformed': predictions_transformed,
        'chunk_boundaries': chunk_boundaries,
        'method': 'RECURSIVE_AUTOREGRESSIVE',
        'n_chunks': len(chunk_boundaries) - 1,
        'model_horizon': model_horizon
    }


# ============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# ============================================================================

def generate_realistic_forecasts(model, X_test_scaled, y_test_scaled, y_test_original, 
                               target_scaler, model_name, forecast_horizon=30):
    """
    Legacy function for backward compatibility.
    Use recursive_multistep_forecast instead.
    """
    import numpy as np
    
    forecasts = []
    
    if hasattr(model, 'predict'):
        try:
            last_price = float(y_test_original[-1])
            y_test_last_10 = y_test_original[-min(10, len(y_test_original)):]
            trend = (y_test_last_10[-1] - y_test_last_10[0]) / len(y_test_last_10) if len(y_test_last_10) > 1 else 0
            
            current_price = last_price
            for day in range(forecast_horizon):
                daily_change = trend + np.random.normal(0, current_price * 0.01)
                current_price = max(current_price + daily_change, current_price * 0.5)
                forecasts.append(float(current_price))
        except:
            forecasts = [float(y_test_original[-1])] * forecast_horizon
    
    return forecasts if forecasts else [float(y_test_original[-1])] * forecast_horizon
