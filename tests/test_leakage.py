import pytest
import pandas as pd
import numpy as np
import logging
from src.modules.feature_engineering.feature_engineering import FeatureEngineering
from src.modules.feature_engineering.indicator_engine import IndicatorEngine

# Helper for leakage detection
def check_pipeline_leakage(pipeline_func, data, target_indices=None):
    """
    Generic leakage checker.
    pipeline_func: function(DataFrame) -> DataFrame (features)
    """
    if target_indices is None:
        target_indices = [len(data) - 1] # Check last row by default

    # 1. Baseline Run
    df_base = pipeline_func(data.copy())
    
    leaks_found = []
    
    for t in target_indices:
        # Modify data at time t ONLY
        data_perturbed = data.copy()
        
        # Perturb Close price significantly (x10) or random noise
        # Use simple multiplication to ensure change
        data_perturbed.iloc[t, data.columns.get_loc('Close')] *= 10
        
        # Recalculate features
        df_perturbed = pipeline_func(data_perturbed)
        
        # features to check (exclude original OHLCV)
        features = [c for c in df_base.columns if c not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Date', 'Ticker', 'Datetime', 'Adj Close']]
        
        row_base = df_base.iloc[t][features]
        row_pert = df_perturbed.iloc[t][features]
        
        # Use pandas comparison which handles NaNs and floats gracefully
        try:
            # check differences
            # If any feature changed at time t, it leaked from t input to t output
            # We allow small float noise but *10 input change causes massive output change if unshifted
            pd.testing.assert_series_equal(row_base, row_pert, rtol=1e-5)
        except AssertionError as e:
            # identify which columns differ
            diff_mask = ~np.isclose(row_base.to_numpy(dtype=float, na_value=np.nan), 
                                  row_pert.to_numpy(dtype=float, na_value=np.nan), 
                                  rtol=1e-5, equal_nan=True)
            leaking_cols = np.array(features)[diff_mask].tolist()
            leaks_found.append({
                'index': t,
                'leaking_columns': leaking_cols
            })

    return leaks_found

def test_leakage_rolling_features(sample_data):
    """
    Audit finding verification:
    Rolling features must NOT change at time t if data at time t changes.
    Applies to all rolling functions (mean, std, median, skew, kurt, etc.).
    """
    config = {
        "target_columns": ["Close"],
        "rolling_windows": [3],
        "rolling_functions": ["mean", "std", "min", "max", "median", "skew", "kurt", "var"],
        "scaler": None 
    }

    def pipeline(df):
        fe = FeatureEngineering(df, config=config)
        # Force basic rolling generation
        return fe.create_rolling_features(df)

    # Use random data from fixture to avoid skew/kurt stability issues
    leaks = check_pipeline_leakage(pipeline, sample_data)
    assert not leaks, f"Leakage detected in rolling features: {leaks}"

def test_leakage_indicators(sample_data):
    """Ensure IndicatorEngine features are shifted and leak-free."""
    config = {
        'leakage_guard': {'enforce_shift_1': True},
        'allow_absolute_scale_features': True 
    }
    
    def pipeline(df):
        engine = IndicatorEngine(config)
        return engine.compute_indicators(df, enabled_groups=['trend', 'momentum', 'volatility'])

    leaks = check_pipeline_leakage(pipeline, sample_data)
    assert not leaks, f"Leakage detected in indicators: {leaks}"

def test_leakage_guard_strict_setting(sample_data, valid_config):
    """
    Verify 'enforce_shift_1' works via the main FeatureEngineering class.
    """
    fe_config = valid_config['feature_engineering']
    fe_config['leakage_guard'] = {'enforce_shift_1': True}
    fe_config['scaler'] = None # Disable scaling to isolate time-series leakage

    def pipeline(df):
        fe = FeatureEngineering(df, config=fe_config)
        return fe.transform(apply_scaling=False, apply_pca_flag=False)

    leaks = check_pipeline_leakage(pipeline, sample_data)
    assert not leaks, f"Leakage detected in full FE pipeline: {leaks}"
