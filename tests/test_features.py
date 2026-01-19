import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'modules', 'feature_engineering'))

from feature_engineering import FeatureEngineering
from indicator_engine import IndicatorEngine

def test_transform_execution_smoke(valid_config, sample_data):
    """Basic smoke test ensuring the pipeline runs end-to-end without error."""
    fe = FeatureEngineering(sample_data.copy(), config=valid_config['feature_engineering'])
    res = fe.transform()
    assert not res.empty
    # Should have original columns + some derived ones
    assert res.shape[1] > sample_data.shape[1]
    assert 'Close' in res.columns

def test_scale_free_mode_indicators(sample_data):
    """Test that strict mode filters absolute price indicators (Scale-Free Mode)."""
    config = {
        'allow_absolute_scale_features': False,
        'feature_engineering': {
            'indicators': {
                'enabled_groups': ['trend', 'volatility']
            }
        }
    }
    
    # We can test IndicatorEngine directly for this logic
    engine = IndicatorEngine(config)
    df_out = engine.compute_indicators(sample_data.copy())
    columns = df_out.columns.tolist()
    
    # Absolute values should be missing
    for col in ['sma_20', 'ema_20', 'atr_14', 'macd_12_26_9', 'true_range']:
        assert col not in columns, f"Absolute indicator {col} found in scale-free mode"
        
    # Relative values should be present
    for col in ['sma_20_ratio', 'atr_normalized_14', 'bbands_width_20']:
         assert col in columns, f"Relative indicator {col} missing in scale-free mode"

def test_scale_free_mode_lags(sample_data):
    """Test that raw price lags are blocked in scale-free mode."""
    config = {
        'allow_absolute_scale_features': False,
        'lag_features': [1],
        'target_columns': ['Close', 'Volume', 'rsi_14'] 
    }
    
    df = sample_data.copy()
    df['rsi_14'] = np.random.random(len(df)) # Mock relative indicator
    
    fe = FeatureEngineering(df, config=config)
    df_out = fe.create_lag_features(df, lags=[1])
    
    columns = df_out.columns.tolist()
    
    # Absolute price lags blocked
    assert 'Close_lag_1' not in columns
    assert 'Volume_lag_1' not in columns
    # Relative lags allowed
    assert 'rsi_14_lag_1' in columns

def test_scale_free_mode_rolling(sample_data):
    """Test that raw price rolling features are blocked in scale-free mode."""
    config = {
        'allow_absolute_scale_features': False,
        'rolling_windows': [5],
        'rolling_functions': ['mean'],
        'target_columns': ['Close', 'rsi_14']
    }
    
    df = sample_data.copy()
    df['rsi_14'] = np.random.random(len(df))
    
    fe = FeatureEngineering(df, config=config)
    df_out = fe.create_rolling_features(df)
    columns = df_out.columns.tolist()
    
    assert 'Close_rolling_5_mean' not in columns
    assert 'rsi_14_rolling_5_mean' in columns

def test_close_only_features(sample_data):
    """Test expert mode Close-Only feature generation."""
    fe = FeatureEngineering(sample_data.copy())
    df_res = fe.create_close_only_features(sample_data.copy())
    
    assert 'Close_lag_1' in df_res.columns
    assert 'Close_return_1' in df_res.columns
    assert 'Close_rolling_5_mean' in df_res.columns
    # Ensure no future leakage in names (we trust the logic test in test_leakage)
