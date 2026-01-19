"""
Tests for base price column restriction and feature scale filtering.

Tests:
1. test_base_price_column_strict - Features only from base_price_column
2. test_absolute_scale_filtering - Remove price-unit features
3. test_combined_with_target_transform - Smoke test with returns target
4. test_classify_feature_columns - Classification logic
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'modules', 'feature_engineering'))

from feature_scale_classifier import (
    classify_feature_columns,
    filter_features_by_scale,
    filter_columns_by_base_price,
    validate_price_column_restriction,
    PRICE_COLUMNS
)
from feature_engineering import FeatureEngineering


def create_sample_ohlcv_data(n_rows: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price series
    base_price = 100
    returns = np.random.randn(n_rows) * 0.02
    close = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Datetime': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'Open': close * (1 + np.random.randn(n_rows) * 0.005),
        'High': close * (1 + np.abs(np.random.randn(n_rows) * 0.01)),
        'Low': close * (1 - np.abs(np.random.randn(n_rows) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, n_rows)
    })
    return df


class TestClassifyFeatureColumns:
    """Test feature column classification."""
    
    def test_classify_raw_prices_as_absolute(self):
        """Raw price columns should be classified as absolute."""
        columns = ['Close', 'Open', 'High', 'Low']
        result = classify_feature_columns(columns)
        
        assert 'Close' in result['absolute_scale']
        assert 'Open' in result['absolute_scale']
        assert 'High' in result['absolute_scale']
        assert 'Low' in result['absolute_scale']
    
    def test_classify_price_lags_as_absolute(self):
        """Price lag columns should be classified as absolute."""
        columns = ['Close_lag_1', 'Close_lag_5', 'High_lag_10']
        result = classify_feature_columns(columns)
        
        assert all(c in result['absolute_scale'] for c in columns)
    
    def test_classify_rolling_mean_as_absolute(self):
        """Rolling mean of price should be classified as absolute."""
        columns = ['Close_rolling_20_mean', 'Close_rolling_5_min']
        result = classify_feature_columns(columns)
        
        assert 'Close_rolling_20_mean' in result['absolute_scale']
        assert 'Close_rolling_5_min' in result['absolute_scale']
    
    def test_classify_sma_ema_as_absolute(self):
        """SMA/EMA values should be classified as absolute."""
        columns = ['sma_20', 'ema_50', 'wma_10']
        result = classify_feature_columns(columns)
        
        assert all(c in result['absolute_scale'] for c in columns)
    
    def test_classify_returns_as_relative(self):
        """Return features should be classified as relative."""
        columns = ['pct_return_1', 'log_return', 'Close_return_5']
        result = classify_feature_columns(columns)
        
        assert all(c in result['relative_scale'] for c in columns)
    
    def test_classify_ratios_as_relative(self):
        """Ratio features should be classified as relative."""
        columns = ['sma_ratio', 'bb_position', 'close_ma_ratio_20']
        result = classify_feature_columns(columns)
        
        assert all(c in result['relative_scale'] for c in columns)
    
    def test_classify_oscillators_as_relative(self):
        """Oscillators should be classified as relative."""
        columns = ['rsi_14', 'stoch_k', 'cci_20', 'mfi_14']
        result = classify_feature_columns(columns)
        
        assert all(c in result['relative_scale'] for c in columns)
    
    def test_classify_volume_separately(self):
        """Volume features should be classified as volume."""
        columns = ['Volume', 'Volume_rolling_20_mean', 'obv']
        result = classify_feature_columns(columns)
        
        assert all(c in result['volume'] for c in columns)


class TestBasePriceColumnStrict:
    """Test that only base_price_column is used when strict mode enabled."""
    
    def test_no_open_high_low_features(self):
        """With allow_additional_price_columns=False, no Open/High/Low features."""
        df = create_sample_ohlcv_data(100)
        
        config = {
            'base_price_column': 'Close',
            'allow_additional_price_columns': False,
            'lag_features': [1, 5],
            'rolling_windows': [5, 10],
        }
        
        fe = FeatureEngineering(df, config)
        result = fe.transform()
        
        # Check no features start with Open_, High_, Low_
        feature_names = [c for c in result.columns if c not in ['Datetime', 'Ticker']]
        
        assert not any(c.startswith('Open_') for c in feature_names), \
            f"Found Open_ features: {[c for c in feature_names if c.startswith('Open_')]}"
        assert not any(c.startswith('High_') for c in feature_names), \
            f"Found High_ features: {[c for c in feature_names if c.startswith('High_')]}"
        assert not any(c.startswith('Low_') for c in feature_names), \
            f"Found Low_ features: {[c for c in feature_names if c.startswith('Low_')]}"
    
    def test_close_features_exist(self):
        """With base_price_column=Close, Close features should exist."""
        df = create_sample_ohlcv_data(100)
        
        config = {
            'base_price_column': 'Close',
            'allow_additional_price_columns': False,
            'lag_features': [1, 5],
            'rolling_windows': [5],
        }
        
        fe = FeatureEngineering(df, config)
        result = fe.transform()
        
        # Should have Close column
        assert 'Close' in result.columns
        
        # Volume is still allowed (not a price column)
        assert 'Volume' in result.columns
    
    def test_filter_columns_by_base_price(self):
        """Test the filter function directly."""
        columns = ['Close_lag_1', 'Open_lag_1', 'High_lag_1', 'Low_lag_1', 'Volume_lag_1']
        
        allowed, dropped = filter_columns_by_base_price(
            columns, 
            base_price_column='Close',
            allow_additional_price_columns=False
        )
        
        assert 'Close_lag_1' in allowed
        assert 'Volume_lag_1' in allowed  # Volume is not a price column
        assert 'Open_lag_1' in dropped
        assert 'High_lag_1' in dropped
        assert 'Low_lag_1' in dropped
    
    def test_backward_compat_allow_additional(self):
        """Default allow_additional_price_columns=True keeps all columns."""
        columns = ['Close_lag_1', 'Open_lag_1', 'High_lag_1', 'Volume_lag_1']
        
        allowed, dropped = filter_columns_by_base_price(
            columns,
            base_price_column='Close',
            allow_additional_price_columns=True
        )
        
        assert len(allowed) == 4
        assert len(dropped) == 0


class TestAbsoluteScaleFiltering:
    """Test that absolute-scale features are removed when disabled."""
    
    def test_no_price_lags_when_scale_free(self):
        """With allow_absolute_scale_features=False, no price lag features."""
        df = create_sample_ohlcv_data(100)
        
        config = {
            'allow_absolute_scale_features': False,
            'lag_features': [1, 5],
        }
        
        fe = FeatureEngineering(df, config)
        result = fe.transform()
        
        # Check no _lag_ columns for price columns
        feature_names = result.columns.tolist()
        
        # Price lags should be filtered out
        absolute_lags = [c for c in feature_names if '_lag_' in c and 
                        any(c.startswith(p + '_') for p in ['Close', 'Open', 'High', 'Low'])]
        
        assert len(absolute_lags) == 0, f"Found absolute lag features: {absolute_lags}"
    
    def test_no_rolling_mean_when_scale_free(self):
        """With allow_absolute_scale_features=False, no price rolling mean."""
        df = create_sample_ohlcv_data(100)
        
        config = {
            'allow_absolute_scale_features': False,
            'rolling_windows': [5, 10],
        }
        
        fe = FeatureEngineering(df, config)
        result = fe.transform()
        
        # Check no rolling mean columns for price columns
        feature_names = result.columns.tolist()
        
        absolute_rolling = [c for c in feature_names if '_rolling_' in c and '_mean' in c and
                           any(c.startswith(p + '_') for p in ['Close', 'Open', 'High', 'Low'])]
        
        assert len(absolute_rolling) == 0, f"Found absolute rolling features: {absolute_rolling}"
    
    def test_filter_features_by_scale_removes_absolute(self):
        """Test filter_features_by_scale removes absolute columns."""
        df = pd.DataFrame({
            'Datetime': pd.date_range('2020-01-01', periods=10),
            'Close': np.linspace(100, 110, 10),
            'Close_lag_5': np.linspace(95, 105, 10),  # Absolute
            'Close_rolling_20_mean': np.linspace(100, 110, 10),  # Absolute
            'sma_20': np.linspace(100, 110, 10),  # Absolute
            'pct_return_1': np.random.randn(10) * 0.01,  # Relative
            'rsi_14': np.random.rand(10) * 100,  # Relative
            'bb_position': np.random.rand(10),  # Relative
            'Volume': np.random.randint(1000, 5000, 10),  # Volume
        })
        
        result = filter_features_by_scale(df, allow_absolute=False, base_col='Close')
        
        # Absolute columns removed
        assert 'Close_lag_5' not in result.columns
        assert 'Close_rolling_20_mean' not in result.columns
        assert 'sma_20' not in result.columns
        
        # Relative columns kept
        assert 'pct_return_1' in result.columns
        assert 'rsi_14' in result.columns
        assert 'bb_position' in result.columns
        
        # Volume kept
        assert 'Volume' in result.columns
        
        # Base column kept
        assert 'Close' in result.columns
    
    def test_return_features_remain(self):
        """Return features should remain with scale-free mode."""
        columns = ['log_return', 'pct_return_1', 'Close_return_5']
        result = classify_feature_columns(columns)
        
        assert all(c in result['relative_scale'] for c in columns)


class TestCombinedWithTargetTransform:
    """Smoke test combining scale-free features with return targets."""
    
    def test_features_non_empty_scale_free(self):
        """With scale-free mode, feature set should still be non-empty."""
        df = create_sample_ohlcv_data(100)
        
        config = {
            'allow_absolute_scale_features': False,
            'base_price_column': 'Close',
            'allow_additional_price_columns': False,
        }
        
        fe = FeatureEngineering(df, config)
        result = fe.transform()
        
        # Should have at least some columns (Datetime, Close, Volume)
        assert len(result.columns) >= 3
        assert 'Close' in result.columns  # Target column preserved
    
    def test_classify_returns_empty_result(self):
        """Classification should return proper structure even for empty input."""
        result = classify_feature_columns([])
        
        assert 'absolute_scale' in result
        assert 'relative_scale' in result
        assert 'volume' in result
        assert 'other' in result
        assert all(len(v) == 0 for v in result.values())


class TestValidatePriceColumnRestriction:
    """Test indicator input validation."""
    
    def test_close_only_indicator_allowed(self):
        """Indicators using only Close should be allowed."""
        allowed, reason = validate_price_column_restriction(
            indicator_inputs=['Close'],
            base_price_column='Close',
            allow_additional=False
        )
        assert allowed
        assert reason == ""
    
    def test_high_low_indicator_blocked(self):
        """Indicators requiring High/Low should be blocked in strict mode."""
        allowed, reason = validate_price_column_restriction(
            indicator_inputs=['High', 'Low'],
            base_price_column='Close',
            allow_additional=False
        )
        assert not allowed
        assert 'requires' in reason
    
    def test_ohlcv_indicator_blocked(self):
        """Indicators requiring OHLCV should be blocked in strict mode."""
        allowed, reason = validate_price_column_restriction(
            indicator_inputs=['Open', 'High', 'Low', 'Close', 'Volume'],
            base_price_column='Close',
            allow_additional=False
        )
        assert not allowed
    
    def test_all_allowed_when_not_strict(self):
        """All indicators allowed when allow_additional=True."""
        allowed, reason = validate_price_column_restriction(
            indicator_inputs=['Open', 'High', 'Low', 'Close'],
            base_price_column='Close',
            allow_additional=True
        )
        assert allowed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
