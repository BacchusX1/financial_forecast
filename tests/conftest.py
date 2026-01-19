import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Create realistic OHLCV stub data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

@pytest.fixture
def valid_config():
    """Minimal valid config dictionary."""
    return {
        'data_loader': {'category': 'stock', 'candle_length': '1d'},
        'feature_engineering': {
            'indicators': {
                'enabled_groups': ['trend'],
                'params': {'sma': [5, 10]}
            },
            'leakage_guard': {'enforce_shift_1': True}
        },
        'data_science': {
            'classifier_models': ['random_forest'],
            'regressor_models': [],
            'test_size': 0.2
        },
        'reporting': {
            'required_outputs': ["benchmark_results.json"]
        }
    }
