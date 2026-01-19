"""
Test Feature Pruning Leakage Prevention

Verifies that pruning decisions are made using training data ONLY.
This test uses a "future-only correlation trap" to detect leakage.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.feature_engineering.feature_pruner import FeaturePruner


def test_pruner_fit_transform_api():
    """Basic API test: fit on train, transform any data."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'Close': np.random.randn(n) * 10 + 100,
        'feature_a': np.random.randn(n),
        'feature_b': np.random.randn(n),
        'constant_col': np.ones(n),  # Should be dropped
    })
    
    config = {'drop_constant': True}
    pruner = FeaturePruner(config)
    
    # Fit on first 80 rows (train)
    pruner.fit(df.iloc[:80])
    
    # Transform full data
    df_pruned = pruner.transform(df)
    
    assert 'constant_col' not in df_pruned.columns
    assert 'feature_a' in df_pruned.columns
    assert 'feature_b' in df_pruned.columns
    assert 'Close' in df_pruned.columns  # Reserved columns kept


def test_pruner_nan_threshold_train_only():
    """Verify NaN ratio is computed on training data only."""
    np.random.seed(42)
    n = 100
    train_end = 80
    
    # Feature with NaN only in test portion
    feature_test_nan = np.random.randn(n)
    feature_test_nan[train_end:] = np.nan  # 100% NaN in test, 0% in train
    
    # Feature with NaN only in train portion
    feature_train_nan = np.random.randn(n)
    feature_train_nan[:int(train_end * 0.5)] = np.nan  # 50% NaN in train
    
    df = pd.DataFrame({
        'Close': np.random.randn(n) * 10 + 100,
        'feature_test_nan': feature_test_nan,
        'feature_train_nan': feature_train_nan,
        'feature_ok': np.random.randn(n),
    })
    
    config = {'drop_high_nan_ratio': 0.3}  # Drop if >30% NaN
    pruner = FeaturePruner(config)
    
    # Fit on training data only (first 80 rows)
    pruner.fit(df.iloc[:train_end])
    
    # feature_test_nan has 0% NaN in train → should NOT be dropped
    # feature_train_nan has 50% NaN in train → should be dropped
    assert 'feature_test_nan' not in pruner.columns_to_drop, \
        "Pruner incorrectly used test data to compute NaN ratio"
    assert 'feature_train_nan' in pruner.columns_to_drop


def test_pruner_correlation_trap():
    """
    CRITICAL: Test "future-only correlation trap".
    
    Scenario:
    - Feature X is correlated with Feature Y ONLY in the test portion.
    - If pruner uses full data, it might drop X due to correlation.
    - If pruner correctly uses train only, X should NOT be dropped.
    """
    np.random.seed(42)
    n = 100
    train_end = 80
    
    # Create uncorrelated features in training
    feature_x = np.random.randn(n)
    feature_y = np.random.randn(n)
    
    # Make them HIGHLY correlated in test portion only
    feature_x[train_end:] = np.linspace(0, 10, n - train_end)
    feature_y[train_end:] = feature_x[train_end:] + 0.001 * np.random.randn(n - train_end)
    
    df = pd.DataFrame({
        'Close': np.random.randn(n) * 10 + 100,
        'feature_x': feature_x,
        'feature_y': feature_y,
        'feature_z': np.random.randn(n),
    })
    
    # Verify: full data has high correlation
    full_corr = df['feature_x'].corr(df['feature_y'])
    train_corr = df.iloc[:train_end]['feature_x'].corr(df.iloc[:train_end]['feature_y'])
    
    # Full data should show high correlation, train should show low
    assert abs(full_corr) > 0.5, f"Test setup issue: full_corr={full_corr:.3f}"
    assert abs(train_corr) < 0.5, f"Test setup issue: train_corr={train_corr:.3f}"
    
    config = {
        'correlation_filter': {
            'enabled': True,
            'method': 'pearson',
            'threshold': 0.8
        }
    }
    pruner = FeaturePruner(config)
    
    # Fit on training data only
    pruner.fit(df.iloc[:train_end])
    
    # Neither feature_x nor feature_y should be dropped (low corr in train)
    dropped = pruner.columns_to_drop
    assert 'feature_x' not in dropped and 'feature_y' not in dropped, \
        f"Pruner leaked test data! Dropped: {dropped}"


def test_pruner_decisions_stable_on_test_perturbation():
    """
    CRITICAL: Pruning decisions must NOT change if test data is perturbed.
    
    This is the ultimate leakage test:
    1. Fit pruner on train
    2. Perturb test data (should have no effect on decisions)
    3. Verify decisions are identical
    """
    np.random.seed(42)
    n = 100
    train_end = 80
    
    df = pd.DataFrame({
        'Close': np.random.randn(n) * 10 + 100,
        'feature_a': np.random.randn(n),
        'feature_b': np.random.randn(n),
        'constant_in_train': np.concatenate([np.ones(train_end), np.random.randn(n - train_end)]),
    })
    
    config = {
        'drop_constant': True,
        'drop_high_nan_ratio': 0.2,
        'correlation_filter': {'enabled': True, 'threshold': 0.95}
    }
    
    # Fit on original train
    pruner1 = FeaturePruner(config)
    pruner1.fit(df.iloc[:train_end])
    decisions1 = pruner1.columns_to_drop.copy()
    
    # Perturb test portion ONLY
    df_perturbed = df.copy()
    df_perturbed.iloc[train_end:, df.columns.get_loc('feature_a')] = 9999.0
    df_perturbed.iloc[train_end:, df.columns.get_loc('feature_b')] = np.nan
    
    # Fit on same train (test perturbation should not matter)
    pruner2 = FeaturePruner(config)
    pruner2.fit(df_perturbed.iloc[:train_end])  # Same train data
    decisions2 = pruner2.columns_to_drop.copy()
    
    # Decisions must be identical
    assert decisions1 == decisions2, \
        f"Pruning decisions changed when test was perturbed! d1={decisions1}, d2={decisions2}"


def test_pruner_artifact_save_load(tmp_path):
    """Test saving and loading pruning artifact."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Close': np.random.randn(50) * 10 + 100,
        'feature_a': np.random.randn(50),
        'constant': np.ones(50),
    })
    
    config = {'drop_constant': True}
    pruner = FeaturePruner(config)
    pruner.fit(df)
    
    # Save
    path = pruner.save_artifact(str(tmp_path))
    assert os.path.exists(path)
    
    # Load
    pruner_loaded = FeaturePruner.load_artifact(path)
    assert pruner_loaded.fitted
    assert 'constant' in pruner_loaded.columns_to_drop
    
    # Transform should work
    df_pruned = pruner_loaded.transform(df)
    assert 'constant' not in df_pruned.columns
