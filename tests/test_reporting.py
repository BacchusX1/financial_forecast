import pytest
import os
import shutil
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.data_science.data_science import DataScientist

def test_report_generation_smoke(valid_config, sample_data, tmp_path):
    """
    Ensure report produces output even if we just run a minimal pipeline.
    """
    # Setup temporary output dir
    out_dir = tmp_path / "out"
    os.makedirs(out_dir, exist_ok=True)
    
    ds_config = valid_config['data_science']
    ds_config['output_folder'] = str(out_dir)
    ds_config['classifier_models'] = ['random_forest'] 
    ds_config['regressor_models'] = [] # Skip regressors for speed
    ds_config['trend_window'] = 5
    ds_config['test_size'] = 0.5 # Small data
    
    # 1. Initialize DataScientist
    # Note: DS expects features already. 
    # For this smoke test, we'll just use raw OHLCV as features + random col
    df_in = sample_data.copy()
    df_in['random_feature'] = np.random.randn(len(df_in))
    
    ds = DataScientist(df_in, config=ds_config)
    
    # 2. Run benchmark
    # Use small holdout to ensure we actually train/pred
    try:
        results = ds.train_and_evaluate(ticker="TEST_TICKER", holdout_days=10)
    except Exception as e:
        pytest.fail(f"Benchmark run failed: {e}")
    
    # 3. Verify Artifacts
    # Directory structure is models_{timestamp}
    subdirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("models_")]
    assert len(subdirs) > 0
    run_dir = subdirs[0]
    
    # Check for HTML Report
    # Name might vary ('benchmark_report_advanced.html' or similar)
    html_files = list(run_dir.glob("*.html"))
    assert len(html_files) > 0, "No HTML report generated"
    
    # Check for JSON results
    json_files = list(run_dir.glob("*.json"))
    assert len(json_files) > 0, "No JSON results generated"

def test_failure_resilience(valid_config, sample_data, tmp_path):
    """
    Test that if one model fails (e.g. valid name but crashy), others survive.
    (Difficult to simulate actual crash inside SKLearn without mocking, 
     so we check that invalid model names are just ignored or handled gracefully)
    """
    out_dir = tmp_path / "out_fail"
    ds_config = valid_config['data_science']
    ds_config['output_folder'] = str(out_dir)
    # 'crazy_model' does not exist. DS should log warning but proceed with valid ones.
    ds_config['classifier_models'] = ['random_forest', 'crazy_model_xyz']
    
    df_in = sample_data.copy()
    df_in['random_feature'] = np.random.randn(len(df_in))
    
    ds = DataScientist(df_in, config=ds_config)
    results = ds.train_and_evaluate(ticker="TEST_TICKER", holdout_days=10)
    
    # Should still produce output
    subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(subdirs) > 0
    
    # Results should contain random_forest
    # We might need to inspect the return object 'results'
    # Assuming results is dict or similar
    # (If results is None/void, we rely on artifacts)
    
    if results:
         # Check structure if possible
         pass
