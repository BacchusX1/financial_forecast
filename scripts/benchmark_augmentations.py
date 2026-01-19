#!/usr/bin/env python3
"""Benchmark all augmentation methods (regressors only).

SIMPLE PIPELINE:
1. Read configuration.yml
2. For each augmentation method:
   a. Write augmentation config to configuration.yml
   b. Call the launcher pipeline (which generates reports)
   c. Collect results
   d. Continue with next augmentation method

Outputs:
- Full reports for each augmentation run in out/augmentation_benchmark_YYYYmmdd_HHMMSS/run_<method>/
- CSV + JSON summary under out/augmentation_benchmark_YYYYmmdd_HHMMSS/
- Console ranking by heldout metrics (rmse_avg, mae_avg, r2_avg)
- Detailed log file with model save status

Usage:
  python scripts/benchmark_augmentations.py
  python scripts/benchmark_augmentations.py --days 365 --horizon 10 --holdout 40
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Match launcher import behavior
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'modules', 'dada_loader'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'modules', 'feature_engineering'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'modules', 'data_science'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from financial_data_loader import DataLoader
from feature_engineering import FeatureEngineering
from data_science import DataScientist

# Augmentor methods listing
try:
    from data_augmentor import list_available_methods
    DATA_AUGMENTOR_AVAILABLE = True
except Exception:
    DATA_AUGMENTOR_AVAILABLE = False


@dataclass(frozen=True)
class AugVariant:
    name: str
    enabled: bool
    methods: List[str]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _today_utc_date() -> datetime:
    return datetime.utcnow()


def _iso_date(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


def _mean_or_nan(values: List[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _setup_benchmark_logger(log_file: str) -> logging.Logger:
    """Setup a dedicated logger for the benchmark that logs to file."""
    logger = logging.getLogger('benchmark_augmentations')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


def _check_model_files(output_dir: str, expected_models: List[str], logger: logging.Logger) -> Dict[str, Any]:
    """Check which model files were saved and log why others might be missing."""
    model_status = {}
    
    for model_name in expected_models:
        # Check for different model file formats
        possible_files = [
            os.path.join(output_dir, f"{model_name}_reg.pt"),    # PyTorch
            os.path.join(output_dir, f"{model_name}_reg.h5"),    # Keras
            os.path.join(output_dir, f"{model_name}_reg.pkl"),   # sklearn/pickle
        ]
        
        found = False
        for filepath in possible_files:
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                model_status[model_name] = {
                    'saved': True,
                    'path': filepath,
                    'size_kb': round(size_kb, 2)
                }
                logger.info(f"  ✓ Model {model_name}: SAVED ({filepath}, {size_kb:.1f} KB)")
                found = True
                break
        
        if not found:
            model_status[model_name] = {
                'saved': False,
                'path': None,
                'reason': 'Model file not found - check training logs for errors'
            }
            logger.warning(f"  ✗ Model {model_name}: NOT SAVED - possible reasons:")
            logger.warning(f"      - Training failed (exception during training)")
            logger.warning(f"      - Model returned None from train method")
            logger.warning(f"      - _save_model() failed (check permissions or disk space)")
            logger.warning(f"      - Model type not in expected list for regressor")
    
    return model_status


def _make_variants() -> List[AugVariant]:
    """Create list of augmentation variants to test."""
    variants = [AugVariant(name='none', enabled=False, methods=[])]

    if not DATA_AUGMENTOR_AVAILABLE:
        return variants

    methods = list_available_methods()
    for method_name in methods.keys():
        # Skip 3D-only methods that require special data shapes
        if method_name in ['time_warp', 'permute_segments']:
            continue
        variants.append(AugVariant(name=method_name, enabled=True, methods=[method_name]))

    return variants


def run_single_augmentation(
    base_config: Dict[str, Any],
    variant: AugVariant,
    run_output_dir: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Run the pipeline for a single augmentation variant.
    This mimics calling launcher.py with modified configuration.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING AUGMENTATION: {variant.name}")
    logger.info(f"{'='*80}")
    
    cfg = copy.deepcopy(base_config)
    
    # Set augmentation config
    cfg.setdefault('data_science', {})
    cfg['data_science'].setdefault('data_augmentation', {})
    cfg['data_science']['data_augmentation'] = {
        'enabled': variant.enabled,
        'preset': None,
        'methods': list(variant.methods),
        'n_augmentations': 1,
        'seed': cfg['data_science'].get('random_state', 42),
    }
    
    # Set output folder for this run
    cfg['data_science']['output_folder'] = run_output_dir
    cfg['output'] = cfg.get('output', {})
    cfg['output']['folder'] = run_output_dir
    
    _ensure_dir(run_output_dir)
    
    # Save the config used for this run
    run_config_path = os.path.join(run_output_dir, 'run_configuration.yml')
    _save_yaml(run_config_path, cfg)
    logger.info(f"  Config saved to: {run_config_path}")
    
    # ========== PIPELINE EXECUTION (similar to launcher.py) ==========
    
    # Step 1: Load data
    logger.info("\n--- Step 1: Loading Data ---")
    dl_config = cfg.get('data_loader', {})
    try:
        loader = DataLoader(dl_config)
        data_dict = loader.assemble_data()
        
        if not data_dict:
            logger.error("No data loaded. Aborting this run.")
            return {'error': 'No data loaded', 'augmentation': variant.name}
        
        combined_data = loader.get_combined_data()
        logger.info(f"  Loaded {len(data_dict)} assets with {len(combined_data)} total records")
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e), 'augmentation': variant.name}
    
    # Step 2: Feature engineering
    logger.info("\n--- Step 2: Feature Engineering ---")
    fe_config = cfg.get('feature_engineering', {})
    try:
        fe = FeatureEngineering(combined_data, fe_config)
        exclude_list = fe_config.get('exclude_from_scaling', [])
        transformed = fe.transform(exclude_from_scaling=exclude_list)
        logger.info(f"  Created {transformed.shape[1]} features from {combined_data.shape[1]} input columns")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e), 'augmentation': variant.name}
    
    # Step 3: Data Science / Model Training
    logger.info("\n--- Step 3: Training ML Models ---")
    ds_config = cfg.get('data_science', {})
    ticker_name = list(data_dict.keys())[0] if data_dict else 'BTC'
    holdout_days = ds_config.get('holdout_days', args.holdout)
    llm_config = cfg.get('llm', {'enabled': False})
    
    regressor_models = ds_config.get('regressor_models', [])
    logger.info(f"  Regressors to train: {regressor_models}")
    logger.info(f"  Forecast horizon: {ds_config.get('forecast_horizon', 10)} steps")
    logger.info(f"  Holdout samples: {holdout_days}")
    logger.info(f"  Augmentation enabled: {variant.enabled}")
    logger.info(f"  Augmentation methods: {variant.methods}")
    
    try:
        ds = DataScientist(transformed, ds_config)
        
        # GENERATE REPORTS = TRUE (user requested reports!)
        results = ds.train_and_evaluate(
            ticker=ticker_name,
            holdout_days=holdout_days,
            llm_config=llm_config,
            generate_reports=True,  # <-- KEY CHANGE: Generate reports!
        )
        
        logger.info(f"  Models trained successfully for augmentation: {variant.name}")
        
    except Exception as e:
        logger.error(f"Data Science training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e), 'augmentation': variant.name}
    
    # Step 4: Check which model files were saved
    logger.info("\n--- Step 4: Checking Model Save Status ---")
    model_status = _check_model_files(run_output_dir, regressor_models, logger)
    
    # Log detailed save status
    saved_count = sum(1 for s in model_status.values() if s.get('saved', False))
    logger.info(f"\n  Model save summary: {saved_count}/{len(regressor_models)} models saved")
    
    if saved_count < len(regressor_models):
        logger.warning("\n  INVESTIGATING MISSING MODELS:")
        # Check for model_errors in results
        if 'model_errors' in results:
            for model_name, error_info in results.get('model_errors', {}).items():
                logger.warning(f"    {model_name}: {error_info.get('exception', 'Unknown error')}")
                if 'traceback' in error_info:
                    for line in error_info['traceback'].split('\n')[:5]:
                        logger.warning(f"      {line}")
        
        # Check regressors dict for missing models
        for model_name in regressor_models:
            if model_name not in results.get('regressors', {}):
                logger.warning(f"    {model_name}: Not in results['regressors'] - training likely failed completely")
    
    # Step 5: Collect holdout metrics
    logger.info("\n--- Step 5: Collecting Holdout Metrics ---")
    holdout = results.get('holdout_forecasts', {})
    per_model_metrics = []
    
    for model_name in regressor_models:
        item = holdout.get(model_name, {})
        metrics = item.get('metrics') if isinstance(item, dict) else None
        
        if metrics:
            per_model_metrics.append({
                'model': model_name,
                'rmse_avg': metrics.get('rmse_avg'),
                'mae_avg': metrics.get('mae_avg'),
                'r2_avg': metrics.get('r2_avg'),
            })
            logger.info(f"    {model_name}: RMSE_avg={metrics.get('rmse_avg', 0):.4f}, R2_avg={metrics.get('r2_avg', 0):.4f}")
        else:
            logger.warning(f"    {model_name}: No holdout metrics available")
    
    # Save run summary
    run_summary = {
        'augmentation': variant.name,
        'enabled': variant.enabled,
        'methods': variant.methods,
        'per_model_metrics': per_model_metrics,
        'model_save_status': model_status,
        'avg_rmse': _mean_or_nan([m['rmse_avg'] for m in per_model_metrics]),
        'avg_mae': _mean_or_nan([m['mae_avg'] for m in per_model_metrics]),
        'avg_r2': _mean_or_nan([m['r2_avg'] for m in per_model_metrics]),
        'n_models_scored': len(per_model_metrics),
        'output_dir': run_output_dir,
    }
    
    with open(os.path.join(run_output_dir, 'run_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)
    
    logger.info(f"\n  Run complete. Reports saved to: {run_output_dir}")
    
    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark augmentation methods for regressor models')
    parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'configuration.yml'),
                        help='Path to base configuration file')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of data to download')
    parser.add_argument('--horizon', type=int, default=10,
                        help='Forecast horizon (number of steps)')
    parser.add_argument('--holdout', type=int, default=40,
                        help='Holdout samples for evaluation')
    parser.add_argument('--output-base', default=os.path.join(REPO_ROOT, 'out'),
                        help='Base output directory')
    parser.add_argument('--regressors', default=None,
                        help='Comma-separated list of regressors (e.g. "dnn,xlstm,tcn")')
    args = parser.parse_args()

    # Setup output directory
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_base, f'augmentation_benchmark_{stamp}')
    _ensure_dir(out_dir)
    
    # Setup logging
    log_file = os.path.join(out_dir, 'benchmark_log.txt')
    logger = _setup_benchmark_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("AUGMENTATION BENCHMARK - STARTING")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {stamp}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Log file: {log_file}")

    # Load base configuration
    logger.info(f"\nLoading base configuration from: {args.config}")
    base_cfg = _load_yaml(args.config)
    
    # Override data loader settings
    now = _today_utc_date()
    start = now - timedelta(days=int(args.days))
    
    base_cfg.setdefault('data_loader', {})
    base_cfg['data_loader'].update({
        'keys': ['BTC'],
        'category': 'crypto',
        'candle_length': '5m',
        'period': None,
        'start_date': _iso_date(start),
        'end_date': _iso_date(now),
        'extended_download': True,
        'chunk_overlap_days': base_cfg.get('data_loader', {}).get('chunk_overlap_days', 1),
        'request_delay_seconds': base_cfg.get('data_loader', {}).get('request_delay_seconds', 2.0),
    })
    
    # Override data science settings
    base_cfg.setdefault('data_science', {})
    base_cfg['data_science']['enabled'] = True
    base_cfg['data_science']['classifier_models'] = []
    base_cfg['data_science']['forecast_horizon'] = int(args.horizon)
    base_cfg['data_science']['holdout_days'] = int(args.holdout)
    base_cfg['data_science']['candle_length'] = '5m'
    
    if args.regressors:
        base_cfg['data_science']['regressor_models'] = [m.strip() for m in args.regressors.split(',') if m.strip()]
    
    regressor_models = base_cfg['data_science'].get('regressor_models', ['dnn', 'xlstm', 'tcn'])
    logger.info(f"Regressors to benchmark: {regressor_models}")
    
    # Disable LLM for benchmarking
    base_cfg.setdefault('llm', {})
    base_cfg['llm']['enabled'] = False
    
    # Build augmentation variants
    variants = _make_variants()
    logger.info(f"\nAugmentation variants to test: {len(variants)}")
    for v in variants:
        logger.info(f"  - {v.name}: enabled={v.enabled}, methods={v.methods}")
    
    # Save benchmark context
    with open(os.path.join(out_dir, 'benchmark_context.json'), 'w') as f:
        json.dump({
            'timestamp': stamp,
            'base_config_path': args.config,
            'data_loader_overrides': base_cfg['data_loader'],
            'data_science_config': base_cfg['data_science'],
            'variants': [{'name': v.name, 'enabled': v.enabled, 'methods': v.methods} for v in variants],
            'regressor_models': regressor_models,
        }, f, indent=2)
    
    # Run benchmarks
    all_results: List[Dict[str, Any]] = []
    
    for i, variant in enumerate(variants, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# VARIANT {i}/{len(variants)}: {variant.name}")
        logger.info(f"{'#'*80}")
        
        run_output_dir = os.path.join(out_dir, f'run_{variant.name}')
        
        result = run_single_augmentation(
            base_config=base_cfg,
            variant=variant,
            run_output_dir=run_output_dir,
            args=args,
            logger=logger,
        )
        
        all_results.append(result)
    
    # Create summary table
    logger.info("\n\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    rows = []
    for r in all_results:
        if 'error' not in r:
            rows.append({
                'augmentation': r['augmentation'],
                'enabled': r['enabled'],
                'methods': ','.join(r.get('methods', [])),
                'n_models_scored': r.get('n_models_scored', 0),
                'holdout_rmse_avg_mean': r.get('avg_rmse', float('nan')),
                'holdout_mae_avg_mean': r.get('avg_mae', float('nan')),
                'holdout_r2_avg_mean': r.get('avg_r2', float('nan')),
            })
        else:
            rows.append({
                'augmentation': r['augmentation'],
                'enabled': False,
                'methods': '',
                'n_models_scored': 0,
                'holdout_rmse_avg_mean': float('nan'),
                'holdout_mae_avg_mean': float('nan'),
                'holdout_r2_avg_mean': float('nan'),
                'error': r['error'],
            })
    
    df = pd.DataFrame(rows)
    df.sort_values(['holdout_rmse_avg_mean', 'holdout_r2_avg_mean'], ascending=[True, False], inplace=True)
    
    # Save results
    df.to_csv(os.path.join(out_dir, 'augmentation_benchmark.csv'), index=False)
    df.to_json(os.path.join(out_dir, 'augmentation_benchmark.json'), orient='records', indent=2)
    
    # Print ranking
    show_cols = ['augmentation', 'n_models_scored', 'holdout_rmse_avg_mean', 'holdout_mae_avg_mean', 'holdout_r2_avg_mean']
    logger.info("\nRanking by holdout RMSE_avg:")
    logger.info(df[show_cols].to_string(index=False))
    
    if len(df) > 0 and not df.iloc[0].get('error'):
        best = df.iloc[0].to_dict()
        logger.info('\nBest augmentation by RMSE_avg:')
        logger.info(json.dumps({k: best.get(k) for k in show_cols}, indent=2, default=str))
    
    logger.info(f"\n\nAll outputs saved to: {out_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("BENCHMARK COMPLETE")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
