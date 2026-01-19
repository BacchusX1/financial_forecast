#!/usr/bin/env python3
"""
Trading Forecast v2 - Launcher
Interactive CLI to configure, download data, and engineer features.

Usage: python launcher.py
"""

import os
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'modules', 'dada_loader'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'modules', 'feature_engineering'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'modules', 'data_science'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src')) # for config_validator

from financial_data_loader import DataLoader
from feature_engineering import FeatureEngineering
from data_science import DataScientist
from config_validator import ConfigValidator


# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def load_config_file(config_path):
    """Load YAML configuration file and validate it."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print_success(f"Loaded configuration from {config_path}")

        # Validate configuration
        print_info("Validating configuration...")
        validator = ConfigValidator(config_dict=config)
        issues = validator.validate()
        
        has_errors = False
        for level, msg in issues:
            if level == "ERROR":
                print_error(f"CONFIG ERROR: {msg}")
                has_errors = True
            elif level == "WARNING":
                print_warning(f"CONFIG WARNING: {msg}")
            else:
                print_info(f"CONFIG SUGGESTION: {msg}")
        
        if has_errors:
            print_warning("Configuration has errors. Some features may not work as expected.")
            # We don't exit here strict, but let user decide?
            # User requirement: "validation rules" - usually implies blocking errors or loud warnings.
            # I'll block if "ERROR" but continue if WARNING.
            # Actually, returning None mimics 'failed load'.
            if any(i[0] == "ERROR" for i in issues):
                return None
        else:
             print_success("Configuration passed validation.")

        return config
    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_path}")
        return None
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return None


def save_config_auto(config, output_path="config_auto.yml"):
    """Save auto-generated configuration to YAML file."""
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print_success(f"Configuration saved to {output_path}")
        return True
    except Exception as e:
        print_error(f"Error saving configuration: {e}")
        return False


def get_user_input(prompt, input_type="str", default=None, options=None):
    """Get validated user input."""
    while True:
        try:
            if options:
                print(f"\n{prompt}")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                choice = input(f"Select (1-{len(options)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    return options[int(choice) - 1]
                print_warning("Invalid selection. Try again.")
                continue
            
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip() or default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if input_type == "int":
                return int(user_input)
            elif input_type == "float":
                return float(user_input)
            elif input_type == "bool":
                return user_input.lower() in ['yes', 'y', 'true', '1']
            elif input_type == "list":
                return [x.strip() for x in user_input.split(',')]
            else:
                return user_input
        except ValueError:
            print_warning(f"Invalid input. Expected {input_type}. Try again.")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)


def interactive_config():
    """Interactively build configuration."""
    print_header("INTERACTIVE CONFIGURATION BUILDER")
    
    config = {
        'data_loader': {},
        'feature_engineering': {},
        'output': {},
        'logging': {}
    }
    
    # ======== DATA LOADER ========
    print(f"\n{Colors.BOLD}DATA LOADER CONFIGURATION{Colors.ENDC}")
    print("-" * 80)
    
    # Keys
    print_info("Enter stock tickers or crypto symbols (comma-separated)")
    print("  Examples: AAPL,GOOGL:NASDAQ,TSLA or BTC,ETH")
    keys_input = get_user_input("Asset symbols")
    config['data_loader']['keys'] = [k.strip() for k in keys_input.split(',')]
    
    # Category
    category = get_user_input(
        "Asset category",
        options=["stock", "crypto"]
    )
    config['data_loader']['category'] = category
    
    # Candle length
    intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo']
    candle = get_user_input(
        "Candle timeframe (1m=1-minute, 1h=hourly, 1d=daily, 1wk=weekly, 1mo=monthly)",
        options=intervals
    )
    config['data_loader']['candle_length'] = candle
    
    # Period
    period = get_user_input("Years of history to download", input_type="int", default="5")
    config['data_loader']['period'] = period
    
    config['data_loader']['start_date'] = None
    config['data_loader']['end_date'] = None
    print_info("Using automatic date range based on period")
    
    # ======== FEATURE ENGINEERING ========
    print(f"\n{Colors.BOLD}FEATURE ENGINEERING CONFIGURATION{Colors.ENDC}")
    print("-" * 80)
    
    # Scaler
    scaler = get_user_input(
        "Scaling method",
        options=["standard (recommended for ML)", "minmax (for [0,1] range)", "robust (handles outliers)", "none (no scaling)"]
    ).split()[0].lower()
    if scaler == "none":
        scaler = None
    config['feature_engineering']['scaler'] = scaler
    
    # Lag features
    lag_input = get_user_input(
        "Lag periods (comma-separated, e.g., 1,5,10)",
        default="1,5,10"
    )
    config['feature_engineering']['lag_features'] = [int(x.strip()) for x in lag_input.split(',') if x.strip()]
    
    # Rolling windows
    rolling_input = get_user_input(
        "Rolling window sizes (comma-separated, e.g., 5,20)",
        default="5,20"
    )
    config['feature_engineering']['rolling_windows'] = [int(x.strip()) for x in rolling_input.split(',') if x.strip()]
    
    # Rolling functions
    rolling_funcs = get_user_input(
        "Rolling aggregation functions",
        options=["mean,std (recommended)", "mean,std,min,max (all)", "mean only"]
    )
    if rolling_funcs.startswith("mean,std ("):
        rolling_funcs = ["mean", "std"]
    elif rolling_funcs.startswith("mean,std,min"):
        rolling_funcs = ["mean", "std", "min", "max"]
    else:
        rolling_funcs = ["mean"]
    config['feature_engineering']['rolling_functions'] = rolling_funcs
    
    # Target columns
    use_auto = get_user_input(
        "Auto-detect target columns?",
        input_type="bool"
    )
    if use_auto:
        config['feature_engineering']['target_columns'] = None
    else:
        cols = get_user_input("Target columns (comma-separated, e.g., Close,Volume,High)")
        config['feature_engineering']['target_columns'] = [c.strip() for c in cols.split(',')]
    
    # PCA
    use_pca = get_user_input(
        "Use PCA dimensionality reduction?",
        input_type="bool"
    )
    if use_pca:
        pca_dim = get_user_input("Number of PCA components", input_type="int", default="10")
        config['feature_engineering']['pca_components'] = pca_dim
    else:
        config['feature_engineering']['pca_components'] = None
    
    # Autoencoder
    use_ae = get_user_input(
        "Use Autoencoder for feature extraction?",
        input_type="bool"
    )
    if use_ae:
        ae_dim = get_user_input("Autoencoder latent dimension", input_type="int", default="20")
        ae_epochs = get_user_input("Training epochs", input_type="int", default="100")
        config['feature_engineering']['autoencoder_latent_dim'] = ae_dim
        config['feature_engineering']['autoencoder_epochs'] = ae_epochs
    else:
        config['feature_engineering']['autoencoder_latent_dim'] = None
        config['feature_engineering']['autoencoder_epochs'] = 100
    
    # Group by ticker
    config['feature_engineering']['group_by_ticker'] = True
    
    # Exclude from scaling - Default: Keep Close in original price range for regression
    config['feature_engineering']['exclude_from_scaling'] = ['Close']
    
    # ======== DATA SCIENCE ========
    print(f"\n{Colors.BOLD}DATA SCIENCE (ML MODELS) CONFIGURATION{Colors.ENDC}")
    print("-" * 80)
    
    use_ds = get_user_input(
        "Enable Data Science module (train ML models)?",
        input_type="bool"
    )
    
    if use_ds:
        # Classifiers
        classifiers = get_user_input(
            "Classifier models (trend classification: UP/DOWN/SIDEWAYS)",
            options=["dnn,random_forest (recommended)", "dnn,svc,random_forest,gradient_boosting (all)"]
        )
        if "recommended" in classifiers:
            config['data_science'] = {'classifier_models': ['dnn', 'random_forest']}
        else:
            config['data_science'] = {'classifier_models': ['dnn', 'svc', 'random_forest', 'gradient_boosting']}
        
        # Regressors
        regressors = get_user_input(
            "Regressor models (time series forecasting)",
            options=["lstm,dnn (recommended)", "lstm,dnn,arima,krr (multiple)", "None (skip regressors)"]
        )
        if "lstm,dnn (recommended)" in regressors:
            config['data_science']['regressor_models'] = ['lstm', 'dnn']
        elif "lstm,dnn,arima" in regressors:
            config['data_science']['regressor_models'] = ['lstm', 'dnn', 'arima', 'krr']
        else:
            config['data_science']['regressor_models'] = []
        
        # Trend window
        trend_window = get_user_input("Trend window (days)", input_type="int", default="5")
        config['data_science']['trend_window'] = trend_window
        
        # Forecast horizon
        forecast_horizon = get_user_input("Forecast horizon (days)", input_type="int", default="30")
        config['data_science']['forecast_horizon'] = forecast_horizon
        
        config['data_science']['test_size'] = 0.2
        config['data_science']['random_state'] = 42
    else:
        config['data_science'] = {
            'enabled': False,
            'classifier_models': [],
            'regressor_models': []
        }
    
    # ======== OUTPUT ========
    print(f"\n{Colors.BOLD}OUTPUT CONFIGURATION{Colors.ENDC}")
    print("-" * 80)
    
    config['output']['folder'] = "out"
    config['output']['save_raw_data'] = True
    config['output']['save_transformed_features'] = True
    config['output']['save_summary'] = True
    
    config['logging']['level'] = "INFO"
    config['logging']['save_to_file'] = False
    
    return config


def extract_config_for_dataloader(config):
    """Extract DataLoader-specific configuration."""
    return config.get('data_loader', {})


def extract_config_for_feature_eng(config):
    """Extract FeatureEngineering-specific configuration."""
    return config.get('feature_engineering', {})


def extract_output_config(config):
    """Extract output configuration."""
    return config.get('output', {})


def create_output_directories(output_config):
    """Create output directory structure."""
    output_folder = output_config.get('folder', 'out')
    
    # Create main output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subfolders
    subfolders = ['raw_data', 'transformed_features', 'summaries']
    for subfolder in subfolders:
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
    
    print_success(f"Created output directory structure: {output_folder}/")
    return output_folder


def run_pipeline(config):
    """Execute data loading and feature engineering pipeline."""
    print_header("EXECUTING PIPELINE")
    
    # Create output directories
    output_config = extract_output_config(config)
    output_folder = create_output_directories(output_config)
    
    # Load data
    print(f"\n{Colors.BOLD}Step 1: Loading Data{Colors.ENDC}")
    print("-" * 80)
    
    dl_config = extract_config_for_dataloader(config)
    try:
        loader = DataLoader(dl_config)
        data_dict = loader.assemble_data()
        
        if not data_dict:
            print_error("No data loaded. Aborting.")
            return False
        
        combined_data = loader.get_combined_data()
        print_success(f"Loaded {len(data_dict)} assets with {len(combined_data)} total records")
        
        # Save raw data
        if output_config.get('save_raw_data', True):
            raw_data_folder = os.path.join(output_folder, 'raw_data')
            for ticker, df in data_dict.items():
                output_file = os.path.join(raw_data_folder, f"{ticker}.csv")
                df.to_csv(output_file, index=False)
                print_info(f"Saved {ticker} → {output_file}")
        
        # Save summary
        if output_config.get('save_summary', True):
            summary_file = os.path.join(output_folder, 'summaries', 'data_summary.csv')
            loader.get_summary().to_csv(summary_file, index=False)
            print_success(f"Saved data summary → {summary_file}")
        
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        return False
    
    # Feature engineering
    print(f"\n{Colors.BOLD}Step 2: Feature Engineering{Colors.ENDC}")
    print("-" * 80)
    
    fe_config = extract_config_for_feature_eng(config)
    
    # Check if indicators are enabled in config
    if 'indicators' in fe_config or 'indicator_set' in fe_config:
        groups = fe_config.get('indicators', {}).get('enabled_groups', [])
        print_info(f"  Advanced indicators enabled: {len(groups) if groups else 'ALL'} groups")
        
    try:
        fe = FeatureEngineering(combined_data, fe_config)
        exclude_list = config.get('feature_engineering', {}).get('exclude_from_scaling', [])
        transformed = fe.transform(exclude_from_scaling=exclude_list)
        print_success(f"Created {transformed.shape[1]} features from {combined_data.shape[1]} input columns")
        print_success(f"Output shape: {transformed.shape[0]} records × {transformed.shape[1]} features")
        
        # Save transformed features WITH LABELS
        if output_config.get('save_transformed_features', True):
            # Create a copy of transformed for adding labels
            features_with_labels = transformed.copy()
            
            # Get data science config for label generation
            ds_cfg = config.get('data_science', {})
            forecast_horizon = ds_cfg.get('forecast_horizon', 10)
            trend_window = ds_cfg.get('trend_window', 5)
            trend_threshold = ds_cfg.get('trend_threshold', 0.02)
            
            # Add multi-step regression targets (Close_t+1, Close_t+2, ..., Close_t+horizon)
            if 'Close' in features_with_labels.columns:
                close_prices = features_with_labels['Close'].values
                n_samples = len(close_prices)
                
                print_info(f"  Generating {forecast_horizon} regression target columns...")
                for step in range(1, forecast_horizon + 1):
                    target_col = f'target_Close_t+{step}'
                    # Shift Close price to get future values
                    target_values = np.full(n_samples, np.nan)
                    if step < n_samples:
                        target_values[:-step] = close_prices[step:]
                    features_with_labels[target_col] = target_values
                
                # Add classification label (trend: UP=2, SIDEWAYS=1, DOWN=0)
                print_info(f"  Generating trend classification label (window={trend_window}, threshold={trend_threshold*100:.1f}%)...")
                trend_labels = np.full(n_samples, 1)  # Default SIDEWAYS
                for i in range(n_samples - trend_window):
                    if close_prices[i] != 0:
                        future_price = close_prices[i + trend_window]
                        current_price = close_prices[i]
                        price_change_pct = (future_price - current_price) / current_price
                        
                        if price_change_pct > trend_threshold:
                            trend_labels[i] = 2  # UP
                        elif price_change_pct < -trend_threshold:
                            trend_labels[i] = 0  # DOWN
                        else:
                            trend_labels[i] = 1  # SIDEWAYS
                
                features_with_labels['target_trend'] = trend_labels
                
                # Add trend label with forecast horizon (for multi-step classification)
                if trend_window != forecast_horizon:
                    print_info(f"  Generating trend label for forecast horizon ({forecast_horizon})...")
                    trend_labels_horizon = np.full(n_samples, 1)
                    for i in range(n_samples - forecast_horizon):
                        if close_prices[i] != 0:
                            future_price = close_prices[i + forecast_horizon]
                            current_price = close_prices[i]
                            price_change_pct = (future_price - current_price) / current_price
                            
                            if price_change_pct > trend_threshold:
                                trend_labels_horizon[i] = 2  # UP
                            elif price_change_pct < -trend_threshold:
                                trend_labels_horizon[i] = 0  # DOWN
                            else:
                                trend_labels_horizon[i] = 1  # SIDEWAYS
                    
                    features_with_labels[f'target_trend_h{forecast_horizon}'] = trend_labels_horizon
                
                # Count label columns
                label_cols = [c for c in features_with_labels.columns if c.startswith('target_')]
                print_success(f"  Added {len(label_cols)} label columns: {label_cols[:3]}..." if len(label_cols) > 3 else f"  Added {len(label_cols)} label columns")
            
            # Save features with labels
            features_file = os.path.join(output_folder, 'transformed_features', 'features_labels.csv')
            features_with_labels.to_csv(features_file, index=False)
            print_success(f"Saved features + labels → {features_file}")
        
        # Save feature engineering summary
        if output_config.get('save_summary', True):
            summary_file = os.path.join(output_folder, 'summaries', 'features_summary.json')
            summary = fe.get_summary()
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print_success(f"Saved feature summary → {summary_file}")
        
    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        return False
    
    # Data Science (ML Models)
    ds_config = config.get('data_science', {})
    if ds_config.get('classifier_models') or ds_config.get('regressor_models'):
        print(f"\n{Colors.BOLD}Step 3: Training ML Models{Colors.ENDC}")
        print("-" * 80)
        
        # Get ticker name from loaded data
        ticker_name = list(data_dict.keys())[0] if data_dict else 'Asset'
        if len(data_dict) > 1:
            ticker_name = f"{ticker_name} (+{len(data_dict)-1} more)"
        
        # Get training parameters from config
        holdout_days = ds_config.get('holdout_days', 50)
        forecast_horizon = ds_config.get('forecast_horizon', 10)
        llm_config = config.get('llm', {})
        
        # Pass candle_length to DataScientist for proper report labels
        ds_config['candle_length'] = dl_config.get('candle_length', '1d')
        ds_config['output_folder'] = output_folder
        
        print_info(f"  Classifiers: {ds_config.get('classifier_models', [])}")
        print_info(f"  Regressors: {ds_config.get('regressor_models', [])}")
        print_info(f"  Forecast horizon: {forecast_horizon} steps")
        print_info(f"  Holdout days: {holdout_days} days")
        if llm_config.get('enabled', False):
            print_info(f"  🤖 LLM Review enabled: {llm_config.get('model', 'llama3.1:8b-instruct')}")
        
        try:
            ds = DataScientist(transformed, ds_config)
            ds_results = ds.train_and_evaluate(
                ticker=ticker_name, 
                holdout_days=holdout_days, 
                llm_config=llm_config,
                generate_reports=True
            )
            
            print_success(f"Models trained successfully")
            
            # ===== RESULTS SUMMARY =====
            print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.BOLD}RESULTS SUMMARY{Colors.ENDC}")
            print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
            
            # Classifier results
            if ds_results.get('classifiers'):
                print(f"\n📊 Trend Classification ({len(ds_results['classifiers'])} models):")
                for model, metrics in ds_results['classifiers'].items():
                    trend = metrics.get('trend_prediction', 'UNKNOWN')
                    trend_emoji = {'UP': '📈', 'DOWN': '📉', 'SIDEWAYS': '➡️'}.get(trend, '❓')
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_score', 0)
                    print(f"  {model.upper():20s}: {trend} {trend_emoji}  (Accuracy: {acc:.2%}, F1: {f1:.4f})")
            
            # Regressor results
            if ds_results.get('regressors'):
                print(f"\n📈 Regressor Performance ({len(ds_results['regressors'])} models):")
                sorted_regs = sorted(ds_results['regressors'].items(), 
                                     key=lambda x: x[1].get('r2_avg', x[1].get('r2', 0)), reverse=True)
                for i, (model, metrics) in enumerate(sorted_regs, 1):
                    r2 = metrics.get('r2_avg', metrics.get('r2', 0))
                    rmse = metrics.get('rmse_avg', metrics.get('rmse', 0))
                    print(f"  {i}. {model.upper():20s}: R²_avg={r2:.4f}  RMSE_avg=${rmse:.2f}")
            
            # Holdout evaluation
            if ds_results.get('holdout_forecasts'):
                print(f"\n🎯 Holdout Evaluation (True Out-of-Sample):")
                for model, data in ds_results['holdout_forecasts'].items():
                    if 'metrics' in data:
                        rmse = data['metrics'].get('rmse_avg', 0)
                        beats = "✓ BEATS NAIVE" if data.get('beats_naive', False) else "✗ below naive"
                        print(f"  {model.upper():20s}: RMSE_avg=${rmse:.2f}  {beats}")
            
            # Baselines
            if ds_results.get('baselines'):
                print(f"\n📊 Baselines ({len(ds_results['baselines'])} strategies):")
                for name, data in ds_results['baselines'].items():
                    rmse = data.get('metrics', {}).get('rmse_avg', 0)
                    print(f"  {name.upper():20s}: RMSE_avg=${rmse:.2f}")
            
            # Future forecasts
            if ds_results.get('future_forecasts'):
                print(f"\n🔮 {forecast_horizon}-Step Price Forecasts:")
                current_price = ds_results.get('last_actual_price')
                if current_price:
                    print(f"  Current price: ${current_price:.2f}")
                
                for model, forecast in ds_results['future_forecasts'].items():
                    if model == 'naive_baseline':
                        continue
                    current = forecast.get('last_price', current_price)
                    preds = forecast.get('predictions', [])
                    if preds:
                        future = preds[-1]
                        change = ((future - current) / current) * 100 if current else 0
                        arrow = "📈" if change > 0 else "📉"
                        print(f"  {model.upper():20s}: Step {len(preds)} -> ${future:.2f}  ({change:+.2f}%) {arrow}")
            
            # LLM Review
            if ds_results.get('llm_review'):
                review = ds_results['llm_review']
                if review.get('content'):
                    print(f"\n🤖 LLM Review ({review.get('model', 'unknown')}):")
                    print(f"  {'-'*60}")
                    content = review['content'][:500] + "..." if len(review.get('content', '')) > 500 else review.get('content', '')
                    print(f"  {content}")
                    print(f"  {'-'*60}")
                elif review.get('error'):
                    print(f"\n⚠️ LLM Review failed: {review['error']}")
            
            try:
                ds_results['output_dir'] = os.path.join(output_folder)
            except:
                print_warning(f"  Warning: Could not set output_dir in ds_results.")

            try:
                print_info(f"\n  Output directory: {ds_results['output_dir']}/")
            except:
                print_info(f"\n  Warning: Output directory: ds_results['output_dir'] variable not found or inaccessible.")
            print_info(f"  Reports: benchmark_report_advanced.html")
            
        except Exception as e:
            print_error(f"Data Science training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    print(f"\n{Colors.BOLD}Pipeline Complete{Colors.ENDC}")
    print("-" * 80)
    print_success(f"All outputs saved to: {output_folder}/")
    print_info(f"  - Raw data: {os.path.join(output_folder, 'raw_data')}/")
    print_info(f"  - Features: {os.path.join(output_folder, 'transformed_features')}/")
    try:
        if ds_config.get('classifier_models') or ds_config.get('regressor_models'):
                print_info(f"  - ML Models: {ds_results['output_dir']}/")
                print_info(f"  - Summaries: {os.path.join(output_folder, 'summaries')}/")
    except:
        print_info(f"  Warning: ML Models or Summaries output directory not found or inaccessible.")
    
    return True


def main():
    """Main launcher entry point."""
    print_header("TRADING FORECAST v2 - LAUNCHER")
    
    # Ask user for config source
    config_choice = get_user_input(
        "Configuration source",
        options=["Use configuration.yml", "Create config_auto.yml interactively"]
    )
    
    if "configuration.yml" in config_choice:
        # Load existing configuration
        config = load_config_file('configuration.yml')
        if config is None:
            print_error("Cannot proceed without configuration.")
            sys.exit(1)
    else:
        # Interactive config builder
        config = interactive_config()
        
        # Save to config_auto.yml
        if save_config_auto(config):
            print_success("Configuration ready!")
        else:
            print_warning("Could not save auto config, but will continue with in-memory config")
    
    # Display loaded configuration
    print(f"\n{Colors.BOLD}Configuration Summary:{Colors.ENDC}")
    print(f"  Data Loader:")
    print(f"    - Assets: {config['data_loader']['keys']}")
    print(f"    - Category: {config['data_loader']['category']}")
    print(f"    - Interval: {config['data_loader']['candle_length']}")
    print(f"    - Period: {config['data_loader']['period']} years")
    
    print(f"  Feature Engineering:")
    print(f"    - Scaler: {config['feature_engineering']['scaler']}")
    print(f"    - Lag features: {config['feature_engineering']['lag_features']}")
    print(f"    - Rolling windows: {config['feature_engineering']['rolling_windows']}")
    print(f"    - PCA: {config['feature_engineering']['pca_components']}")
    print(f"    - Autoencoder: {config['feature_engineering']['autoencoder_latent_dim']}")
    
    ds_config = config.get('data_science', {})
    if ds_config.get('classifier_models') or ds_config.get('regressor_models'):
        print(f"  Data Science:")
        print(f"    - Classifiers: {ds_config.get('classifier_models', [])}")
        print(f"    - Regressors: {ds_config.get('regressor_models', [])}")
        print(f"    - Trend window: {ds_config.get('trend_window', 5)} days")
        print(f"    - Forecast horizon: {ds_config.get('forecast_horizon', 30)} days")
    
    print(f"  Output: {config['output']['folder']}/")
    
    # Confirm and proceed
    proceed = get_user_input("\nProceed with pipeline execution?", input_type="bool")
    if not proceed:
        print_warning("Pipeline cancelled.")
        sys.exit(0)
    
    # Run pipeline
    success = run_pipeline(config)
    
    if success:
        print_success("\n" + "=" * 80)
        print_success("All tasks completed successfully!")
        print_success("=" * 80)
        sys.exit(0)
    else:
        print_error("\n" + "=" * 80)
        print_error("Pipeline failed. Check logs above for details.")
        print_error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Pipeline interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)



