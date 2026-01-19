"""
Data Science Module: Time Series Classification and Regression for Financial Data

Supports:
  - Classifiers: DNN, SVC, Random Forest (Trend: UP/DOWN/SIDEWAYS)
  - Regressors: LSTM, DNN, Kernel Ridge Regression, Linear
  - Automatic visualization and metrics
  - Timestamp-based output organization

NOTE: ARIMA, SVR, Gradient Boosting removed due to poor performance.

Usage:
  from data_science import DataScientist
  
  config = {
      "classifier_models": ["dnn", "svc", "random_forest"],
      "regressor_models": ["lstm", "dnn", "krr"],
      "trend_window": 5,  # days to classify trend
      "forecast_horizon": 30  # days to forecast
  }
  
  ds = DataScientist(features_df, config)
  results = ds.train_and_evaluate()
"""

import os
import json
import logging
import warnings
import pickle
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                            mean_squared_error, mean_absolute_error, r2_score, roc_auc_score)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

# Import forecast utilities
try:
    from forecast_utils import (
        build_multistep_targets, 
        evaluate_multistep, 
        compute_multistep_baselines,
        validate_no_leakage,
        create_multistep_trend_labels,
        postprocess_forecasts,
        roll_multistep_predictions,
        aggregate_rolling_predictions,
        compute_rolling_baselines,
        evaluate_rolling_forecast,
        create_10day_trend_labels,
        recursive_holdout_forecast,  # NEW: True autoregressive forecasting
        # Target transforms
        get_target_transform,
        to_target,
        from_target,
        from_target_cumulative,
        evaluate_in_both_spaces,
        VALID_TARGET_TRANSFORMS
    )
    FORECAST_UTILS_AVAILABLE = True
except ImportError:
    FORECAST_UTILS_AVAILABLE = False

# Import report generator
try:
    from report_generator import ReportGenerator, ensure_required_reports
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False

try:
    import tensorflow as tf
    # Disable GPU to avoid CUDA errors
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Advanced feature processing
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_engineering'))
    from feature_processor import FeatureProcessor
    FEATURE_PROCESSOR_AVAILABLE = True
except ImportError:
    FEATURE_PROCESSOR_AVAILABLE = False

# Data augmentation
try:
    from data_augmentor import DataAugmentor
    DATA_AUGMENTOR_AVAILABLE = True
except ImportError:
    DATA_AUGMENTOR_AVAILABLE = False

# Feature pruner for leakage-free pruning
try:
    from feature_pruner import FeaturePruner
    FEATURE_PRUNER_AVAILABLE = True
except ImportError:
    FEATURE_PRUNER_AVAILABLE = False

# Advanced models
try:
    from model_zoo import (
        LightGBMMultiOutput, TCN, NBeatsLite, 
        MixtureOfExperts, MultiTaskModel, train_pytorch_model,
        xLSTMRegressor, xLSTMRegressorUnified, get_device, XLSTM_AVAILABLE as MODEL_ZOO_XLSTM,
        create_model, DNNRegressor, LSTMRegressor, MODEL_REGISTRY
    )
    MODEL_ZOO_AVAILABLE = True
except ImportError:
    MODEL_ZOO_AVAILABLE = False
    MODEL_ZOO_XLSTM = False

# xLSTM - Extended LSTM (parallelizable, better performance than traditional LSTM)
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False

# Configure logger with colored console output
logger = logging.getLogger('data_science')
logger.setLevel(logging.DEBUG)
logger.handlers = []  # Clear any existing handlers

# Console handler with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for warnings/errors."""
    COLORS = {
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'DEBUG': '\033[90m',    # Gray
        'INFO': '',             # Default
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET'] if color else ''
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(console_handler)

warnings.filterwarnings('ignore')

# Suppress matplotlib GUI warnings
import matplotlib
matplotlib.use('Agg')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)


class DataScientist:
    """Main class for time series classification and regression."""
    
    VALID_CLASSIFIERS = ['dnn', 'svc', 'random_forest']
    # All available regressors (availability checked at runtime)
    # Core: lstm, xlstm, dnn, krr, linear
    # Model Zoo (requires MODEL_ZOO_AVAILABLE): lightgbm, tcn, nbeats, moe, multitask
    VALID_REGRESSORS = ['lstm', 'xlstm', 'dnn', 'krr', 'linear', 'lightgbm', 'tcn', 'nbeats', 'moe', 'multitask']
    TREND_LABELS = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
    TREND_COLORS = {0: '#FF6B6B', 1: '#808080', 2: '#51CF66'}  # Red, Gray, Green
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize DataScientist with features and configuration.
        
        Args:
            data: DataFrame with features (must include 'Close')
            config: Configuration dict with keys:
                - classifier_models: List of classifier names
                - regressor_models: List of regressor names
                - trend_window: Days for trend classification (default: 5)
                - forecast_horizon: Days to forecast (default: 30)
                - test_size: Train/test split ratio (default: 0.2)
                - random_state: Random seed (default: 42)
                - output_folder: Output directory (default: 'out')
        """
        if config is None:
            config = {}
        
        self.data = data.copy()
        self.config = config
        
        # Extract original Close prices BEFORE scaling happens in FeatureEngineering
        # The Close column in features is typically scaled, so we need to recover original prices
        # by using the relationship: original = scaled * std + mean
        if 'Close' in self.data.columns:
            close_values = self.data['Close'].values
            # Detect if Close is already scaled (values typically in range [-3, 3] for StandardScaler)
            close_min, close_max = np.nanmin(close_values), np.nanmax(close_values)
            close_range = close_max - close_min
            
            # If Close appears to be in scaled range (small range), we need to recover original
            # For now, store what we have - will handle in train_regressors
            self.close_scaled = close_values
            
            # Estimate original close range from data if possible
            # We'll store statistics to recover original scale
            self.close_min_scaled = close_min
            self.close_max_scaled = close_max
        
        # Model selection (None = not specified, [] = explicitly empty)
        self.classifier_models = config.get('classifier_models', None)
        self.regressor_models = config.get('regressor_models', None)
        
        # Parameters
        self.trend_window = config.get('trend_window', 5)
        self.forecast_horizon = config.get('forecast_horizon', 30)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.output_base = config.get('output_folder', 'out')
        
        # Rolling holdout parameters (NEW)
        self.holdout_agg_method = config.get('holdout_agg', 'mean')
        self.holdout_stride = config.get('holdout_stride', 1)
        
        # 10-day trend assessment parameters (NEW)
        self.trend_horizon = config.get('trend_horizon', 10)
        self.trend_threshold = config.get('trend_threshold', 0.02)
        
        # Candle/interval information for report labels
        self.candle_length = config.get('candle_length', '1d')
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(self.random_state)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_base, f"models_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup file logging to output_dir/log.txt
        self._setup_file_logging()
        
        # Validate data
        if self.data.empty:
            raise ValueError("Data is empty")
        if 'Close' not in self.data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Storage
        self.classifier_results = {}
        self.regressor_results = {}
        self.scalers = {}
        self.y_scalers = {}  # Separate scalers for Close prices (targets)
        self.predictions = {}
        self.models = {}  # Store trained models
        self.feature_processor = None  # Advanced feature processor (PCA/AE)
        
        # Initialize data augmentor
        self.augmentor = None
        augment_config = config.get('data_augmentation', {})
        if DATA_AUGMENTOR_AVAILABLE and augment_config.get('enabled', False):
            self.augmentor = DataAugmentor(augment_config)
            logger.info(f"Data augmentation enabled: {self.augmentor.get_augmentation_summary()}")
        elif augment_config.get('enabled', False) and not DATA_AUGMENTOR_AVAILABLE:
            logger.warning("Data augmentation requested but DataAugmentor not available")
        
        # Store original y values for inverse transform
        self.y_test_original = {}
        self.y_train_original = {}
        self.X_test_indices = {}  # Store indices for visualization context
        
        logger.info(f"DataScientist initialized")
        logger.info(f"  Data shape: {self.data.shape}")
        logger.info(f"  Classifiers: {self.classifier_models}")
        logger.info(f"  Regressors: {self.regressor_models}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    # ==================== UTILITY METHODS ====================
    
    def _get_model_config(self, model_name: str) -> dict:
        """Get configuration for a model from unified config structure.
        
        Checks in order: model_hyperparameters, model_zoo, advanced_models (legacy)
        """
        # Check model_hyperparameters first (for core models like lstm, dnn, krr, etc.)
        hp_key = f'regressor_{model_name}'
        hp_config = self.config.get('model_hyperparameters', {}).get(hp_key, {})
        if hp_config:
            return hp_config
        
        # Check model_zoo (new unified location for model zoo models)
        zoo_config = self.config.get('model_zoo', {}).get(model_name, {})
        if zoo_config:
            # Return params if present, otherwise the whole config
            return zoo_config.get('params', zoo_config)
        
        # Check advanced_models for backward compatibility
        adv_config = self.config.get('advanced_models', {}).get(model_name, {})
        if adv_config:
            return adv_config.get('params', adv_config)
        
        return {}
    
    def _setup_file_logging(self):
        """Setup file logging to output_dir/log.txt with ANSI colors preserved."""
        log_file = os.path.join(self.output_dir, 'log.txt')
        
        # File handler with ANSI color codes (for viewing in terminal with 'cat' or 'less -R')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Use the same colored formatter as console
        class ColoredFileFormatter(logging.Formatter):
            """Formatter with ANSI colors for file output."""
            COLORS = {
                'WARNING': '\033[93m',  # Yellow
                'ERROR': '\033[91m',    # Red
                'CRITICAL': '\033[91m\033[1m',  # Bold Red
                'DEBUG': '\033[90m',    # Gray
                'INFO': '',             # Default
                'RESET': '\033[0m'
            }
            
            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                reset = self.COLORS['RESET'] if color else ''
                # Format with timestamp for file
                timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
                return f"{timestamp} {color}{record.levelname}:{record.name}: {record.getMessage()}{reset}"
        
        file_handler.setFormatter(ColoredFileFormatter())
        logger.addHandler(file_handler)
        
        # Store reference for cleanup
        self._file_handler = file_handler
        self._log_file = log_file
        
        logger.info(f"Logging to file: {log_file}")
    
    def _clean_data(self, X: np.ndarray, y: np.ndarray = None, method: str = 'drop') -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Clean data by handling NaN and Inf values."""
        # Replace inf with nan
        X = np.where(np.isinf(X), np.nan, X)
        
        if method == 'drop':
            # Create mask for rows with NaN in X
            valid_mask = ~np.any(np.isnan(X), axis=1)
            if y is not None:
                valid_mask &= ~np.isnan(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask] if y is not None else None
        else:  # method == 'impute'
            imputer = SimpleImputer(strategy='mean')
            X_clean = imputer.fit_transform(X)
            y_clean = y
        
        return (X_clean, y_clean) if y is not None else X_clean
    
    def _save_model(self, model, model_name: str, model_type: str):
        """Save trained model to disk."""
        try:
            if model_type == 'classifier':
                suffix = 'clf'
            else:
                suffix = 'reg'
            
            # Handle TensorFlow/Keras models
            if TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
                model_path = os.path.join(self.output_dir, f"{model_name}_{suffix}.h5")
                model.save(model_path)
            # Handle PyTorch models
            elif hasattr(model, 'state_dict'):
                import torch
                model_path = os.path.join(self.output_dir, f"{model_name}_{suffix}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                }, model_path)
            else:
                # Use pickle for sklearn models
                model_path = os.path.join(self.output_dir, f"{model_name}_{suffix}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"  Saved model: {model_path}")
            self.models[f"{model_name}_{suffix}"] = model_path
        except Exception as e:
            logger.warning(f"  Could not save {model_name}: {str(e)}")
    
    def _get_period_unit(self) -> str:
        """Get human-readable unit for the data interval (days, hours, 5m candles, etc.)."""
        interval_units = {
            '1m': '1m candles', '2m': '2m candles', '5m': '5m candles',
            '15m': '15m candles', '30m': '30m candles', '60m': '1h candles', '90m': '90m candles',
            '1h': 'hours', '1d': 'days', '5d': '5-day periods',
            '1wk': 'weeks', '1mo': 'months', '3mo': 'quarters'
        }
        return interval_units.get(self.candle_length, 'periods')

    def _predict_with_model(self, model, model_name: str, X_input: np.ndarray, config: dict = None) -> np.ndarray:
        """
        Unified prediction method handling different model backends (Sklearn, Keras, PyTorch).
        Handles input reshaping for RNNs and device management for PyTorch.
        """
        try:
            config = config or {}
            
            # Handle LSTM/xLSTM 3D input requirements
            if model_name in ['lstm', 'xlstm']:
                is_xlstm = (model_name == 'xlstm')
                model_config = config.get(f'{model_name}_config')
                
                if model_config:
                    timesteps = model_config['timesteps']
                    features_per_step = model_config['features_per_step']
                    total_needed = timesteps * features_per_step
                    n_features = X_input.shape[1]
                    
                    if n_features < total_needed:
                        # Pad if features missing
                        X_padded = np.pad(X_input, ((0, 0), (0, total_needed - n_features)), mode='constant')
                    else:
                        X_padded = X_input[:, :total_needed]
                    
                    X_3d = X_padded.reshape((len(X_input), timesteps, features_per_step))
                else:
                    # Fallback for simple 3D reshape
                    X_3d = X_input.reshape((len(X_input), 1, -1))
                
                if is_xlstm:
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # xLSTM is PyTorch
                    model.eval()
                    with torch.no_grad():
                        X_t = torch.FloatTensor(X_3d).to(device)
                        return model(X_t).cpu().numpy()
                else:
                    # LSTM is Keras
                    return model.predict(X_3d, verbose=0)
            
            # Handle PyTorch Model Zoo models
            elif model_name in ['tcn', 'nbeats', 'moe', 'multitask']:
                import torch
                use_gpu = self.config.get('use_gpu', True)
                device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_input).to(device)
                    if model_name == 'multitask':
                        return model(X_t)[0].cpu().numpy() # Return regression part only
                    else:
                        return model(X_t).cpu().numpy()
            
            # Handle Keras/DNN models
            elif model_name in ['dnn', 'keras'] or hasattr(model, 'predict') and 'tensorflow' in str(type(model)):
                 return model.predict(X_input, verbose=0)
            
            # Handle Standard Sklearn/Other
            else:
                return model.predict(X_input)
                
        except Exception as e:
            logger.warning(f"Prediction failed for {model_name}: {e}")
            raise e

    
    def _save_training_datasets(self, 
                                 X_train_clf: np.ndarray = None, y_train_clf: np.ndarray = None,
                                 X_train_reg: np.ndarray = None, y_train_reg: np.ndarray = None,
                                 feature_cols: list = None, horizon: int = None,
                                 target_transform: str = 'price'):
        """
        Save the exact training datasets used for classifiers and regressors.
        
        This creates CSV files that show EXACTLY what features and labels are used,
        including any transformations (scaling, PCA, percentage returns, etc.).
        
        Args:
            X_train_clf: Classifier features (scaled/processed)
            y_train_clf: Classifier labels (trend: 0=DOWN, 1=SIDEWAYS, 2=UP)
            X_train_reg: Regressor features (scaled/processed)
            y_train_reg: Regressor targets (transformed: pct_change, log_return, or price)
            feature_cols: Original feature column names (before processing)
            horizon: Forecast horizon for multi-step targets
            target_transform: Target transformation method used
        """
        output_dir = os.path.join(self.output_dir, 'training_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classifier training data
        if X_train_clf is not None and y_train_clf is not None:
            try:
                n_features = X_train_clf.shape[1]
                # Generate feature column names
                # Note: classifier may use different features than regressor
                clf_feature_names = [f'feature_{i}_scaled' for i in range(n_features)]
                
                clf_df = pd.DataFrame(X_train_clf, columns=clf_feature_names)
                clf_df['target_trend'] = y_train_clf
                clf_df['target_trend_label'] = [self.TREND_LABELS.get(int(t), 'UNKNOWN') for t in y_train_clf]
                
                clf_path = os.path.join(output_dir, 'classifier_training_data.csv')
                clf_df.to_csv(clf_path, index=False)
                logger.info(f"  Saved classifier training data: {clf_path}")
                logger.info(f"    Shape: {clf_df.shape}, Trend distribution: DOWN={np.sum(y_train_clf==0)}, SIDEWAYS={np.sum(y_train_clf==1)}, UP={np.sum(y_train_clf==2)}")
            except Exception as e:
                logger.warning(f"  Could not save classifier training data: {e}")
        
        # Save regressor training data
        if X_train_reg is not None and y_train_reg is not None:
            try:
                n_features = X_train_reg.shape[1]
                # Generate feature column names
                # Note: After PCA/AE, dimensions may differ from original feature_cols
                if feature_cols and len(feature_cols) == n_features:
                    reg_feature_names = [f'{c}_scaled' for c in feature_cols]
                else:
                    # Features may have been transformed (PCA, AE, etc.)
                    reg_feature_names = [f'feature_{i}_processed' for i in range(n_features)]
                
                reg_df = pd.DataFrame(X_train_reg, columns=reg_feature_names)
                
                # Add multi-step targets with appropriate naming
                horizon = horizon or y_train_reg.shape[1] if len(y_train_reg.shape) > 1 else 1
                
                if len(y_train_reg.shape) > 1:  # Multi-step targets
                    for step in range(y_train_reg.shape[1]):
                        col_name = f'target_{target_transform}_t+{step+1}'
                        reg_df[col_name] = y_train_reg[:, step]
                else:  # Single-step target
                    reg_df[f'target_{target_transform}'] = y_train_reg
                
                reg_path = os.path.join(output_dir, 'regressor_training_data.csv')
                reg_df.to_csv(reg_path, index=False)
                logger.info(f"  Saved regressor training data: {reg_path}")
                logger.info(f"    Shape: {reg_df.shape}, Target transform: {target_transform.upper()}")
                
                # Also save a metadata file explaining the targets
                metadata = {
                    'target_transform': target_transform,
                    'horizon': horizon,
                    'n_features': n_features,
                    'n_samples': len(reg_df),
                    'target_columns': [c for c in reg_df.columns if c.startswith('target_')],
                    'note': 'Targets are TRANSFORMED values. For pct_change: 0.02 = +2% return. For log_return: log(future/current).'
                }
                if target_transform == 'pct_change':
                    metadata['interpretation'] = 'target_pct_change_t+k = (Close_{t+k} - Close_t) / Close_t'
                elif target_transform == 'log_return':
                    metadata['interpretation'] = 'target_log_return_t+k = log(Close_{t+k} / Close_t)'
                else:
                    metadata['interpretation'] = 'target_price_t+k = Close_{t+k} (raw price)'
                
                meta_path = os.path.join(output_dir, 'regressor_metadata.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"  Could not save regressor training data: {e}")

    def save_feature_processor(self, processor):
        """Save fitted FeatureProcessor to disk."""
        try:
            if not processor or not processor.fitted:
                return

            path = os.path.join(self.output_dir, 'feature_processor.pkl')
            with open(path, 'wb') as f:
                pickle.dump(processor, f)
            logger.info(f"  Saved FeatureProcessor: {path}")
        except Exception as e:
            logger.warning(f"  Could not save FeatureProcessor: {e}")

    
    def _inverse_transform_y(self, y_scaled: np.ndarray, scaler_key: str = 'regressor_y') -> np.ndarray:
        """
        Inverse transform scaled y values back to original price range.
        
        Args:
            y_scaled: Scaled target values (shape: (n,) or (n, 1))
            scaler_key: Key to access the correct y scaler
        
        Returns:
            Original scale values
        """
        if scaler_key not in self.y_scalers:
            logger.warning(f"No scaler found for {scaler_key}, returning original values")
            return y_scaled
        
        scaler = self.y_scalers[scaler_key]
        
        # Reshape if needed
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        y_original = scaler.inverse_transform(y_scaled)
        return y_original.flatten()
    
    def _scale_y(self, y: np.ndarray, fit: bool = True, scaler_key: str = 'regressor_y') -> np.ndarray:
        """
        Scale y values using StandardScaler.
        
        Args:
            y: Original target values
            fit: If True, fit the scaler; if False, only transform
            scaler_key: Key to store/retrieve the scaler
        
        Returns:
            Scaled values
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if fit:
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y)
            self.y_scalers[scaler_key] = scaler
        else:
            if scaler_key not in self.y_scalers:
                logger.warning(f"No fitted scaler for {scaler_key}, fitting new scaler")
                scaler = StandardScaler()
                y_scaled = scaler.fit_transform(y)
                self.y_scalers[scaler_key] = scaler
            else:
                scaler = self.y_scalers[scaler_key]
                y_scaled = scaler.transform(y)
        
        return y_scaled.flatten()
    
    def _plot_to_base64(self) -> str:
        """Convert current matplotlib figure to base64 string."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        return image_base64
    
    def _recursive_forecast(self, model, close_history: np.ndarray, 
                            feature_scaler: StandardScaler, y_scaler: StandardScaler,
                            horizon: int, lags: List[int] = None, 
                            windows: List[int] = None,
                            model_type: str = 'sklearn') -> np.ndarray:
        """
        TRUE recursive multi-step ahead forecasting.
        
        Day 1: Uses actual Close history to create features → predict Close_t+1
        Day 2: Uses actual history + Day 1 prediction → predict Close_t+2
        Day N: Uses actual history + all prior predictions → predict Close_t+N
        
        This is the correct approach for multi-step price forecasting where
        each prediction depends on previous predictions.
        
        Args:
            model: Trained regression model
            close_history: Array of historical Close prices (unscaled, actual prices)
            feature_scaler: Fitted StandardScaler for X features
            y_scaler: Fitted StandardScaler for y (Close prices)
            horizon: Number of days to forecast
            lags: Lag periods used in feature engineering (default: [1, 5, 10, 20])
            windows: Rolling window sizes (default: [5, 20])
            model_type: 'sklearn', 'keras', or 'lstm'
        
        Returns:
            Array of predicted prices (unscaled, actual price values)
        """
        if lags is None:
            lags = [1, 5, 10, 20]
        if windows is None:
            windows = [5, 20]
        
        # Start with actual price history (make a copy to extend)
        prices = list(close_history.copy())
        predictions = []
        
        logger.info(f"  Recursive forecast: {horizon} steps ahead")
        logger.info(f"  Starting from last price: ${prices[-1]:.2f}")
        
        for step in range(1, horizon + 1):
            # Build features from current price history (including prior predictions)
            features = self._build_features_from_prices(
                np.array(prices), lags=lags, windows=windows
            )
            
            # Scale features using the training scaler
            features_scaled = feature_scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            if model_type == 'lstm':
                # LSTM needs 3D input: (samples, timesteps, features)
                features_3d = features_scaled.reshape(1, 1, -1)
                pred_scaled = model.predict(features_3d, verbose=0).flatten()[0]
            elif model_type == 'keras':
                pred_scaled = model.predict(features_scaled, verbose=0).flatten()[0]
            else:
                # sklearn models
                pred_scaled = model.predict(features_scaled)[0]
            
            # Inverse transform to get actual price
            pred_price = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
            
            # Sanity check: price should be reasonable (not negative, not >10x history)
            last_actual = close_history[-1]
            if pred_price < 0:
                logger.warning(f"  Step {step}: Negative price ${pred_price:.2f} → clipped to ${last_actual * 0.5:.2f}")
                pred_price = last_actual * 0.5  # Floor at 50% of last price
            elif pred_price > last_actual * 2:
                logger.warning(f"  Step {step}: Price spike ${pred_price:.2f} → clipped to ${last_actual * 1.2:.2f}")
                pred_price = last_actual * 1.2  # Cap at 20% above last price
            
            predictions.append(pred_price)
            
            # CRITICAL: Add prediction to price history for next step's features
            prices.append(pred_price)
        
        logger.info(f"  Recursive forecast complete: ${predictions[0]:.2f} → ${predictions[-1]:.2f}")
        return np.array(predictions)
    
    def _build_features_from_prices(self, prices: np.ndarray, 
                                     lags: List[int] = None,
                                     windows: List[int] = None) -> np.ndarray:
        """
        Build feature vector from price history (same structure as training).
        
        CRITICAL: All features are LAGGED by 1 day to match training data.
        At time t, we only use information available up to t-1.
        
        Features (same order as create_close_only_features with lag=1):
        1. Close_lag_1, Close_lag_5, Close_lag_10 (prices at t-1, t-5, t-10)
        2. Close_return_1, Close_return_5, Close_return_10 (returns lagged by 1)
        3. Close_log_return (lagged by 1)
        4. Close_rolling_5_mean, Close_rolling_5_std, Close_ma_ratio_5 (lagged by 1)
        5. Close_rolling_20_mean, Close_rolling_20_std, Close_ma_ratio_20 (lagged by 1)
        6. Close_ROC_10, Close_momentum_10, Close_acceleration (lagged by 1)
        
        Args:
            prices: Array of price history (most recent is last)
            lags: Lag periods for the original features
            windows: Rolling window sizes
        
        Returns:
            Feature vector (1D array) - 16 features
        """
        if lags is None:
            lags = [1, 5, 10]  # Match training: 3 lags not 4
        if windows is None:
            windows = [5, 20]
        
        features = []
        # Use price at t-1 as reference (all features are lagged by 1)
        last_price = prices[-1]  # This is the price we want to predict
        ref_price = prices[-2] if len(prices) > 1 else prices[-1]  # t-1 price
        
        # 1. LAG FEATURES: Close_lag_1, Close_lag_5, Close_lag_10
        # Close_lag_k at time t = Close at time t-k
        for lag in lags:
            if len(prices) > lag:
                features.append(prices[-1 - lag])
            else:
                features.append(prices[0])  # Use oldest available
        
        # 2. RETURN FEATURES (lagged by 1 day)
        # Close_return_k at time t = return from t-k-1 to t-1
        for lag in [1, 5, 10]:
            idx_start = -1 - lag - 1  # t - lag - 1
            idx_end = -1 - 1  # t - 1
            if len(prices) > lag + 1 and prices[idx_start] != 0:
                ret = (prices[idx_end] - prices[idx_start]) / prices[idx_start]
            else:
                ret = 0.0
            features.append(ret)
        
        # 3. LOG RETURN (lagged by 1 day)
        # log_return at t = log(price_{t-1} / price_{t-2})
        if len(prices) > 2 and prices[-3] > 0:
            log_ret = np.log(prices[-2] / prices[-3])
        else:
            log_ret = 0.0
        features.append(log_ret)
        
        # 4. ROLLING WINDOW FEATURES (lagged by 1 day)
        # Rolling features at t use prices up to t-1
        for window in windows:
            # Get window prices ENDING at t-1 (not including current price)
            end_idx = -1  # up to t-1 (exclusive of last element which is t)
            start_idx = max(0, len(prices) - 1 - window)
            window_prices = prices[start_idx:-1] if len(prices) > 1 else prices
            
            # Rolling mean (lagged)
            rolling_mean = np.mean(window_prices) if len(window_prices) > 0 else ref_price
            features.append(rolling_mean)
            
            # Rolling std (lagged)
            rolling_std = np.std(window_prices) if len(window_prices) > 1 else 0.0
            features.append(rolling_std)
            
            # MA ratio (lagged): (price_{t-1} - MA_{t-1}) / MA_{t-1}
            if rolling_mean != 0:
                ma_ratio = (ref_price - rolling_mean) / rolling_mean
            else:
                ma_ratio = 0.0
            features.append(ma_ratio)
        
        # 5. TECHNICAL INDICATORS (all lagged by 1 day)
        # ROC_10 at t: return from t-12 to t-2 (lagged version)
        if len(prices) > 12 and prices[-13] != 0:
            roc = ((prices[-3] - prices[-13]) / prices[-13]) * 100
        else:
            roc = 0.0
        features.append(roc)
        
        # Momentum_10 at t: price_{t-2} - price_{t-12} (lagged version)
        if len(prices) > 12:
            momentum = prices[-3] - prices[-13]
        else:
            momentum = 0.0
        features.append(momentum)
        
        # Acceleration (second derivative of returns, already uses lagged returns)
        if len(prices) > 3:
            ret_t1 = (prices[-2] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
            ret_t2 = (prices[-3] - prices[-4]) / prices[-4] if prices[-4] != 0 else 0
            acceleration = ret_t1 - ret_t2
        else:
            acceleration = 0.0
        features.append(acceleration)
        
        return np.array(features)
    
    def _sanity_check_features(self, feature_cols: List[str], target_col: str = 'Close') -> Tuple[bool, List[str]]:
        """
        Perform expert-level sanity checks on feature set.
        
        Checks:
        1. Target column (Close) is NOT in feature columns
        2. No "future" columns (e.g., Close_t+1) in features
        3. All features are backward-looking (lag, rolling use past data)
        4. Feature names follow expected pattern
        
        Args:
            feature_cols: List of feature column names
            target_col: Target column name (default: 'Close')
        
        Returns:
            Tuple of (passed: bool, warnings: List[str])
        """
        warnings = []
        passed = True
        
        logger.info("=" * 60)
        logger.info("SANITY CHECK: Feature Set Validation")
        logger.info("=" * 60)
        
        # Check 1: Target not in features
        if target_col in feature_cols:
            warnings.append(f"CRITICAL: Target '{target_col}' found in feature columns - DATA LEAKAGE!")
            passed = False
            logger.error(f"  ✗ FAIL: {target_col} in features (leakage)")
        else:
            logger.info(f"  ✓ PASS: Target '{target_col}' not in features")
        
        # Check 2: No future-looking columns
        future_patterns = ['_t+', '_future', '_next', '_ahead', '_forward']
        future_cols = [c for c in feature_cols if any(p in c.lower() for p in future_patterns)]
        if future_cols:
            warnings.append(f"CRITICAL: Future-looking columns found: {future_cols}")
            passed = False
            logger.error(f"  ✗ FAIL: Future columns found: {future_cols}")
        else:
            logger.info(f"  ✓ PASS: No future-looking columns")
        
        # Check 3: Verify lag features are properly named
        lag_cols = [c for c in feature_cols if '_lag_' in c]
        if lag_cols:
            logger.info(f"  ✓ INFO: {len(lag_cols)} lag features found (backward-looking)")
        
        # Check 4: Verify rolling features don't use future (center=False is default)
        rolling_cols = [c for c in feature_cols if '_rolling_' in c]
        if rolling_cols:
            logger.info(f"  ✓ INFO: {len(rolling_cols)} rolling features found")
        
        # Check 5: Warn if too few features
        if len(feature_cols) < 5:
            warnings.append(f"WARNING: Only {len(feature_cols)} features - may underfit")
            logger.warning(f"  ⚠ WARN: Only {len(feature_cols)} features")
        
        # Check 6: Warn if Close-derived features exist but target is Close
        close_derived = [c for c in feature_cols if c.startswith('Close_')]
        if close_derived and target_col == 'Close':
            logger.info(f"  ✓ INFO: {len(close_derived)} Close-derived features (valid for Close prediction)")
        
        # Summary
        if passed:
            logger.info("  ✓ ALL SANITY CHECKS PASSED")
        else:
            logger.error("  ✗ SANITY CHECKS FAILED - See warnings above")
        
        logger.info("=" * 60)
        
        return passed, warnings
    
    def _get_close_only_feature_columns(self) -> List[str]:
        """
        Get feature column names for Close-only prediction.
        
        These are the columns created by FeatureEngineering.create_close_only_features().
        Order must match _build_features_from_prices().
        
        Returns:
            List of feature column names in correct order
        """
        lags = [1, 5, 10, 20]
        windows = [5, 20]
        
        features = []
        
        # 1. Lag features
        for lag in lags:
            features.append(f"Close_lag_{lag}")
        
        # 2. Return features
        for lag in [1, 5, 10]:
            features.append(f"Close_return_{lag}")
        
        # 3. Log return
        features.append("Close_log_return")
        
        # 4. Rolling features
        for window in windows:
            features.append(f"Close_rolling_{window}_mean")
            features.append(f"Close_rolling_{window}_std")
            features.append(f"Close_ma_ratio_{window}")
        
        # 5. Technical indicators
        features.append("Close_ROC_10")
        features.append("Close_momentum_10")
        features.append("Close_acceleration")
        
        return features
    
    def _time_series_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform proper time series train/test split (NO SHUFFLING).
        Test set is always the last portion of the data.
        
        Args:
            X: Feature matrix
            y: Target array
            test_size: Fraction of data for testing
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"  Time series split: train={len(X_train)}, test={len(X_test)}")
        logger.info(f"  Train indices: 0-{split_idx-1}, Test indices: {split_idx}-{n_samples-1}")
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_features_excluding_target(self, target_col: str = 'Close') -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix EXCLUDING the target column and any directly derived features.
        This prevents data leakage from target being in features.
        
        Args:
            target_col: Target column name to exclude
        
        Returns:
            Feature matrix and list of feature names
        """
        # Get all numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude: target column, columns that contain target name (lag/rolling of target)
        # CRITICAL: We must exclude Close and all Close-derived features for regression
        exclude_patterns = [target_col]
        
        # Also exclude lag and rolling features of the target to prevent leakage
        # (lag features are valid for features, but Close_lag_1 predicting Close is leakage)
        feature_cols = []
        for col in numeric_cols:
            # Skip the exact target column
            if col == target_col:
                continue
            # Keep the column if it's not a lag/rolling of the target
            # Note: We DO keep lag/rolling features of OTHER columns (like Volume)
            feature_cols.append(col)
        
        X = self.data[feature_cols].values
        logger.info(f"  Features prepared: {len(feature_cols)} columns (excluded target: {target_col})")
        
        return X, feature_cols
    
    def _compute_naive_baseline(self, y_test: np.ndarray) -> Dict:
        """
        Compute naive baseline metrics (last value predictor / random walk).
        This baseline assumes tomorrow's price = today's price.
        
        Args:
            y_test: Actual test values
        
        Returns:
            Dictionary with baseline metrics
        """
        # Naive prediction: previous value (shift by 1)
        y_pred_naive = np.roll(y_test, 1)
        y_pred_naive[0] = y_test[0]  # First prediction = first actual
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
        mae = mean_absolute_error(y_test, y_pred_naive)
        r2 = r2_score(y_test, y_pred_naive)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'name': 'Naive (Last Value)'
        }
    
    def _validate_predictions(self, y_pred: np.ndarray, y_true: np.ndarray, model_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Validate predictions for reasonableness and log warnings.
        
        Args:
            y_pred: Predicted values
            y_true: Actual values
            model_name: Name of the model
        
        Returns:
            Validated predictions and list of warnings
        """
        warnings_list = []
        
        # Check for negative prices in price-space
        if np.any(y_pred < 0):
            neg_count = np.sum(y_pred < 0)
            warnings_list.append(f"WARNING: {model_name} produced {neg_count} negative price predictions")
            logger.warning(f"  {model_name}: {neg_count} negative predictions detected")
        
        # Check for unrealistic predictions (>50% change from mean)
        mean_true = np.mean(y_true)
        unrealistic_mask = np.abs(y_pred - mean_true) > 0.5 * mean_true
        if np.any(unrealistic_mask):
            unrealistic_count = np.sum(unrealistic_mask)
            warnings_list.append(f"WARNING: {model_name} has {unrealistic_count} predictions >50% from mean")
        
        # Check for suspiciously perfect metrics
        r2 = r2_score(y_true, y_pred)
        if r2 > 0.999:
            warnings_list.append(f"WARNING: {model_name} has R²={r2:.6f} - possible data leakage!")
            logger.warning(f"  {model_name}: R²={r2:.6f} is suspiciously high - check for data leakage")
        
        return y_pred, warnings_list
    
    # ==================== TREND CLASSIFICATION ====================
    
    def create_trend_labels(self, data: pd.DataFrame = None, window: int = None) -> np.ndarray:
        """
        Create trend labels based on FUTURE price movement: UP (2), SIDEWAYS (1), DOWN (0).
        
        This creates labels based on what WILL happen in the next `window` days,
        which is correct for classification (predict future trend from current features).
        
        Note: The last `window` samples will have NaN labels and should be excluded.
        
        Args:
            data: DataFrame with 'Close' column
            window: Window size for trend calculation (days to look ahead)
        
        Returns:
            Array of trend labels (last `window` entries are 1=SIDEWAYS as placeholder)
        """
        if data is None:
            data = self.data
        if window is None:
            window = self.trend_window
        
        # Calculate trend as percentage change FORWARD (what will happen)
        close_prices = data['Close'].values
        n = len(close_prices)
        trends = np.ones(n)  # Default to SIDEWAYS
        
        # For each point, look FORWARD to determine trend
        for i in range(n - window):
            if close_prices[i] == 0:
                trends[i] = 1  # SIDEWAYS if division by zero
            else:
                # Future price change: (price in `window` days) / (current price) - 1
                future_price = close_prices[i + window]
                current_price = close_prices[i]
                price_change_pct = (future_price - current_price) / current_price
                
                if price_change_pct > 0.02:  # Up threshold: 2%
                    trends[i] = 2  # UP
                elif price_change_pct < -0.02:  # Down threshold: -2%
                    trends[i] = 0  # DOWN
                else:
                    trends[i] = 1  # SIDEWAYS
        
        # Last `window` entries have no future data, mark as SIDEWAYS (will be excluded in training)
        logger.info(f"  Created trend labels: {np.sum(trends == 0)} DOWN, {np.sum(trends == 1)} SIDEWAYS, {np.sum(trends == 2)} UP")
        logger.info(f"  Note: Last {window} samples have no future data (set to SIDEWAYS)")
        
        return trends.astype(int)
    
    # ==================== CLASSIFIERS ====================
    
    def train_classifier_dnn(self, X_train, X_test, y_train, y_test, model_name='dnn'):
        """Train Deep Neural Network classifier with improved architecture and regularization."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for DNN classifier")
            return None
        
        logger.info(f"Training {model_name.upper()} classifier...")
        
        try:
            # Load hyperparameters from config (with defaults)
            hp = self.config.get('model_hyperparameters', {}).get('classifier_dnn', {})
            layer_sizes = hp.get('layers', [128, 64, 32])
            dropouts = hp.get('dropout', [0.4, 0.3, 0.2])
            activation = hp.get('activation', 'relu')
            l2_reg = hp.get('l2_reg', 0.001)
            learning_rate = hp.get('learning_rate', 0.001)
            epochs = hp.get('epochs', 100)
            batch_size = hp.get('batch_size', 16)
            patience = hp.get('early_stopping_patience', 10)
            
            # Ensure dropouts list matches layers
            if len(dropouts) < len(layer_sizes):
                dropouts = dropouts + [0.2] * (len(layer_sizes) - len(dropouts))
            
            # Build model dynamically from config
            model = keras.Sequential()
            for i, (units, dropout) in enumerate(zip(layer_sizes, dropouts)):
                if i == 0:
                    model.add(layers.Dense(units, activation=activation, input_shape=(X_train.shape[1],),
                                          kernel_regularizer=keras.regularizers.l2(l2_reg)))
                else:
                    model.add(layers.Dense(units, activation=activation,
                                          kernel_regularizer=keras.regularizers.l2(l2_reg)))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(dropout))
            
            # Output layer for 3 classes: DOWN, SIDEWAYS, UP
            model.add(layers.Dense(3, activation='softmax'))
            
            # Improved optimizer with learning rate scheduler
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping callback
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            
            # Learning rate scheduler
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(patience // 2, 3),
                min_lr=0.00001,
                verbose=1
            )
            
            # Train with callbacks
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stop, lr_scheduler],
                verbose=1
            )
            
            # Evaluate
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_pred_proba = model.predict(X_test, verbose=0)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"  Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'cm': confusion_matrix(y_test, y_pred),
                'y_test': y_test
            }
        except Exception as e:
            logger.warning(f"  dnn failed: {str(e)}")
            return None
    
    def train_classifier_svc(self, X_train, X_test, y_train, y_test, model_name='svc'):
        """Train Support Vector Classifier with optimized hyperparameters."""
        logger.info(f"Training {model_name.upper()} classifier...")
        
        try:
            # Load hyperparameters from config (with defaults)
            hp = self.config.get('model_hyperparameters', {}).get('classifier_svc', {})
            use_grid_search = hp.get('use_grid_search', True)
            probability = hp.get('probability', True)
            
            if use_grid_search:
                # GridSearchCV for hyperparameter tuning
                gs_params = hp.get('grid_search', {})
                param_grid = {
                    'C': gs_params.get('C', [0.1, 1, 10, 100]),
                    'gamma': gs_params.get('gamma', ['scale', 'auto', 0.001, 0.01]),
                    'kernel': gs_params.get('kernel', ['rbf', 'linear'])
                }
                cv_folds = gs_params.get('cv_folds', 5)
                
                svc = SVC(probability=probability, random_state=self.random_state)
                grid_search = GridSearchCV(svc, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                logger.info(f"  Best SVC parameters: {grid_search.best_params_}")
            else:
                # Use fixed parameters from config
                fixed_params = hp.get('fixed', {})
                model = SVC(
                    C=fixed_params.get('C', 1.0),
                    gamma=fixed_params.get('gamma', 'scale'),
                    kernel=fixed_params.get('kernel', 'rbf'),
                    probability=probability,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
                logger.info(f"  SVC trained with fixed params: C={model.C}, gamma={model.gamma}, kernel={model.kernel}")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if probability else None
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"  Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
            
            return {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'cm': confusion_matrix(y_test, y_pred),
                'y_test': y_test
            }
        except Exception as e:
            logger.warning(f"  svc failed: {str(e)}")
            return None
    
    def train_classifier_random_forest(self, X_train, X_test, y_train, y_test, model_name='random_forest'):
        """Train Random Forest Classifier with tuned hyperparameters."""
        logger.info(f"Training {model_name.upper()} classifier...")
        
        try:
            # Load hyperparameters from config (with defaults)
            hp = self.config.get('model_hyperparameters', {}).get('classifier_random_forest', {})
            use_random_search = hp.get('use_random_search', True)
            
            if use_random_search:
                # RandomizedSearchCV for hyperparameter search
                rs_params = hp.get('random_search', {})
                param_dist = {
                    'n_estimators': rs_params.get('n_estimators', [100, 200, 300, 400, 500]),
                    'max_depth': rs_params.get('max_depth', [10, 15, 20, 25, 30, None]),
                    'min_samples_split': rs_params.get('min_samples_split', [2, 5, 10]),
                    'min_samples_leaf': rs_params.get('min_samples_leaf', [1, 2, 4]),
                    'max_features': rs_params.get('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': rs_params.get('bootstrap', [True, False])
                }
                n_iter = rs_params.get('n_iter', 50)
                cv_folds = rs_params.get('cv_folds', 5)
                
                rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
                random_search = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=cv_folds, 
                                                   n_jobs=-1, verbose=1, random_state=self.random_state)
                random_search.fit(X_train, y_train)
                
                model = random_search.best_estimator_
                logger.info(f"  Best RF parameters: {random_search.best_params_}")
            else:
                # Use fixed parameters from config
                fixed_params = hp.get('fixed', {})
                model = RandomForestClassifier(
                    n_estimators=fixed_params.get('n_estimators', 200),
                    max_depth=fixed_params.get('max_depth', 20),
                    min_samples_split=fixed_params.get('min_samples_split', 5),
                    min_samples_leaf=fixed_params.get('min_samples_leaf', 2),
                    max_features=fixed_params.get('max_features', 'sqrt'),
                    bootstrap=fixed_params.get('bootstrap', True),
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                logger.info(f"  RF trained with fixed params: n_estimators={model.n_estimators}, max_depth={model.max_depth}")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"  Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
            
            return {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'cm': confusion_matrix(y_test, y_pred),
                'feature_importance': model.feature_importances_,
                'y_test': y_test
            }
        except Exception as e:
            logger.warning(f"  random_forest failed: {str(e)}")
            return None
    
    # ==================== MULTI-OUTPUT REGRESSORS ====================
    
    def train_multioutput_linear(self, X_train, X_test, y_train, y_test, model_name='linear'):
        """Train Multi-Output Linear Regression for multi-step forecasting."""
        logger.info(f"Training {model_name.upper()} multi-output regressor...")
        
        try:
            # Load hyperparameters from config (with defaults)
            hp = self.config.get('model_hyperparameters', {}).get('regressor_linear', {})
            fit_intercept = hp.get('fit_intercept', True)
            use_regularization = hp.get('use_regularization', False)
            
            if use_regularization:
                # Use ElasticNet for regularized linear regression
                from sklearn.linear_model import ElasticNet
                from sklearn.multioutput import MultiOutputRegressor
                elasticnet_params = hp.get('elasticnet', {})
                base_model = ElasticNet(
                    alpha=elasticnet_params.get('alpha', 1.0),
                    l1_ratio=elasticnet_params.get('l1_ratio', 0.5),
                    max_iter=elasticnet_params.get('max_iter', 1000),
                    fit_intercept=fit_intercept,
                    random_state=self.random_state
                )
                model = MultiOutputRegressor(base_model, n_jobs=-1)
                logger.info(f"  Using ElasticNet regularization: alpha={base_model.alpha}, l1_ratio={base_model.l1_ratio}")
            else:
                model = LinearRegression(fit_intercept=fit_intercept)
            
            model.fit(X_train, y_train)  # y_train is (n_samples, horizon)
            
            y_pred = model.predict(X_test)
            
            # Per-step metrics
            horizon = y_train.shape[1]
            rmse_per_step = []
            for step in range(horizon):
                rmse_per_step.append(np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step])))
            
            rmse_avg = np.mean(rmse_per_step)
            mae_avg = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f} | Horizon: {horizon}")
            
            return {
                'model': model,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'mae_avg': mae_avg,
                'r2_avg': r2_avg,
                'rmse_per_step': rmse_per_step,
                'horizon': horizon
            }
        except Exception as e:
            logger.warning(f"  {model_name} multi-output failed: {str(e)}")
            return None
    
    def train_multioutput_krr(self, X_train, X_test, y_train, y_test, model_name='krr'):
        """Train Multi-Output Kernel Ridge Regression."""
        logger.info(f"Training {model_name.upper()} multi-output regressor...")
        
        try:
            # Load hyperparameters from config (with defaults)
            hp = self.config.get('model_hyperparameters', {}).get('regressor_krr', {})
            use_grid_search = hp.get('use_grid_search', True)
            
            if use_grid_search:
                # GridSearchCV for hyperparameter tuning
                gs_params = hp.get('grid_search', {})
                param_grid = {
                    'alpha': gs_params.get('alpha', [0.001, 0.01, 0.1, 1.0, 10.0]),
                    'gamma': gs_params.get('gamma', [0.001, 0.01, 0.1, 1.0]),
                    'kernel': gs_params.get('kernel', ['rbf', 'linear', 'polynomial'])
                }
                cv_folds = gs_params.get('cv_folds', 3)
                
                # Use GridSearchCV on a single output for hyperparameter selection
                # (full multi-output grid search would be too slow)
                from sklearn.model_selection import GridSearchCV
                base_krr = KernelRidge()
                grid_search = GridSearchCV(base_krr, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
                # Use first output for hyperparameter tuning
                grid_search.fit(X_train, y_train[:, 0])
                best_params = grid_search.best_params_
                logger.info(f"  Best KRR parameters: {best_params}")
                
                # Create multi-output regressor with best params
                base_krr = KernelRidge(**best_params)
                model = MultiOutputRegressor(base_krr, n_jobs=-1)
            else:
                # Use fixed parameters from config
                fixed_params = hp.get('fixed', {})
                base_krr = KernelRidge(
                    alpha=fixed_params.get('alpha', 1.0),
                    gamma=fixed_params.get('gamma', 0.1),
                    kernel=fixed_params.get('kernel', 'rbf')
                )
                model = MultiOutputRegressor(base_krr, n_jobs=-1)
                logger.info(f"  KRR using fixed params: alpha={base_krr.alpha}, gamma={base_krr.gamma}, kernel={base_krr.kernel}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            horizon = y_train.shape[1]
            rmse_per_step = []
            for step in range(horizon):
                rmse_per_step.append(np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step])))
            
            rmse_avg = np.mean(rmse_per_step)
            mae_avg = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f} | Horizon: {horizon}")
            
            return {
                'model': model,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'mae_avg': mae_avg,
                'r2_avg': r2_avg,
                'rmse_per_step': rmse_per_step,
                'horizon': horizon
            }
        except Exception as e:
            logger.warning(f"  {model_name} multi-output failed: {str(e)}")
            return None
    
    def train_multioutput_dnn(self, X_train, X_test, y_train, y_test, model_name='dnn'):
        """
        Train Multi-Output DNN for multi-step forecasting.
        
        Uses the UNIFIED DNNRegressor from model_zoo.py - the same architecture
        used by MoE as a DNN expert.
        
        Architecture (PyTorch):
        - Deep residual network with skip connections
        - GELU activation for smoother gradients
        - Layer normalization for stable training
        - Squeeze-and-Excitation attention for feature importance
        - Separate prediction heads for short/medium/long term horizons
        """
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available for DNN")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()} (Unified PyTorch)...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            # Get hyperparameters from config
            hp = self.config.get('model_hyperparameters', {}).get('regressor_dnn', {})
            epochs = hp.get('epochs', 300)
            batch_size = hp.get('batch_size', 32)
            learning_rate = hp.get('learning_rate', 0.001)
            patience = hp.get('early_stopping_patience', 50)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            horizon = output_dim
            
            # Create model using unified factory
            from src.modules.data_science.model_zoo import create_model
            model = create_model(
                model_type='dnn',
                input_dim=input_dim,
                output_dim=output_dim,
                hyperparameters=self.config.get('model_hyperparameters', {}),
                model_zoo_config=self.config.get('model_zoo', {})
            )
            
            # Train using standard PyTorch training loop
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                use_gpu=use_gpu,
                patience=patience,
                verbose=True  # Explicit verbose for training progress
            )
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            rmse_per_step = []
            for step in range(horizon):
                rmse_per_step.append(np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step])))
            
            rmse_avg = np.mean(rmse_per_step)
            mae_avg = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            # Log RMSE per horizon segment
            short_rmse = np.mean(rmse_per_step[:horizon//3]) if horizon >= 3 else rmse_avg
            medium_rmse = np.mean(rmse_per_step[horizon//3:2*horizon//3]) if horizon >= 3 else rmse_avg
            long_rmse = np.mean(rmse_per_step[2*horizon//3:]) if horizon >= 3 else rmse_avg
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f} | Horizon: {horizon}")
            logger.info(f"  Architecture: DNNRegressor (ResNet + SE-Attention + Multi-Head)")
            logger.info(f"  RMSE by segment: Short={short_rmse:.4f}, Medium={medium_rmse:.4f}, Long={long_rmse:.4f}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'mae_avg': mae_avg,
                'r2_avg': r2_avg,
                'rmse_per_step': rmse_per_step,
                'horizon': horizon,
                'rmse_by_segment': {
                    'short': short_rmse,
                    'medium': medium_rmse,
                    'long': long_rmse
                }
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_multioutput_lstm(self, X_train, X_test, y_train, y_test, model_name='lstm'):
        """
        Train Multi-Output LSTM for multi-step forecasting.
        
        Uses the UNIFIED LSTMRegressor from model_zoo.py - the same architecture
        used by MoE as an LSTM expert.
        
        Architecture (PyTorch):
        - Bidirectional LSTM layers for capturing temporal patterns
        - Multi-head self-attention for long-range dependencies
        - Layer normalization for stable training
        - Dense layers with BatchNorm
        - Multi-output prediction head
        """
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available for LSTM")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()} (Unified PyTorch)...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            # Get hyperparameters from config
            hp = self.config.get('model_hyperparameters', {}).get('regressor_lstm', {})
            epochs = hp.get('epochs', 200)
            batch_size = hp.get('batch_size', 32)
            learning_rate = hp.get('learning_rate', 0.002)
            patience = hp.get('early_stopping_patience', 30)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            horizon = output_dim
            
            # Create model using unified factory
            from src.modules.data_science.model_zoo import create_model
            model = create_model(
                model_type='lstm',
                input_dim=input_dim,
                output_dim=output_dim,
                hyperparameters=self.config.get('model_hyperparameters', {}),
                model_zoo_config=self.config.get('model_zoo', {})
            )
            
            # Train using standard PyTorch training loop
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                use_gpu=use_gpu,
                patience=patience,
                verbose=True  # Explicit verbose for training progress
            )
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            rmse_per_step = []
            for step in range(horizon):
                rmse_per_step.append(np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step])))
            
            rmse_avg = np.mean(rmse_per_step)
            mae_avg = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f} | Horizon: {horizon}")
            logger.info(f"  Architecture: LSTMRegressor (Bidirectional LSTM + Multi-Head Attention)")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'mae_avg': mae_avg,
                'r2_avg': r2_avg,
                'rmse_per_step': rmse_per_step,
                'horizon': horizon
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_multioutput_xlstm(self, X_train, X_test, y_train, y_test, model_name='xlstm'):
        """
        Train xLSTM (Extended LSTM) multi-output regressor.
        
        Uses the UNIFIED xLSTMRegressorUnified from model_zoo.py - the same 
        architecture used by MoE as an xLSTM expert.
        
        xLSTM provides better parallelization and performance than traditional LSTM
        through exponential gating and matrix memory (mLSTM variant).
        """
        if not XLSTM_AVAILABLE:
            logger.warning("xLSTM not available. Install with: pip install xlstm")
            return None
        
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available for xLSTM")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()} (Unified PyTorch)...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            logger.info(f"    Using device: {device}")
            
            # Get hyperparameters from config
            hp = self.config.get('model_hyperparameters', {}).get('regressor_xlstm', {})
            epochs = hp.get('epochs', 100)
            batch_size = hp.get('batch_size', 32)
            learning_rate = hp.get('learning_rate', 0.001)
            patience = hp.get('patience', 10)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            horizon = output_dim
            
            # Create model using unified factory
            from src.modules.data_science.model_zoo import create_model
            model = create_model(
                model_type='xlstm',
                input_dim=input_dim,
                output_dim=output_dim,
                hyperparameters=self.config.get('model_hyperparameters', {}),
                model_zoo_config=self.config.get('model_zoo', {})
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"    Model parameters: {trainable_params:,} trainable / {total_params:,} total")
            
            # Train using standard PyTorch training loop
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                use_gpu=use_gpu,
                patience=patience,
                verbose=True  # Explicit verbose for training progress
            )
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            # Calculate metrics
            rmse_per_step = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(horizon)]
            rmse_avg = np.mean(rmse_per_step)
            mae_avg = np.mean(np.abs(y_test - y_pred))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | MAE_avg: {mae_avg:.4f} | R²_avg: {r2_avg:.4f}")
            logger.info(f"  Architecture: xLSTMRegressorUnified (mLSTM blocks)")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'mae_avg': mae_avg,
                'r2_avg': r2_avg,
                'rmse_per_step': rmse_per_step,
                'horizon': horizon
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_lightgbm_multioutput(self, X_train, X_test, y_train, y_test, model_name='lightgbm'):
        """Train LightGBM multi-output regressor."""
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available, skipping LightGBM")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()}...")
            
            params = self._get_model_config(model_name)
            n_outputs = y_train.shape[1]
            
            model = LightGBMMultiOutput(
                n_outputs=n_outputs,
                num_leaves=params.get('num_leaves', 31),
                learning_rate=params.get('learning_rate', 0.05),
                n_estimators=params.get('n_estimators', 100)
            )
            
            model.fit(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)
            
            rmse_avg = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f}")
            
            return {
                'model': model,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'r2_avg': r2_avg,
                'feature_importance': model.get_feature_importance()
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_tcn(self, X_train, X_test, y_train, y_test, model_name='tcn'):
        """Train Temporal Convolutional Network."""
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available, skipping TCN")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()}...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            params = self._get_model_config(model_name)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            
            model = TCN(
                input_dim=input_dim,
                output_dim=output_dim,
                num_channels=params.get('num_channels', [64, 64, 32]),
                kernel_size=params.get('kernel_size', 3),
                dropout=params.get('dropout', 0.2)
            )
            
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                lr=params.get('lr', 1e-3),
                use_gpu=use_gpu,
                verbose=True  # Explicit verbose for training progress
            )
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            rmse_avg = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'r2_avg': r2_avg
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_nbeats(self, X_train, X_test, y_train, y_test, model_name='nbeats'):
        """Train N-BEATS Lite model."""
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available, skipping N-BEATS")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()}...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            params = self._get_model_config(model_name)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            
            model = NBeatsLite(
                input_dim=input_dim,
                output_dim=output_dim,
                num_blocks=params.get('num_blocks', 2),
                hidden_dim=params.get('hidden_dim', 64)
            )
            
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                lr=params.get('lr', 1e-3),
                use_gpu=use_gpu,
                verbose=True  # Explicit verbose for training progress
            )
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            rmse_avg = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'r2_avg': r2_avg
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_moe(self, X_train, X_test, y_train, y_test, model_name='moe'):
        """
        Train Mixture of Experts model with extensible expert types.
        
        Supports all model architectures from data_science.py as experts:
        - linear: Simple linear regression
        - dnn: Deep NN with residual connections + SE attention  
        - lstm: Bidirectional LSTM with multi-head attention
        - xlstm: Extended LSTM with matrix memory
        - tcn: Temporal Convolutional Network
        - nbeats: N-BEATS interpretable forecast
        
        Configuration (in configuration.yml under model_zoo.moe):
            expert_types: ['dnn', 'tcn', 'xlstm']  # Select which experts to use
            expert_configs:  # Optional per-expert overrides
                dnn:
                    layers: [512, 256, 128]
                tcn:
                    num_channels: [128, 64, 32]
            epochs: 100
            batch_size: 32
            lr: 0.001
            gating_hidden_dim: 64
        
        Each expert automatically uses hyperparameters from:
        - model_hyperparameters.regressor_* for core models
        - model_zoo.* for model zoo models
        """
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available, skipping MoE")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()} (Extensible)...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            params = self._get_model_config(model_name)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            
            # Get expert types from config (defaults to ['dnn', 'tcn'])
            expert_types = params.get('expert_types', ['dnn', 'tcn'])
            expert_configs = params.get('expert_configs', {})
            gating_hidden_dim = params.get('gating_hidden_dim', 64)
            
            # Log available experts
            from src.modules.data_science.model_zoo import MixtureOfExperts as MoE
            available = MoE.get_available_experts()
            logger.info(f"  Available experts: {available}")
            logger.info(f"  Selected experts: {expert_types}")
            
            # Get hyperparameters for experts
            hyperparameters = self.config.get('model_hyperparameters', {})
            model_zoo_config = self.config.get('model_zoo', {})
            
            # Create MoE with selected experts
            model = MixtureOfExperts(
                input_dim=input_dim,
                output_dim=output_dim,
                expert_types=expert_types,
                expert_configs=expert_configs,
                hyperparameters=hyperparameters,
                model_zoo_config=model_zoo_config,
                gating_hidden_dim=gating_hidden_dim
            )
            
            history = train_pytorch_model(
                model, X_train, y_train, X_test, y_test,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                lr=params.get('lr', 1e-3),
                use_gpu=use_gpu,
                verbose=True  # Explicit verbose for training progress
            )
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred = model(X_test_tensor).cpu().numpy()
            
            # Get average gate weights
            gate_weights = model.get_gate_weights(X_test_tensor).cpu().numpy()
            avg_gate_weights = gate_weights.mean(axis=0)
            
            rmse_avg = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            # Build gate weights log with expert names
            expert_names = model.get_expert_names()
            gate_log = ", ".join([f"{name.upper()}={avg_gate_weights[i]:.3f}" 
                                  for i, name in enumerate(expert_names)])
            
            logger.info(f"  RMSE_avg: {rmse_avg:.4f} | R²_avg: {r2_avg:.4f}")
            logger.info(f"  Gate weights: {gate_log}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'r2_avg': r2_avg,
                'gate_weights': avg_gate_weights.tolist(),
                'expert_names': expert_names
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_multitask(self, X_train, X_test, y_train, y_test, y_trend, model_name='multitask'):
        """Train Multi-Task model (regression + classification)."""
        if not MODEL_ZOO_AVAILABLE:
            logger.warning("Model zoo not available, skipping Multi-Task")
            return None
        
        try:
            logger.info(f"  Training {model_name.upper()}...")
            
            # GPU configuration
            use_gpu = self.config.get('use_gpu', True)
            device = get_device(use_gpu)
            
            params = self._get_model_config(model_name)
            
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
            num_classes = params.get('num_classes', 3)
            
            # Align trend labels with training data
            y_trend_train = y_trend[:len(X_train)]
            y_trend_test = y_trend[len(X_train):len(X_train)+len(X_test)]
            
            model = MultiTaskModel(
                input_dim=input_dim,
                output_dim=output_dim,
                num_classes=num_classes
            )
            
            # Train with both regression and classification targets
            history = train_pytorch_model(
                model, X_train, (y_train, y_trend_train), 
                X_test, (y_test, y_trend_test),
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32),
                lr=params.get('lr', 1e-3),
                task='multitask',
                use_gpu=use_gpu,
                verbose=True  # Explicit verbose for training progress
            )
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_pred_reg, y_pred_clf = model(X_test_tensor)
                y_pred = y_pred_reg.cpu().numpy()
                trend_logits = y_pred_clf.cpu().numpy()
                trend_pred = trend_logits.argmax(axis=1)
            
            # Regression metrics
            rmse_avg = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            r2_avg = r2_score(y_test.flatten(), y_pred.flatten())
            
            # Classification metrics
            clf_acc = accuracy_score(y_trend_test, trend_pred)
            clf_f1 = f1_score(y_trend_test, trend_pred, average='weighted')
            
            logger.info(f"  Regression: RMSE_avg={rmse_avg:.4f}, R²_avg={r2_avg:.4f}")
            logger.info(f"  Classification: Accuracy={clf_acc:.3f}, F1={clf_f1:.3f}")
            
            return {
                'model': model,
                'history': history,
                'y_pred': y_pred,
                'rmse_avg': rmse_avg,
                'r2_avg': r2_avg,
                'clf_accuracy': clf_acc,
                'clf_f1': clf_f1,
                'trend_predictions': trend_pred.tolist()
            }
        except Exception as e:
            logger.warning(f"  {model_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== TRAINING & EVALUATION ====================
    
    def train_classifiers(self):
        """Train all selected classifiers with proper time-series splitting."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING CLASSIFIERS")
        logger.info("="*80)
        
        if not self.classifier_models:
            logger.info("No classifiers selected")
            return
        
        # Get numeric features (exclude Datetime, Ticker)
        X = self.data.select_dtypes(include=[np.number]).values
        
        # Create labels based on Close prices
        y = self.create_trend_labels()
        
        # Clean data
        X, y = self._clean_data(X, y, method='drop')
        
        if len(X) < 10:
            logger.warning("Not enough valid data for classifier training")
            return
        
        # FIXED: Time-series split (NO SHUFFLE)
        X_train, X_test, y_train, y_test = self._time_series_split(
            X, y, test_size=self.test_size
        )
        
        # Impute any remaining NaN values - fit on train only
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Scale features - fit on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.scalers['classifier'] = scaler
        
        # ADVANCED FEATURE PROCESSING (PCA/AE)
        if FEATURE_PROCESSOR_AVAILABLE:
            fp_config = self.config.get('feature_processing', {})
            # Map legacy keys if they exist in main config
            if 'pca_components' in self.config:
                 fp_config['pca'] = {'enabled': True, 'n_components': self.config['pca_components']}
            if 'autoencoder_latent_dim' in self.config:
                 fp_config['autoencoder'] = {
                     'enabled': True, 
                     'latent_dim': self.config['autoencoder_latent_dim'],
                     'epochs': self.config.get('autoencoder_epochs', 100)
                 }

            if fp_config:
                logger.info("  Applying Advanced Feature Processing (PCA/AE)...")
                try:
                    self.feature_processor = FeatureProcessor(fp_config)
                    self.feature_processor.fit(X_train)
                    X_train = self.feature_processor.transform(X_train)
                    X_test = self.feature_processor.transform(X_test)
                except Exception as fp_e:
                     logger.warning(f"  Feature Processing Failed: {fp_e}")
        
        # Train models
        for model_name in self.classifier_models:
            try:
                if model_name == 'dnn':
                    result = self.train_classifier_dnn(X_train, X_test, y_train, y_test, model_name)
                elif model_name == 'svc':
                    result = self.train_classifier_svc(X_train, X_test, y_train, y_test, model_name)
                elif model_name == 'random_forest':
                    result = self.train_classifier_random_forest(X_train, X_test, y_train, y_test, model_name)
                else:
                    logger.warning(f"Unknown classifier: {model_name}")
                    continue
                
                if result:
                    self.classifier_results[model_name] = result
                    self.predictions[f'{model_name}_classifier'] = result['y_pred']
                    # Save model
                    self._save_model(result['model'], model_name, 'classifier')
            except Exception as e:
                logger.warning(f"  {model_name} failed: {str(e)}")
                continue
    
    def train_and_evaluate(self, ticker: str = None, holdout_days: int = 50, 
                             llm_config: dict = None, generate_reports: bool = True) -> Dict:
        """
        Train and evaluate selected models with MULTI-STEP forecasting.
        
        This is the unified training method that includes:
        - Holdout period with true out-of-sample evaluation
        - Multi-step future forecasts
        - Baseline comparisons (naive, drift, rolling mean)
        - 10-day trend assessment with probabilities
        - Training data export for reproducibility
        - Optional LLM-powered analysis
        
        CRITICAL DATA MANAGEMENT:
        - Forecast horizon: configurable via self.forecast_horizon (default: 10)
        - Multi-step targets: y[t] = [Close_{t+1}, ..., Close_{t+horizon}]
        - Training uses data before holdout period
        - Holdout evaluation: DIRECT multi-step prediction (not recursive)
        
        NO DATA LEAKAGE GUARANTEES:
        1. All features are lagged by 1 day (use t-1 data to predict t)
        2. Scaler is fit ONLY on training data
        3. Target indices are strictly > feature indices
        4. Holdout period is completely excluded from training
        
        Args:
            ticker: Asset name for report title (optional)
            holdout_days: Number of days to hold out for validation (default: 50)
            llm_config: LLM configuration for automated review (optional)
            generate_reports: Whether to generate HTML reports (default: True)
        
        Returns:
            Dictionary with all results including holdout metrics and forecasts
        """
        logger.info("\n" + "="*80)
        logger.info(f"TRAIN AND EVALUATE - {ticker or 'Asset'}")
        logger.info("="*80)
        logger.info(f"  FORECAST HORIZON: {self.forecast_horizon} steps ahead")
        logger.info(f"  HOLDOUT PERIOD: Last {holdout_days} days reserved for validation")
        logger.info(f"  NO DATA LEAKAGE: Training uses only data before holdout period")
        
        horizon = self.forecast_horizon
        
        # Use models from config, fallback to all valid models only if None (not specified)
        # Empty list [] means explicitly no models wanted
        if self.classifier_models is None:
            logger.info(f"  No classifiers specified in config, using all: {self.VALID_CLASSIFIERS}")
            self.classifier_models = self.VALID_CLASSIFIERS
        elif not self.classifier_models:
            logger.info("  Classifiers explicitly disabled (empty list)")
        else:
            logger.info(f"  Classifiers from config: {self.classifier_models}")
        
        if self.regressor_models is None:
            logger.info(f"  No regressors specified in config, using all: {self.VALID_REGRESSORS}")
            self.regressor_models = self.VALID_REGRESSORS
        elif not self.regressor_models:
            logger.info("  Regressors explicitly disabled (empty list)")
        else:
            logger.info(f"  Regressors from config: {self.regressor_models}")
        
        benchmark_results = {
            'ticker': ticker or 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'forecast_horizon': horizon,
            'holdout_days': holdout_days,
            'forecast_mode': 'multioutput',
            'classifiers': {},
            'regressors': {},
            'future_forecasts': {},
            'holdout_forecasts': {},
            'baselines': {},
            'warnings': [],
            'pipeline_info': {},
            'llm_review': None,
            'model_errors': {},  # Capture failed model errors with stack traces
            'config': self.config  # Pass config for reporting options
        }
        
        # Storage for detailed metrics
        classifier_details = {}
        regressor_details = {}
        
        # ===== STORE ORIGINAL CLOSE PRICES =====
        original_close_prices = self.data['Close'].values.copy()
        valid_close_mask = ~np.isnan(original_close_prices)
        original_close_prices_clean = original_close_prices[valid_close_mask]
        
        total_samples = len(original_close_prices_clean)
        
        # Ensure holdout is large enough for multi-step evaluation
        min_holdout = horizon + 10
        if holdout_days < min_holdout:
            logger.warning(f"Holdout ({holdout_days}) too small for horizon ({horizon}), adjusting to {min_holdout}")
            holdout_days = min_holdout
        
        holdout_start_idx = total_samples - holdout_days
        
        # Split data into training and holdout
        train_close = original_close_prices_clean[:holdout_start_idx]
        holdout_close = original_close_prices_clean[holdout_start_idx:]
        
        last_train_price = float(train_close[-1])
        last_actual_price = float(original_close_prices_clean[-1])
        
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Training samples: {holdout_start_idx} (indices 0-{holdout_start_idx-1})")
        logger.info(f"  Holdout samples: {holdout_days} (indices {holdout_start_idx}-{total_samples-1})")
        logger.info(f"  Last training price: ${last_train_price:.2f}")
        logger.info(f"  Last actual price: ${last_actual_price:.2f}")
        
        benchmark_results['last_actual_price'] = last_actual_price
        benchmark_results['last_train_price'] = last_train_price
        benchmark_results['holdout_actual'] = holdout_close.tolist()
        
        # ===== GET FEATURE COLUMNS =====
        expected_close_features = self._get_close_only_feature_columns()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        close_features_available = [c for c in expected_close_features if c in numeric_cols]
        
        if len(close_features_available) >= 5:
            feature_cols = close_features_available
        else:
            feature_cols = [col for col in numeric_cols 
                           if col.startswith('Close_') and col != 'Close']
            if not feature_cols:
                feature_cols = [col for col in numeric_cols 
                               if col != 'Close' and not col.startswith('Close_')]
        
        # Validate features for leakage
        if FORECAST_UTILS_AVAILABLE:
            is_valid, leakage_warnings = validate_no_leakage(feature_cols)
            benchmark_results['warnings'].extend(leakage_warnings)
            if not is_valid:
                logger.error("LEAKAGE DETECTED - check feature columns!")
        
        # Store pipeline info
        benchmark_results['pipeline_info'] = {
            'raw_data_shape': self.data.shape,
            'total_samples': total_samples,
            'training_samples': holdout_start_idx,
            'holdout_samples': holdout_days,
            'horizon': horizon,
            'feature_columns': feature_cols,
            'target_column': 'Close (multi-step)',
            'feature_count': len(feature_cols),
            'model_type': 'multi-output regression',
            'data_split_method': f'Holdout last {holdout_days} days (no shuffle)',
            'scaling': 'StandardScaler (fit on training only)',
            'no_leakage_guarantee': True,
            'lag_features': [1, 5, 10],
            'rolling_windows': [5, 20],
            # Scale-free mode settings
            'target_transform': self.config.get('target_transform', 
                                                'pct_change' if self.config.get('use_percentage_returns', False) else 'price'),
            'base_price_column': self.config.get('base_price_column', 'Close'),
            'allow_additional_price_columns': self.config.get('allow_additional_price_columns', False),
            'allow_absolute_scale_features': self.config.get('allow_absolute_scale_features', False)
        }
        
        logger.info(f"  Features ({len(feature_cols)}): {feature_cols[:5]}...")
        
        # ===== BUILD MULTI-STEP TARGETS =====
        if FORECAST_UTILS_AVAILABLE:
            # Use forecast_utils for multi-step targets
            y_multistep_all, valid_indices = build_multistep_targets(original_close_prices_clean, horizon)
            logger.info(f"  Multi-step targets: {y_multistep_all.shape} (samples x horizon)")
        else:
            # Fallback: build targets manually
            n_valid = total_samples - horizon
            y_multistep_all = np.zeros((n_valid, horizon))
            for t in range(n_valid):
                y_multistep_all[t, :] = original_close_prices_clean[t + 1 : t + 1 + horizon]
            valid_indices = np.arange(n_valid)
        
        # ===== TARGET TRANSFORM (unified: price | pct_change | log_return) =====
        # Get context prices (last known price before forecast window)
        last_known_prices_all = original_close_prices_clean[valid_indices]
        
        # Resolve target transform with backward compatibility
        if FORECAST_UTILS_AVAILABLE:
            target_transform = get_target_transform(self.config)
            # Logic fix: Ensure we don't fallback to price if not intended, prefer log_return for stability
            ds_config = self.config.get('data_science', self.config)
            if 'target_transform' not in ds_config:
                logger.info("  Target transform not explicitly set, defaulting to 'log_return' for stability")
                target_transform = 'log_return'
        else:
            # Fallback for legacy use_percentage_returns
            use_pct = self.config.get('use_percentage_returns', False)
            target_transform = 'pct_change' if use_pct else 'price'
        
        logger.info(f"  Target transform: {target_transform.upper()}")
        self._target_transform = target_transform  # Store for later inverse
        
        # Apply forward transform
        if FORECAST_UTILS_AVAILABLE and target_transform != 'price':
            y_targets_all = to_target(y_multistep_all, target_transform, last_known_prices_all)
        elif target_transform == 'pct_change':
            # Fallback pct_change
            safe_last_prices = last_known_prices_all.copy()
            safe_last_prices[safe_last_prices == 0] = 1e-8
            y_targets_all = (y_multistep_all - safe_last_prices[:, None]) / safe_last_prices[:, None]
        elif target_transform == 'log_return':
            # Fallback log_return
            safe_last_prices = last_known_prices_all.copy()
            safe_last_prices[safe_last_prices == 0] = 1e-8
            y_targets_all = np.log(y_multistep_all / safe_last_prices[:, None])
        else:
            y_targets_all = y_multistep_all
        
        # Backward compat flag (for downstream code that checks use_pct)
        use_pct = target_transform != 'price'
        
        # ===== FEATURE PRUNING (train-only fit) =====
        # Split into training and holdout FIRST (for pruning)
        train_end_idx = holdout_start_idx - horizon
        if train_end_idx < 50:
            logger.warning(f"Training set too small ({train_end_idx}), reducing holdout")
            train_end_idx = max(50, total_samples - 2 * horizon)
        
        # Apply feature pruning if enabled and available
        pruning_config = self.config.get('feature_pruning', {})
        self.feature_pruner = None
        
        if FEATURE_PRUNER_AVAILABLE and pruning_config:
            logger.info("\n--- FEATURE PRUNING (train-only fit) ---")
            
            # Create DataFrame for train portion ONLY
            df_train_for_pruning = self.data[feature_cols].iloc[:train_end_idx].copy()
            
            # Fit pruner on training data only
            self.feature_pruner = FeaturePruner(pruning_config)
            self.feature_pruner.fit(df_train_for_pruning, target_col='Close')
            
            # Get remaining feature columns after pruning
            feature_cols_pruned = [c for c in feature_cols if c not in self.feature_pruner.columns_to_drop]
            
            logger.info(f"  Pruned {len(self.feature_pruner.columns_to_drop)} features")
            logger.info(f"  Remaining features: {len(feature_cols_pruned)}")
            
            # Save pruning artifact
            pruner_path = self.feature_pruner.save_artifact(self.output_dir)
            
            # Update feature_cols
            feature_cols = feature_cols_pruned
            
            # Update pipeline info
            benchmark_results['pipeline_info']['pruning'] = self.feature_pruner.get_summary()
            benchmark_results['pipeline_info']['feature_count'] = len(feature_cols)
            benchmark_results['pipeline_info']['feature_columns'] = feature_cols
            # Store artifact path
            benchmark_results['artifacts'] = benchmark_results.get('artifacts', {})
            benchmark_results['artifacts']['feature_pruner'] = pruner_path
        
        # Align features with valid target indices
        X_all = self.data[feature_cols].values[valid_close_mask][:len(valid_indices)]
        
        # Split into training and holdout using pre-computed train_end_idx
        X_train_full = X_all[:train_end_idx]
        y_train_full = y_targets_all[:train_end_idx]
        
        # Keep original prices and last known prices for evaluation/reconstruction
        y_train_full_original = y_multistep_all[:train_end_idx]
        last_known_train_full = last_known_prices_all[:train_end_idx]
        
        X_holdout = X_all[train_end_idx:len(valid_indices)]
        y_holdout = y_targets_all[train_end_idx:len(valid_indices)]
        
        # Keep original prices and last known prices for holdout evaluation
        y_holdout_original_prices = y_multistep_all[train_end_idx:len(valid_indices)]
        last_known_holdout = last_known_prices_all[train_end_idx:len(valid_indices)]
        
        logger.info(f"  Training set: X={X_train_full.shape}, y={y_train_full.shape}")
        logger.info(f"  Holdout set: X={X_holdout.shape}, y={y_holdout.shape}")

        
        # ===== COMPUTE MULTI-STEP BASELINES =====
        if FORECAST_UTILS_AVAILABLE and len(holdout_close) > horizon:
            baselines = compute_multistep_baselines(train_close, holdout_close, horizon)
            for name, data in baselines.items():
                benchmark_results['baselines'][name] = {
                    'metrics': data['metrics'],
                    'description': data['description']
                }
        else:
            # Simple baselines
            naive_pred = np.full_like(y_holdout, last_train_price)
            naive_rmse = np.sqrt(np.mean((y_holdout - naive_pred) ** 2))
            benchmark_results['baselines']['naive'] = {
                'metrics': {'rmse_avg': float(naive_rmse)},
                'description': 'Last training price for all steps'
            }
        
        # ===== CLASSIFIERS (use horizon-based trend) =====
        logger.info("\n--- CLASSIFIERS (Trend Classification) ---")
        
        if FORECAST_UTILS_AVAILABLE:
            y_trend = create_multistep_trend_labels(original_close_prices_clean, horizon=horizon)
        else:
            y_trend = self.create_trend_labels()
        
        X_clf_all = self.data.select_dtypes(include=[np.number]).values[valid_close_mask]
        y_clf_all = y_trend[valid_close_mask]
        
        X_clf_train = X_clf_all[:train_end_idx]
        y_clf_train = y_clf_all[:train_end_idx]
        
        # Further split for validation
        split_idx = int(len(X_clf_train) * 0.8)
        X_train_clf = X_clf_train[:split_idx]
        X_test_clf = X_clf_train[split_idx:]
        y_train_clf = y_clf_train[:split_idx]
        y_test_clf = y_clf_train[split_idx:]
        
        # Clean, impute, scale
        X_train_clf, y_train_clf = self._clean_data(X_train_clf, y_train_clf, method='drop')
        X_test_clf, y_test_clf = self._clean_data(X_test_clf, y_test_clf, method='drop')
        
        # Store classifier training data before any further processing
        # (for saving to file later - will be updated after scaling)
        saved_X_train_clf = None
        saved_y_train_clf = None
        
        if len(X_train_clf) >= 10:
            imputer_clf = SimpleImputer(strategy='mean')
            X_train_clf = imputer_clf.fit_transform(X_train_clf)
            X_test_clf = imputer_clf.transform(X_test_clf)
            
            scaler_clf = StandardScaler()
            X_train_clf = scaler_clf.fit_transform(X_train_clf)
            X_test_clf = scaler_clf.transform(X_test_clf)
            
            # Save scaled classifier training data for file export
            saved_X_train_clf = X_train_clf.copy()
            saved_y_train_clf = y_train_clf.copy()
            
            for model_name in self.classifier_models:
                try:
                    if model_name == 'dnn':
                        result = self.train_classifier_dnn(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
                    elif model_name == 'svc':
                        result = self.train_classifier_svc(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
                    elif model_name == 'random_forest':
                        result = self.train_classifier_random_forest(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
                    else:
                        result = None
                    
                    if result:
                        benchmark_results['classifiers'][model_name] = {
                            'accuracy': float(result.get('accuracy', 0)),
                            'f1_score': float(result.get('f1_score', 0)),
                            'trend_prediction': self.TREND_LABELS.get(int(result['y_pred'][-1]) if len(result['y_pred']) > 0 else 1, 'UNKNOWN')
                        }
                        classifier_details[model_name] = {
                            'cm': result.get('cm'),
                            'y_test': result.get('y_test'),
                            'y_pred': result.get('y_pred'),
                            'history': result.get('history')
                        }
                        self._save_model(result['model'], model_name, 'classifier')
                        
                        # Store model + scaler for 10-day trend assessment
                        classifier_details[model_name]['model'] = result['model']
                        classifier_details[model_name]['scaler'] = scaler_clf
                        classifier_details[model_name]['imputer'] = imputer_clf
                        
                except Exception as e:
                    logger.warning(f"  {model_name} failed: {str(e)}")
        
        # ===== 10-DAY TREND ASSESSMENT (NEW) =====
        # For each classifier: predict trend for next 10 days with expected % change
        if FORECAST_UTILS_AVAILABLE and len(benchmark_results['classifiers']) > 0:
            logger.info("\n--- 10-DAY TREND ASSESSMENT ---")
            
            try:
                # Get trend horizon and threshold from config
                trend_horizon = getattr(self, 'trend_horizon', 10)
                trend_threshold = getattr(self, 'trend_threshold', 0.02)
                
                # Create labels with pct_changes
                # Note: create_10day_trend_labels uses fixed 10-day horizon
                trend_labels, pct_changes = create_10day_trend_labels(
                    close_series=original_close_prices_clean,
                    threshold=trend_threshold
                )
                
                benchmark_results['trend_10day'] = {}
                
                for model_name, details in classifier_details.items():
                    if 'model' in details and 'scaler' in details:
                        try:
                            model = details['model']
                            scaler = details['scaler']
                            imputer = details['imputer']
                            
                            # Get latest features for prediction
                            X_latest = X_clf_all[-1:].copy()
                            X_latest = imputer.transform(X_latest)
                            X_latest_scaled = scaler.transform(X_latest)
                            
                            # Predict trend
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_latest_scaled)[0]
                                trend_pred = model.predict(X_latest_scaled)[0]
                                proba_dict = {
                                    self.TREND_LABELS.get(i, f'Class_{i}'): float(p)
                                    for i, p in enumerate(proba)
                                }
                            else:
                                # DNN or model without predict_proba
                                if hasattr(model, 'predict'):
                                    raw_pred = model.predict(X_latest_scaled)
                                    if len(raw_pred.shape) > 1 and raw_pred.shape[1] > 1:
                                        # Multi-class output
                                        proba = raw_pred[0]
                                        trend_pred = int(np.argmax(proba))
                                        proba_dict = {
                                            self.TREND_LABELS.get(i, f'Class_{i}'): float(p)
                                            for i, p in enumerate(proba)
                                        }
                                    else:
                                        trend_pred = int(np.round(raw_pred.flatten()[0]))
                                        proba_dict = {}
                                else:
                                    trend_pred = 1  # HOLD
                                    proba_dict = {}
                            
                            trend_label = self.TREND_LABELS.get(trend_pred, 'HOLD')
                            
                            benchmark_results['trend_10day'][model_name] = {
                                'trend_label': trend_label,
                                'trend_value': int(trend_pred),
                                'probabilities': proba_dict,
                                'horizon_days': trend_horizon,
                                'threshold_pct': trend_threshold * 100
                            }
                            
                            logger.info(f"    {model_name}: {trend_label} (proba={proba_dict})")
                            
                        except Exception as trend_err:
                            logger.warning(f"    {model_name} 10-day trend failed: {trend_err}")
                            benchmark_results['trend_10day'][model_name] = {'error': str(trend_err)}
                
                # Add expected % change from best regressor
                if benchmark_results.get('future_forecasts'):
                    # Find best regressor by R²
                    best_reg_name = None
                    best_r2 = -float('inf')
                    for reg_name, reg_data in benchmark_results['regressors'].items():
                        r2 = reg_data.get('r2_avg', 0)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_reg_name = reg_name
                    
                    if best_reg_name and best_reg_name in benchmark_results['future_forecasts']:
                        future_preds = benchmark_results['future_forecasts'][best_reg_name]['predictions']
                        if len(future_preds) >= trend_horizon:
                            price_day_10 = future_preds[trend_horizon - 1]
                            expected_pct_change = ((price_day_10 - last_actual_price) / last_actual_price) * 100
                            
                            benchmark_results['trend_10day']['expected_pct_change'] = {
                                'value': float(expected_pct_change),
                                'source_model': best_reg_name,
                                'current_price': float(last_actual_price),
                                'predicted_price_day_10': float(price_day_10)
                            }
                            
                            logger.info(f"    Expected % change (10d): {expected_pct_change:+.2f}% "
                                       f"(${last_actual_price:.2f} → ${price_day_10:.2f}, via {best_reg_name})")
                
            except Exception as trend_section_err:
                logger.warning(f"  10-day trend assessment failed: {trend_section_err}")
                import traceback
                traceback.print_exc()
        
        # ===== REGRESSORS (Multi-Output) =====
        logger.info("\n--- REGRESSORS (Multi-Step Forecasting) ---")
        logger.info(f"  Mode: Multi-output (predicting {horizon} steps at once)")
        
        if not feature_cols or len(X_train_full) < 20:
            logger.error("Insufficient data for regression training")
            benchmark_results['warnings'].append("Insufficient data for regression")
        else:
            # Clean training data
            X_train_reg, y_train_reg = self._clean_data(X_train_full, y_train_full[:, 0], method='drop')
            # Re-align y to match cleaned X (keep all horizon columns)
            valid_mask_train = ~np.any(np.isnan(X_train_full) | np.isinf(X_train_full), axis=1)
            X_train_reg = X_train_full[valid_mask_train]
            y_train_reg = y_train_full[valid_mask_train]
            
            # Filter auxiliary arrays for reconstruction
            last_known_train_reg = last_known_train_full[valid_mask_train]
            y_train_original_reg = y_train_full_original[valid_mask_train]
            
            # ===== OUTLIER FILTERING (Training Data Only) =====
            outlier_config = self.config.get('outlier_filter', {})
            if outlier_config.get('enabled', False):
                percentile = outlier_config.get('percentile', 1.0)
                method = outlier_config.get('method', 'max_abs')
                
                n_before = len(X_train_reg)
                
                # Compute extreme value per sample based on method
                if method == 'max_abs':
                    # Max of absolute values across horizon
                    extreme_vals = np.max(np.abs(y_train_reg), axis=1)
                elif method == 'mean':
                    # Mean across horizon
                    extreme_vals = np.mean(y_train_reg, axis=1)
                elif method == 'max':
                    # Max across horizon
                    extreme_vals = np.max(y_train_reg, axis=1)
                elif method == 'min':
                    # Min across horizon
                    extreme_vals = np.min(y_train_reg, axis=1)
                else:
                    extreme_vals = np.max(np.abs(y_train_reg), axis=1)
                
                # Compute percentile thresholds
                lower_thresh = np.percentile(extreme_vals, percentile)
                upper_thresh = np.percentile(extreme_vals, 100 - percentile)
                
                # Create mask to keep samples within thresholds
                if method in ['max_abs', 'max']:
                    # For these methods, we only filter the upper tail
                    keep_mask = extreme_vals <= upper_thresh
                elif method == 'min':
                    # For min, we only filter the lower tail
                    keep_mask = extreme_vals >= lower_thresh
                else:  # 'mean' - filter both tails
                    keep_mask = (extreme_vals >= lower_thresh) & (extreme_vals <= upper_thresh)
                
                # Apply filter
                X_train_reg = X_train_reg[keep_mask]
                y_train_reg = y_train_reg[keep_mask]
                last_known_train_reg = last_known_train_reg[keep_mask]
                y_train_original_reg = y_train_original_reg[keep_mask]
                
                n_after = len(X_train_reg)
                n_removed = n_before - n_after
                
                logger.info(f"\n--- OUTLIER FILTERING (Training Data) ---")
                logger.info(f"  Method: {method}, Percentile: {percentile}%")
                logger.info(f"  Threshold: {lower_thresh:.6f} to {upper_thresh:.6f}")
                logger.info(f"  Removed: {n_removed} samples ({100*n_removed/n_before:.1f}%)")
                logger.info(f"  Remaining: {n_after} samples")
                
                benchmark_results['outlier_filter'] = {
                    'enabled': True,
                    'method': method,
                    'percentile': percentile,
                    'lower_threshold': float(lower_thresh),
                    'upper_threshold': float(upper_thresh),
                    'samples_removed': n_removed,
                    'samples_remaining': n_after,
                    'removal_rate_pct': float(100 * n_removed / n_before) if n_before > 0 else 0
                }
            
            # Split for internal validation
            split_idx = int(len(X_train_reg) * 0.85)
            X_train = X_train_reg[:split_idx]
            X_val = X_train_reg[split_idx:]
            y_train = y_train_reg[:split_idx]
            y_val = y_train_reg[split_idx:]
            
            # Split auxiliary arrays
            last_known_val = last_known_train_reg[split_idx:]
            y_val_original_prices = y_train_original_reg[split_idx:]
            
            logger.info(f"  Internal split: train={len(X_train)}, val={len(X_val)}")
            
            # ===== DATA AUGMENTATION (Training Data Only) =====
            # Apply BEFORE scaling to preserve data distribution
            if self.augmentor is not None:
                n_before = len(X_train)
                # Flatten multi-step y for augmentation if needed
                if len(y_train.shape) > 1:
                    # For multi-step: augment each sample's (X, y) pair together
                    # We need to handle y as a 2D array (n_samples, horizon)
                    X_train_aug_list = [X_train]
                    y_train_aug_list = [y_train]
                    
                    for method in self.augmentor.methods:
                        for aug_idx in range(self.augmentor.n_augmentations):
                            # Augment X (features) with the method
                            if method == 'jitter':
                                X_aug, _ = self.augmentor._jitter(X_train, y_train[:, 0])
                                y_aug = y_train.copy()  # Keep y unchanged for jitter
                            elif method == 'scale':
                                X_aug, _ = self.augmentor._magnitude_scale(X_train, y_train[:, 0])
                                y_aug = y_train.copy()  # Keep y unchanged for scale
                            elif method == 'window_crop':
                                # For window crop, we need to crop both X and y together
                                crop_ratio = self.augmentor.config.get('window_crop_ratio', 0.9)
                                crop_size = int(len(X_train) * crop_ratio)
                                if crop_size >= 10:
                                    start = np.random.randint(0, len(X_train) - crop_size + 1)
                                    X_aug = X_train[start:start+crop_size]
                                    y_aug = y_train[start:start+crop_size]
                                else:
                                    continue
                            elif method == 'mixup':
                                # Mixup for both X and y
                                alpha = self.augmentor.config.get('mixup_alpha', 0.2)
                                lam = np.random.beta(alpha, alpha, size=(len(X_train), 1))
                                perm = np.random.permutation(len(X_train))
                                X_aug = lam * X_train + (1 - lam) * X_train[perm]
                                y_aug = lam * y_train + (1 - lam) * y_train[perm]
                            else:
                                continue
                            
                            X_train_aug_list.append(X_aug)
                            y_train_aug_list.append(y_aug)
                    
                    X_train = np.vstack(X_train_aug_list)
                    y_train = np.vstack(y_train_aug_list)
                else:
                    # Single-step target
                    X_train, y_train = self.augmentor.augment(X_train, y_train)
                
                n_after = len(X_train)
                logger.info(f"\n--- DATA AUGMENTATION (Training Data) ---")
                logger.info(f"  Methods: {self.augmentor.methods}")
                logger.info(f"  Samples: {n_before} → {n_after} ({n_after/n_before:.1f}x)")
                
                benchmark_results['data_augmentation'] = {
                    'enabled': True,
                    'methods': self.augmentor.methods,
                    'samples_before': n_before,
                    'samples_after': n_after,
                    'multiplier': float(n_after / n_before)
                }
            
            # Impute and scale features - FIT ON TRAIN ONLY
            imputer_reg = SimpleImputer(strategy='mean')
            X_train_imputed = imputer_reg.fit_transform(X_train)
            X_val_imputed = imputer_reg.transform(X_val)
            
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_imputed)
            X_val_scaled = scaler_X.transform(X_val_imputed)
            
            # Scale y (target prices) - FIT ON TRAIN ONLY
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled = scaler_y.transform(y_val)
            
            self.scalers['benchmark_X'] = scaler_X
            self.y_scalers['benchmark_y'] = scaler_y
            
            # ===== ADVANCED FEATURE PROCESSING (PCA/AE/Gating) =====
            feature_proc_metadata = {}
            X_train_processed = X_train_scaled
            X_val_processed = X_val_scaled
            
            if FEATURE_PROCESSOR_AVAILABLE:
                feature_processing_config = self.config.get('feature_processing', {})
                if feature_processing_config.get('mode', 'raw') != 'raw':
                    logger.info(f"\n--- ADVANCED FEATURE PROCESSING ---")
                    logger.info(f"  Mode: {feature_processing_config.get('mode', 'raw')}")
                    
                    try:
                        feature_processor = FeatureProcessor(feature_processing_config)
                        
                        # Fit on TRAINING data only
                        feature_processor.fit(X_train_scaled)
                        
                        # Transform both train and validation
                        X_train_processed = feature_processor.transform(X_train_scaled)
                        X_val_processed = feature_processor.transform(X_val_scaled)
                        
                        # Collect metadata for report
                        feature_proc_metadata = feature_processor.get_metadata()
                        feature_proc_metadata['input_features'] = X_train_scaled.shape[1]
                        feature_proc_metadata['output_features'] = X_train_processed.shape[1]
                        
                        logger.info(f"  Input features: {X_train_scaled.shape[1]}")
                        logger.info(f"  Output features: {X_train_processed.shape[1]}")
                        if 'pca_variance' in feature_proc_metadata:
                            logger.info(f"  PCA explained variance: {feature_proc_metadata['pca_variance']:.1f}%")
                        if 'ae_loss' in feature_proc_metadata:
                            logger.info(f"  Autoencoder loss: {feature_proc_metadata['ae_loss']:.6f}")
                        
                        # Store for holdout processing
                        self.scalers['feature_processor'] = feature_processor
                        benchmark_results['feature_processing'] = feature_proc_metadata
                        
                    except Exception as fp_err:
                        logger.warning(f"  Feature processing failed: {fp_err}")
                        import traceback
                        traceback.print_exc()
                        X_train_processed = X_train_scaled
                        X_val_processed = X_val_scaled
            
            # Use processed features for modeling
            X_train_for_models = X_train_processed
            X_val_for_models = X_val_processed
            
            # ===== SAVE TRAINING DATASETS =====
            # Save the exact data used for training (for reproducibility and debugging)
            # Can be disabled for large benchmarking sweeps.
            if self.config.get('save_training_datasets', True):
                logger.info("\n--- SAVING TRAINING DATASETS ---")
                self._save_training_datasets(
                    X_train_clf=saved_X_train_clf,
                    y_train_clf=saved_y_train_clf,
                    X_train_reg=X_train_for_models,
                    y_train_reg=y_train_scaled,
                    feature_cols=feature_cols,
                    horizon=horizon,
                    target_transform=target_transform
                )
            
            # Prepare holdout
            valid_mask_holdout = ~np.any(np.isnan(X_holdout) | np.isinf(X_holdout), axis=1)
            X_holdout_clean = X_holdout[valid_mask_holdout]
            y_holdout_clean = y_holdout[valid_mask_holdout]
            
            if len(X_holdout_clean) > 0:
                X_holdout_imputed = imputer_reg.transform(X_holdout_clean)
                X_holdout_scaled = scaler_X.transform(X_holdout_imputed)
                
                # Apply feature processing to holdout (TRANSFORM ONLY, no fit)
                X_holdout_for_models = X_holdout_scaled
                if 'feature_processor' in self.scalers:
                    try:
                        X_holdout_for_models = self.scalers['feature_processor'].transform(X_holdout_scaled)
                    except Exception as fp_ho_err:
                        logger.warning(f"  Feature processing on holdout failed: {fp_ho_err}")
                        X_holdout_for_models = X_holdout_scaled
            else:
                X_holdout_scaled = np.array([])
                X_holdout_for_models = np.array([])
                y_holdout_clean = np.array([])
            
            # ===== SHAPE VALIDATION (CRITICAL for detecting feature dimension mismatches) =====
            # Training and holdout MUST have the same feature dimensions
            if len(X_holdout_for_models) > 0 and len(X_train_for_models) > 0:
                train_features = X_train_for_models.shape[1]
                holdout_features = X_holdout_for_models.shape[1]
                if train_features != holdout_features:
                    logger.error(f"  ❌ FEATURE DIMENSION MISMATCH!")
                    logger.error(f"     Training features: {train_features}")
                    logger.error(f"     Holdout features: {holdout_features}")
                    logger.error(f"     This will cause prediction failures!")
                    raise ValueError(f"Feature dimension mismatch: train={train_features} vs holdout={holdout_features}")
                else:
                    logger.info(f"  ✓ Feature dimensions validated: {train_features} features for both train and holdout")
            
            # Get naive baseline RMSE for comparison
            naive_rmse = benchmark_results['baselines'].get('naive', {}).get('metrics', {}).get('rmse_avg', 999)
            
            # Use configured regressors (availability checked per-model)
            all_regressors = self.regressor_models.copy() if self.regressor_models else []
            
            for model_name in all_regressors:
                try:
                    # Train multi-output model
                    if model_name == 'lstm':
                        result = self.train_multioutput_lstm(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif model_name == 'xlstm':
                        result = self.train_multioutput_xlstm(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif model_name == 'dnn':
                        result = self.train_multioutput_dnn(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif model_name == 'krr':
                        result = self.train_multioutput_krr(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif model_name == 'linear':
                        result = self.train_multioutput_linear(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif MODEL_ZOO_AVAILABLE and model_name == 'lightgbm':
                        result = self.train_lightgbm_multioutput(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif MODEL_ZOO_AVAILABLE and model_name == 'tcn':
                        result = self.train_tcn(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif MODEL_ZOO_AVAILABLE and model_name == 'nbeats':
                        result = self.train_nbeats(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif MODEL_ZOO_AVAILABLE and model_name == 'moe':
                        result = self.train_moe(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled)
                    elif MODEL_ZOO_AVAILABLE and model_name == 'multitask':
                        result = self.train_multitask(X_train_for_models, X_val_for_models, y_train_scaled, y_val_scaled, y_trend)
                    else:
                        result = None
                    
                    if result is not None:
                        # Inverse transform validation predictions
                        y_pred_scaled = result['y_pred']
                        y_pred_transformed = scaler_y.inverse_transform(y_pred_scaled)  # Unscaled but still in transform space
                        y_val_transformed = scaler_y.inverse_transform(y_val_scaled)  # Unscaled but still in transform space
                        
                        # Convert to price space
                        y_pred_original = y_pred_transformed.copy()
                        
                        if use_pct:
                            # Convert transformed values back to price
                            if FORECAST_UTILS_AVAILABLE:
                                y_pred_original = from_target(y_pred_original, target_transform, last_known_val)
                            elif target_transform == 'pct_change':
                                y_pred_original = last_known_val[:, None] * (1 + y_pred_original)
                            elif target_transform == 'log_return':
                                y_pred_original = last_known_val[:, None] * np.exp(y_pred_original)
                            y_val_original = y_val_original_prices
                        else:
                            y_val_original = scaler_y.inverse_transform(y_val_scaled)
                        
                        # Calculate metrics on original scale
                        if FORECAST_UTILS_AVAILABLE:
                            val_metrics = evaluate_multistep(y_val_original, y_pred_original)
                        else:
                            val_metrics = {
                                'rmse_avg': float(np.sqrt(mean_squared_error(y_val_original.flatten(), y_pred_original.flatten()))),
                                'r2_avg': float(r2_score(y_val_original.flatten(), y_pred_original.flatten()))
                            }
                        
                        benchmark_results['regressors'][model_name] = {
                            'rmse': val_metrics.get('rmse_avg', 0),
                            'rmse_avg': val_metrics.get('rmse_avg', 0),
                            'mae': val_metrics.get('mae_avg', 0),
                            'r2': val_metrics.get('r2_avg', 0),
                            'r2_avg': val_metrics.get('r2_avg', 0),
                            'rmse_step_1': val_metrics.get('rmse_step_1', 0),
                            'rmse_step_5': val_metrics.get('rmse_step_5', 0),
                            'rmse_step_10': val_metrics.get(f'rmse_step_{horizon}', 0)
                        }
                        
                        # Holdout evaluation (DIRECT prediction)
                        # CRITICAL: Use X_holdout_for_models (processed features) for ALL models
                        if len(X_holdout_for_models) > 0:
                            try:
                                holdout_pred_scaled = self._predict_with_model(
                                    result['model'], model_name, X_holdout_for_models, config=result
                                )
                                
                                holdout_pred_original = scaler_y.inverse_transform(holdout_pred_scaled)
                                
                                if use_pct:
                                    # Convert transformed values back to price
                                    if FORECAST_UTILS_AVAILABLE:
                                        holdout_pred_original = from_target(holdout_pred_original, target_transform, last_known_holdout)
                                    elif target_transform == 'pct_change':
                                        holdout_pred_original = last_known_holdout[:, None] * (1 + holdout_pred_original)
                                    elif target_transform == 'log_return':
                                        holdout_pred_original = last_known_holdout[:, None] * np.exp(holdout_pred_original)
                                    y_holdout_original = y_holdout_original_prices
                                else:
                                    y_holdout_original = y_holdout_clean
                                
                                if FORECAST_UTILS_AVAILABLE:
                                    holdout_metrics = evaluate_multistep(y_holdout_original, holdout_pred_original)
                                else:
                                    holdout_metrics = {
                                        'rmse_avg': float(np.sqrt(mean_squared_error(y_holdout_original.flatten(), holdout_pred_original.flatten()))),
                                        'r2_avg': float(r2_score(y_holdout_original.flatten(), holdout_pred_original.flatten()))
                                    }
                                
                                beats_naive = holdout_metrics['rmse_avg'] < naive_rmse
                                
                                # Extract first-step predictions for holdout plot visualization
                                # Each sample's prediction for step 1 (immediate next value)
                                holdout_pred_step1 = holdout_pred_original[:, 0] if holdout_pred_original.ndim > 1 else holdout_pred_original.flatten()
                                holdout_actual_step1 = y_holdout_original[:, 0] if y_holdout_original.ndim > 1 else y_holdout_original.flatten()
                                
                                # Compute first-step metrics for visualization
                                step1_rmse = float(np.sqrt(mean_squared_error(holdout_actual_step1, holdout_pred_step1)))
                                step1_mae = float(mean_absolute_error(holdout_actual_step1, holdout_pred_step1))
                                step1_r2 = float(r2_score(holdout_actual_step1, holdout_pred_step1))
                                
                                benchmark_results['holdout_forecasts'][model_name] = {
                                    'metrics': holdout_metrics,
                                    'beats_naive': bool(beats_naive),
                                    'method': 'DIRECT_MULTIOUTPUT',
                                    'n_samples': len(X_holdout_clean),
                                    'predictions_sample': holdout_pred_original[:3].tolist() if len(holdout_pred_original) > 0 else [],
                                    # Add fields needed for holdout plot visualization
                                    'predictions': holdout_pred_step1.tolist(),
                                    'actual': holdout_actual_step1.tolist(),
                                    'rmse': step1_rmse,
                                    'mae': step1_mae,
                                    'r2': step1_r2
                                }
                                
                                logger.info(f"    Holdout: RMSE_avg=${holdout_metrics['rmse_avg']:.2f}, beats_naive={beats_naive}")
                                
                            except Exception as ho_err:
                                logger.warning(f"  {model_name} holdout failed: {ho_err}")
                                benchmark_results['holdout_forecasts'][model_name] = {'error': str(ho_err)}
                        
                        # Future forecast - Use LATEST available features (from holdout if available, else train)
                        try:
                            # Determine input for future forecast
                            if len(X_holdout_for_models) > 0:
                                X_last = X_holdout_for_models[-1:].copy()
                            else:
                                X_last = X_train_for_models[-1:].copy()
                            
                            # Make prediction
                            future_pred_scaled = self._predict_with_model(
                                result['model'], model_name, X_last, config=result
                            )
                            
                            # Inverse transform
                            if len(future_pred_scaled.flatten()) >= horizon:
                                # Multi-output model - direct prediction
                                future_pred_scaled = future_pred_scaled.flatten()[:horizon]
                                future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(1, -1)).flatten()
                                
                                if use_pct:
                                    # Convert transformed values back to price
                                    if FORECAST_UTILS_AVAILABLE:
                                        future_pred = from_target(future_pred.reshape(1, -1), target_transform, np.array([last_actual_price])).flatten()
                                    elif target_transform == 'pct_change':
                                        future_pred = last_actual_price * (1 + future_pred)
                                    elif target_transform == 'log_return':
                                        future_pred = last_actual_price * np.exp(future_pred)
                            else:
                                # Single output - repeat last prediction
                                val = scaler_y.inverse_transform([[future_pred_scaled.flatten()[0]]])[0, 0]
                                if use_pct:
                                    if target_transform == 'pct_change':
                                        val = last_actual_price * (1 + val)
                                    elif target_transform == 'log_return':
                                        val = last_actual_price * np.exp(val)
                                future_pred = np.full(horizon, val)
                            
                            benchmark_results['future_forecasts'][model_name] = {
                                'predictions': future_pred.tolist(),
                                'last_price': last_actual_price,
                                'horizon': horizon,
                                'method': 'DIRECT_FROM_LAST_FEATURES',
                                'note': 'Using last feature vector to avoid feature mismatch'
                            }
                            logger.info(f"    Future forecast: ${future_pred[0]:.2f} (day 1) → ${future_pred[-1]:.2f} (day {horizon})")
                            
                        except Exception as fut_err:
                            logger.warning(f"  {model_name} future forecast failed: {fut_err}")
                            benchmark_results['future_forecasts'][model_name] = {
                                'predictions': [last_actual_price] * horizon,
                                'last_price': last_actual_price,
                                'error': str(fut_err),
                                'method': 'FALLBACK_NAIVE'
                            }
                        
                        # Store details for report (including model for rolling holdout)
                        regressor_details[model_name] = {
                            'model': result['model'],  # Keep model in memory for rolling holdout
                            'y_val': y_val_original,
                            'y_pred': y_pred_original,
                            'y_val_transformed': y_val_transformed,  # Values in training space (log_return/pct_change)
                            'y_pred_transformed': y_pred_transformed,  # Predictions in training space
                            'y_val_scaled': y_val_scaled,  # StandardScaler-scaled values (what model actually sees)
                            'y_pred_scaled': y_pred_scaled,  # Model's raw scaled output
                            'target_transform': target_transform,  # Transform type for labeling
                            'history': result.get('history'),
                            'rmse_per_step': result.get('rmse_per_step', []),
                            'lstm_config': result.get('lstm_config'),  # For enhanced LSTM
                            'xlstm_config': result.get('xlstm_config'),  # For xLSTM
                            'rmse_by_segment': result.get('rmse_by_segment'),  # For enhanced DNN
                            'pipeline_info': {
                                'raw_data_shape': (len(X_train_for_models), X_train_for_models.shape[1]) if len(X_train_for_models) > 0 else (0, 0),
                                'model_input_shape': X_train_for_models.shape,
                                'model_output_shape': y_train.shape,
                                'model_type': 'regressor',
                                'forecast_horizon': horizon,
                                'feature_count': X_train_for_models.shape[1] if len(X_train_for_models) > 0 else 0
                            }
                        }
                        
                        self._save_model(result['model'], model_name, 'regressor')
                        logger.info(f"  {model_name}: R²_avg={val_metrics.get('r2_avg', 0):.4f}, RMSE_avg=${val_metrics.get('rmse_avg', 0):.2f}")
                        
                except Exception as e:
                    import traceback
                    error_tb = traceback.format_exc()
                    logger.warning(f"  {model_name} failed: {str(e)}")
                    traceback.print_exc()
                    # Store error for report
                    benchmark_results['model_errors'][model_name] = {
                        'exception': str(e),
                        'exception_type': type(e).__name__,
                        'traceback': error_tb
                    }
            
            # Add naive baseline to forecasts
            benchmark_results['future_forecasts']['naive_baseline'] = {
                'predictions': [last_actual_price] * horizon,
                'last_price': last_actual_price,
                'note': 'Naive: last price for all steps'
            }
            
            # ===== SAMPLE FORECASTS (NEW) =====
            # Generate 10 random test cases for visualization
            logger.info("\n--- GENERATING SAMPLE FORECASTS ---")
            benchmark_results['sample_forecasts'] = []
            
            if len(X_holdout_clean) > 0:
                try:
                    num_samples = min(10, len(X_holdout_clean))
                    # Use fixed seed for reproducibility of samples
                    rng = np.random.RandomState(42)
                    sample_indices_clean = rng.choice(len(X_holdout_clean), num_samples, replace=False)
                    sample_indices_clean.sort()
                    
                    # Map back to original indices
                    holdout_indices_map = np.where(valid_mask_holdout)[0]
                    
                    for idx_clean in sample_indices_clean:
                        idx_holdout = holdout_indices_map[idx_clean]
                        global_idx = train_end_idx + idx_holdout
                        
                        # Get history (30 days)
                        hist_start = max(0, global_idx - 29)
                        history = original_close_prices_clean[hist_start : global_idx + 1].tolist()
                        
                        # Get actual future in PRICE space (not log returns)
                        # Use y_holdout_original_prices which contains actual prices
                        actual_future = y_holdout_original_prices[idx_holdout].tolist()
                        
                        # Get last known price for this sample (for inverse transform)
                        sample_last_known_price = last_known_holdout[idx_holdout]
                        
                        sample_data = {
                            'index': int(idx_clean),
                            'history': history,
                            'actual_future': actual_future,
                            'last_known_price': float(sample_last_known_price),
                            'model_predictions': {}
                        }
                        
                        # Get predictions from each model
                        X_sample = X_holdout_for_models[idx_clean:idx_clean+1]
                        
                        for model_name, details in regressor_details.items():
                            if 'model' in details and details['model'] is not None:
                                model = details['model']
                                try:
                                    # Handle different model types
                                    if model_name == 'lstm':
                                        lstm_config = details.get('lstm_config')
                                        if lstm_config:
                                            timesteps = lstm_config['timesteps']
                                            features_per_step = lstm_config['features_per_step']
                                            total_needed = timesteps * features_per_step
                                            n_features = X_sample.shape[1]
                                            
                                            if n_features < total_needed:
                                                X_s_padded = np.pad(X_sample, 
                                                                   ((0, 0), (0, total_needed - n_features)), 
                                                                   mode='constant')
                                            else:
                                                X_s_padded = X_sample[:, :total_needed]
                                            
                                            X_s_3d = X_s_padded.reshape((1, timesteps, features_per_step))
                                        else:
                                            X_s_3d = X_sample.reshape((1, 1, -1))
                                        pred_scaled = model.predict(X_s_3d, verbose=0)
                                    elif model_name == 'xlstm':
                                        import torch
                                        xlstm_config = details.get('xlstm_config')
                                        if xlstm_config:
                                            timesteps = xlstm_config['timesteps']
                                            features_per_step = xlstm_config['features_per_step']
                                            total_needed = timesteps * features_per_step
                                            n_features = X_sample.shape[1]
                                            
                                            if n_features < total_needed:
                                                X_s_padded = np.pad(X_sample, 
                                                                   ((0, 0), (0, total_needed - n_features)), 
                                                                   mode='constant')
                                            else:
                                                X_s_padded = X_sample[:, :total_needed]
                                            
                                            X_s_3d = X_s_padded.reshape((1, timesteps, features_per_step))
                                        else:
                                            X_s_3d = X_sample.reshape((1, 1, -1))
                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                        model.eval()
                                        with torch.no_grad():
                                            X_t = torch.FloatTensor(X_s_3d).to(device)
                                            pred_scaled = model(X_t).cpu().numpy()
                                    elif model_name == 'dnn':
                                        pred_scaled = model.predict(X_sample, verbose=0)
                                    elif model_name in ['tcn', 'nbeats', 'moe', 'multitask']:
                                        import torch
                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                        model.to(device)
                                        model.eval()
                                        with torch.no_grad():
                                            X_t = torch.FloatTensor(X_sample).to(device)
                                            if model_name == 'multitask':
                                                pred_scaled = model(X_t)[0].cpu().numpy()
                                            else:
                                                pred_scaled = model(X_t).cpu().numpy()
                                    else:
                                        # sklearn models
                                        pred_scaled = model.predict(X_sample)
                                    
                                    # Inverse transform predictions (from scaled space to target space)
                                    pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1) if pred_scaled.ndim == 1 else pred_scaled)
                                    
                                    # Handle different output shapes
                                    if len(pred.shape) > 1:
                                        pred = pred[0]
                                    
                                    # CRITICAL: Apply from_target to convert log_return/pct_change to prices
                                    if use_pct:
                                        base_price = np.array([sample_last_known_price])
                                        pred_2d = pred.reshape(1, -1)  # Shape (1, horizon)
                                        if FORECAST_UTILS_AVAILABLE:
                                            pred = from_target(pred_2d, target_transform, base_price).flatten()
                                        elif target_transform == 'pct_change':
                                            pred = (base_price * (1 + pred)).flatten()
                                        elif target_transform == 'log_return':
                                            pred = (base_price * np.exp(pred)).flatten()
                                    
                                    sample_data['model_predictions'][model_name] = pred.tolist()
                                except Exception as e:
                                    logger.debug(f"Failed to predict sample for {model_name}: {e}")
                        
                        benchmark_results['sample_forecasts'].append(sample_data)
                    logger.info(f"  Generated {len(benchmark_results['sample_forecasts'])} sample forecasts")
                except Exception as e:
                    logger.warning(f"Failed to generate sample forecasts: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ===== ROLLING HOLDOUT AGGREGATION (NEW) =====
            # For better visualization: aggregate overlapping predictions per date
            logger.info("\n--- ROLLING HOLDOUT AGGREGATION ---")
            
            benchmark_results['holdout_rolling'] = {}
            
            # CRITICAL: Use X_holdout_for_models (processed features) for consistency
            if FORECAST_UTILS_AVAILABLE and len(X_holdout_for_models) > horizon:
                try:
                    # Get holdout aggregation method from config
                    holdout_agg_method = getattr(self, 'holdout_agg_method', 'mean')
                    stride = getattr(self, 'holdout_stride', 1)
                    
                    # Use models directly from regressor_details (in memory with attributes)
                    # instead of loading from files (which loses custom attributes)
                    trained_models = {}
                    lstm_configs = {}  # Store LSTM config separately
                    xlstm_configs = {}  # Store xLSTM config separately
                    
                    for model_name, details in regressor_details.items():
                        if model_name in self.VALID_REGRESSORS:
                            # Get model from regressor_details if stored (preferred - in memory)
                            if 'model' in details and details['model'] is not None:
                                trained_models[model_name] = details['model']
                                if details.get('lstm_config'):
                                    lstm_configs[model_name] = details['lstm_config']
                                if details.get('xlstm_config'):
                                    xlstm_configs[model_name] = details['xlstm_config']
                                logger.info(f"    Using in-memory model for {model_name}")
                            else:
                                # Load from file as fallback
                                if model_name in ['lstm', 'dnn']:
                                    model_path = os.path.join(self.output_dir, f'{model_name}_reg.h5')
                                    if os.path.exists(model_path):
                                        try:
                                            from tensorflow.keras.models import load_model
                                            trained_models[model_name] = load_model(model_path)
                                            # Restore LSTM config from regressor_details
                                            if details.get('lstm_config'):
                                                lstm_configs[model_name] = details['lstm_config']
                                            logger.info(f"    Loaded {model_name} from file")
                                        except Exception as load_err:
                                            logger.warning(f"Could not load {model_name}: {load_err}")
                                elif model_name in ['xlstm', 'tcn', 'nbeats', 'moe', 'multitask']:
                                    # PyTorch models saved with .pt extension
                                    model_path = os.path.join(self.output_dir, f'{model_name}_reg.pt')
                                    if os.path.exists(model_path):
                                        try:
                                            import torch
                                            trained_models[model_name] = torch.load(model_path)
                                            if details.get('xlstm_config'):
                                                xlstm_configs[model_name] = details['xlstm_config']
                                            logger.info(f"    Loaded {model_name} from file")
                                        except Exception as load_err:
                                            logger.warning(f"Could not load {model_name}: {load_err}")
                                else:
                                    model_path = os.path.join(self.output_dir, f'{model_name}_reg.pkl')
                                    if os.path.exists(model_path):
                                        try:
                                            with open(model_path, 'rb') as f:
                                                trained_models[model_name] = pickle.load(f)
                                            logger.info(f"    Loaded {model_name} from file")
                                        except Exception as load_err:
                                            logger.warning(f"Could not load {model_name}: {load_err}")
                    
                    # Log loaded models summary
                    logger.info(f"    Loaded {len(trained_models)} models for rolling holdout: {list(trained_models.keys())}")
                    
                    # Holdout Close prices for rolling prediction
                    holdout_close_for_roll = holdout_close[:len(X_holdout_clean)]
                    holdout_length = len(holdout_close_for_roll)
                    
                    for model_name, model in trained_models.items():
                        try:
                            # Map model names to model types for roll_multistep_predictions
                            if model_name == 'lstm':
                                model_type = 'lstm'
                            elif model_name == 'xlstm':
                                model_type = 'xlstm'
                            elif model_name == 'dnn':
                                model_type = 'keras'
                            elif model_name in ['tcn', 'nbeats', 'moe', 'multitask']:
                                model_type = model_name  # PyTorch Model Zoo
                            else:
                                model_type = 'sklearn'
                            
                            # Get LSTM/xLSTM config if available
                            model_lstm_config = lstm_configs.get(model_name) or xlstm_configs.get(model_name)
                            
                            # Generate rolling multi-step predictions
                            # CRITICAL: Use X_holdout_for_models (processed features) to match training
                            pred_matrix, origin_indices = roll_multistep_predictions(
                                model=model,
                                X_holdout=X_holdout_for_models,
                                y_scaler=scaler_y,
                                horizon=horizon,
                                stride=stride,
                                model_type=model_type,
                                lstm_config=model_lstm_config
                            )
                            
                            if len(pred_matrix) == 0:
                                logger.warning(f"    {model_name}: No rolling predictions generated")
                                continue
                            
                            # NEW: Handle target transforms for rolling predictions
                            if use_pct:
                                # pred_matrix contains transformed values (pct_change or log_return)
                                # Convert to prices using appropriate inverse transform
                                base_prices = last_known_holdout[origin_indices]
                                if FORECAST_UTILS_AVAILABLE:
                                    pred_matrix = from_target(pred_matrix, target_transform, base_prices)
                                elif target_transform == 'pct_change':
                                    pred_matrix = base_prices[:, None] * (1 + pred_matrix)
                                elif target_transform == 'log_return':
                                    pred_matrix = base_prices[:, None] * np.exp(pred_matrix)
                            
                            # Aggregate overlapping predictions to daily values
                            agg_result = aggregate_rolling_predictions(
                                pred_matrix=pred_matrix,
                                origin_indices=origin_indices,
                                holdout_length=holdout_length,
                                horizon=horizon,
                                method=holdout_agg_method
                            )
                            
                            # Extract aggregated predictions
                            daily_pred = agg_result['pred_agg']
                            daily_std = agg_result['pred_std']
                            valid_mask = agg_result['valid_mask']
                            
                            # Evaluate against actual holdout Close
                            roll_metrics = evaluate_rolling_forecast(
                                actual=holdout_close_for_roll,
                                aggregated=agg_result
                            )
                            
                            # Align to valid dates for storage
                            n_daily = min(len(daily_pred), len(holdout_close_for_roll))
                            daily_actual = holdout_close_for_roll[:n_daily]
                            daily_pred_aligned = daily_pred[:n_daily]
                            daily_std_aligned = daily_std[:n_daily]
                            
                            # Translate metrics keys for consistency
                            if 'rmse_agg' in roll_metrics:
                                roll_metrics['rmse'] = roll_metrics['rmse_agg']
                            if 'mae_agg' in roll_metrics:
                                roll_metrics['mae'] = roll_metrics['mae_agg']
                            if 'r2_agg' in roll_metrics:
                                roll_metrics['r2'] = roll_metrics['r2_agg']
                            
                            benchmark_results['holdout_rolling'][model_name] = {
                                'daily_predictions': daily_pred_aligned.tolist(),
                                'daily_std': daily_std_aligned.tolist(),
                                'daily_actual': daily_actual.tolist(),
                                'metrics': roll_metrics,
                                'agg_method': holdout_agg_method,
                                'stride': stride,
                                'n_predictions': len(daily_pred_aligned)
                            }
                            
                            logger.info(f"    {model_name}: RMSE=${roll_metrics.get('rmse', roll_metrics.get('rmse_agg', 0)):.2f}, "
                                       f"MAE=${roll_metrics.get('mae', roll_metrics.get('mae_agg', 0)):.2f}, "
                                       f"R²={roll_metrics.get('r2', roll_metrics.get('r2_agg', 0)):.4f}")
                        
                        except Exception as roll_err:
                            logger.warning(f"    {model_name} rolling holdout failed: {roll_err}")
                            benchmark_results['holdout_rolling'][model_name] = {'error': str(roll_err)}
                    
                    # Compute rolling baselines (same aggregation method)
                    logger.info("  Computing rolling baselines...")
                    baseline_results = compute_rolling_baselines(
                        close_train=train_close,
                        close_holdout=holdout_close_for_roll,
                        horizon=horizon,
                        stride=stride,
                        agg_method=holdout_agg_method
                    )
                    
                    # Process baseline results to add metrics
                    processed_baselines = {}
                    for baseline_name, baseline_data in baseline_results.items():
                        if 'aggregated' in baseline_data:
                            # Evaluate the baseline using the aggregated dict
                            baseline_metrics = evaluate_rolling_forecast(
                                actual=holdout_close_for_roll,
                                aggregated=baseline_data['aggregated']
                            )
                            # Normalize metric keys
                            if 'rmse_agg' in baseline_metrics:
                                baseline_metrics['rmse'] = baseline_metrics['rmse_agg']
                            if 'mae_agg' in baseline_metrics:
                                baseline_metrics['mae'] = baseline_metrics['mae_agg']
                            if 'r2_agg' in baseline_metrics:
                                baseline_metrics['r2'] = baseline_metrics['r2_agg']
                            
                            processed_baselines[baseline_name] = {
                                'metrics': baseline_metrics,
                                'description': baseline_data.get('description', '')
                            }
                            logger.info(f"    {baseline_name}: RMSE=${baseline_metrics.get('rmse', 0):.2f}")
                    
                    benchmark_results['rolling_baselines'] = processed_baselines
                    
                except Exception as roll_err:
                    logger.warning(f"  Rolling holdout aggregation failed: {roll_err}")
                    import traceback
                    traceback.print_exc()
                    benchmark_results['holdout_rolling'] = {'error': str(roll_err)}
            else:
                logger.info("  Skipping rolling holdout (forecast_utils not available or insufficient data)")
        
            # ===== RECURSIVE HOLDOUT FORECAST (TRUE AUTOREGRESSIVE) =====
            # This is the real-world scenario: predict native horizon, feed back, continue
            logger.info("\n--- RECURSIVE HOLDOUT FORECAST (TRUE AUTOREGRESSIVE) ---")
            
            benchmark_results['recursive_forecasts'] = {}
            benchmark_results['holdout_actual_transformed'] = []
            
            if FORECAST_UTILS_AVAILABLE and len(X_holdout_for_models) > 0:
                try:
                    # Get the first feature vector in holdout (start of recursive forecast)
                    X_start = X_holdout_for_models[0:1].copy()
                    holdout_length = len(holdout_close)
                    
                    # Compute actual transformed values for holdout period
                    if use_pct and len(holdout_close) > 1:
                        # Get log returns for actual holdout
                        actual_transformed_list = []
                        for i in range(1, len(holdout_close)):
                            if target_transform == 'log_return':
                                val = np.log(holdout_close[i] / holdout_close[i-1])
                            elif target_transform == 'pct_change':
                                val = (holdout_close[i] - holdout_close[i-1]) / holdout_close[i-1]
                            else:
                                val = holdout_close[i]
                            actual_transformed_list.append(val)
                        benchmark_results['holdout_actual_transformed'] = actual_transformed_list
                    
                    # Last known price before holdout starts
                    last_price_before_holdout = train_close[-1] if len(train_close) > 0 else holdout_close[0]
                    
                    for model_name, details in regressor_details.items():
                        if 'model' not in details or details['model'] is None:
                            continue
                        
                        model = details['model']
                        
                        try:
                            # Determine model type
                            if model_name == 'lstm':
                                model_type = 'lstm'
                                lstm_config = details.get('lstm_config')
                            elif model_name == 'xlstm':
                                model_type = 'xlstm'
                                lstm_config = details.get('xlstm_config')
                            elif model_name == 'dnn':
                                model_type = 'keras'
                                lstm_config = None
                            elif model_name in ['tcn', 'nbeats', 'moe', 'multitask']:
                                model_type = model_name
                                lstm_config = None
                            else:
                                model_type = 'sklearn'
                                lstm_config = None
                            
                            # Generate recursive forecast
                            recursive_result = recursive_holdout_forecast(
                                model=model,
                                X_start=X_start,
                                y_scaler=scaler_y,
                                holdout_length=holdout_length,
                                model_horizon=horizon,
                                model_type=model_type,
                                lstm_config=lstm_config,
                                target_transform=target_transform,
                                last_known_price=last_price_before_holdout
                            )
                            
                            # Calculate metrics
                            pred_prices = recursive_result['predictions']
                            
                            # Check if we got any predictions
                            if len(pred_prices) == 0:
                                raise ValueError("No predictions generated - model may have failed to produce output")
                            
                            valid_len = min(len(pred_prices), len(holdout_close))
                            
                            rmse = float(np.sqrt(mean_squared_error(
                                holdout_close[:valid_len], pred_prices[:valid_len])))
                            mae = float(mean_absolute_error(
                                holdout_close[:valid_len], pred_prices[:valid_len]))
                            r2 = float(r2_score(
                                holdout_close[:valid_len], pred_prices[:valid_len]))
                            
                            benchmark_results['recursive_forecasts'][model_name] = {
                                'predictions': pred_prices.tolist() if hasattr(pred_prices, 'tolist') else pred_prices,
                                'predictions_transformed': recursive_result['predictions_transformed'].tolist() 
                                    if hasattr(recursive_result['predictions_transformed'], 'tolist') 
                                    else recursive_result['predictions_transformed'],
                                'chunk_boundaries': recursive_result['chunk_boundaries'],
                                'method': recursive_result['method'],
                                'n_chunks': recursive_result['n_chunks'],
                                'model_horizon': recursive_result['model_horizon'],
                                'metrics': {
                                    'rmse': rmse,
                                    'mae': mae,
                                    'r2': r2
                                }
                            }
                            
                            logger.info(f"    {model_name}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}")
                            
                        except Exception as rec_err:
                            logger.warning(f"    {model_name} recursive forecast failed: {rec_err}")
                            import traceback
                            traceback.print_exc()
                            benchmark_results['recursive_forecasts'][model_name] = {'error': str(rec_err)}
                    
                    # Store holdout actual prices for plotting
                    benchmark_results['holdout_actual_prices'] = holdout_close.tolist()
                    benchmark_results['target_transform'] = target_transform
                    
                except Exception as rec_all_err:
                    logger.warning(f"  Recursive holdout forecast failed: {rec_all_err}")
                    import traceback
                    traceback.print_exc()
                    benchmark_results['recursive_forecasts'] = {'error': str(rec_all_err)}
        
        # ===== LLM REVIEW (if enabled) =====
        if llm_config and llm_config.get('enabled', False):
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm'))
                from ollama_client import OllamaClient
                
                client = OllamaClient(
                    base_url=llm_config.get('base_url', 'http://localhost:11434'),
                    model=llm_config.get('model', 'llama3.1:8b-instruct'),
                    timeout_s=llm_config.get('timeout_s', 120)
                )
                
                if client.is_available():
                    user_prompt = client.generate_review_prompt(benchmark_results)
                    llm_response = client.chat(user_prompt=user_prompt)
                    
                    if llm_response['success']:
                        benchmark_results['llm_review'] = {
                            'content': llm_response['content'],
                            'model': llm_response['model'],
                            'duration_s': llm_response.get('duration_s', 0)
                        }
                        logger.info(f"  LLM review completed in {llm_response.get('duration_s', 0):.1f}s")
                    else:
                        benchmark_results['llm_review'] = {
                            'error': llm_response['error'],
                            'model': llm_response['model']
                        }
                else:
                    benchmark_results['llm_review'] = {
                        'error': 'Ollama server not available. Start with: ollama serve'
                    }
            except Exception as llm_err:
                benchmark_results['llm_review'] = {'error': str(llm_err)}
                logger.warning(f"  LLM review failed: {llm_err}")
        
        # ===== GENERATE REPORTS =====
        if generate_reports:
            self._generate_all_reports(
                benchmark_results, 
                ticker=ticker,
                classifier_details=classifier_details,
                regressor_details=regressor_details,
                y_all=original_close_prices_clean
            )
        
        return benchmark_results
    
    def _generate_all_reports(self, results: Dict, ticker: str = None,
                              classifier_details: Dict = None, 
                              regressor_details: Dict = None,
                              y_all: np.ndarray = None):
        """
        Generate all reports (advanced + legacy) with error handling and stub fallback.
        
        This is the unified report generation method used by train_and_evaluate().
        
        Args:
            results: Benchmark or training results dictionary
            ticker: Asset name for report titles
            classifier_details: Detailed classifier metrics (optional)
            regressor_details: Detailed regressor metrics (optional)
            y_all: Full timeseries for context plots (optional)
        """
        ticker = ticker or results.get('ticker', 'Asset')
        generation_errors = {}
        
        # Use stored classifier/regressor results if not provided
        if classifier_details is None:
            classifier_details = results.get('classifiers', {})
        if regressor_details is None:
            regressor_details = results.get('regressors', {})
        
        # Try to get y_all from data if not provided
        if y_all is None and hasattr(self, 'data') and 'Close' in self.data.columns:
            y_all = self.data['Close'].dropna().values
        
        horizon = results.get('forecast_horizon', self.config.get('forecast_horizon', 30))
        
        # Add period_unit to results for report labels (days, hours, 5m candles, etc.)
        results['period_unit'] = self._get_period_unit()
        results['candle_length'] = self.candle_length
        
        # ===== GENERATE ADVANCED REPORT =====
        if REPORT_GENERATOR_AVAILABLE:
            try:
                report_gen = ReportGenerator(self.output_dir)
                report_path = report_gen.generate_comprehensive_report(
                    results, classifier_details, regressor_details, ticker,
                    forecast_horizon=horizon, y_all=y_all
                )
                logger.info(f"Advanced benchmark report saved: {report_path}")
            except Exception as e:
                import traceback
                error_tb = traceback.format_exc()
                logger.warning(f"Could not generate advanced report: {str(e)}")
                generation_errors['benchmark_report_advanced.html'] = {
                    'exception': str(e),
                    'traceback': error_tb
                }
        
        # ===== SAVE RESULTS JSON =====
        results_json_path = os.path.join(self.output_dir, 'benchmark_results.json')
        
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            return obj
        
        try:
            serializable_results = make_serializable(results)
            with open(results_json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Results JSON saved: {results_json_path}")
        except Exception as e:
            logger.warning(f"Could not save results JSON: {e}")
        
        # ===== SAVE CONFIG SNAPSHOT =====
        config_snapshot_path = os.path.join(self.output_dir, 'config_snapshot.json')
        try:
            config_snapshot = make_serializable(self.config)
            with open(config_snapshot_path, 'w') as f:
                json.dump(config_snapshot, f, indent=2)
            logger.info(f"Config snapshot saved: {config_snapshot_path}")
            results['artifacts'] = results.get('artifacts', {})
            results['artifacts']['config_snapshot'] = config_snapshot_path
        except Exception as cfg_err:
            logger.warning(f"Could not save config snapshot: {cfg_err}")
        
        # ===== ENSURE REQUIRED REPORTS EXIST =====
        if REPORT_GENERATOR_AVAILABLE:
            ensure_result = ensure_required_reports(
                self.output_dir,
                benchmark_results=results,
                generation_errors=generation_errors
            )
            if ensure_result['created_stubs']:
                logger.warning(f"Created {len(ensure_result['created_stubs'])} stub report(s)")
