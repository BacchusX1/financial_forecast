"""
FeatureEngineering: Apply transformations to financial data for ML/analysis.

Supports:
  - Scaling (Standard, MinMax, Robust)
  - Time series features (lag, rolling windows, technical indicators)
  - Dimensionality reduction (PCA, Autoencoder latent space)
  - Categorical encoding

Usage:
  from financial_data_loader import DataLoader
  
  # Load data
  loader = DataLoader(config)
  data = loader.assemble_data()
  
  # Apply features
  fe_config = {
      "scaler": "standard",  # or "minmax", "robust"
      "lag_features": [1, 5, 10],
      "rolling_windows": [5, 20],
      "target_columns": ["Close", "Volume"]
  }
  fe = FeatureEngineering(data, fe_config)
  transformed = fe.transform()
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
# PyTorch is optional for autoencoder features
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from typing import Dict, List, Optional, Union
import logging
import warnings
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import indicator engine (optional dependency)
try:
    from indicator_engine import IndicatorEngine
    INDICATOR_ENGINE_AVAILABLE = True
except ImportError:
    INDICATOR_ENGINE_AVAILABLE = False
    logger.warning("IndicatorEngine not available. Advanced indicator features disabled.")

# Try to import feature scale classifier (optional dependency)
try:
    from feature_scale_classifier import (
        classify_feature_columns,
        filter_features_by_scale,
        filter_columns_by_base_price,
        PRICE_COLUMNS
    )
    SCALE_CLASSIFIER_AVAILABLE = True
except ImportError:
    SCALE_CLASSIFIER_AVAILABLE = False
    PRICE_COLUMNS = {'Close', 'Open', 'High', 'Low', 'Adj Close'}
    logger.warning("FeatureScaleClassifier not available. Scale filtering disabled.")


if TORCH_AVAILABLE:
    class AutoencoderNetwork(nn.Module):
        """PyTorch autoencoder for dimensionality reduction."""
        
        def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
            super(AutoencoderNetwork, self).__init__()
            
            if hidden_dims is None:
                hidden_dims = [max(latent_dim * 4, input_dim // 2), latent_dim * 2]
            
            # Encoder
            encoder_layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = h_dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Decoder
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
else:
    # Dummy class when torch not available
    class AutoencoderNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Cannot use AutoencoderNetwork.")


class FeatureEngineering:
    """Transform financial data with scaling, lag/rolling features, and autoencoder."""
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    RESERVED_COLUMNS = ['Datetime', 'Ticker', 'Date']
    
    def __init__(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame], config: dict = None):
        """
        Initialize with data from DataLoader or DataFrame.
        
        Args:
            data: Dict from DataLoader.assemble_data() or combined DataFrame
            config: Configuration with optional keys:
                - scaler: "standard", "minmax", "robust" (default: None)
                - target_columns: List of columns to transform (default: numeric columns)
                - lag_features: List of lag periods (e.g., [1, 5, 10])
                - rolling_windows: List of window sizes (e.g., [5, 20])
                - rolling_functions: List of functions (default: ["mean", "std"])
                - pca_components: Number of PCA components (default: None)
                - autoencoder_latent_dim: Latent space dimension (default: None)
                - autoencoder_epochs: Training epochs (default: 100)
                - group_by_ticker: Apply transformations per ticker (default: True)
        """
        if config is None:
            config = {}
        
        # Handle input data
        if isinstance(data, dict):
            self.data_dict = data
            self.data = pd.concat(data.values(), ignore_index=True)
            logger.info(f"Loaded {len(data)} assets, {self.data.shape[0]} total records")
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.data_dict = None
            logger.info(f"Loaded DataFrame: {self.data.shape[0]} records")
        else:
            raise ValueError("Data must be Dict[str, DataFrame] or DataFrame")
        
        if self.data.empty:
            raise ValueError("Data is empty")
        
        # Configuration
        self.config = config
        self.scaler_type = config.get("scaler")
        self.target_columns = config.get("target_columns")
        self.lag_features = config.get("lag_features", [])
        self.rolling_windows = config.get("rolling_windows", [])
        self.rolling_functions = config.get("rolling_functions", ["mean", "std"])
        
        # DEPRECATED: PCA/AE in FeatureEngineering causes leakage
        # These are now handled by DataScientist/FeatureProcessor on training split only
        self.pca_components = config.get("pca_components")
        self.autoencoder_latent_dim = config.get("autoencoder_latent_dim")
        self.autoencoder_epochs = config.get("autoencoder_epochs", 100)
        
        if self.pca_components is not None:
            logger.warning("DEPRECATION: 'pca_components' in FeatureEngineering is deprecated to prevent data leakage.")
            logger.warning("Please configure PCA in the Data Science module instead.")
            self.pca_components = None # Disable it

        if self.autoencoder_latent_dim is not None:
             logger.warning("DEPRECATION: 'autoencoder_latent_dim' in FeatureEngineering is deprecated.")
             logger.warning("Please configure Autoencoder in the Data Science module instead.")
             self.autoencoder_latent_dim = None # Disable it

        self.group_by_ticker = config.get("group_by_ticker", True)
        self.allow_absolute_features = config.get("allow_absolute_scale_features", True)
        self.enforce_shift_1 = config.get("leakage_guard", {}).get("enforce_shift_1", True)
        
        # New: Base price column restriction
        self.base_price_column = config.get("base_price_column", "Close")
        self.allow_additional_price_columns = config.get("allow_additional_price_columns", True)

        # Storage for fitted objects
        self.scaler = None
        self.pca = None
        self.autoencoder = None
        self.fitted = False
        self.exclude_from_scaling = config.get("exclude_from_scaling", [])  # Read from config, default empty list

        # Auto-detect target columns if not specified
        if self.target_columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.target_columns = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]

        if not self.target_columns:
            raise ValueError("No numeric columns found for feature engineering")
        
        logger.info(f"FeatureEngineering initialized - Target columns: {self.target_columns}")
    
    def create_indicator_features(self, data: pd.DataFrame = None, 
                                  enabled_groups: List[str] = None,
                                  apply_pruning: bool = True) -> pd.DataFrame:
        """
        Create technical indicator features using the IndicatorEngine.
        
        This method computes 70+ technical indicators with automatic leakage prevention:
        - All indicators are automatically shifted by 1 period (.shift(1))
        - Features at time t only contain information up to time t-1
        - Supports feature pruning (constant, high NaN, correlation)
        
        Args:
            data: Input DataFrame (uses self.data if None)
            enabled_groups: List of indicator groups to enable
                Options: ['returns', 'trend', 'volatility', 'momentum', 'volume', 'candle', 'regime']
            apply_pruning: Whether to apply feature pruning
        
        Returns:
            DataFrame with original columns + indicator features (all shifted for leakage safety)
        
        Example:
            fe = FeatureEngineering(data, config)
            df_with_indicators = fe.create_indicator_features(
                enabled_groups=['returns', 'trend', 'momentum']
            )
        """
        if not INDICATOR_ENGINE_AVAILABLE:
            logger.error("IndicatorEngine not available. Cannot create indicator features.")
            logger.error("Make sure indicator_engine.py and indicator_registry.py are in the same directory.")
            raise ImportError("IndicatorEngine not available")
        
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Build config for indicator engine from feature_engineering config
        indicator_config = self.config.get('feature_engineering', {}) if 'feature_engineering' in self.config else self.config
        
        # Create indicator engine
        engine = IndicatorEngine(indicator_config)
        
        # Compute indicators
        logger.info("Creating indicator features with IndicatorEngine...")
        df_with_indicators = engine.compute_indicators(data, enabled_groups=enabled_groups)
        
        # Apply pruning if requested
        # WARNING: Pruning here uses FULL dataset which causes leakage!
        # Recommended: Set apply_pruning=False and use FeaturePruner after train/test split.
        if apply_pruning:
            logger.warning("⚠️  WARNING: Pruning on full dataset causes DATA LEAKAGE!")
            logger.warning("   Pruning decisions (correlation, NaN ratio) use test data statistics.")
            logger.warning("   Recommended: Set apply_pruning=False and prune after train/test split.")
            df_pruned = engine.prune_features(df_with_indicators)
        else:
            df_pruned = df_with_indicators
        
        # Validate leakage prevention
        engine.assert_no_leakage(df_pruned)
        
        # Update target columns to include new features
        numeric_cols = df_pruned.select_dtypes(include=[np.number]).columns.tolist()
        self.target_columns = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]
        
        # Print summary
        summary = engine.get_summary()
        logger.info("=" * 80)
        logger.info("INDICATOR FEATURES SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Original columns: {data.shape[1]}")
        logger.info(f"New feature columns: {len(summary['new_feature_columns'])}")
        logger.info(f"Total columns: {df_pruned.shape[1]}")
        logger.info(f"Rows: {df_pruned.shape[0]}")
        logger.info(f"Enabled groups: {summary['enabled_groups']}")
        logger.info("=" * 80)
        
        return df_pruned
    
    def scale_data(self, data: pd.DataFrame, scaler_type: str = None, fit: bool = True, 
                   exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Apply scaling to numeric columns.
        
        Args:
            data: Input DataFrame
            scaler_type: "standard", "minmax", or "robust"
            fit: Whether to fit scaler (True) or use existing (False)
            exclude_columns: List of column names to exclude from scaling
        
        Returns:
            Scaled DataFrame
        """

        if scaler_type is None:
            scaler_type = self.scaler_type
        
        if scaler_type is None or scaler_type.lower() == 'none':
            logger.info("No scaling applied")
            return data.copy()
        
        if scaler_type not in self.SCALERS:
            raise ValueError(f"Invalid scaler: {scaler_type}. Choose from {list(self.SCALERS.keys())}")
        
        df = data.copy()
        
        # Get all numeric columns to scale (including lag and rolling features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]
        
        # Exclude specified columns
        if exclude_columns:
            cols_to_scale = [col for col in cols_to_scale if col not in exclude_columns]
        
        if not cols_to_scale:
            logger.warning("No columns to scale after exclusions")
            return df
        
        if fit:
            self.scaler = self.SCALERS[scaler_type]()
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            logger.info(f"Applied {scaler_type} scaling (fit) to {len(cols_to_scale)} columns")
            if exclude_columns:
                logger.info(f"Excluded {len(exclude_columns)} columns from scaling: {exclude_columns}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first")
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            logger.info(f"Applied {scaler_type} scaling (transform) to {len(cols_to_scale)} columns")
        
        return df
    
    def create_close_only_features(self, data: pd.DataFrame, lags: List[int] = None,
                                    windows: List[int] = None) -> pd.DataFrame:
        """
        Create features ONLY from Close price for pure price forecasting.
        
        This method creates a clean feature set with:
        1. Close_lag_1, Close_lag_2, ... (lagged prices - past values only)
        2. Close_return_1, Close_return_5, ... (percentage returns)
        3. Close_rolling_N_mean (moving averages)
        4. Close_rolling_N_std (volatility)
        
        CRITICAL: All features are strictly BACKWARD-looking (no future data leakage).
        
        Args:
            data: DataFrame with 'Close' column
            lags: List of lag periods (default: config's lag_features)
            windows: List of rolling window sizes (default: config's rolling_windows)
        
        Returns:
            DataFrame with Close-only features + original Close column
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column for Close-only features")
        
        if lags is None:
            lags = self.lag_features if self.lag_features else [1, 5, 10, 20]
        if windows is None:
            windows = self.rolling_windows if self.rolling_windows else [5, 20]
        
        df = data.copy()
        close = df['Close']
        
        logger.info("=" * 60)
        logger.info("CREATING CLOSE-ONLY FEATURES (Expert Mode)")
        logger.info("=" * 60)
        logger.info(f"  Input: {len(df)} samples")
        logger.info(f"  Close price range: ${close.min():.2f} - ${close.max():.2f}")
        
        # 1. LAG FEATURES: Past Close prices (strictly backward-looking)
        for lag in lags:
            col_name = f"Close_lag_{lag}"
            if self.group_by_ticker and 'Ticker' in df.columns:
                df[col_name] = df.groupby('Ticker')['Close'].shift(lag)
            else:
                df[col_name] = close.shift(lag)
        
        logger.info(f"  Created {len(lags)} lag features: Close_lag_{lags}")
        
        # 2. RETURN FEATURES: Percentage change (measures momentum/direction)
        # CRITICAL: We lag the returns by 1 day to avoid data leakage
        # Close_return_1 at time t = return from t-2 to t-1 (not including current price)
        for lag in [1, 5, 10]:
            col_name = f"Close_return_{lag}"
            if self.group_by_ticker and 'Ticker' in df.columns:
                # Lagged returns: shift by 1 to avoid using current Close
                df[col_name] = df.groupby('Ticker')['Close'].pct_change(periods=lag).shift(1)
            else:
                df[col_name] = close.pct_change(periods=lag).shift(1)
        
        logger.info(f"  Created 3 return features: Close_return_1, 5, 10 (lagged by 1 day)")
        
        # 3. LOG RETURN (better for statistical properties) - also lagged
        df['Close_log_return'] = np.log(close / close.shift(1)).shift(1)
        logger.info(f"  Created log return feature: Close_log_return (lagged by 1 day)")
        
        # 4. ROLLING WINDOW FEATURES (moving averages & volatility)
        # CRITICAL: We shift rolling features by 1 day to avoid data leakage
        # Rolling mean at time t = mean of prices from t-window to t-1 (excludes current)
        for window in windows:
            # Rolling mean (simple moving average) - SHIFTED to exclude current price
            mean_col = f"Close_rolling_{window}_mean"
            if self.group_by_ticker and 'Ticker' in df.columns:
                df[mean_col] = df.groupby('Ticker')['Close'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
            else:
                df[mean_col] = close.rolling(window, min_periods=1).mean().shift(1)
            
            # Rolling std (volatility) - SHIFTED to exclude current price
            std_col = f"Close_rolling_{window}_std"
            if self.group_by_ticker and 'Ticker' in df.columns:
                df[std_col] = df.groupby('Ticker')['Close'].transform(
                    lambda x: x.rolling(window, min_periods=1).std().shift(1)
                )
            else:
                df[std_col] = close.rolling(window, min_periods=1).std().shift(1)
            
            # Price relative to moving average - use lagged values only
            # ma_ratio at t = (Close_{t-1} - MA_{t-1}) / MA_{t-1}
            ratio_col = f"Close_ma_ratio_{window}"
            df[ratio_col] = (close.shift(1) - df[mean_col]) / df[mean_col].replace(0, np.nan)
        
        logger.info(f"  Created {len(windows)*3} rolling features for windows: {windows} (lagged by 1 day)")
        
        # 5. TECHNICAL INDICATORS (all based on Close only)
        # Rate of Change (ROC) - lagged to avoid leakage
        # ROC at t-1: change from t-11 to t-1 (no current price used)
        df['Close_ROC_10'] = ((close.shift(1) - close.shift(11)) / close.shift(11).replace(0, np.nan)) * 100
        
        # Momentum - LAGGED by 1 day to avoid using current Close
        # Momentum at t-1: price change from t-11 to t-1
        df['Close_momentum_10'] = close.shift(1) - close.shift(11)
        
        # Price acceleration (second derivative) - already lagged via return features
        df['Close_acceleration'] = df['Close_return_1'] - df['Close_return_1'].shift(1)
        
        logger.info(f"  Created 3 technical indicators: ROC, momentum, acceleration (all lagged)")
        
        # Summary of created features
        feature_cols = [c for c in df.columns if c.startswith('Close_') and c != 'Close']
        logger.info(f"  TOTAL: {len(feature_cols)} Close-only features created")
        logger.info(f"  Features: {feature_cols}")
        
        # Update target_columns to reflect new feature set
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.target_columns = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = None, 
                           columns: List[str] = None) -> pd.DataFrame:
        """
        Create lagged features for time series.
        
        Args:
            data: Input DataFrame (must have Ticker if group_by_ticker=True)
            lags: List of lag periods (e.g., [1, 5, 10])
            columns: Columns to create lags for (default: target_columns)
        
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = self.lag_features
        
        if not lags:
            return data.copy()
        
        if columns is None:
            columns = self.target_columns
        
        # Filter for base price column restriction
        if not self.allow_additional_price_columns:
            other_price_cols = PRICE_COLUMNS - {self.base_price_column}
            original_len = len(columns)
            columns = [c for c in columns if c not in other_price_cols]
            if len(columns) < original_len:
                logger.info(f"Base-price-only mode: Dropped {original_len - len(columns)} non-{self.base_price_column} columns from lag generation")
        
        # Filter for Scale-Free Mode
        if not self.allow_absolute_features:
            ABSOLUTE_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
            original_len = len(columns)
            columns = [c for c in columns if c not in ABSOLUTE_COLS]
            if len(columns) < original_len:
                logger.info(f"Scale-Free Mode: Dropped {original_len - len(columns)} absolute columns from lag generation")
        
        df = data.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not in data, skipping lags")
                continue
            
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                if self.group_by_ticker and 'Ticker' in df.columns:
                    df[col_name] = df.groupby('Ticker')[col].shift(lag)
                else:
                    df[col_name] = df[col].shift(lag)
        
        created_lags = len(columns) * len(lags)
        logger.info(f"Created {created_lags} lag features (lags: {lags})")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.target_columns = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, windows: List[int] = None,
                               functions: List[str] = None, columns: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Uses min_periods=1, so first entry uses only itself, entries before window size use 
        partial windows, and entries from window size onward use full windows.
        Example: window=5 on 50 entries → row[0] uses 1 value, row[4] uses 5 values, row[49] uses last 5 values.
        
        Args:
            data: Input DataFrame
            windows: Window sizes (e.g., [5, 20])
            functions: Aggregation functions (e.g., ["mean", "std", "min", "max"])
            columns: Columns to create rolling features for
        
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = self.rolling_windows
        
        if not windows:
            return data.copy()
        
        if functions is None:
            functions = self.rolling_functions
        
        if columns is None:
            columns = self.target_columns

        # Filter for base price column restriction
        if not self.allow_additional_price_columns:
            other_price_cols = PRICE_COLUMNS - {self.base_price_column}
            original_len = len(columns)
            columns = [c for c in columns if c not in other_price_cols]
            if len(columns) < original_len:
                logger.info(f"Base-price-only mode: Dropped {original_len - len(columns)} non-{self.base_price_column} columns from rolling generation")

        # Filter for Scale-Free Mode
        if not self.allow_absolute_features:
            ABSOLUTE_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
            original_len = len(columns)
            columns = [c for c in columns if c not in ABSOLUTE_COLS]
            if len(columns) < original_len:
                logger.info(f"Scale-Free Mode: Dropped {original_len - len(columns)} absolute columns from rolling generation")
        
        df = data.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not in data, skipping rolling features")
                continue
            
            for window in windows:
                for func in functions:
                    col_name = f"{col}_rolling_{window}_{func}"
                    rolled = None
                    
                    # Enforce leakage prevention: Shift by 1
                    # Rolling features must be computed from data up to t-1 only.
                    
                    if self.group_by_ticker and 'Ticker' in df.columns:
                        try:
                            # Apply generic rolling function
                            rolled = df.groupby('Ticker')[col].transform(
                                lambda x: getattr(x.rolling(window, min_periods=1), func)()
                            )
                            # Shift grouped results to prevent leakage
                            rolled = rolled.groupby(df['Ticker']).shift(1)
                        except AttributeError:
                           # Fallback or skip if function not found
                           logger.warning(f"Rolling function '{func}' not supported, skipping {col_name}")
                           continue

                    else:
                        try:
                            # Apply generic rolling function
                            r = df[col].rolling(window, min_periods=1)
                            rolled = getattr(r, func)()
                            # Shift result to prevent leakage
                            rolled = rolled.shift(1)
                        except AttributeError:
                            logger.warning(f"Rolling function '{func}' not supported, skipping {col_name}")
                            continue

                    df[col_name] = rolled
        
        created_features = len(columns) * len(windows) * len(functions)
        logger.info(f"Created {created_features} rolling features (windows: {windows}, funcs: {functions})")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.target_columns = [col for col in numeric_cols if col not in self.RESERVED_COLUMNS]

        return df
    
    def apply_pca(self, data: pd.DataFrame, n_components: int = None, 
                  fit: bool = True, columns: List[str] = None) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            data: Input DataFrame
            n_components: Number of components to keep
            fit: Whether to fit PCA or use existing
            columns: Columns to apply PCA to (default: target_columns)
        
        Returns:
            DataFrame with PCA components
        """
        if n_components is None:
            n_components = self.pca_components
        
        if n_components is None:
            return data.copy()
        
        if columns is None:
            columns = self.target_columns
        
        df = data.copy()
        feature_data = df[columns].fillna(0)  # PCA requires no NaN
        
        if fit:
            self.pca = PCA(n_components=n_components)
            components = self.pca.fit_transform(feature_data)
            variance_explained = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA fitted: {n_components} components, {variance_explained:.2%} variance explained")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first")
            components = self.pca.transform(feature_data)
            logger.info(f"PCA transformed: {n_components} components")
        
        # Add PCA components to dataframe
        for i in range(n_components):
            df[f'PCA_{i+1}'] = components[:, i]
        
        return df
    
    def apply_autoencoder(self, data: pd.DataFrame, latent_dim: int = None,
                         epochs: int = None, fit: bool = True, 
                         columns: List[str] = None) -> pd.DataFrame:
        """
        Apply autoencoder for latent space feature extraction.
        
        Args:
            data: Input DataFrame
            latent_dim: Latent space dimension
            epochs: Training epochs
            fit: Whether to train autoencoder or use existing
            columns: Columns to encode (default: target_columns)
        
        Returns:
            DataFrame with latent features and reconstruction error
        """
        if latent_dim is None:
            latent_dim = self.autoencoder_latent_dim
        
        if latent_dim is None:
            return data.copy()
        
        if epochs is None:
            epochs = self.autoencoder_epochs
        
        if columns is None:
            columns = self.target_columns
        
        df = data.copy()
        feature_data = df[columns].fillna(0).values.astype(np.float32)
        
        input_dim = feature_data.shape[1]
        
        if fit:
            logger.info(f"Training autoencoder: input_dim={input_dim}, latent_dim={latent_dim}, epochs={epochs}")
            
            self.autoencoder = AutoencoderNetwork(input_dim, latent_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(feature_data)
            
            # Training loop
            self.autoencoder.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                reconstructed, latent = self.autoencoder(X_tensor)
                loss = criterion(reconstructed, X_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
            logger.info(f"Autoencoder training complete. Final loss: {loss.item():.6f}")
        
        if self.autoencoder is None:
            raise ValueError("Autoencoder not fitted. Call with fit=True first")
        
        # Generate latent features
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(feature_data)
            reconstructed, latent = self.autoencoder(X_tensor)
            
            # Add latent features
            latent_np = latent.numpy()
            for i in range(latent_dim):
                df[f'Latent_{i+1}'] = latent_np[:, i]
            
            # Add reconstruction error
            reconstruction_error = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()
            df['Reconstruction_Error'] = reconstruction_error
        
        logger.info(f"Generated {latent_dim} latent features + reconstruction error")
        
        return df
    
    def transform(self, apply_scaling: bool = None, apply_lags: bool = None,
                  apply_rolling: bool = None, apply_pca_flag: bool = None,
                  apply_autoencoder_flag: bool = None, exclude_from_scaling: List[str] = None) -> pd.DataFrame:
        """
        Apply all configured transformations.
        
        Args:
            apply_scaling: Override config for scaling
            apply_lags: Override config for lag features
            apply_rolling: Override config for rolling features
            apply_pca_flag: Override config for PCA
            apply_autoencoder_flag: Override config for autoencoder
            exclude_from_scaling: List of column names to exclude from scaling (e.g., ['Volume', 'Open'])
        
        Returns:
            Transformed DataFrame
        """
        df = self.data.copy()
        
        # Store exclude list for transform_new_data
        if exclude_from_scaling is not None:
            self.exclude_from_scaling = exclude_from_scaling
        
        # Determine what to apply
        do_scaling = self.scaler_type is not None if apply_scaling is None else apply_scaling
        do_lags = len(self.lag_features) > 0 if apply_lags is None else apply_lags
        do_rolling = len(self.rolling_windows) > 0 if apply_rolling is None else apply_rolling
        do_pca = self.pca_components is not None if apply_pca_flag is None else apply_pca_flag
        do_autoencoder = self.autoencoder_latent_dim is not None if apply_autoencoder_flag is None else apply_autoencoder_flag
        
        # Check for indicators configuration
        indicators_config = self.config.get('indicators')
        indicator_set = self.config.get('indicator_set')
        use_indicators = (indicators_config is not None or indicator_set is not None) and INDICATOR_ENGINE_AVAILABLE
        
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)
        
        # 1. Create time series features first (before scaling)
        if use_indicators:
            logger.info("Using advanced indicator features (replacing basic lag/rolling)")
            enabled_groups = indicators_config.get('enabled_groups') if isinstance(indicators_config, dict) else None
            # Default groups if not specified
            if enabled_groups is None:
                enabled_groups = ['returns', 'trend', 'volatility', 'momentum', 'volume', 'candle', 'regime']
            
            try:
                df = self.create_indicator_features(data=df, enabled_groups=enabled_groups, apply_pruning=True)
            except Exception as e:
                logger.error(f"Failed to create indicator features: {e}")
                logger.info("Falling back to basic lag/rolling features")
                use_indicators = False  # Fallback to basic
        
        if not use_indicators:
            if do_lags:
                df = self.create_lag_features(df)
            
            if do_rolling:
                df = self.create_rolling_features(df)
        
        # 2. Apply scaling to ALL numeric features (including lag/rolling)
        # WARNING: This causes DATA LEAKAGE if used before train/test split!
        # For ML pipelines, set scaler=null here and let DataScientist handle scaling.
        if do_scaling:
            logger.warning("⚠️  WARNING: Scaling applied to FULL dataset before train/test split!")
            logger.warning("   This causes DATA LEAKAGE - test data statistics used in scaling.")
            logger.warning("   For ML pipelines: set scaler=null and let DataScientist scale after split.")
            df = self.scale_data(df, fit=True, exclude_columns=self.exclude_from_scaling)
        
        # 3. Apply dimensionality reduction
        if do_pca:
            df = self.apply_pca(df, fit=True)
        
        if do_autoencoder:
            df = self.apply_autoencoder(df, fit=True)
        
        # 4. Apply final scale filtering (if allow_absolute_scale_features=False)
        if not self.allow_absolute_features:
            df = self.filter_by_scale(df)
        
        self.fitted = True
        
        logger.info("=" * 80)
        logger.info(f"TRANSFORMATION COMPLETE: {df.shape[1]} features, {df.shape[0]} records")
        logger.info("=" * 80)
        
        return df
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            data: New DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Must call transform() first to fit transformers")
        
        df = data.copy()
        
        # Apply same transformations
        if self.lag_features:
            df = self.create_lag_features(df)
        
        if self.rolling_windows:
            df = self.create_rolling_features(df)
        
        if self.scaler_type:
            df = self.scale_data(df, fit=False, exclude_columns=self.exclude_from_scaling)
        
        if self.pca_components:
            df = self.apply_pca(df, fit=False)
        
        if self.autoencoder_latent_dim:
            df = self.apply_autoencoder(df, fit=False)
        
        logger.info(f"Transformed new data: {df.shape[1]} features, {df.shape[0]} records")
        
        return df
    
    def filter_by_scale(self, df: pd.DataFrame, keep_columns: List[str] = None) -> pd.DataFrame:
        """
        Filter features based on scale type (absolute vs relative).
        
        When allow_absolute_scale_features=False, removes price-unit features
        and keeps only relative/normalized features.
        
        Args:
            df: Input DataFrame with features
            keep_columns: Columns to always keep (e.g., target, metadata)
            
        Returns:
            Filtered DataFrame
        """
        if self.allow_absolute_features:
            return df
        
        if not SCALE_CLASSIFIER_AVAILABLE:
            logger.warning("Scale classifier not available, cannot filter by scale")
            return df
        
        if keep_columns is None:
            keep_columns = ['Datetime', 'Date', 'Ticker', self.base_price_column]
        
        return filter_features_by_scale(
            df, 
            allow_absolute=False,
            base_col=self.base_price_column,
            keep_columns=keep_columns
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature column names."""
        if not self.fitted:
            logger.warning("Transformers not fitted yet")
            return self.target_columns
        
        # Would need to track created features - simplified version
        return [col for col in self.data.columns if col not in self.RESERVED_COLUMNS]
    
    def get_summary(self) -> Dict:
        """Get summary of applied transformations."""
        summary = {
            'Input Shape': self.data.shape,
            'Target Columns': self.target_columns,
            'Scaler': self.scaler_type or 'None',
            'Lag Features': self.lag_features if self.lag_features else 'None',
            'Rolling Windows': self.rolling_windows if self.rolling_windows else 'None',
            'PCA Components': self.pca_components or 'None',
            'Autoencoder Latent Dim': self.autoencoder_latent_dim or 'None',
            'Fitted': self.fitted
        }
        return summary
