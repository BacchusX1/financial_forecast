# Trading Forecast v2 - System Architecture Documentation

## 1. Project Overview

A **modular, configuration-driven** ML pipeline for financial time series forecasting with **strict data leakage prevention**.

```
trading_forecast_v2/
├── launcher.py                    # ← ENTRY POINT (CLI)
├── configuration.yml              # ← Master config file
├── config_auto.yml                # ← Auto-generated from interactive mode
├── requirements.txt
├── doc/
│   ├── DOCS_SYSTEM_ARCHITECTURE.md      # This document
│   ├── generator_fromMD_FILE.py         # Markdown → HTML generator
│   ├── readme                           # How to regenerate HTML
│   └── trading_forecast_v2.html         # Generated HTML output
├── src/
│   ├── config_validator.py        # Configuration validation
│   └── modules/
│       ├── dada_loader/
│       │   └── financial_data_loader.py    # DataLoader class
│       ├── feature_engineering/
│       │   ├── feature_engineering.py      # FeatureEngineering class
│       │   ├── feature_processor.py        # PCA/Autoencoder post-split
│       │   ├── feature_pruner.py           # Correlation/NaN filtering
│       │   ├── feature_scale_classifier.py # Scale-free feature detection
│       │   ├── indicator_engine.py         # Technical indicators
│       │   └── indicator_registry.py       # 112+ indicator definitions
│       ├── data_science/
│       │   ├── data_science.py             # DataScientist class
│       │   ├── data_augmentor.py           # Training data augmentation
│       │   ├── model_zoo.py                # Advanced models (xLSTM, TCN, N-BEATS, MoE)
│       │   ├── report_generator.py         # HTML report generation
│       │   └── forecast_utils.py           # Multi-step utilities
│       └── llm/
│           └── ollama_client.py            # Local LLM integration (disabled)
└── out/                           # ← OUTPUT (auto-created)
    ├── raw_data/                  # Downloaded OHLCV CSVs
    ├── transformed_features/      # Engineered features + labels
    ├── models_YYYYMMDD_HHMMSS/    # Model outputs & reports
    │   ├── benchmark_report_advanced.html
    │   ├── benchmark_results.json
    │   ├── training_data/
    │   └── *.pt, *.pkl (saved models)
    └── summaries/
```

---

## 2. Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           launcher.py (main)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Load configuration.yml  ──►  ConfigValidator.validate()                │
│  2. Create output directories                                               │
│  3. Run pipeline: run_pipeline(config)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Step 1: Data Load  │  │ Step 2: Feature Eng │  │ Step 3: ML Training │
│    DataLoader       │  │ FeatureEngineering  │  │   DataScientist     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
    data_dict[ticker]        transformed_df          benchmark_results
    (raw OHLCV)              (features + labels)     (models + reports)
```

---

## 3. Configuration Structure (configuration.yml)

### 3.1 Logging Section
```yaml
logging:
  level: "DEBUG"              # DEBUG | INFO | WARNING | ERROR
  # Training logs auto-saved to out/models_*/log.txt with colors
```

### 3.2 Data Loader Section
```yaml
data_loader:
  keys: ["BTC"]               # Tickers to download
  category: "crypto"          # "crypto" | "stock"
  candle_length: "5m"         # 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
  period: null                # Years (null = auto-max for interval)
  start_date: null            # Override: "YYYY-MM-DD"
  end_date: null              # Override: "YYYY-MM-DD"
  extended_download: false    # Bypass API limits with chunked requests
  chunk_overlap_days: 1       # Overlap for deduplication
  request_delay_seconds: 2.0  # Delay between API requests
```

**API Limits:** 1m=7d, 5m-90m=60d, 1h=730d, 1d+=unlimited

**Extended Download:** Set `extended_download: true` to fetch data beyond API limits by making multiple chunked requests automatically.

### 3.3 Feature Engineering Section
```yaml
feature_engineering:
  scaler: null                        # MUST be null for ML (prevents leakage)
  target_columns: ["Close"]
  allow_absolute_scale_features: false # Scale-free mode (recommended)
  
  # Lag and rolling features (tuned for 5-min candles)
  lag_features: [1, 3, 6, 12, 24, 48]     # 48 = 4 hours
  rolling_windows: [6, 12, 24, 48, 96]    # 96 = 8 hours
  rolling_functions: ["mean", "std", "min", "max"]
  
  # Dimensionality reduction
  pca_components: 10
  autoencoder_latent_dim: null  # Set to integer to enable
  autoencoder_epochs: 50
  
  group_by_ticker: true
  exclude_from_scaling: []
  
  # Technical indicators (112+ available)
  indicator_set: "expanded_v1"
  indicators:
    enabled_groups: ["returns", "trend", "volatility", "momentum", "volume", "candle", "regime"]
    params:
      rsi: [14, 28, 48]
      atr: [14, 28]
      bbands: [{length: 20, std: 2}]
      macd: [{fast: 12, slow: 26, signal: 9}]
      sma: [6, 12, 24, 48, 144]
      ema: [6, 12, 24, 48, 144]
  
  # Leakage prevention
  leakage_guard:
    enforce_shift_1: true             # Auto-shift all indicators by 1 period
    fail_on_unshifted_columns: true
  
  # Feature pruning
  feature_pruning:
    drop_constant: true
    drop_high_nan_ratio: 0.2
    correlation_filter:
      enabled: true
      method: "spearman"
      threshold: 0.98
```

### 3.4 Data Science Section
```yaml
data_science:
  enabled: true
  use_gpu: true                 # GPU acceleration for PyTorch models
  
  # Models to train
  classifier_models: []         # ["dnn", "svc", "random_forest"]
  regressor_models: ["dnn", "xlstm", "tcn"]
  
  # Core parameters
  forecast_horizon: 10          # Steps to predict
  holdout_days: 20              # Last N samples for validation
  target_transform: "log_return" # "price" | "pct_change" | "log_return"
  forecast_mode: "multioutput"  # "multioutput" | "recursive"
  
  # Outlier filtering (training only)
  outlier_filter:
    enabled: true
    percentile: 2.0
    method: "max_abs"           # max_abs | mean | max | min
  
  # Data augmentation (training only)
  data_augmentation:
    enabled: true
    preset: "financial"         # minimal | standard | aggressive | financial | volatility | tail_risk
    methods: ["jitter", "volatility_regime", "block_bootstrap"]
    n_augmentations: 3
    # GARCH parameters
    garch_alpha: 0.1
    garch_beta: 0.85
    block_size: 20
  
  # Feature processing (post-split)
  feature_processing:
    mode: "raw+pca+ae"          # raw | raw+pca | raw+ae | raw+pca+ae
    pca:
      enabled: true
      n_components: 10
    autoencoder:
      enabled: false
      latent_dim: 8
      epochs: 10
  
  # Model hyperparameters
  model_hyperparameters:
    # Classifier DNN
    classifier_dnn:
      layers: [64, 32]
      dropout: [0.3, 0.2]
      activation: "relu"
      epochs: 5
      batch_size: 64
    
    # Regressor DNN (ResNet + SE-Attention)
    regressor_dnn:
      layers: [128, 64, 32]
      dropout: [0.2, 0.15, 0.1]
      activation: "gelu"
      epochs: 10
      use_layer_norm: true
      use_se_attention: true
      use_residual: true
    
    # LSTM (Bidirectional + Multi-Head Attention)
    regressor_lstm:
      timesteps: 4
      lstm_units: [64, 32, 16]
      attention: {num_heads: 4, key_dim: 16}
      epochs: 10
      bidirectional: true
    
    # xLSTM (Extended LSTM with Matrix Memory)
    regressor_xlstm:
      timesteps: 4
      num_blocks: 1             # Keep small (1-4) for fast training
      num_heads: 2              # Keep small (2-4)
      dropout: 0.15
      epochs: 50
      patience: 10
    
    # KRR (Kernel Ridge Regression)
    regressor_krr:
      use_grid_search: true
      grid_search:
        alpha: [0.001, 0.01, 0.1, 1.0, 10.0]
        gamma: [0.001, 0.01, 0.1]
        kernel: ["rbf"]
  
  # Model Zoo hyperparameters
  model_zoo:
    lightgbm:
      params:
        num_leaves: 31
        learning_rate: 0.1
        n_estimators: 50
    
    tcn:
      params:
        num_channels: [512, 512, 256, 64, 10]
        kernel_size: 50
        dropout: 0.2
        epochs: 500
    
    nbeats:
      params:
        num_blocks: 64
        hidden_dim: 128
        epochs: 150
    
    moe:
      params:
        expert_types: ['linear', 'dnn', 'tcn']
        gating_hidden_dim: 32
        epochs: 10
    
    multitask:
      params:
        num_classes: 3
        epochs: 10
  
  # Validation
  trend_window: 10
  trend_threshold: 0.001
  test_size: 0.15
  random_state: 42
```

### 3.5 Reporting Section
```yaml
reporting:
  use_collapsible_sections: true
  dark_mode: false
  holdout_plot_style: "both"      # overlay | faceted | both
  show_per_step_metrics: true
  show_predictions_vs_actual: true
  show_error_distribution: true
  show_residual_analysis: true
  show_model_comparison_table: true
  show_model_ranking_chart: true
  show_feature_importance: true
  feature_importance_top_k: 20
  show_training_curves: true
  show_confusion_matrix: true
  num_sample_forecasts: 5
  show_prediction_intervals: true
  show_pipeline_overview: true
  show_data_statistics: true
  show_outlier_filter_summary: true
```

### 3.6 Output Section
```yaml
output:
  folder: "out"
  save_raw_data: true
  save_transformed_features: true
  save_summary: true
```

---

## 4. Class Reference

### 4.1 DataLoader
**File:** `src/modules/dada_loader/financial_data_loader.py`

```python
class DataLoader:
    """Download OHLCV data via yfinance with auto-interval validation."""
    
    def __init__(self, config: dict)
    def assemble_data() -> Dict[str, pd.DataFrame]
    def get_combined_data() -> pd.DataFrame
    def get_summary() -> pd.DataFrame
    def download_data(ticker) -> pd.DataFrame  # Single ticker
    def download_extended(ticker) -> pd.DataFrame  # Chunked download
```

**Key Features:**
- Normalizes intervals (`5min` → `5m`)
- Auto-adjusts period if exceeds API limit
- Adds `-USD` suffix for crypto
- Extended download mode for bypassing API limits
- 60-second timeout for large downloads

---

### 4.2 FeatureEngineering
**File:** `src/modules/feature_engineering/feature_engineering.py`

```python
class FeatureEngineering:
    """Transform OHLCV data into ML-ready features."""
    
    def __init__(self, data: Union[Dict, pd.DataFrame], config: dict)
    def transform(exclude_from_scaling: List[str] = None) -> pd.DataFrame
    def create_lag_features(df) -> pd.DataFrame
    def create_rolling_features(df) -> pd.DataFrame
    def create_indicator_features(df, enabled_groups) -> pd.DataFrame
    def apply_pca(df, fit: bool) -> pd.DataFrame
    def apply_autoencoder(df, fit: bool) -> pd.DataFrame
    def filter_by_scale(df) -> pd.DataFrame  # Scale-free mode
    def get_summary() -> Dict
```

**Pipeline Order:**
1. Indicator features (with `.shift(1)` for leakage prevention)
2. Lag features
3. Rolling features
4. Feature pruning (constant, NaN, correlation)
5. Scale filtering (if `allow_absolute_scale_features: false`)
6. PCA compression (optional)
7. Autoencoder latent features (optional)

---

### 4.3 IndicatorEngine
**File:** `src/modules/feature_engineering/indicator_engine.py`

```python
class IndicatorEngine:
    """Compute 112+ technical indicators with leakage prevention."""
    
    def __init__(self, config: dict)
    def compute_indicators(data, enabled_groups, custom_params) -> pd.DataFrame
    def prune_features(df) -> pd.DataFrame
    def assert_no_leakage(df, original_cols) -> bool
```

**Indicator Groups:**
| Group | Examples |
|-------|----------|
| returns | log_return, pct_change, realized_vol, zscore |
| trend | sma, ema, wma, kama, dema, tema, hma, alma, t3, slope |
| volatility | atr, bbands, keltner, donchian, parkinson_vol, gk_vol |
| momentum | rsi, macd, stoch, cci, williams_r, roc, trix, adx, aroon, vortex, kst, tsi, fisher |
| volume | obv, cmf, mfi, force_index, eom, pvt, kvo, vwap_proxy |
| candle | body_ratio, upper_wick, lower_wick, engulfing, breakout_flag |
| regime | hurst, entropy, autocorr, skew, kurtosis, drawdown |

---

### 4.4 DataScientist
**File:** `src/modules/data_science/data_science.py`

```python
class DataScientist:
    """Main ML training and evaluation class."""
    
    VALID_CLASSIFIERS = ['dnn', 'svc', 'random_forest']
    VALID_REGRESSORS = ['lstm', 'xlstm', 'dnn', 'krr', 'linear', 
                        'lightgbm', 'tcn', 'nbeats', 'moe', 'multitask']
    
    def __init__(self, data: pd.DataFrame, config: dict)
    def train_and_evaluate(ticker, holdout_days, llm_config, generate_reports) -> Dict
    
    # Classifier training
    def train_classifier_dnn(X_train, X_test, y_train, y_test)
    def train_classifier_svc(X_train, X_test, y_train, y_test)
    def train_classifier_random_forest(X_train, X_test, y_train, y_test)
    
    # Regressor training
    def train_regressor_dnn(X_train, X_test, y_train, y_test)
    def train_multioutput_lstm(X_train, X_test, y_train, y_test)
    def train_multioutput_xlstm(X_train, X_test, y_train, y_test)
    def train_regressor_krr(X_train, X_test, y_train, y_test)
    def train_regressor_linear(X_train, X_test, y_train, y_test)
    def train_lightgbm_multioutput(X_train, X_test, y_train, y_test)
    def train_tcn(X_train, X_test, y_train, y_test)
    def train_nbeats(X_train, X_test, y_train, y_test)
    def train_moe(X_train, X_test, y_train, y_test)
    def train_multitask(X_train, X_test, y_train, y_test, y_trend)
```

**train_and_evaluate Steps:**
1. Split data (training vs holdout)
2. Build multi-step targets
3. Apply target transform (price/pct_change/log_return)
4. Apply outlier filtering (training only)
5. Apply data augmentation (training only)
6. Apply feature processing (PCA/Autoencoder)
7. Train classifiers & regressors
8. Evaluate on holdout
9. Generate future forecasts
10. Create HTML reports

---

### 4.5 Model Zoo
**File:** `src/modules/data_science/model_zoo.py`

```python
# Available models (all PyTorch)
MODEL_REGISTRY = {
    'linear': LinearRegressor,
    'dnn': DNNRegressor,        # ResNet + SE-Attention + Multi-Head
    'lstm': LSTMRegressor,      # Bidirectional + Multi-Head Attention
    'xlstm': xLSTMRegressorUnified,  # Extended LSTM with Matrix Memory
    'tcn': TCN,                 # Temporal Convolutional Network
    'nbeats': NBeatsLite,       # Interpretable forecast
}

class LightGBMMultiOutput:
    """LightGBM wrapper - trains one model per forecast step."""

class MixtureOfExperts(nn.Module):
    """
    Extensible MoE with regime-aware gating.
    Supports ANY model architecture as experts.
    """

class MultiTaskModel(nn.Module):
    """Joint regression + classification head."""

def create_model(model_type, input_dim, output_dim, config) -> nn.Module
    """Factory function to create models by name."""

def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                        epochs, batch_size, lr, verbose=True) -> Dict
    """Generic training loop with early stopping. Logs every epoch."""
```

---

### 4.6 Data Augmentation
**File:** `src/modules/data_science/data_augmentor.py`

```python
class DataAugmentor:
    """Training data augmentation for time series."""
    
    # Basic methods
    def jitter(X, y, std)           # Gaussian noise
    def scale(X, y, range)          # Magnitude scaling
    def window_crop(X, y, ratio)    # Subsequences
    def mixup(X, y, alpha)          # Sample interpolation
    
    # Advanced methods (financial)
    def volatility_regime(X, y, alpha, beta)  # GARCH-like clustering
    def block_bootstrap(X, y, block_size)     # Preserve autocorrelation
    def frequency_mask(X, y, ratio, band)     # FFT masking
    def pattern_morph(X, y, n_neighbors)      # DTW interpolation
    def tail_augment(X, y, percentile)        # Extreme event amplification
    
    def augment(X, y, methods, n_augmentations) -> Tuple[X_aug, y_aug]
```

**Presets:**
- `minimal`: jitter + scale
- `standard`: jitter, scale, magnitude_warp, pattern_morph
- `financial`: volatility_regime, block_bootstrap, jitter (recommended for crypto)
- `volatility`: volatility_regime, regime_noise
- `tail_risk`: tail_augment, volatility_regime

---

### 4.7 Forecast Utilities
**File:** `src/modules/data_science/forecast_utils.py`

```python
# Target construction
def build_multistep_targets(close_series, horizon) -> Tuple[np.ndarray, np.ndarray]

# Evaluation
def evaluate_multistep(y_true, y_pred, last_known_prices) -> Dict

# Baselines
def compute_multistep_baselines(train_close, holdout_close, horizon) -> Dict
    # Returns: naive, drift, rolling_mean baselines

# Target transforms
VALID_TARGET_TRANSFORMS = ["price", "pct_change", "log_return"]
def to_target(prices, transform) -> np.ndarray
def from_target(targets, base_prices, transform) -> np.ndarray
def from_target_cumulative(returns, base_price, transform) -> np.ndarray
```

---

## 5. Data Flow Diagram

```
configuration.yml
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  launcher.py::run_pipeline(config)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ Step 1: Data Loading
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataLoader(config['data_loader'])                                          │
│  ├── download_data(ticker) via yfinance (60s timeout)                      │
│  ├── download_extended(ticker) for chunked downloads                       │
│  └── assemble_data() → data_dict = {ticker: DataFrame}                     │
│      get_combined_data() → combined_df (all tickers)                        │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ combined_df [Datetime, Open, High, Low, Close, Volume, Ticker]
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FeatureEngineering(combined_df, config['feature_engineering'])             │
│  ├── create_indicator_features() via IndicatorEngine                       │
│  │   └── .shift(1) applied to ALL indicator columns                        │
│  ├── create_lag_features() → Close_lag_1, Close_lag_5, ...                  │
│  ├── create_rolling_features() → Close_rolling_20_mean, ...                 │
│  ├── prune_features() → Remove constant, NaN, correlated features          │
│  ├── filter_by_scale() (if allow_absolute_scale_features: false)            │
│  ├── apply_pca() → PC_1, PC_2, ... (if enabled)                             │
│  └── apply_autoencoder() → Latent_1, Latent_2, ... (if enabled)             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ transformed_df [Close, feature_1, feature_2, ..., feature_N]
       │
       │ launcher.py adds target columns:
       │   target_Close_t+1, ..., target_Close_t+{horizon}
       │   target_trend (UP=2, SIDEWAYS=1, DOWN=0)
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataScientist(transformed_df, config['data_science'])                      │
│  └── train_and_evaluate(ticker, holdout_days, llm_config)                   │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ Inside train_and_evaluate():
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. SPLIT DATA                                                              │
│     train_data = data[:-holdout_days]                                       │
│     holdout_data = data[-holdout_days:]                                     │
│                                                                             │
│  2. BUILD MULTI-STEP TARGETS                                                │
│     y_multistep = build_multistep_targets(close, horizon)                   │
│     Shape: (N - horizon, horizon)                                           │
│                                                                             │
│  3. APPLY TARGET TRANSFORM                                                  │
│     if target_transform == "log_return":                                    │
│         y_train = log(y / base_price) for each step                         │
│                                                                             │
│  4. APPLY OUTLIER FILTERING (training only)                                 │
│     Remove top/bottom percentile of extreme target values                   │
│                                                                             │
│  5. APPLY DATA AUGMENTATION (training only)                                 │
│     DataAugmentor.augment() with configured methods                         │
│     Typical 4x data increase with "financial" preset                        │
│                                                                             │
│  6. APPLY FEATURE PROCESSING (fit on train only)                            │
│     FeatureProcessor: StandardScaler + PCA + Autoencoder                    │
│                                                                             │
│  7. TRAIN CLASSIFIERS                                                       │
│     for model in classifier_models: train_classifier_{model}()              │
│                                                                             │
│  8. TRAIN REGRESSORS                                                        │
│     for model in regressor_models: train_{model}()                          │
│     Verbose logging: every epoch shows Train/Val loss                       │
│                                                                             │
│  9. HOLDOUT EVALUATION                                                      │
│     predictions = model.predict(X_holdout)                                  │
│     metrics = evaluate_multistep(y_holdout, predictions)                    │
│     Compare against naive, drift, rolling_mean baselines                    │
│                                                                             │
│  10. FUTURE FORECASTS                                                       │
│      X_future = last_known_features                                         │
│      future_pred = model.predict(X_future)                                  │
│                                                                             │
│  11. GENERATE REPORTS                                                       │
│      ReportGenerator.generate_comprehensive_report()                        │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ benchmark_results dict + HTML report
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT FILES (out/models_YYYYMMDD_HHMMSS/)                                 │
│  ├── benchmark_results.json        # All metrics & predictions             │
│  ├── benchmark_report_advanced.html # Comprehensive report with all plots  │
│  ├── training_data/                                                         │
│  │   ├── classifier_training_data.csv                                      │
│  │   └── regressor_training_data.csv                                        │
│  ├── {model}_reg.pt                # Saved PyTorch models                  │
│  └── {model}_regressor.pkl         # Saved sklearn models                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Leakage Prevention Guarantees

| Stage | Mechanism | Code Location |
|-------|-----------|---------------|
| Indicators | All outputs shifted by 1 period | `IndicatorEngine.compute_indicators()` |
| Lag Features | Explicit `df.shift(lag)` | `FeatureEngineering.create_lag_features()` |
| Scaler | Fit ONLY on training data | `FeatureProcessor.fit()` |
| PCA/AE | Fit ONLY on training data | `FeatureProcessor.fit()` |
| Holdout | Completely excluded from training | `train_and_evaluate()` split logic |
| Augmentation | Applied ONLY to training data | `DataAugmentor.augment()` |
| Outlier Filter | Applied ONLY to training data | `DataScientist._filter_outliers()` |
| Target | Built from FUTURE prices only | `build_multistep_targets()` |

**Validation:** `leakage_guard.fail_on_unshifted_columns: true` enforces validation.

---

## 7. Model Hyperparameters Reference

### Classifiers

| Model | Key Parameters | Config Path |
|-------|----------------|-------------|
| DNN | layers, dropout, activation, epochs | `model_hyperparameters.classifier_dnn` |
| SVC | C, gamma, kernel (GridSearch) | `model_hyperparameters.classifier_svc` |
| Random Forest | n_estimators, max_depth | `model_hyperparameters.classifier_random_forest` |

### Regressors (Core)

| Model | Key Parameters | Config Path |
|-------|----------------|-------------|
| DNN | layers, dropout, use_residual, use_se_attention | `model_hyperparameters.regressor_dnn` |
| LSTM | timesteps, lstm_units, attention, bidirectional | `model_hyperparameters.regressor_lstm` |
| xLSTM | timesteps, num_blocks, num_heads, dropout | `model_hyperparameters.regressor_xlstm` |
| KRR | alpha, gamma, kernel | `model_hyperparameters.regressor_krr` |
| Linear | fit_intercept, elasticnet | `model_hyperparameters.regressor_linear` |

### Model Zoo (Advanced)

| Model | Key Parameters | Config Path |
|-------|----------------|-------------|
| LightGBM | num_leaves, learning_rate, n_estimators | `model_zoo.lightgbm.params` |
| TCN | num_channels, kernel_size, dropout, epochs | `model_zoo.tcn.params` |
| N-BEATS | num_blocks, hidden_dim, epochs | `model_zoo.nbeats.params` |
| MoE | expert_types, gating_hidden_dim | `model_zoo.moe.params` |
| MultiTask | num_classes | `model_zoo.multitask.params` |

---

## 8. Target Transforms

| Transform | Formula (per step) | Inverse | Use Case |
|-----------|-------------------|---------|----------|
| `price` | `y = Close_{t+k}` | Identity | Direct price prediction |
| `pct_change` | `y = (Close_{t+k} - Close_t) / Close_t` | `pred × base + base` | Stocks |
| `log_return` | `y = log(Close_{t+k} / Close_t)` | `base × exp(pred)` | Crypto (recommended) |

**Note:** `log_return` is more numerically stable and treats up/down moves symmetrically.

---

## 9. Data Augmentation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| jitter | Gaussian noise injection | General |
| scale | Random magnitude scaling | Volatility robustness |
| volatility_regime | GARCH-like clustering | Financial data |
| block_bootstrap | Block resampling | Preserving autocorrelation |
| frequency_mask | FFT band masking | Removing noise |
| pattern_morph | DTW interpolation | Pattern diversity |
| tail_augment | Extreme event amplification | Tail risk |

---

## 10. Quick Reference: Key Config Paths

| Purpose | Config Path |
|---------|-------------|
| Tickers to download | `data_loader.keys` |
| Candle interval | `data_loader.candle_length` |
| Extended download | `data_loader.extended_download` |
| Scale-free mode | `feature_engineering.allow_absolute_scale_features` |
| Indicators | `feature_engineering.indicators.enabled_groups` |
| Models to train | `data_science.regressor_models` |
| Forecast steps | `data_science.forecast_horizon` |
| Holdout size | `data_science.holdout_days` |
| Target transform | `data_science.target_transform` |
| Outlier filter | `data_science.outlier_filter` |
| Augmentation | `data_science.data_augmentation` |
| GPU usage | `data_science.use_gpu` |

---

## 11. Running the Pipeline

```bash
# Interactive mode
python launcher.py
# Select "Use configuration.yml" or "Create config_auto.yml interactively"

# Outputs
out/
├── raw_data/BTC.csv
├── transformed_features/features_labels.csv
├── models_20260108_120000/
│   ├── benchmark_report_advanced.html  ← Main report
│   ├── benchmark_results.json
│   ├── training_data/
│   │   ├── classifier_training_data.csv
│   │   └── regressor_training_data.csv
│   └── *.pt (saved PyTorch models)
└── summaries/
    ├── data_summary.csv
    └── features_summary.json
```

---

## 12. Testing

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_reporting_robustness.py -v
pytest tests/test_config.py -v
pytest tests/test_features.py -v
pytest tests/test_leakage.py -v
pytest tests/test_pruning.py -v
pytest tests/test_target_transforms.py -v
```

---

## 13. Regenerating This Documentation

```bash
# Convert Markdown to HTML
python doc/generator_fromMD_FILE.py doc/DOCS_SYSTEM_ARCHITECTURE.md \
  -o doc/trading_forecast_v2.html --title "Trading Forecast v2"
```
