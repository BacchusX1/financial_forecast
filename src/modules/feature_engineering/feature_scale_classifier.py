"""
Feature Scale Classifier: Classify and filter features by scale type.

Supports:
- Identifying absolute-scale (price-unit) features vs relative-scale (unitless)
- Filtering features based on allow_absolute_scale_features config
- Base price column enforcement

Feature Classification Rules:
- ABSOLUTE SCALE: Values in price units (e.g., $150.25)
  - Raw price columns: Close, Open, High, Low
  - Price lags: Close_lag_5, High_lag_10
  - Rolling price aggregates: Close_rolling_20_mean, Close_rolling_5_min
  - SMA/EMA values: sma_20, ema_50
  - Bollinger bands: bb_upper, bb_lower
  - ATR values: atr_14 (dollar amount)
  
- RELATIVE SCALE: Unitless or normalized values
  - Returns: pct_return_1, log_return
  - Ratios: sma_ratio, bb_position, rsi_14
  - Normalized indicators: zscore_close, atr_normalized_14
  - Momentum oscillators: stoch_k, cci_20
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
import logging
import re

logger = logging.getLogger(__name__)


# Canonical price columns in OHLCV data
PRICE_COLUMNS = {'Close', 'Open', 'High', 'Low', 'Adj Close'}

# Volume is NOT a price column
VOLUME_COLUMNS = {'Volume'}


def get_price_columns_used(columns: List[str]) -> Set[str]:
    """
    Determine which price columns appear in feature names.
    
    Args:
        columns: List of column names
        
    Returns:
        Set of price column prefixes found (e.g., {'Close', 'High'})
    """
    used = set()
    for col in columns:
        for price_col in PRICE_COLUMNS:
            if col.startswith(price_col + '_') or col == price_col:
                used.add(price_col)
    return used


def filter_columns_by_base_price(columns: List[str], base_price_column: str = 'Close',
                                  allow_additional_price_columns: bool = True) -> Tuple[List[str], List[str]]:
    """
    Filter columns to only those derived from base_price_column.
    
    Args:
        columns: List of column names
        base_price_column: The allowed price column (default: 'Close')
        allow_additional_price_columns: If False, restrict to base column only
        
    Returns:
        Tuple of (allowed_columns, dropped_columns)
    """
    if allow_additional_price_columns:
        return columns, []
    
    allowed = []
    dropped = []
    
    # Non-base price columns to filter out
    other_price_cols = PRICE_COLUMNS - {base_price_column}
    
    for col in columns:
        is_other_price = False
        for other_col in other_price_cols:
            # Check if column starts with another price column name
            if col.startswith(other_col + '_') or col == other_col:
                is_other_price = True
                break
        
        if is_other_price:
            dropped.append(col)
        else:
            allowed.append(col)
    
    return allowed, dropped


def classify_feature_columns(columns: List[str], base_col: str = 'Close') -> Dict[str, List[str]]:
    """
    Classify feature columns into scale categories.
    
    Categories:
    - absolute_scale: Features with values in price units
    - relative_scale: Unitless/normalized features (returns, ratios, oscillators)
    - volume: Volume-based features (separate category)
    - other: Unclassified (e.g., metadata columns)
    
    Args:
        columns: List of column names to classify
        base_col: Base price column name (for pattern matching)
        
    Returns:
        Dict with keys: absolute_scale, relative_scale, volume, other
    """
    result = {
        'absolute_scale': [],
        'relative_scale': [],
        'volume': [],
        'other': []
    }
    
    # Patterns indicating ABSOLUTE scale (price units)
    # These are features whose raw values are in dollar amounts
    ABSOLUTE_PATTERNS = [
        # Raw price columns
        r'^(Close|Open|High|Low|Adj Close)$',
        # Price lags
        r'^(Close|Open|High|Low)_lag_\d+$',
        # Rolling aggregates of prices (mean, min, max, median are price-level)
        r'^(Close|Open|High|Low)_rolling_\d+_(mean|min|max|median)$',
        # SMA/EMA values (price level)
        r'^(sma|ema|wma|dema|tema|kama|hma|vwma)_\d+$',
        # ATR (price level)
        r'^(atr)_\d+$',
        r'^true_range$',
        # Bollinger bands absolute levels
        r'^bb_(upper|lower)_\d+$',
        # Keltner/Donchian channel levels (when absolute)
        r'^(keltner|donchian)_(upper|lower)_\d+$',
        # MACD histogram and signal (price differences)
        r'^macd_(line|signal|hist)$',
        # Momentum raw (price difference)
        r'^momentum_\d+$',
        # Intraday range (High - Low, price units)
        r'^intraday_range$',
        # Gap open-close
        r'^gap_open_close$',
        # DPO (detrended price oscillator, price units)
        r'^dpo_\d+$',
        # Elder ray (price level)
        r'^elder_ray_(bull|bear)_\d+$',
        # Awesome oscillator (price level)
        r'^awesome_osc$',
        # Close acceleration/momentum in raw form
        r'^Close_momentum_\d+$',
    ]
    
    # Patterns indicating RELATIVE scale (unitless)
    RELATIVE_PATTERNS = [
        # Returns
        r'_return_\d+$',
        r'^log_return',
        r'^pct_return',
        r'_log_return$',
        r'_pct_return$',
        # Ratios
        r'_ratio',
        r'^sma_ratio',
        r'^ema_ratio',
        r'^price_to_sma',
        r'^close_ma_ratio',
        # Normalized indicators
        r'_normalized',
        r'_zscore',
        r'^zscore_',
        r'_percentile',
        # Relative position indicators
        r'_position',
        r'^bb_position',
        r'^bb_width',
        r'^keltner_position',
        r'^keltner_width',
        r'^donchian_position',
        r'^donchian_width',
        # Oscillators (bounded 0-100 or -100 to 100)
        r'^rsi_\d+$',
        r'^stoch_[kd]$',
        r'^stoch_rsi',
        r'^cci_\d+$',
        r'^mfi_\d+$',
        r'^williams_r',
        r'^adx_\d+$',
        r'^di_plus_\d+$',
        r'^di_minus_\d+$',
        # Volatility ratios
        r'^realized_vol',
        r'^parkinson_vol',
        r'^garman_klass',
        r'^chaikin_vol',
        r'^volatility_ratio',
        # Trend strength
        r'^trend_strength',
        r'^aroon_',
        # Rolling std when it's of returns (relative)
        r'_return.*_rolling.*_std',
        # Acceleration (second derivative of returns)
        r'_acceleration$',
        r'^price_acceleration',
        # ROC (rate of change, percentage)
        r'^roc_\d+$',
        r'_ROC_\d+$',
        # CMF, OBV trends
        r'^cmf_',
        r'^obv_zscore',
        r'^vpt_zscore',
        # Regime indicators
        r'^regime_',
        r'^hurst_',
        r'^fractal_dim',
    ]
    
    # Volume patterns
    VOLUME_PATTERNS = [
        r'^Volume$',
        r'^Volume_',
        r'_Volume_',
        r'^obv$',
        r'^vpt$',
        r'^ad_line$',
        r'^force_index',
        r'^ease_of_movement',
    ]
    
    # Compile patterns
    abs_regex = [re.compile(p, re.IGNORECASE) for p in ABSOLUTE_PATTERNS]
    rel_regex = [re.compile(p, re.IGNORECASE) for p in RELATIVE_PATTERNS]
    vol_regex = [re.compile(p, re.IGNORECASE) for p in VOLUME_PATTERNS]
    
    for col in columns:
        # Check volume first
        if any(r.search(col) for r in vol_regex):
            result['volume'].append(col)
            continue
        
        # Check relative (prioritize relative over absolute for safety)
        if any(r.search(col) for r in rel_regex):
            result['relative_scale'].append(col)
            continue
        
        # Check absolute
        if any(r.search(col) for r in abs_regex):
            result['absolute_scale'].append(col)
            continue
        
        # Check if it's a rolling std (ambiguous - could be price or return std)
        # Rolling std of price is absolute; rolling std of returns is relative
        if '_rolling_' in col and '_std' in col:
            # If column starts with a price column name, it's absolute
            if any(col.startswith(p + '_') for p in PRICE_COLUMNS):
                result['absolute_scale'].append(col)
                continue
        
        # Default: classify based on heuristics
        # If name contains price column prefix, assume absolute
        if any(col.startswith(p + '_') or col == p for p in PRICE_COLUMNS):
            result['absolute_scale'].append(col)
        else:
            result['other'].append(col)
    
    return result


def filter_features_by_scale(df: pd.DataFrame, allow_absolute: bool = True,
                              base_col: str = 'Close',
                              keep_columns: List[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame columns based on scale type.
    
    Args:
        df: Input DataFrame
        allow_absolute: If False, remove absolute-scale features
        base_col: Base price column (always kept if present)
        keep_columns: Columns to always keep (e.g., metadata, target)
        
    Returns:
        Filtered DataFrame
    """
    if allow_absolute:
        return df
    
    if keep_columns is None:
        keep_columns = ['Datetime', 'Date', 'Ticker', base_col]
    
    # Classify all columns
    classification = classify_feature_columns(df.columns.tolist(), base_col)
    
    logger.info("=" * 60)
    logger.info("FEATURE SCALE FILTERING (allow_absolute_scale_features=False)")
    logger.info("=" * 60)
    logger.info(f"Absolute-scale features to drop: {len(classification['absolute_scale'])}")
    logger.info(f"Relative-scale features to keep: {len(classification['relative_scale'])}")
    logger.info(f"Volume features: {len(classification['volume'])}")
    logger.info(f"Other/unclassified: {len(classification['other'])}")
    
    # Log which absolute features are being dropped
    if classification['absolute_scale']:
        logger.info(f"Dropping: {classification['absolute_scale'][:10]}...")
        if len(classification['absolute_scale']) > 10:
            logger.info(f"  ... and {len(classification['absolute_scale']) - 10} more")
    
    # Build list of columns to keep
    cols_to_keep = set(keep_columns)
    cols_to_keep.update(classification['relative_scale'])
    cols_to_keep.update(classification['volume'])  # Keep volume features
    cols_to_keep.update(classification['other'])    # Keep other/metadata
    
    # Filter DataFrame
    final_cols = [c for c in df.columns if c in cols_to_keep]
    
    dropped_count = len(df.columns) - len(final_cols)
    logger.info(f"Result: Kept {len(final_cols)} columns, dropped {dropped_count}")
    
    return df[final_cols]


def validate_price_column_restriction(indicator_inputs: List[str], 
                                       base_price_column: str,
                                       allow_additional: bool) -> Tuple[bool, str]:
    """
    Validate if an indicator can be computed given price column restrictions.
    
    Args:
        indicator_inputs: Required input columns for the indicator
        base_price_column: The allowed price column
        allow_additional: If False, only base column is allowed
        
    Returns:
        Tuple of (is_allowed, reason_if_not)
    """
    if allow_additional:
        return True, ""
    
    other_price_cols = PRICE_COLUMNS - {base_price_column}
    
    for required in indicator_inputs:
        if required in other_price_cols:
            return False, f"requires {required} (only {base_price_column} allowed)"
    
    return True, ""


def get_base_price_column(config: dict) -> str:
    """
    Get base price column from config with default.
    
    Checks both top-level and feature_engineering sub-config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Base price column name (default: 'Close')
    """
    # Check top-level first
    if 'base_price_column' in config:
        return config['base_price_column']
    # Then check feature_engineering sub-config
    fe_config = config.get('feature_engineering', {})
    return fe_config.get('base_price_column', 'Close')


def get_allow_additional_price_columns(config: dict) -> bool:
    """
    Get whether additional price columns are allowed.
    
    Checks both top-level and feature_engineering sub-config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if additional columns allowed (default: True for backward compat)
    """
    # Check top-level first
    if 'allow_additional_price_columns' in config:
        return config['allow_additional_price_columns']
    # Then check feature_engineering sub-config
    fe_config = config.get('feature_engineering', {})
    return fe_config.get('allow_additional_price_columns', True)


def get_allow_absolute_scale_features(config: dict) -> bool:
    """
    Get whether absolute-scale features are allowed.
    
    Checks both top-level and feature_engineering sub-config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if absolute features allowed (default: True)
    """
    # Check top-level first
    if 'allow_absolute_scale_features' in config:
        return config['allow_absolute_scale_features']
    # Then check feature_engineering sub-config
    fe_config = config.get('feature_engineering', {})
    return fe_config.get('allow_absolute_scale_features', True)
