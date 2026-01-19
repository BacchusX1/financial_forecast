"""
Indicator Engine: Compute technical indicators with strict leakage prevention.

This engine:
1. Computes indicators from OHLCV data using the indicator registry
2. Automatically applies .shift(1) to ALL new indicator columns (leakage-safe)
3. Validates that no future data leaks into features
4. Supports feature pruning (constant, high NaN, correlation)
5. Exports features to configured output directory

CRITICAL RULE: Every indicator output gets .shift(1) applied so that features
at time t only contain information up to time t-1.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
import os

from indicator_registry import INDICATOR_REGISTRY

# Import feature scale classifier
try:
    from feature_scale_classifier import (
        validate_price_column_restriction,
        filter_features_by_scale,
        get_base_price_column,
        get_allow_additional_price_columns,
        get_allow_absolute_scale_features,
        PRICE_COLUMNS
    )
    SCALE_CLASSIFIER_AVAILABLE = True
except ImportError:
    SCALE_CLASSIFIER_AVAILABLE = False
    PRICE_COLUMNS = {'Close', 'Open', 'High', 'Low', 'Adj Close'}

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Compute and manage technical indicators with leakage prevention.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize indicator engine.
        
        Args:
            config: Configuration dict with keys:
                - indicator_set: Name of indicator set (e.g., "expanded_v1")
                - indicators: Dict with enabled_groups and params
                - leakage_guard: Dict with enforcement settings
                - feature_pruning: Dict with pruning settings
        """
        self.config = config or {}
        self.registry = INDICATOR_REGISTRY
        self.computed_indicators = {}
        self.original_columns = []
        self.leakage_guard_enabled = self.config.get('leakage_guard', {}).get('enforce_shift_1', True)
        self.fail_on_unshifted = self.config.get('leakage_guard', {}).get('fail_on_unshifted_columns', True)
        
        # New: Price column restrictions
        if SCALE_CLASSIFIER_AVAILABLE:
            self.base_price_column = get_base_price_column(self.config)
            self.allow_additional_price_columns = get_allow_additional_price_columns(self.config)
            self.allow_absolute_features = get_allow_absolute_scale_features(self.config)
        else:
            self.base_price_column = self.config.get('base_price_column', 'Close')
            self.allow_additional_price_columns = self.config.get('allow_additional_price_columns', True)
            self.allow_absolute_features = self.config.get('allow_absolute_scale_features', True)
        
        logger.info(f"IndicatorEngine initialized with {len(self.registry.indicators)} available indicators")
        logger.info(f"Leakage guard: {'ENABLED' if self.leakage_guard_enabled else 'DISABLED'}")
        logger.info(f"Base price column: {self.base_price_column}")
        logger.info(f"Additional price columns: {'ALLOWED' if self.allow_additional_price_columns else 'RESTRICTED'}")
        logger.info(f"Absolute Scale Features: {'ALLOWED' if self.allow_absolute_features else 'FORBIDDEN (Scale-Free Mode)'}")
    
    def compute_indicators(self, data: pd.DataFrame, enabled_groups: List[str] = None,
                          custom_params: Dict = None) -> pd.DataFrame:
        """
        Compute indicators and apply .shift(1) for leakage prevention.
        
        Args:
            data: Input DataFrame with OHLCV columns
            enabled_groups: List of indicator groups to enable (e.g., ['returns', 'trend'])
            custom_params: Custom parameters override for specific indicators
        
        Returns:
            DataFrame with original columns + shifted indicators
        """
        df = data.copy()
        self.original_columns = list(df.columns)
        
        if enabled_groups is None:
            enabled_groups = self.config.get('indicators', {}).get('enabled_groups', 
                ['returns', 'trend', 'volatility', 'momentum', 'volume', 'candle', 'regime'])
        
        logger.info("=" * 80)
        logger.info("COMPUTING TECHNICAL INDICATORS (LEAKAGE-SAFE)")
        logger.info("=" * 80)
        logger.info(f"Input data: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Enabled groups: {enabled_groups}")
        
        # Get all indicators for enabled groups
        indicators_to_compute = []
        for group in enabled_groups:
            group_indicators = self.registry.get_indicators_by_category(group)
            indicators_to_compute.extend(group_indicators)
        
        # FILTER: Base price column restriction
        if not self.allow_additional_price_columns:
            filtered_indicators = []
            skipped_count = 0
            other_price_cols = PRICE_COLUMNS - {self.base_price_column}
            
            for ind in indicators_to_compute:
                ind_meta = self.registry.indicators.get(ind, {})
                required_cols = ind_meta.get('inputs', [])
                
                # Check if any required column is a non-base price column
                requires_other_price = any(col in other_price_cols for col in required_cols)
                
                if requires_other_price:
                    logger.debug(f"Base-price-only mode: Skipping '{ind}' (requires {required_cols})")
                    skipped_count += 1
                else:
                    filtered_indicators.append(ind)
            
            if skipped_count > 0:
                logger.info(f"Base-price-only mode: Skipped {skipped_count} indicators requiring Open/High/Low")
            
            indicators_to_compute = filtered_indicators
        
        # FILTER: Scale-Free Mode
        if not self.allow_absolute_features:
            filtered_indicators = []
            
            # Indicators that depend on absolute price scale
            # (Matches prefixes)
            ABSOLUTE_PREFIXES = [
                'sma_', 'ema_', 'wma_', 'vwma_', 'hma_', 'dema_', 'tema_', 'kama_',
                'true_range', 'atr_', 
                'momentum_', 'macd_', 'elder_ray_', 'awesome_osc', 'dpo_',
                'gap_open_close', 'intraday_range'
            ]
            
            # Allow these even if they match prefixes (e.g. sma_ratio)
            SAFE_SUFFIXES_STRICT = [
                '_ratio', '_normalized', '_pct', '_percent', '_width', '_position'
            ]

            for ind in indicators_to_compute:
                is_absolute = False
                for prefix in ABSOLUTE_PREFIXES:
                    if ind.startswith(prefix):
                        is_absolute = True
                        break
                
                # Check for safe suffixes if flagged as absolute
                if is_absolute:
                    for suffix in SAFE_SUFFIXES_STRICT:
                        # Check if suffix exists in the name
                        if suffix in ind: 
                            is_absolute = False
                            break
                
                if not is_absolute:
                    filtered_indicators.append(ind)
                else:
                    logger.debug(f"Scale-Free Mode: Dropping absolute indicator '{ind}'")
            
            if len(indicators_to_compute) != len(filtered_indicators):
                logger.info(f"Scale-Free Mode: Dropped {len(indicators_to_compute) - len(filtered_indicators)} absolute indicators")
            
            indicators_to_compute = filtered_indicators

        logger.info(f"Will compute {len(indicators_to_compute)} indicators")
        
        # Compute each indicator
        computed_count = 0
        failed_count = 0
        new_columns = []
        
        for ind_name in indicators_to_compute:
            try:
                ind_meta = self.registry.indicators[ind_name]
                required_cols = ind_meta['inputs']
                
                # Check if required columns exist
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Skipping {ind_name}: missing required columns {required_cols}")
                    failed_count += 1
                    continue
                
                # Compute indicator
                result = self.registry.compute(ind_name, df)
                
                # CRITICAL: Apply .shift(1) for leakage prevention
                # This ensures indicator at time t only uses data up to t-1
                if self.leakage_guard_enabled:
                    result = result.shift(1)
                
                df[ind_name] = result
                new_columns.append(ind_name)
                computed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to compute {ind_name}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Successfully computed: {computed_count} indicators")
        if failed_count > 0:
            logger.warning(f"Failed to compute: {failed_count} indicators")
        
        logger.info(f"New feature columns: {len(new_columns)}")
        logger.info(f"Total columns: {df.shape[1]} (original: {len(self.original_columns)} + new: {len(new_columns)})")
        
        # Store metadata
        self.computed_indicators = {
            'total_computed': computed_count,
            'failed': failed_count,
            'new_columns': new_columns,
            'enabled_groups': enabled_groups
        }
        
        return df
    
    def prune_features(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Prune features based on configuration.
        
        Args:
            data: DataFrame with features
            target_col: Target column name (excluded from pruning)
        
        Returns:
            DataFrame with pruned features
        """
        pruning_config = self.config.get('feature_pruning', {})
        
        if not pruning_config:
            logger.info("Feature pruning disabled")
            return data
        
        df = data.copy()
        initial_cols = df.shape[1]
        
        logger.info("=" * 80)
        logger.info("FEATURE PRUNING")
        logger.info("=" * 80)
        
        # Get feature columns (exclude metadata and target)
        reserved_cols = ['Datetime', 'Date', 'Ticker', target_col]
        feature_cols = [col for col in df.columns if col not in reserved_cols]
        
        logger.info(f"Initial features: {len(feature_cols)}")
        
        to_drop = set()
        
        # 1. Drop constant features
        if pruning_config.get('drop_constant', True):
            for col in feature_cols:
                if col in to_drop:
                    continue
                nunique = df[col].nunique()
                if nunique <= 1:
                    to_drop.add(col)
                    logger.debug(f"Constant feature: {col} (nunique={nunique})")
            
            if to_drop:
                logger.info(f"Dropping {len(to_drop)} constant features")
        
        # 2. Drop high NaN ratio features
        nan_threshold = pruning_config.get('drop_high_nan_ratio', 0.2)
        if nan_threshold:
            for col in feature_cols:
                if col in to_drop:
                    continue
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio > nan_threshold:
                    to_drop.add(col)
                    logger.debug(f"High NaN feature: {col} (ratio={nan_ratio:.2%})")
            
            logger.info(f"Dropping features with NaN ratio > {nan_threshold:.0%}: {len([c for c in to_drop if c not in to_drop])} features")
        
        # 3. Correlation filter
        corr_config = pruning_config.get('correlation_filter', {})
        if corr_config.get('enabled', False):
            method = corr_config.get('method', 'spearman')
            threshold = corr_config.get('threshold', 0.98)
            
            # Get features that haven't been dropped yet
            remaining_features = [col for col in feature_cols if col not in to_drop]
            
            if len(remaining_features) > 1:
                logger.info(f"Computing {method} correlation matrix for {len(remaining_features)} features...")
                
                # Compute correlation matrix
                corr_matrix = df[remaining_features].corr(method=method).abs()
                
                # Find highly correlated pairs
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Drop features with correlation > threshold
                corr_dropped = []
                for column in upper_tri.columns:
                    if column in to_drop:
                        continue
                    if any(upper_tri[column] > threshold):
                        to_drop.add(column)
                        corr_dropped.append(column)
                
                if corr_dropped:
                    logger.info(f"Dropping {len(corr_dropped)} highly correlated features (>{threshold:.2%})")
        
        # Apply pruning
        if to_drop:
            logger.info(f"Total features to drop: {len(to_drop)}")
            df = df.drop(columns=list(to_drop))
        
        final_cols = df.shape[1]
        logger.info(f"Final features: {final_cols - len(reserved_cols)} (dropped {initial_cols - final_cols})")
        logger.info("=" * 80)
        
        return df
    
    def assert_no_leakage(self, data: pd.DataFrame, original_cols: List[str] = None) -> bool:
        """
        Validate that no data leakage exists in features.
        
        This is a heuristic check that ensures:
        1. New feature columns have been shifted (NaN in first row)
        2. Original OHLCV columns are unchanged
        
        Args:
            data: DataFrame to validate
            original_cols: List of original column names
        
        Returns:
            True if validation passes, False otherwise
        
        Raises:
            ValueError: If fail_on_unshifted_columns=True and leakage detected
        """
        if not self.leakage_guard_enabled:
            logger.info("Leakage guard disabled, skipping validation")
            return True
        
        if original_cols is None:
            original_cols = self.original_columns
        
        logger.info("Validating leakage prevention...")
        
        # Get new feature columns
        new_cols = [col for col in data.columns if col not in original_cols]
        
        if not new_cols:
            logger.info("No new features to validate")
            return True
        
        # Check if first row of new features is NaN (evidence of shift)
        # Note: This is a heuristic - some indicators may have valid first values
        # More robust: check that each indicator doesn't correlate perfectly with t=0 data
        
        first_row_nans = data[new_cols].iloc[0].isna().sum()
        total_new = len(new_cols)
        
        logger.info(f"Leakage check: {first_row_nans}/{total_new} features have NaN in first row")
        
        # We expect most features to be NaN in first row due to shift(1)
        # But rolling/lag features may have partial windows
        if first_row_nans < total_new * 0.5:
            msg = (f"Potential leakage detected: Only {first_row_nans}/{total_new} "
                   f"features have NaN in first row. Expected most features to be shifted.")
            
            if self.fail_on_unshifted:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False
        
        logger.info("✓ Leakage validation passed")
        return True
    
    def export_features(self, data: pd.DataFrame, output_path: str, 
                       ticker: str = None) -> str:
        """
        Export computed features to CSV.
        
        Args:
            data: DataFrame with features
            output_path: Base output directory
            ticker: Optional ticker name for filename
        
        Returns:
            Path to exported file
        """
        # Create output directory
        feature_dir = Path(output_path) / 'transformed_features'
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if ticker:
            filename = f"{ticker}_features.csv"
        else:
            filename = "features.csv"
        
        filepath = feature_dir / filename
        
        # Export
        data.to_csv(filepath, index=False)
        logger.info(f"Exported features to {filepath}")
        
        return str(filepath)
    
    def get_summary(self) -> Dict:
        """
        Get summary of computed indicators.
        
        Returns:
            Dict with indicator statistics
        """
        return {
            'total_available_indicators': len(self.registry.indicators),
            'computed_indicators': self.computed_indicators.get('total_computed', 0),
            'failed_indicators': self.computed_indicators.get('failed', 0),
            'new_feature_columns': self.computed_indicators.get('new_columns', []),
            'enabled_groups': self.computed_indicators.get('enabled_groups', []),
            'leakage_guard_enabled': self.leakage_guard_enabled,
            'categories': self.registry.get_categories()
        }


def create_indicator_features(data: pd.DataFrame, config: dict = None,
                             output_path: str = None) -> pd.DataFrame:
    """
    Convenience function to create indicator features.
    
    Args:
        data: Input OHLCV DataFrame
        config: Configuration dict
        output_path: Optional output directory for export
    
    Returns:
        DataFrame with indicators
    """
    engine = IndicatorEngine(config)
    
    # Get enabled groups from config
    enabled_groups = None
    if config:
        enabled_groups = config.get('indicators', {}).get('enabled_groups')
    
    # Compute indicators
    df_with_indicators = engine.compute_indicators(data, enabled_groups=enabled_groups)
    
    # Prune features
    df_pruned = engine.prune_features(df_with_indicators)
    
    # Validate leakage
    engine.assert_no_leakage(df_pruned)
    
    # Export if requested
    if output_path:
        ticker = data['Ticker'].iloc[0] if 'Ticker' in data.columns else None
        engine.export_features(df_pruned, output_path, ticker=ticker)
    
    # Print summary
    summary = engine.get_summary()
    logger.info("=" * 80)
    logger.info("INDICATOR ENGINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Available indicators: {summary['total_available_indicators']}")
    logger.info(f"Computed indicators: {summary['computed_indicators']}")
    logger.info(f"Failed indicators: {summary['failed_indicators']}")
    logger.info(f"Feature columns created: {len(summary['new_feature_columns'])}")
    logger.info(f"Enabled groups: {summary['enabled_groups']}")
    logger.info("=" * 80)
    
    return df_pruned
