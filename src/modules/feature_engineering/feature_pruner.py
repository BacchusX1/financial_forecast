"""
Feature Pruner: Train-only fit/transform for safe feature selection.

CRITICAL: This module enforces that all pruning decisions are made using
TRAINING DATA ONLY. The fitted pruner stores columns to drop and applies
the same decisions to any dataset (train, validation, test, holdout).

Supports:
- Constant/near-constant feature removal
- High NaN ratio feature removal  
- Correlation-based redundancy removal

Usage:
    pruner = FeaturePruner(config)
    pruner.fit(X_train)  # Fit on training data ONLY
    X_train_pruned = pruner.transform(X_train)
    X_test_pruned = pruner.transform(X_test)  # Same columns dropped
    pruner.save_artifact(output_dir)  # Save decisions for reproducibility
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PruningDecision:
    """Record of a single pruning decision."""
    column: str
    reason: str
    detail: str  # e.g., "nan_ratio=0.45" or "corr_with=feature_x (0.99)"


class FeaturePruner:
    """
    Feature pruner with fit/transform API for leakage-free pruning.
    
    All pruning decisions are made during fit() using training data only.
    transform() applies the same decisions to any dataset.
    """
    
    RESERVED_COLUMNS = ['Datetime', 'Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    
    def __init__(self, config: dict = None):
        """
        Initialize pruner with configuration.
        
        Args:
            config: Dict with keys:
                - drop_constant: bool (default: True)
                - drop_high_nan_ratio: float threshold (default: 0.2, set to None to disable)
                - correlation_filter: dict with {enabled, method, threshold}
        """
        self.config = config or {}
        
        # Pruning settings
        self.drop_constant = self.config.get('drop_constant', True)
        self.nan_threshold = self.config.get('drop_high_nan_ratio', 0.2)
        
        corr_config = self.config.get('correlation_filter', {})
        self.corr_enabled = corr_config.get('enabled', False)
        self.corr_method = corr_config.get('method', 'spearman')
        self.corr_threshold = corr_config.get('threshold', 0.98)
        
        # State (populated by fit)
        self.fitted = False
        self.columns_to_drop: Set[str] = set()
        self.decisions: List[PruningDecision] = []
        self.original_feature_count: int = 0
        self.final_feature_count: int = 0
        self.feature_columns: List[str] = []  # Columns seen during fit
        
    def fit(self, df: pd.DataFrame, target_col: str = 'Close') -> 'FeaturePruner':
        """
        Fit pruner on training data ONLY.
        
        Analyzes the training data to determine which columns to drop.
        This method MUST be called with training data only to prevent leakage.
        
        Args:
            df: Training DataFrame
            target_col: Target column name (excluded from pruning)
        
        Returns:
            self (for method chaining)
        """
        logger.info("=" * 60)
        logger.info("FEATURE PRUNER: FIT (train-only)")
        logger.info("=" * 60)
        
        # Reset state
        self.columns_to_drop = set()
        self.decisions = []
        
        # Identify feature columns (exclude reserved)
        reserved = set(self.RESERVED_COLUMNS) | {target_col}
        self.feature_columns = [col for col in df.columns if col not in reserved]
        self.original_feature_count = len(self.feature_columns)
        
        logger.info(f"Input features: {self.original_feature_count}")
        
        # 1. Drop constant/near-constant features
        if self.drop_constant:
            self._fit_constant_filter(df)
        
        # 2. Drop high NaN ratio features
        if self.nan_threshold is not None:
            self._fit_nan_filter(df)
        
        # 3. Correlation filter
        if self.corr_enabled:
            self._fit_correlation_filter(df)
        
        self.final_feature_count = self.original_feature_count - len(self.columns_to_drop)
        self.fitted = True
        
        logger.info(f"Columns to drop: {len(self.columns_to_drop)}")
        logger.info(f"Final features: {self.final_feature_count}")
        logger.info("=" * 60)
        
        return self
    
    def _fit_constant_filter(self, df: pd.DataFrame):
        """Identify constant or near-constant features."""
        count = 0
        for col in self.feature_columns:
            if col in self.columns_to_drop:
                continue
            nunique = df[col].nunique()
            if nunique <= 1:
                self.columns_to_drop.add(col)
                self.decisions.append(PruningDecision(
                    column=col,
                    reason="constant",
                    detail=f"nunique={nunique}"
                ))
                count += 1
        
        if count > 0:
            logger.info(f"  Constant features: {count}")
    
    def _fit_nan_filter(self, df: pd.DataFrame):
        """Identify features with high NaN ratio."""
        count = 0
        for col in self.feature_columns:
            if col in self.columns_to_drop:
                continue
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > self.nan_threshold:
                self.columns_to_drop.add(col)
                self.decisions.append(PruningDecision(
                    column=col,
                    reason="high_nan",
                    detail=f"nan_ratio={nan_ratio:.2%}"
                ))
                count += 1
        
        if count > 0:
            logger.info(f"  High NaN features (>{self.nan_threshold:.0%}): {count}")
    
    def _fit_correlation_filter(self, df: pd.DataFrame):
        """Identify highly correlated features (keep first, drop second)."""
        # Get remaining features
        remaining = [col for col in self.feature_columns if col not in self.columns_to_drop]
        
        if len(remaining) < 2:
            return
        
        logger.info(f"  Computing {self.corr_method} correlation for {len(remaining)} features...")
        
        # Compute correlation matrix
        corr_matrix = df[remaining].corr(method=self.corr_method).abs()
        
        # Get upper triangle (avoid duplicate pairs)
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # Find pairs above threshold
        count = 0
        for i, col1 in enumerate(remaining):
            for j, col2 in enumerate(remaining):
                if not upper_tri[i, j]:
                    continue
                if col2 in self.columns_to_drop:
                    continue
                    
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > self.corr_threshold:
                    # Drop the second column (keep first)
                    self.columns_to_drop.add(col2)
                    self.decisions.append(PruningDecision(
                        column=col2,
                        reason="high_correlation",
                        detail=f"corr_with={col1} ({corr_val:.3f})"
                    ))
                    count += 1
        
        if count > 0:
            logger.info(f"  Correlated features (>{self.corr_threshold:.0%}): {count}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted pruning decisions to a dataset.
        
        Args:
            df: DataFrame to transform (can be train, test, or any split)
        
        Returns:
            DataFrame with dropped columns removed
        """
        if not self.fitted:
            raise RuntimeError("Pruner not fitted. Call fit() first with training data.")
        
        # Only drop columns that exist in this dataframe
        cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        
        if cols_to_drop:
            return df.drop(columns=cols_to_drop)
        return df.copy()
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """Convenience method: fit and transform in one call."""
        return self.fit(df, target_col).transform(df)
    
    def get_summary(self) -> Dict:
        """Get summary of pruning decisions."""
        if not self.fitted:
            return {"error": "Pruner not fitted"}
        
        # Group by reason
        by_reason = {}
        for d in self.decisions:
            if d.reason not in by_reason:
                by_reason[d.reason] = []
            by_reason[d.reason].append(d.column)
        
        return {
            "original_features": self.original_feature_count,
            "dropped_features": len(self.columns_to_drop),
            "final_features": self.final_feature_count,
            "by_reason": {k: len(v) for k, v in by_reason.items()},
            "dropped_columns": list(self.columns_to_drop)
        }
    
    def save_artifact(self, output_dir: str, filename: str = "pruning_decisions.json") -> str:
        """
        Save pruning decisions as JSON artifact.
        
        Args:
            output_dir: Directory to save to
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        if not self.fitted:
            raise RuntimeError("Pruner not fitted. Call fit() first.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_dir) / filename
        
        artifact = {
            "config": self.config,
            "summary": self.get_summary(),
            "decisions": [asdict(d) for d in self.decisions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(artifact, f, indent=2)
        
        logger.info(f"Saved pruning artifact: {filepath}")
        return str(filepath)
    
    @classmethod
    def load_artifact(cls, filepath: str) -> 'FeaturePruner':
        """
        Load a fitted pruner from artifact.
        
        Args:
            filepath: Path to pruning_decisions.json
        
        Returns:
            Fitted FeaturePruner instance
        """
        with open(filepath, 'r') as f:
            artifact = json.load(f)
        
        pruner = cls(config=artifact.get('config', {}))
        pruner.columns_to_drop = set(artifact['summary']['dropped_columns'])
        pruner.decisions = [
            PruningDecision(**d) for d in artifact.get('decisions', [])
        ]
        pruner.original_feature_count = artifact['summary']['original_features']
        pruner.final_feature_count = artifact['summary']['final_features']
        pruner.fitted = True
        
        return pruner
