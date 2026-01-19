import logging
import yaml
import sys
import os

# Add src to path to import registry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules/feature_engineering')))

try:
    from indicator_registry import INDICATOR_REGISTRY
except ImportError:
    # Fallack if run from different location
    sys.path.append('/home/bacchus/coding/trading_forecast_v2/src/modules/feature_engineering')
    from indicator_registry import INDICATOR_REGISTRY

logger = logging.getLogger("ConfigValidator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

class ConfigValidator:
    """
    Validates configuration.yml against the codebase capabilities.
    Enforces schemas, detects dead parameters, and estimates compute costs.
    """
    
    VALID_GROUPS = {'returns', 'trend', 'volatility', 'momentum', 'volume', 'candle', 'regime'}
    
    def __init__(self, config_path=None, config_dict=None):
        if config_dict is not None: # Check for not None, as empty dict is valid input type
            self.config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Must provide config_path or config_dict")
            
        self.registry = INDICATOR_REGISTRY
        self.fe_config = self.config.get('feature_engineering', {})
        self.ind_config = self.fe_config.get('indicators', {})

    def validate(self):
        """Run all validation checks."""
        issues = []
        issues.extend(self._validate_groups())
        issues.extend(self._validate_params_alignment())
        issues.extend(self._validate_compute_complexity())
        issues.extend(self._validate_leakage_settings())
        
        return issues

    def _validate_groups(self):
        """Ensure enabled_groups are valid."""
        issues = []
        enabled = self.ind_config.get('enabled_groups', [])
        
        if not isinstance(enabled, list):
            issues.append(("ERROR", f"enabled_groups must be a list, got {type(enabled)}"))
            return issues
            
        for group in enabled:
            if group not in self.VALID_GROUPS:
                issues.append(("ERROR", f"Invalid indicator group: '{group}'. Valid options: {self.VALID_GROUPS}"))
        
        return issues

    def _validate_params_alignment(self):
        """
        Check if configured params actually exist in the registry.
        This detects the 'Implementation Gap' where config asks for RSI 100 but registry only has 14.
        """
        issues = []
        params = self.ind_config.get('params', {})
        
        # Helper to reconstruct expected names, e.g. rsi -> rsi_14
        # This is heuristics based on standard naming in registry
        
        for algo, settings in params.items():
            # Handle List[int] (simple windows)
            if isinstance(settings, list) and all(isinstance(x, int) for x in settings):
                for window in settings:
                    expected_name = f"{algo}_{window}"
                    # Try to find exactly this, or with suffix
                    # This is fuzzy because registry naming varies (sma_20 vs rsi_14)
                    
                    found = False
                    for reg_name in self.registry.indicators.keys():
                        if reg_name == expected_name:
                            found = True
                            break
                        if reg_name.startswith(expected_name + "_"): # e.g. rsi_14_lag...
                            found = True
                            break
                    
                    if not found:
                        # Check if it exists with reversed format (unlikely but possible) or different name
                        issues.append(("WARNING", f"Config expects '{expected_name}', but it is NOT in the Registry. Feature will be missing unless Registry is updated."))

        return issues

    def _validate_compute_complexity(self):
        """Estimate feature explosion."""
        issues = []
        enabled = self.ind_config.get('enabled_groups', [])
        
        # Count enabled indicators
        count = 0
        for group in enabled:
            count += len(self.registry.get_indicators_by_category(group))
            
        # Add lag multipliers
        lags = self.fe_config.get('lag_features', [])
        lag_multiplier = 1 + len(lags)
        
        # Add rolling multipliers
        rolling_wins = self.fe_config.get('rolling_windows', [])
        rolling_funcs = self.fe_config.get('rolling_functions', [])
        rolling_multiplier = len(rolling_wins) * len(rolling_funcs)
        
        total_features_per_col = count * lag_multiplier  # Indicators * Lags
        # Note: Rolling is usually applied to Target Cols, whilst Indicators are separate. 
        # But this is a rough heuristic.
        
        dl_config = self.config.get('data_loader', {})
        candle = dl_config.get('candle_length', '1h')
        
        # Rough row estimator
        intervals_per_day = {
            '1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1
        }
        rows_per_day = intervals_per_day.get(candle, 24) 
        # Assume 1 year default if period not set (params often in years)
        # But dl_config might use start_date. Assuming 2 years for safety.
        estimated_rows = rows_per_day * 365 * 2 
        
        total_cells = estimated_rows * total_features_per_col
        
        if total_cells > 100_000_000:
            issues.append(("WARNING", f"High Compute Load: Est. {total_cells/1e6:.1f}M cells. May require >1GB RAM."))
            
        if count > 100 and rows_per_day >= 288: # 5m or lower
             issues.append(("WARNING", f"High Indicator Count ({count}) with low timeframe ({candle}). Processing will be slow."))
             
        return issues

    def _validate_leakage_settings(self):
        """Check for conflicting leakage settings."""
        issues = []
        allow_abs = self.fe_config.get('allow_absolute_scale_features', True)
        
        if allow_abs:
            issues.append(("SUGGESTION", "allow_absolute_scale_features is TRUE. Set to FALSE for robust Deep Learning models."))
            
        return issues

if __name__ == "__main__":
    # Test run
    try:
        v = ConfigValidator(config_path='configuration.yml')
        results = v.validate()
        print("\n--- Validation Results ---")
        for level, msg in results:
            print(f"[{level}] {msg}")
            
        if not results:
            print("Configuration is valid.")
            
    except Exception as e:
        print(f"Error loading config: {e}")
