"""
Data Augmentation Module for Financial Time Series
===================================================

This module provides ADVANCED data augmentation techniques specifically designed
for financial time series forecasting. All techniques are designed to:
1. Only augment TRAINING data (never holdout/test data)
2. Preserve temporal structure, autocorrelation, and market microstructure
3. Generate realistic synthetic samples that respect financial properties
4. Prevent data leakage

BASIC Augmentation Techniques:
------------------------------
1. **Jittering**: Add Gaussian noise to features (not target)
2. **Magnitude Scaling**: Scale features by random factor to simulate
   different volatility regimes
3. **Window Cropping**: Create overlapping subsequences
4. **Mixup**: Interpolate between similar time periods

ADVANCED Augmentation Techniques:
---------------------------------
5. **Volatility Regime Shift**: Apply volatility clustering (GARCH-like)
6. **Block Bootstrap**: Preserve autocorrelation via block resampling
7. **Frequency Masking**: Mask/modify specific frequency bands (FFT-based)
8. **Pattern Morphing**: DTW-based pattern interpolation
9. **Synthetic Trend Injection**: Add realistic trend components
10. **Regime-Conditional Noise**: Add regime-aware noise (high vol = more noise)
11. **Tail Risk Augmentation**: Generate extreme event scenarios
12. **Feature Rotation**: Rotate in PCA space to create diverse samples

Usage:
------
```python
from data_augmentor import DataAugmentor

augmentor = DataAugmentor(config={
    'enabled': True,
    'methods': ['jitter', 'volatility_regime', 'block_bootstrap', 'pattern_morph'],
    'jitter_std': 0.01,
    'scale_range': [0.9, 1.1],
    'n_augmentations': 2,
    'seed': 42,
    # Advanced options
    'garch_alpha': 0.1,
    'garch_beta': 0.85,
    'block_size': 20,
    'freq_mask_ratio': 0.1,
    'morph_n_neighbors': 5,
})

# Augment training data only (after train/test split)
X_train_aug, y_train_aug = augmentor.augment(X_train, y_train)
```

CRITICAL: Only apply to training data, never to holdout/validation/test sets!
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional imports for advanced methods
try:
    from scipy import signal
    from scipy.fft import fft, ifft
    from scipy.spatial.distance import cdist
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==================== AUGMENTATION PRESETS (defined before class) ====================

AUGMENTATION_PRESETS = {
    'minimal': {
        'enabled': True,
        'methods': ['jitter', 'scale'],
        'n_augmentations': 1,
        'jitter_std': 0.01,
        'scale_range': [0.95, 1.05],
        'seed': 42,
    },
    'standard': {
        'enabled': True,
        'methods': ['jitter', 'scale', 'magnitude_warp', 'pattern_morph'],
        'n_augmentations': 1,
        'jitter_std': 0.02,
        'scale_range': [0.9, 1.1],
        'morph_n_neighbors': 5,
        'morph_alpha': 0.3,
        'seed': 42,
    },
    'aggressive': {
        'enabled': True,
        'methods': ['jitter', 'scale', 'volatility_regime', 'block_bootstrap',
                   'pattern_morph', 'smote_ts', 'magnitude_warp'],
        'n_augmentations': 2,
        'jitter_std': 0.03,
        'scale_range': [0.8, 1.2],
        'garch_alpha': 0.15,
        'garch_beta': 0.80,
        'block_size': 15,
        'seed': 42,
    },
    'financial': {
        'enabled': True,
        'methods': ['volatility_regime', 'block_bootstrap', 'regime_noise',
                   'tail_augment', 'magnitude_warp'],
        'n_augmentations': 1,
        'garch_alpha': 0.1,
        'garch_beta': 0.85,
        'block_size': 20,
        'regime_vol_multiplier': [0.5, 2.0],
        'tail_percentile': 5,
        'tail_amplify': 1.5,
        'seed': 42,
    },
    'volatility': {
        'enabled': True,
        'methods': ['volatility_regime', 'regime_noise', 'magnitude_warp', 'scale'],
        'n_augmentations': 2,
        'garch_alpha': 0.12,
        'garch_beta': 0.82,
        'scale_range': [0.7, 1.3],
        'regime_vol_multiplier': [0.3, 3.0],
        'seed': 42,
    },
    'tail_risk': {
        'enabled': True,
        'methods': ['tail_augment', 'volatility_regime', 'jitter', 'smote_ts'],
        'n_augmentations': 2,
        'tail_percentile': 3,
        'tail_amplify': 2.0,
        'garch_alpha': 0.15,
        'garch_beta': 0.80,
        'jitter_std': 0.02,
        'seed': 42,
    },
}


def get_augmentation_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get recommended augmentation configuration for different scenarios.
    
    Presets:
        - 'minimal': Light augmentation, minimal data distortion
        - 'standard': Balanced augmentation for general use
        - 'aggressive': Heavy augmentation for small datasets
        - 'financial': Optimized for financial time series
        - 'volatility': Focus on volatility regime handling
        - 'tail_risk': Focus on extreme event handling
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Configuration dictionary for DataAugmentor
    """
    if preset_name not in AUGMENTATION_PRESETS:
        available = list(AUGMENTATION_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return AUGMENTATION_PRESETS[preset_name].copy()


class DataAugmentor:
    """
    Advanced data augmentation for financial time series.
    
    Designed for supervised learning where we have (X, y) pairs.
    All augmentation preserves the X-y relationship and respects
    financial time series properties (volatility clustering, fat tails, etc.)
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'enabled': False,
        'methods': ['jitter', 'scale'],
        'n_augmentations': 1,  # How many augmented copies per method
        
        # Basic augmentation params
        'jitter_std': 0.02,    # Std of Gaussian noise as fraction of feature std
        'scale_range': [0.9, 1.1],  # Min/max scaling factors
        'window_crop_ratio': 0.9,  # Fraction of sequence to keep
        'mixup_alpha': 0.2,    # Beta distribution parameter for mixup
        
        # Advanced augmentation params
        'garch_alpha': 0.1,    # GARCH alpha (news impact)
        'garch_beta': 0.85,    # GARCH beta (persistence)
        'block_size': 20,      # Block size for bootstrap
        'freq_mask_ratio': 0.15,  # Fraction of frequencies to mask
        'freq_band': 'mid',    # Which band to mask: 'low', 'mid', 'high', 'random'
        'morph_n_neighbors': 5,  # Neighbors for pattern morphing
        'morph_alpha': 0.3,    # Interpolation strength for morphing
        'trend_magnitude': 0.02,  # Max trend injection magnitude
        'regime_vol_multiplier': [0.5, 2.0],  # Volatility regime range
        'tail_percentile': 5,  # Percentile for tail events
        'tail_amplify': 1.5,   # Amplification factor for tails
        'pca_components': 0.95,  # Variance to retain for PCA rotation
        'rotation_angle_std': 0.1,  # Std of rotation angles (radians)
        
        'seed': None,
        # Which columns to augment (None = all feature columns)
        'augment_columns': None,
        # Columns to never augment (e.g., categorical, time-based)
        'exclude_columns': [],
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DataAugmentor with basic and advanced augmentation methods.
        
        Args:
            config: Configuration dictionary. Keys:
                Basic:
                - enabled: bool, whether augmentation is active
                - preset: str, use predefined config ('minimal', 'standard', 'aggressive',
                          'financial', 'volatility', 'tail_risk')
                - methods: list of method names (see AVAILABLE_METHODS)
                - n_augmentations: int, number of augmented copies per method
                - jitter_std: float, noise level for jittering
                - scale_range: [min, max], range for magnitude scaling
                - seed: int or None, random seed for reproducibility
                
                Advanced:
                - garch_alpha/beta: GARCH parameters for volatility regime
                - block_size: size of blocks for bootstrap
                - freq_mask_ratio: fraction of frequencies to mask
                - morph_n_neighbors: neighbors for DTW morphing
                - trend_magnitude: max trend injection strength
                - tail_percentile: percentile threshold for tail events
                - pca_components: variance retained for feature rotation
        """
        # Handle preset first
        base_config = self.DEFAULT_CONFIG.copy()
        user_config = config or {}
        
        preset_name = user_config.get('preset')
        if preset_name:
            try:
                preset_config = get_augmentation_preset(preset_name)
                base_config.update(preset_config)
                logger.info(f"Using augmentation preset: '{preset_name}'")
            except ValueError as e:
                logger.warning(f"Invalid preset: {e}, using defaults")
        
        # Override with user config (preset values can be overridden)
        base_config.update(user_config)
        self.config = base_config
        
        self.enabled = self.config['enabled']
        self.methods = self.config['methods']
        self.n_augmentations = self.config['n_augmentations']
        self.seed = self.config['seed']
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Method dispatch - Basic methods
        self._method_map = {
            'jitter': self._jitter,
            'scale': self._magnitude_scale,
            'window_crop': self._window_crop,
            'mixup': self._mixup,
        }
        
        # Advanced methods
        self._method_map.update({
            'volatility_regime': self._volatility_regime_shift,
            'block_bootstrap': self._block_bootstrap,
            'frequency_mask': self._frequency_mask,
            'pattern_morph': self._pattern_morph,
            'trend_inject': self._synthetic_trend_injection,
            'regime_noise': self._regime_conditional_noise,
            'tail_augment': self._tail_risk_augmentation,
            'feature_rotation': self._feature_rotation,
            'smote_ts': self._smote_timeseries,
            'magnitude_warp': self._magnitude_warp,
        })
        
        # Check which advanced methods are available
        self._check_dependencies()
        
        if self.enabled:
            logger.info(f"DataAugmentor initialized: methods={self.methods}, "
                       f"n_augmentations={self.n_augmentations}")
            logger.info(f"  Advanced methods available: scipy={SCIPY_AVAILABLE}, sklearn={SKLEARN_AVAILABLE}")
    
    def _check_dependencies(self):
        """Check which advanced methods can run based on available dependencies."""
        self._unavailable_methods = []
        
        scipy_methods = ['frequency_mask', 'pattern_morph']
        sklearn_methods = ['feature_rotation', 'smote_ts', 'pattern_morph']
        
        for method in self.methods:
            if method in scipy_methods and not SCIPY_AVAILABLE:
                self._unavailable_methods.append((method, 'scipy'))
            if method in sklearn_methods and not SKLEARN_AVAILABLE:
                if method not in [m[0] for m in self._unavailable_methods]:
                    self._unavailable_methods.append((method, 'sklearn'))
        
        if self._unavailable_methods:
            logger.warning(f"Some methods unavailable due to missing dependencies: "
                          f"{self._unavailable_methods}")
    
    def _estimate_volatility(self, X: np.ndarray) -> np.ndarray:
        """Estimate local volatility from features using rolling std."""
        # Use feature variance as proxy for volatility
        if X.shape[0] < 10:
            return np.ones(X.shape[0])
        
        window = min(20, X.shape[0] // 5)
        vol = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            start = max(0, i - window)
            vol[i] = np.std(X[start:i+1, :].mean(axis=1)) if i > 0 else 1.0
        
        # Normalize to [0, 1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol
    
    def augment(self, X: np.ndarray, y: np.ndarray, 
                feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
            feature_names: Optional list of feature names for selective augmentation
        
        Returns:
            X_augmented: Concatenated original + augmented features
            y_augmented: Concatenated original + augmented targets
        """
        if not self.enabled:
            logger.debug("Data augmentation disabled, returning original data")
            return X, y
        
        if len(X) == 0:
            logger.warning("Empty input data, skipping augmentation")
            return X, y
        
        original_size = len(X)
        logger.info(f"Starting data augmentation: {original_size} samples, "
                   f"methods={self.methods}")
        
        # Start with original data
        X_all = [X.copy()]
        y_all = [y.copy()]
        
        for method_name in self.methods:
            if method_name not in self._method_map:
                logger.warning(f"Unknown augmentation method: {method_name}, skipping")
                continue
            
            method_func = self._method_map[method_name]
            
            for aug_idx in range(self.n_augmentations):
                try:
                    X_aug, y_aug = method_func(X, y, feature_names)
                    if X_aug is not None and len(X_aug) > 0:
                        X_all.append(X_aug)
                        y_all.append(y_aug)
                        logger.debug(f"  {method_name} #{aug_idx+1}: +{len(X_aug)} samples")
                except Exception as e:
                    logger.error(f"Error in {method_name}: {e}")
        
        # Concatenate all
        X_augmented = np.vstack(X_all)
        y_augmented = np.concatenate(y_all)
        
        # Shuffle augmented data (but keep original at start for reproducibility)
        # Actually, for time series we should NOT shuffle - just append
        # The model training should handle the ordering
        
        augmented_size = len(X_augmented)
        multiplier = augmented_size / original_size
        
        logger.info(f"Augmentation complete: {original_size} → {augmented_size} samples "
                   f"({multiplier:.1f}x multiplier)")
        
        return X_augmented, y_augmented
    
    def _jitter(self, X: np.ndarray, y: np.ndarray, 
                feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise to features.
        
        Noise is proportional to each feature's standard deviation to maintain
        relative scales. Target y is NOT modified.
        """
        jitter_std = self.config['jitter_std']
        
        # Calculate per-feature std
        feature_stds = np.std(X, axis=0, keepdims=True)
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)  # Avoid div by zero
        
        # Generate noise proportional to feature std
        noise = np.random.randn(*X.shape) * jitter_std * feature_stds
        
        X_jittered = X + noise
        
        # Keep y unchanged - we're just adding noise to features
        return X_jittered, y.copy()
    
    def _magnitude_scale(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features by a random factor to simulate different volatility regimes.
        
        Each sample gets a different scaling factor, applied consistently
        across all features to maintain relative relationships.
        """
        scale_min, scale_max = self.config['scale_range']
        
        # Random scale factor per sample (broadcast across features)
        scales = np.random.uniform(scale_min, scale_max, size=(len(X), 1))
        
        X_scaled = X * scales
        
        # For regression targets, we should also scale y proportionally
        # if the target is a price/return that would be affected by volatility scaling
        # However, for log returns and pct_change, scaling might not make sense
        # We'll keep y unchanged to be safe - the relationship is preserved
        return X_scaled, y.copy()
    
    def _window_crop(self, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create subsequences by cropping windows.
        
        For non-sequential data, this randomly samples a contiguous block.
        This is useful when combined with lookback features that capture
        local patterns.
        """
        crop_ratio = self.config['window_crop_ratio']
        n_samples = len(X)
        crop_size = int(n_samples * crop_ratio)
        
        if crop_size < 10:  # Too small to crop
            return X.copy(), y.copy()
        
        # Random start position
        start_idx = np.random.randint(0, n_samples - crop_size + 1)
        end_idx = start_idx + crop_size
        
        return X[start_idx:end_idx].copy(), y[start_idx:end_idx].copy()
    
    def _mixup(self, X: np.ndarray, y: np.ndarray,
               feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: interpolate between pairs of samples.
        
        Creates synthetic samples by linearly interpolating between
        two training examples. Uses same-label pairs for classification
        or nearby-target pairs for regression.
        """
        alpha = self.config['mixup_alpha']
        n_samples = len(X)
        
        if n_samples < 2:
            return X.copy(), y.copy()
        
        # Sample mixing weights from Beta distribution
        lam = np.random.beta(alpha, alpha, size=n_samples)
        lam = lam.reshape(-1, 1)  # For broadcasting with X
        
        # Random permutation for pairing
        perm = np.random.permutation(n_samples)
        
        # Mixup
        X_mixed = lam * X + (1 - lam) * X[perm]
        y_mixed = lam.ravel() * y + (1 - lam.ravel()) * y[perm]
        
        return X_mixed, y_mixed
    
    # ==================== ADVANCED AUGMENTATION METHODS ====================
    
    def _volatility_regime_shift(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply GARCH-like volatility clustering to features.
        
        Creates realistic volatility regime shifts by applying time-varying
        scaling based on simulated GARCH(1,1) process. This captures the
        empirical observation that volatility clusters in financial data.
        
        σ²_t = α * ε²_{t-1} + β * σ²_{t-1}
        
        High volatility periods get amplified, low volatility periods get damped.
        """
        alpha = self.config['garch_alpha']
        beta = self.config['garch_beta']
        n_samples = len(X)
        
        # Initialize volatility process
        vol = np.ones(n_samples)
        innovations = np.random.randn(n_samples) ** 2  # Squared innovations
        
        # Simulate GARCH(1,1) volatility
        for t in range(1, n_samples):
            vol[t] = 0.01 + alpha * innovations[t-1] + beta * vol[t-1]
        
        # Normalize volatility to reasonable range
        vol = np.sqrt(vol)
        vol = vol / vol.mean()  # Center around 1
        vol = np.clip(vol, 0.5, 2.0)  # Prevent extreme values
        
        # Apply volatility scaling to features
        X_vol = X * vol.reshape(-1, 1)
        
        return X_vol, y.copy()
    
    def _block_bootstrap(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Block bootstrap resampling for time series.
        
        Preserves autocorrelation structure by sampling contiguous blocks
        and concatenating them. Better than random sampling for time series.
        
        Uses circular block bootstrap to handle edge effects.
        """
        block_size = self.config['block_size']
        n_samples = len(X)
        
        if n_samples < block_size * 2:
            # Fall back to regular resampling for small datasets
            indices = np.random.choice(n_samples, n_samples, replace=True)
            return X[indices].copy(), y[indices].copy()
        
        # Calculate number of blocks needed
        n_blocks = int(np.ceil(n_samples / block_size))
        
        # Sample block starting positions
        starts = np.random.randint(0, n_samples, n_blocks)
        
        # Build resampled dataset from blocks
        X_blocks = []
        y_blocks = []
        
        for start in starts:
            end = start + block_size
            if end <= n_samples:
                X_blocks.append(X[start:end])
                y_blocks.append(y[start:end])
            else:
                # Circular: wrap around to beginning
                X_blocks.append(np.vstack([X[start:], X[:end - n_samples]]))
                y_blocks.append(np.concatenate([y[start:], y[:end - n_samples]]))
        
        X_boot = np.vstack(X_blocks)[:n_samples]
        y_boot = np.concatenate(y_blocks)[:n_samples]
        
        return X_boot, y_boot
    
    def _frequency_mask(self, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mask specific frequency bands using FFT.
        
        Removes or attenuates specific frequency components to create
        augmented samples that focus on different time scales.
        - 'low': mask low frequencies (removes trends)
        - 'mid': mask mid frequencies (removes medium-term patterns)
        - 'high': mask high frequencies (removes noise)
        
        This is particularly useful for models that need to be robust
        to different frequency components.
        """
        if not SCIPY_AVAILABLE:
            logger.debug("frequency_mask requires scipy, skipping")
            return X.copy(), y.copy()
        
        mask_ratio = self.config['freq_mask_ratio']
        band = self.config['freq_band']
        n_samples, n_features = X.shape
        
        X_masked = np.zeros_like(X)
        
        for feat_idx in range(n_features):
            # FFT of feature column
            freq = fft(X[:, feat_idx])
            n_freq = len(freq)
            
            # Create frequency mask
            mask = np.ones(n_freq)
            n_mask = int(n_freq * mask_ratio)
            
            if band == 'low':
                # Mask low frequencies (DC + first few)
                mask[1:n_mask+1] = 0
                mask[-n_mask:] = 0  # Symmetric for real signal
            elif band == 'mid':
                # Mask middle frequencies
                mid_start = n_freq // 4
                mask[mid_start:mid_start + n_mask] = 0
                mask[-(mid_start + n_mask):-mid_start] = 0
            elif band == 'high':
                # Mask high frequencies
                mask[n_freq//2 - n_mask:n_freq//2 + n_mask] = 0
            else:  # 'random'
                # Random frequency masking
                mask_indices = np.random.choice(n_freq, n_mask, replace=False)
                mask[mask_indices] = 0
            
            # Apply mask and inverse FFT
            freq_masked = freq * mask
            X_masked[:, feat_idx] = np.real(ifft(freq_masked))
        
        return X_masked, y.copy()
    
    def _pattern_morph(self, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create new samples by morphing between similar patterns.
        
        Uses nearest neighbors to find similar samples, then interpolates
        between them to create realistic synthetic patterns.
        
        Similar to SMOTE but respects temporal locality in feature space.
        """
        if not SKLEARN_AVAILABLE:
            logger.debug("pattern_morph requires sklearn, skipping")
            return X.copy(), y.copy()
        
        n_neighbors = self.config['morph_n_neighbors']
        alpha = self.config['morph_alpha']
        n_samples = len(X)
        
        if n_samples < n_neighbors + 1:
            return X.copy(), y.copy()
        
        # Find nearest neighbors for each sample
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        X_morphed = np.zeros_like(X)
        y_morphed = np.zeros_like(y)
        
        for i in range(n_samples):
            # Skip self (index 0), pick random neighbor
            neighbor_idx = indices[i, np.random.randint(1, n_neighbors + 1)]
            
            # Random interpolation weight
            lam = np.random.uniform(0, alpha)
            
            # Morph between sample and neighbor
            X_morphed[i] = (1 - lam) * X[i] + lam * X[neighbor_idx]
            y_morphed[i] = (1 - lam) * y[i] + lam * y[neighbor_idx]
        
        return X_morphed, y_morphed
    
    def _synthetic_trend_injection(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synthetic trend components into features.
        
        Adds linear or polynomial trend components to simulate
        different market regimes (trending up, down, or range-bound).
        
        Useful for making models robust to trend changes.
        """
        magnitude = self.config['trend_magnitude']
        n_samples, n_features = X.shape
        
        # Random trend type: linear, quadratic, or sinusoidal
        trend_type = np.random.choice(['linear', 'quadratic', 'sine'])
        trend_direction = np.random.choice([-1, 1])
        
        t = np.linspace(0, 1, n_samples)
        
        if trend_type == 'linear':
            trend = trend_direction * magnitude * t
        elif trend_type == 'quadratic':
            trend = trend_direction * magnitude * (t ** 2)
        else:  # sine
            freq = np.random.uniform(0.5, 3)  # Random frequency
            trend = magnitude * np.sin(2 * np.pi * freq * t)
        
        # Apply trend to all features (scaled by feature std)
        feature_stds = np.std(X, axis=0, keepdims=True)
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        
        X_trended = X + trend.reshape(-1, 1) * feature_stds
        
        return X_trended, y.copy()
    
    def _regime_conditional_noise(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise proportional to local volatility regime.
        
        High volatility periods get more noise, low volatility periods
        get less. This respects the heteroskedastic nature of financial data.
        """
        vol_range = self.config['regime_vol_multiplier']
        base_std = self.config['jitter_std']
        
        # Estimate local volatility
        local_vol = self._estimate_volatility(X)
        
        # Scale volatility to multiplier range
        vol_multiplier = vol_range[0] + local_vol * (vol_range[1] - vol_range[0])
        
        # Calculate per-feature std
        feature_stds = np.std(X, axis=0, keepdims=True)
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        
        # Generate volatility-scaled noise
        noise = np.random.randn(*X.shape) * base_std * feature_stds
        noise *= vol_multiplier.reshape(-1, 1)
        
        X_noisy = X + noise
        
        return X_noisy, y.copy()
    
    def _tail_risk_augmentation(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment tail/extreme events to improve model robustness.
        
        Identifies extreme samples (based on target y or feature magnitude)
        and creates amplified versions to increase their representation.
        
        Helps models learn from rare but important extreme events.
        """
        percentile = self.config['tail_percentile']
        amplify = self.config['tail_amplify']
        
        # Identify tail samples based on y magnitude
        y_abs = np.abs(y)
        threshold_low = np.percentile(y_abs, percentile)
        threshold_high = np.percentile(y_abs, 100 - percentile)
        
        # Find tail indices
        tail_mask = (y_abs <= threshold_low) | (y_abs >= threshold_high)
        tail_indices = np.where(tail_mask)[0]
        
        if len(tail_indices) < 5:
            # Not enough tail samples, return copy
            return X.copy(), y.copy()
        
        # Create augmented tails
        X_tails = X[tail_indices].copy()
        y_tails = y[tail_indices].copy()
        
        # Amplify: push features further in their direction from mean
        feature_means = X.mean(axis=0)
        X_tails_aug = feature_means + amplify * (X_tails - feature_means)
        
        # Amplify y as well (for extreme moves)
        y_mean = y.mean()
        y_tails_aug = y_mean + amplify * (y_tails - y_mean)
        
        # Combine original with amplified tails
        X_aug = np.vstack([X, X_tails_aug])
        y_aug = np.concatenate([y, y_tails_aug])
        
        return X_aug, y_aug
    
    def _feature_rotation(self, X: np.ndarray, y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate features in PCA space to create diverse samples.
        
        Projects features to PCA space, applies small random rotations,
        then projects back. Creates samples that are statistically similar
        but occupy different regions of feature space.
        """
        if not SKLEARN_AVAILABLE:
            logger.debug("feature_rotation requires sklearn, skipping")
            return X.copy(), y.copy()
        
        n_components = self.config['pca_components']
        angle_std = self.config['rotation_angle_std']
        n_samples, n_features = X.shape
        
        if n_features < 3:
            return X.copy(), y.copy()
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        n_pca = X_pca.shape[1]
        
        if n_pca < 2:
            return X.copy(), y.copy()
        
        # Generate random rotation matrix (small perturbation from identity)
        # Use Givens rotations for numerical stability
        rotation = np.eye(n_pca)
        
        for i in range(n_pca - 1):
            for j in range(i + 1, n_pca):
                angle = np.random.randn() * angle_std
                c, s = np.cos(angle), np.sin(angle)
                givens = np.eye(n_pca)
                givens[i, i] = c
                givens[j, j] = c
                givens[i, j] = -s
                givens[j, i] = s
                rotation = rotation @ givens
        
        # Apply rotation in PCA space
        X_pca_rotated = X_pca @ rotation
        
        # Project back to original space
        X_rotated = pca.inverse_transform(X_pca_rotated)
        
        return X_rotated, y.copy()
    
    def _smote_timeseries(self, X: np.ndarray, y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE-like oversampling adapted for time series regression.
        
        Creates synthetic samples by interpolating between samples with
        similar target values, preserving the X-y relationship.
        
        Unlike standard SMOTE, this groups samples by target quantiles
        rather than class labels.
        """
        if not SKLEARN_AVAILABLE:
            logger.debug("smote_ts requires sklearn, skipping")
            return X.copy(), y.copy()
        
        n_samples = len(X)
        n_neighbors = min(5, n_samples // 10)
        
        if n_samples < n_neighbors * 2:
            return X.copy(), y.copy()
        
        # Group samples by target quantile
        n_groups = 5
        quantiles = np.percentile(y, np.linspace(0, 100, n_groups + 1))
        
        X_synth_list = []
        y_synth_list = []
        
        for i in range(n_groups):
            mask = (y >= quantiles[i]) & (y < quantiles[i+1])
            if i == n_groups - 1:  # Include max value in last group
                mask = (y >= quantiles[i]) & (y <= quantiles[i+1])
            
            group_indices = np.where(mask)[0]
            
            if len(group_indices) < n_neighbors:
                continue
            
            X_group = X[group_indices]
            y_group = y[group_indices]
            
            # Find neighbors within group
            nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(group_indices)))
            nn.fit(X_group)
            
            # Generate synthetic samples
            n_synth = len(group_indices) // 2
            for _ in range(n_synth):
                idx = np.random.randint(len(group_indices))
                _, neighbors = nn.kneighbors([X_group[idx]])
                neighbor_idx = neighbors[0, np.random.randint(1, len(neighbors[0]))]
                
                lam = np.random.uniform(0, 1)
                X_synth_list.append((1 - lam) * X_group[idx] + lam * X_group[neighbor_idx])
                y_synth_list.append((1 - lam) * y_group[idx] + lam * y_group[neighbor_idx])
        
        if not X_synth_list:
            return X.copy(), y.copy()
        
        X_synth = np.vstack([X] + [np.array(X_synth_list)])
        y_synth = np.concatenate([y, np.array(y_synth_list)])
        
        return X_synth, y_synth
    
    def _magnitude_warp(self, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply smooth magnitude warping using cubic splines.
        
        Multiplies features by a smooth random curve, simulating
        gradual regime changes in volatility/magnitude.
        
        Different from simple scaling: the scaling factor changes
        smoothly over time rather than being constant.
        """
        n_samples, n_features = X.shape
        
        # Generate smooth warping curve using random knots
        n_knots = min(4, n_samples // 20)
        if n_knots < 2:
            return X.copy(), y.copy()
        
        knot_positions = np.linspace(0, n_samples - 1, n_knots).astype(int)
        knot_values = np.random.uniform(0.8, 1.2, n_knots)  # Random multipliers
        
        # Interpolate to create smooth curve
        warp_curve = np.interp(np.arange(n_samples), knot_positions, knot_values)
        
        # Apply warping
        X_warped = X * warp_curve.reshape(-1, 1)
        
        return X_warped, y.copy()
    
    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Return summary of augmentation configuration."""
        return {
            'enabled': self.enabled,
            'methods': self.methods if self.enabled else [],
            'n_augmentations': self.n_augmentations if self.enabled else 0,
            'expected_multiplier': 1 + len(self.methods) * self.n_augmentations if self.enabled else 1,
            'config': self.config
        }


class TimeSeriesAugmentor(DataAugmentor):
    """
    Extended augmentor for sequential/time-series specific augmentations.
    
    Adds techniques that work on 3D sequence data (n_samples, seq_len, n_features)
    used by LSTM/Transformer models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Additional time-series specific methods
        self._method_map.update({
            'time_warp': self._time_warp,
            'permute_segments': self._permute_segments,
        })
    
    def _time_warp(self, X: np.ndarray, y: np.ndarray,
                   feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply smooth time warping using cubic spline interpolation.
        
        Stretches/compresses different parts of the sequence slightly.
        Only meaningful for 3D sequential data (n_samples, seq_len, n_features).
        """
        # Check if data is 3D (sequential)
        if X.ndim != 3:
            logger.debug("Time warp requires 3D data, skipping")
            return X.copy(), y.copy()
        
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            logger.warning("scipy required for time_warp, skipping")
            return X.copy(), y.copy()
        
        n_samples, seq_len, n_features = X.shape
        X_warped = np.zeros_like(X)
        
        for i in range(n_samples):
            # Create warping curve (4 random control points)
            warp_steps = 4
            warp_points = np.sort(np.random.randint(0, seq_len, size=warp_steps))
            warp_values = warp_points + np.random.randn(warp_steps) * 2
            warp_values = np.clip(warp_values, 0, seq_len - 1)
            
            # Interpolate warping function
            original_steps = np.linspace(0, seq_len - 1, seq_len)
            spline = CubicSpline(warp_points, warp_values)
            warped_steps = spline(original_steps)
            warped_steps = np.clip(warped_steps, 0, seq_len - 1).astype(int)
            
            X_warped[i] = X[i, warped_steps, :]
        
        return X_warped, y.copy()
    
    def _permute_segments(self, X: np.ndarray, y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly permute segments within each sequence.
        
        Breaks sequence into k segments and shuffles their order.
        Useful for capturing patterns that are position-invariant.
        
        WARNING: May break temporal causality - use with caution!
        """
        if X.ndim != 3:
            logger.debug("Permute segments requires 3D data, skipping")
            return X.copy(), y.copy()
        
        n_samples, seq_len, n_features = X.shape
        n_segments = 4  # Number of segments
        
        if seq_len < n_segments:
            return X.copy(), y.copy()
        
        segment_len = seq_len // n_segments
        X_permuted = np.zeros_like(X)
        
        for i in range(n_samples):
            # Split into segments
            segments = []
            for j in range(n_segments):
                start = j * segment_len
                end = start + segment_len if j < n_segments - 1 else seq_len
                segments.append(X[i, start:end, :])
            
            # Shuffle segments
            np.random.shuffle(segments)
            
            # Reassemble
            X_permuted[i] = np.vstack(segments)[:seq_len]
        
        return X_permuted, y.copy()


def list_available_methods() -> Dict[str, str]:
    """
    List all available augmentation methods with descriptions.
    
    Returns:
        Dictionary mapping method name to description
    """
    return {
        # Basic methods
        'jitter': 'Add Gaussian noise proportional to feature std',
        'scale': 'Random magnitude scaling (constant per sample)',
        'window_crop': 'Random contiguous subsequence sampling',
        'mixup': 'Linear interpolation between sample pairs',
        
        # Advanced methods
        'volatility_regime': 'GARCH-like volatility clustering simulation',
        'block_bootstrap': 'Block resampling preserving autocorrelation',
        'frequency_mask': 'FFT-based frequency band masking (requires scipy)',
        'pattern_morph': 'Nearest-neighbor based pattern interpolation (requires sklearn)',
        'trend_inject': 'Synthetic linear/polynomial/sinusoidal trend addition',
        'regime_noise': 'Volatility-proportional noise injection',
        'tail_augment': 'Amplification of extreme/tail events',
        'feature_rotation': 'PCA-space rotation for diversity (requires sklearn)',
        'smote_ts': 'SMOTE adapted for regression (requires sklearn)',
        'magnitude_warp': 'Smooth time-varying magnitude scaling',
        
        # TimeSeriesAugmentor only (3D data)
        'time_warp': 'Cubic spline time warping (3D data, requires scipy)',
        'permute_segments': 'Random segment permutation (3D data)',
    }
