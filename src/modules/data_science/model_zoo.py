"""
Model Zoo: Advanced forecasting architectures for trading_forecast_v2.

Includes:
- LightGBM (tree-based multioutput)
- TCN (Temporal Convolutional Network)
- N-BEATS-Lite (interpretable forecast)
- MoE (Mixture-of-Experts with regime awareness)
- Multi-Task (joint regression + classification)

All models support multioutput forecasting (10-day horizon).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# xLSTM imports (optional)
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

# Try LightGBM import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


# ============================================================================
# 1. LIGHTGBM MULTIOUTPUT REGRESSOR
# ============================================================================

class LightGBMMultiOutput:
    """
    LightGBM wrapper for multioutput regression.
    
    Trains separate model for each output step.
    """
    
    def __init__(self, n_outputs: int = 10, **lgb_params):
        """
        Args:
            n_outputs: Number of forecast steps
            **lgb_params: Parameters for LightGBM
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        self.n_outputs = n_outputs
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        self.models = []
        self.feature_importances_ = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Fit LightGBM models (one per output step).
        
        Args:
            X_train: Training features (N, features)
            y_train: Training targets (N, n_outputs)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        self.models = []
        
        for i in range(self.n_outputs):
            if verbose and i % 5 == 0:
                logger.info(f"Training LightGBM for step {i+1}/{self.n_outputs}...")
            
            # Create dataset
            train_data = lgb.Dataset(X_train, label=y_train[:, i])
            valid_sets = [train_data]
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val[:, i], reference=train_data)
                valid_sets.append(val_data)
            
            # Train
            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=100,
                valid_sets=valid_sets,
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]
            )
            
            self.models.append(model)
        
        # Get feature importances (average across models)
        importances = np.array([m.feature_importance(importance_type='gain') for m in self.models])
        self.feature_importances_ = np.mean(importances, axis=0)
        
        if verbose:
            logger.info(f"LightGBM training complete ({len(self.models)} models)")
    
    def predict(self, X):
        """
        Predict all output steps.
        
        Args:
            X: Features (N, features)
        
        Returns:
            Predictions (N, n_outputs)
        """
        predictions = np.zeros((X.shape[0], self.n_outputs))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        return predictions
    
    def get_feature_importance(self, feature_names: List[str] = None, top_k: int = 20) -> Dict:
        """Get top feature importances."""
        if self.feature_importances_ is None:
            return {}
        
        indices = np.argsort(self.feature_importances_)[::-1][:top_k]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        
        return {
            feature_names[i]: float(self.feature_importances_[i])
            for i in indices
        }


# ============================================================================
# GPU DEVICE HELPER
# ============================================================================

def get_device(use_gpu: Union[bool, str] = True) -> torch.device:
    """
    Get the appropriate device for PyTorch models.
    
    Args:
        use_gpu: True/False or 'auto' to auto-detect
    
    Returns:
        torch.device for CPU or CUDA
    """
    if use_gpu == 'auto' or use_gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
    elif use_gpu is False or use_gpu == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(use_gpu)  # Allow explicit device string
    return device


# ============================================================================
# 2. xLSTM REGRESSOR (Extended LSTM with Matrix Memory)
# ============================================================================

class xLSTMRegressor(nn.Module):
    """
    xLSTM-based regressor for multi-step forecasting.
    
    Uses mLSTM (matrix LSTM) blocks which are fully parallelizable
    with exponential gating and matrix memory.
    
    Args:
        xlstm_config: xLSTMBlockStackConfig for the model
        input_dim: Input feature dimension per timestep
        adjusted_dim: Adjusted dimension (divisible by num_heads)
        output_dim: Number of forecast steps (horizon)
        dropout: Dropout rate for output projection
    """
    
    def __init__(self, xlstm_config, input_dim: int, adjusted_dim: int, 
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        
        if not XLSTM_AVAILABLE:
            raise ImportError("xLSTM not installed. Install with: pip install xlstm")
        
        self.input_proj = nn.Linear(input_dim, adjusted_dim) if input_dim != adjusted_dim else nn.Identity()
        self.xlstm = xLSTMBlockStack(xlstm_config)
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(adjusted_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
        # Store config for serialization
        self._input_dim = input_dim
        self._adjusted_dim = adjusted_dim
        self._output_dim = output_dim
        self._dropout = dropout
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, timesteps, features)
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # x: (batch, timesteps, features)
        x = self.input_proj(x)  # Project to adjusted dimension
        x = self.xlstm(x)  # (batch, timesteps, adjusted_features)
        x = x[:, -1, :]  # Take last timestep: (batch, adjusted_features)
        x = self.output_proj(x)  # (batch, output_dim)
        return x


class xLSTMRegressorUnified(nn.Module):
    """
    Unified xLSTM Regressor with consistent interface.
    
    This wrapper provides the same interface as other regressors
    (input_dim, output_dim + config params) and internally creates
    the xLSTMBlockStackConfig.
    
    Used by both train_multioutput_xlstm and MoE experts.
    
    Args:
        input_dim: Total input dimension (will be reshaped to sequence)
        output_dim: Number of forecast steps (horizon)
        timesteps: Number of timesteps for sequence reshaping
        num_blocks: Number of mLSTM blocks
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for xLSTM (embedding_dim)
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 timesteps: int = 4, num_blocks: int = 3, 
                 num_heads: int = 4, hidden_dim: int = 128,
                 dropout: float = 0.2, **kwargs):
        super(xLSTMRegressorUnified, self).__init__()
        
        if not XLSTM_AVAILABLE:
            raise ImportError("xLSTM not installed. Install with: pip install xlstm")
        
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Calculate features per timestep
        self.features_per_step = input_dim // timesteps if input_dim >= timesteps else input_dim
        
        # Use explicit hidden_dim that is guaranteed to be divisible by num_heads
        # This is the embedding dimension for xLSTM - must be multiple of num_heads
        self.adjusted_features = hidden_dim
        # Ensure divisibility by num_heads
        if self.adjusted_features % num_heads != 0:
            self.adjusted_features = ((self.adjusted_features // num_heads) + 1) * num_heads
        
        # Calculate qkv_proj_blocksize - must divide adjusted_features evenly
        # Common values: 4, 8, 16, 32
        self.qkv_blocksize = 4
        while self.adjusted_features % self.qkv_blocksize != 0 and self.qkv_blocksize > 1:
            self.qkv_blocksize -= 1
        
        # Input projection from features_per_step to adjusted_features
        self.input_proj = nn.Linear(self.features_per_step, self.adjusted_features)
        
        # Configure xLSTM with validated dimensions
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=min(4, timesteps),
                    qkv_proj_blocksize=self.qkv_blocksize,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            ),
            context_length=timesteps,
            num_blocks=num_blocks,
            embedding_dim=self.adjusted_features,
        )
        
        self.xlstm = xLSTMBlockStack(xlstm_config)
        
        # Output projection with larger hidden layer for better capacity
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.adjusted_features, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to sequence: (batch, features) -> (batch, timesteps, features_per_step)
        if self.input_dim >= self.timesteps:
            used_features = self.timesteps * self.features_per_step
            x_seq = x[:, :used_features].reshape(batch_size, self.timesteps, self.features_per_step)
        else:
            x_seq = x.unsqueeze(1).repeat(1, self.timesteps, 1)
        
        # Project to adjusted dimension
        x_seq = self.input_proj(x_seq)
        
        # xLSTM forward
        x_seq = self.xlstm(x_seq)
        
        # Take last timestep
        x_last = x_seq[:, -1, :]
        
        return self.output_proj(x_last)
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features) or (N, timesteps, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            # Flatten 3D to 2D if needed (reshaping handled in forward)
            if X_t.dim() == 3:
                X_t = X_t.reshape(X_t.size(0), -1)
            output = self.forward(X_t)
            return output.cpu().numpy()


# ============================================================================
# 3. TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================

class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):
    """
    Temporal Convolutional Network for multioutput forecasting.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps
        num_channels: List of channel sizes per layer
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim: int, output_dim: int = 10, 
                 num_channels: List[int] = None, kernel_size: int = 3, dropout: float = 0.2):
        super(TCN, self).__init__()
        
        if num_channels is None:
            num_channels = [64, 64, 32]
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, 
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) or (batch, features) for single timestep
        
        Returns:
            (batch, output_dim)
        """
        if x.dim() == 2:
            # Single timestep: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        y = self.network(x)
        
        # Take last timestep
        y = y[:, :, -1]
        
        return self.linear(y)
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            output = self.forward(X_t)
            return output.cpu().numpy()


# ============================================================================
# 4. N-BEATS LITE (Interpretable Forecast)
# ============================================================================

class NBeatsBlock(nn.Module):
    """Single block for N-BEATS."""
    
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int, num_layers: int = 4):
        super(NBeatsBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_dim, theta_dim)
    
    def forward(self, x):
        h = self.layers(x)
        theta = self.theta(h)
        return theta


class NBeatsLite(nn.Module):
    """
    Simplified N-BEATS for multioutput forecasting.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps
        num_blocks: Number of stacks/blocks
        hidden_dim: Hidden layer size
    """
    
    def __init__(self, input_dim: int, output_dim: int = 10, 
                 num_blocks: int = 2, hidden_dim: int = 64):
        super(NBeatsLite, self).__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, output_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch, features)
        
        Returns:
            (batch, output_dim)
        """
        forecast = 0
        for block in self.blocks:
            theta = block(x)
            forecast = forecast + theta
        
        return forecast
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            output = self.forward(X_t)
            return output.cpu().numpy()


# ============================================================================
# 5. UNIFIED MODEL ARCHITECTURES
# ============================================================================
# These are the CANONICAL implementations of all model architectures.
# Used by BOTH:
#   - Standalone train_* methods in data_science.py
#   - MoE (Mixture of Experts) as expert modules
#
# Available models:
#   - LinearRegressor: Simple linear regression (PyTorch)
#   - DNNRegressor: Deep NN with residual connections + SE attention
#   - LSTMRegressor: Bidirectional LSTM with multi-head attention
#   - xLSTMRegressor: Extended LSTM with matrix memory (already defined above)
#   - TCN: Temporal Convolutional Network (already defined above)
#   - NBeatsLite: N-BEATS interpretable forecast (already defined above)
# ============================================================================

class LinearRegressor(nn.Module):
    """
    Simple Linear Regressor for multi-output forecasting.
    
    This is the PyTorch equivalent of sklearn LinearRegression,
    used for both standalone training and as an MoE expert.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps (horizon)
        use_bias: Whether to include bias term
    """
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, **kwargs):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
    
    def forward(self, x):
        return self.linear(x)
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            output = self.forward(X_t)
            return output.cpu().numpy()


class DNNRegressor(nn.Module):
    """
    Deep Neural Network Regressor with residual connections and SE attention.
    
    This is the CANONICAL DNN implementation used by both:
    - train_multioutput_dnn() in data_science.py
    - MoE as a DNN expert
    
    Architecture:
    - Initial projection layer
    - 3 Residual blocks with LayerNorm and dropout
    - Squeeze-and-Excitation (SE) attention for feature weighting
    - Multi-head output (short/medium/long term)
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps (horizon)
        layers: Hidden layer sizes [default: [256, 128, 64]]
        dropout: Dropout rates per layer [default: [0.25, 0.2, 0.15]]
        activation: Activation function ('gelu', 'relu', 'swish')
        l2_reg: L2 regularization (for reference, not directly used in PyTorch)
        use_layer_norm: Whether to use LayerNorm
        use_residual: Whether to use residual connections
        use_se_attention: Whether to use SE attention
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 layers: List[int] = None, dropout: List[float] = None,
                 activation: str = 'gelu', l2_reg: float = 0.0001,
                 use_layer_norm: bool = True, use_residual: bool = True,
                 use_se_attention: bool = True, **kwargs):
        super(DNNRegressor, self).__init__()
        
        if layers is None:
            layers = [256, 128, 64]
        if dropout is None:
            dropout = [0.25, 0.2, 0.15]
        
        # Ensure we have 3 layers
        while len(layers) < 3:
            layers.append(64)
        while len(dropout) < 3:
            dropout.append(0.15)
        
        self.use_residual = use_residual
        self.use_se_attention = use_se_attention
        self.use_layer_norm = use_layer_norm
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, layers[0])
        self.norm0 = nn.LayerNorm(layers[0]) if use_layer_norm else nn.Identity()
        
        # Residual Block 1
        self.block1 = nn.Sequential(
            nn.Linear(layers[0], layers[0]),
            nn.LayerNorm(layers[0]) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout[0]),
            nn.Linear(layers[0], layers[0]),
            nn.LayerNorm(layers[0]) if use_layer_norm else nn.Identity(),
        )
        
        # SE Attention
        if use_se_attention:
            self.se = nn.Sequential(
                nn.Linear(layers[0], 32),
                nn.ReLU(),
                nn.Linear(32, layers[0]),
                nn.Sigmoid()
            )
        
        # Residual Block 2
        self.proj1 = nn.Linear(layers[0], layers[1])
        self.block2 = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.LayerNorm(layers[1]) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout[1]),
            nn.Linear(layers[1], layers[1]),
            nn.LayerNorm(layers[1]) if use_layer_norm else nn.Identity(),
        )
        
        # Residual Block 3
        self.proj2 = nn.Linear(layers[1], layers[2])
        self.block3 = nn.Sequential(
            nn.Linear(layers[1], layers[2]),
            nn.LayerNorm(layers[2]) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout[2]),
            nn.Linear(layers[2], layers[2]),
            nn.LayerNorm(layers[2]) if use_layer_norm else nn.Identity(),
        )
        
        # Multi-head output (short/medium/long term)
        self.shared = nn.Sequential(
            nn.Linear(layers[2], output_dim * 2),
            self.activation,
            nn.Dropout(0.1)
        )
        
        # Split horizon into short/medium/long term
        short_term = output_dim // 3
        medium_term = output_dim // 3
        long_term = output_dim - short_term - medium_term
        
        self.short_head = nn.Sequential(nn.Linear(output_dim * 2, 64), self.activation, nn.Linear(64, short_term))
        self.medium_head = nn.Sequential(nn.Linear(output_dim * 2, 64), self.activation, nn.Linear(64, medium_term))
        self.long_head = nn.Sequential(nn.Linear(output_dim * 2, 64), self.activation, nn.Linear(64, long_term))
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        x = self.norm0(x)
        x = self.activation(x)
        
        # Block 1 with residual
        residual = x
        x = self.block1(x)
        if self.use_residual:
            x = x + residual
        x = self.activation(x)
        
        # SE attention
        if self.use_se_attention:
            se_weights = self.se(x)
            x = x * se_weights
        
        # Block 2 with residual
        residual = self.proj1(x)
        x = self.block2(x)
        if self.use_residual:
            x = x + residual
        x = self.activation(x)
        
        # Block 3 with residual
        residual = self.proj2(x)
        x = self.block3(x)
        if self.use_residual:
            x = x + residual
        x = self.activation(x)
        
        # Multi-head output
        shared = self.shared(x)
        short_out = self.short_head(shared)
        medium_out = self.medium_head(shared)
        long_out = self.long_head(shared)
        
        return torch.cat([short_out, medium_out, long_out], dim=1)
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            output = self.forward(X_t)
            return output.cpu().numpy()


class LSTMRegressor(nn.Module):
    """
    Bidirectional LSTM Regressor with multi-head attention.
    
    This is the CANONICAL LSTM implementation used by both:
    - train_multioutput_lstm() in data_science.py
    - MoE as an LSTM expert
    
    Architecture:
    - 2 Bidirectional LSTM layers
    - Multi-head self-attention
    - Final LSTM for aggregation
    - Dense layers with BatchNorm
    - Multi-output prediction head
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps (horizon)
        timesteps: Number of timesteps for sequence reshaping
        lstm_units: LSTM hidden sizes per layer [default: [128, 64, 64]]
        lstm_dropout: Dropout per LSTM layer [default: [0.2, 0.15, 0.1]]
        num_heads: Number of attention heads
        key_dim: Attention key dimension
        dense_layers: Dense layer sizes [default: [128, 64]]
        dense_dropout: Dense layer dropout [default: [0.25, 0.2]]
        bidirectional: Whether to use bidirectional LSTM
        use_layer_norm: Whether to use LayerNorm
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 timesteps: int = 4, lstm_units: List[int] = None,
                 lstm_dropout: List[float] = None, 
                 num_heads: int = 4, key_dim: int = 32,
                 dense_layers: List[int] = None, dense_dropout: List[float] = None,
                 bidirectional: bool = True, use_layer_norm: bool = True, **kwargs):
        super(LSTMRegressor, self).__init__()
        
        if lstm_units is None:
            lstm_units = [128, 64, 64]
        if lstm_dropout is None:
            lstm_dropout = [0.2, 0.15, 0.1]
        if dense_layers is None:
            dense_layers = [128, 64]
        if dense_dropout is None:
            dense_dropout = [0.25, 0.2]
        
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.bidirectional = bidirectional
        
        # Calculate features per timestep
        self.features_per_step = input_dim // timesteps if input_dim >= timesteps else input_dim
        
        # LSTM layers
        lstm1_output = lstm_units[0] * 2 if bidirectional else lstm_units[0]
        self.lstm1 = nn.LSTM(self.features_per_step, lstm_units[0], batch_first=True,
                             dropout=lstm_dropout[0] if len(lstm_units) > 1 else 0, 
                             bidirectional=bidirectional)
        self.norm1 = nn.LayerNorm(lstm1_output) if use_layer_norm else nn.Identity()
        
        lstm2_output = lstm_units[1] * 2 if bidirectional else lstm_units[1]
        self.lstm2 = nn.LSTM(lstm1_output, lstm_units[1], batch_first=True,
                             dropout=lstm_dropout[1] if len(lstm_units) > 2 else 0, 
                             bidirectional=bidirectional)
        self.norm2 = nn.LayerNorm(lstm2_output) if use_layer_norm else nn.Identity()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(lstm2_output, num_heads, dropout=0.1, batch_first=True)
        self.norm_attn = nn.LayerNorm(lstm2_output) if use_layer_norm else nn.Identity()
        
        # Final LSTM to aggregate
        self.lstm3 = nn.LSTM(lstm2_output, lstm_units[2], batch_first=True, dropout=lstm_dropout[2])
        
        # Dense layers
        self.dense1 = nn.Sequential(
            nn.Linear(lstm_units[2], dense_layers[0]),
            nn.BatchNorm1d(dense_layers[0]),
            nn.ReLU(),
            nn.Dropout(dense_dropout[0])
        )
        self.dense2 = nn.Sequential(
            nn.Linear(dense_layers[0], dense_layers[1]),
            nn.BatchNorm1d(dense_layers[1]),
            nn.ReLU(),
            nn.Dropout(dense_dropout[1])
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(dense_layers[1], output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to sequence
        if self.input_dim >= self.timesteps:
            used_features = self.timesteps * self.features_per_step
            x_seq = x[:, :used_features].reshape(batch_size, self.timesteps, self.features_per_step)
        else:
            x_seq = x.unsqueeze(1).repeat(1, self.timesteps, 1)
        
        # LSTM 1
        x, _ = self.lstm1(x_seq)
        x = self.norm1(x)
        
        # LSTM 2
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm_attn(x)
        
        # LSTM 3
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.output(x)
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features) or (N, timesteps, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            # Flatten 3D to 2D if needed (LSTM reshapes internally)
            if X_t.dim() == 3:
                X_t = X_t.reshape(X_t.size(0), -1)
            output = self.forward(X_t)
            return output.cpu().numpy()


# xLSTMRegressor is already defined above (line ~270)
# TCN is already defined above (line ~310)
# NBeatsLite is already defined above (line ~380)


# ============================================================================
# 6. MODEL REGISTRY - Factory for creating models by name
# ============================================================================

MODEL_REGISTRY = {
    'linear': {
        'class': LinearRegressor,
        'default_config': {'use_bias': True},
        'config_path': 'regressor_linear',
        'description': 'Simple linear regression'
    },
    'dnn': {
        'class': DNNRegressor,
        'default_config': {
            'layers': [512, 256, 128],       # Larger layers for more capacity
            'dropout': [0.3, 0.25, 0.2],     # More regularization
            'activation': 'gelu',
            'l2_reg': 0.0001,
            'use_layer_norm': True,
            'use_residual': True,
            'use_se_attention': True
        },
        'config_path': 'regressor_dnn',
        'description': 'Deep NN with residual + SE attention'
    },
    'lstm': {
        'class': LSTMRegressor,
        'default_config': {
            'timesteps': 8,                    # Longer context
            'lstm_units': [256, 128, 64],      # Larger LSTM layers
            'lstm_dropout': [0.25, 0.2, 0.15],
            'num_heads': 8,                    # More attention heads
            'key_dim': 32,
            'dense_layers': [256, 128],        # Larger dense layers
            'dense_dropout': [0.3, 0.25],
            'bidirectional': True,
            'use_layer_norm': True
        },
        'config_path': 'regressor_lstm',
        'description': 'Bidirectional LSTM with multi-head attention'
    },
    'tcn': {
        'class': TCN,
        'default_config': {
            'num_channels': [128, 128, 64, 64],  # Deeper network
            'kernel_size': 5,                     # Larger kernel
            'dropout': 0.2
        },
        'config_path': 'tcn',
        'description': 'Temporal Convolutional Network'
    },
    'nbeats': {
        'class': NBeatsLite,
        'default_config': {
            'num_blocks': 4,       # More blocks for capacity
            'hidden_dim': 128      # Larger hidden dimension
        },
        'config_path': 'nbeats',
        'description': 'N-BEATS interpretable forecast'
    },
}

# Add xLSTM if available
if XLSTM_AVAILABLE:
    MODEL_REGISTRY['xlstm'] = {
        'class': xLSTMRegressorUnified,
        'default_config': {
            'timesteps': 4,       # Sequence length
            'num_blocks': 3,      # mLSTM layers (increased for capacity)
            'num_heads': 4,       # Attention heads
            'hidden_dim': 128,    # Embedding dimension (must be divisible by num_heads)
            'dropout': 0.15
        },
        'config_path': 'regressor_xlstm',
        'description': 'Extended LSTM with matrix memory (mLSTM)'
    }


def create_model(model_type: str, input_dim: int, output_dim: int, 
                 config: Dict = None, hyperparameters: Dict = None,
                 model_zoo_config: Dict = None) -> nn.Module:
    """
    Factory function to create a model by name.
    
    This is the UNIFIED way to create models, used by both:
    - train_* methods in data_science.py
    - MoE for creating experts
    
    Args:
        model_type: Model type ('linear', 'dnn', 'lstm', 'xlstm', 'tcn', 'nbeats')
        input_dim: Number of input features
        output_dim: Number of forecast steps
        config: Explicit config overrides (highest priority)
        hyperparameters: Config from model_hyperparameters section
        model_zoo_config: Config from model_zoo section
    
    Returns:
        PyTorch nn.Module instance
    
    Example:
        model = create_model('dnn', input_dim=100, output_dim=10,
                            hyperparameters=config.get('model_hyperparameters', {}))
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    model_info = MODEL_REGISTRY[model_type]
    model_class = model_info['class']
    config_path = model_info['config_path']
    
    # Build config: defaults -> YAML config -> explicit overrides
    model_config = {**model_info['default_config']}
    
    if hyperparameters is None:
        hyperparameters = {}
    if model_zoo_config is None:
        model_zoo_config = {}
    if config is None:
        config = {}
    
    # Check model_hyperparameters for core models
    if config_path in hyperparameters:
        yaml_config = hyperparameters[config_path]
        for key, value in yaml_config.items():
            if key in model_config:
                model_config[key] = value
    
    # Check model_zoo for model_zoo models
    if model_type in model_zoo_config:
        zoo_params = model_zoo_config[model_type].get('params', {})
        for key, value in zoo_params.items():
            if key in model_config:
                model_config[key] = value
    
    # Apply explicit overrides
    model_config.update(config)
    
    # Create model
    return model_class(input_dim=input_dim, output_dim=output_dim, **model_config)


def get_available_models() -> List[str]:
    """Return list of all available model types."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_type: str) -> Dict:
    """Get information about a model type."""
    if model_type not in MODEL_REGISTRY:
        return None
    info = MODEL_REGISTRY[model_type].copy()
    info['name'] = model_type
    return info


# ============================================================================
# 7. MIXTURE OF EXPERTS (MoE) - Uses unified MODEL_REGISTRY
# ============================================================================

class RegimeGating(nn.Module):
    """
    Gating network for regime-aware MoE.
    Uses a deeper network for better regime detection.
    """
    
    def __init__(self, input_dim: int, num_experts: int = 2, hidden_dim: int = 64):
        super(RegimeGating, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """Return expert weights (batch, num_experts)."""
        return self.network(x)


class MixtureOfExperts(nn.Module):
    """
    Extensible Mixture of Experts with regime-aware gating.
    
    Uses the UNIFIED MODEL_REGISTRY - same model classes used by 
    standalone train_* methods in data_science.py.
    
    Supports all model architectures:
    - linear: Simple linear regression
    - dnn: Deep NN with residual connections + SE attention
    - lstm: Bidirectional LSTM with multi-head attention
    - xlstm: Extended LSTM with matrix memory (if installed)
    - tcn: Temporal Convolutional Network
    - nbeats: N-BEATS interpretable forecast
    
    Each expert uses hyperparameters from configuration.yml, matching the
    standalone model implementations.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of forecast steps (horizon)
        expert_types: List of expert types to use (default: ['dnn', 'tcn'])
        expert_configs: Dict of expert-specific config overrides (optional)
        hyperparameters: Full hyperparameters dict from config (optional)
        model_zoo_config: Config from model_zoo section (optional)
        gating_hidden_dim: Hidden dimension for gating network
        
    Example:
        # Use 3 experts: DNN, TCN, and xLSTM with config from YAML
        model = MixtureOfExperts(
            input_dim=100,
            output_dim=10,
            expert_types=['dnn', 'tcn', 'xlstm'],
            hyperparameters=config.get('model_hyperparameters', {})
        )
    """
    
    def __init__(self, input_dim: int, output_dim: int = 10,
                 expert_types: List[str] = None, expert_configs: Dict = None,
                 hyperparameters: Dict = None, model_zoo_config: Dict = None,
                 gating_hidden_dim: int = 64, num_experts: int = None):
        super(MixtureOfExperts, self).__init__()
        
        # Default experts if not specified
        if expert_types is None:
            expert_types = ['dnn', 'tcn']
        
        # Filter to only available experts using MODEL_REGISTRY
        available_experts = []
        for expert_type in expert_types:
            if expert_type in MODEL_REGISTRY:
                available_experts.append(expert_type)
            else:
                logger.warning(f"Expert type '{expert_type}' not available, skipping")
        
        if not available_experts:
            raise ValueError("No valid experts specified. Available: " + ", ".join(MODEL_REGISTRY.keys()))
        
        self.expert_types = available_experts
        self.num_experts = len(available_experts)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if expert_configs is None:
            expert_configs = {}
        if hyperparameters is None:
            hyperparameters = {}
        if model_zoo_config is None:
            model_zoo_config = {}
        
        # Build experts using the unified create_model factory
        self.experts = nn.ModuleList()
        for expert_type in self.expert_types:
            # Get explicit overrides for this expert
            config_override = expert_configs.get(expert_type, {})
            
            # Create expert using unified factory
            try:
                expert = create_model(
                    model_type=expert_type,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    config=config_override,
                    hyperparameters=hyperparameters,
                    model_zoo_config=model_zoo_config
                )
                self.experts.append(expert)
                logger.info(f"  MoE: Added {expert_type.upper()} expert")
            except Exception as e:
                logger.warning(f"  MoE: Failed to create {expert_type} expert: {e}")
                self.expert_types.remove(expert_type)
        
        # Update num_experts after potential failures
        self.num_experts = len(self.experts)
        
        if self.num_experts == 0:
            raise ValueError("No experts could be created")
        
        # Gating network
        self.gating = RegimeGating(input_dim, self.num_experts, gating_hidden_dim)
        
        # Track gate weights for analysis
        self.gate_weights_history = []
        
        logger.info(f"  MoE initialized with {self.num_experts} experts: {self.expert_types}")
    
    def forward(self, x, return_gates=False):
        """
        Forward pass through all experts with gated combination.
        
        Args:
            x: (batch, features)
            return_gates: If True, also return gate weights
        
        Returns:
            predictions (batch, output_dim) [and optionally gate weights]
        """
        # Get predictions from all experts
        expert_preds = []
        for expert in self.experts:
            pred = expert(x)
            expert_preds.append(pred)
        
        # Stack predictions: (batch, num_experts, output_dim)
        predictions = torch.stack(expert_preds, dim=1)
        
        # Get gate weights: (batch, num_experts)
        gates = self.gating(x)
        
        # Weighted combination: (batch, output_dim)
        output = torch.sum(predictions * gates.unsqueeze(2), dim=1)
        
        if return_gates:
            return output, gates
        return output
    
    def get_gate_weights(self, x):
        """Get gate weights for input without making predictions."""
        with torch.no_grad():
            return self.gating(x)
    
    def get_expert_predictions(self, x):
        """Get individual predictions from each expert."""
        with torch.no_grad():
            preds = {}
            for i, (expert_type, expert) in enumerate(zip(self.expert_types, self.experts)):
                preds[expert_type] = expert(x).cpu().numpy()
        return preds
    
    def get_average_gate_weights(self):
        """Get average gate weights across all predictions."""
        if not self.gate_weights_history:
            return None
        return np.mean(self.gate_weights_history, axis=0)
    
    def get_expert_names(self) -> List[str]:
        """Return list of expert names in order."""
        return self.expert_types.copy()
    
    @classmethod
    def get_available_experts(cls) -> List[str]:
        """Return list of all available expert types."""
        return list(MODEL_REGISTRY.keys())
    
    def predict(self, X, verbose=0):
        """
        Sklearn/Keras-style predict method for compatibility.
        
        Args:
            X: Input features as numpy array (N, features)
            verbose: Ignored, for API compatibility
        
        Returns:
            Predictions as numpy array (N, output_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(device)
            else:
                X_t = X.to(device)
            output = self.forward(X_t)
            return output.cpu().numpy()


# ============================================================================
# 8. MULTI-TASK MODEL (Regression + Classification)
# ============================================================================

class MultiTaskModel(nn.Module):
    """
    Multi-task model for joint regression and classification.
    
    Tasks:
    - Regression: Predict 10-day price forecast
    - Classification: Predict trend direction (UP/DOWN/SIDEWAYS)
    
    Shared backbone: TCN
    """
    
    def __init__(self, input_dim: int, output_dim: int = 10, num_classes: int = 3):
        super(MultiTaskModel, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Regression head
        self.regression_head = nn.Linear(64, output_dim)
        
        # Classification head
        self.classification_head = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, features)
        
        Returns:
            Tuple of (regression_output, classification_logits)
        """
        features = self.backbone(x)
        
        regression = self.regression_head(features)
        classification = self.classification_head(features)
        
        return regression, classification


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_pytorch_model(model, X_train, y_train, X_val=None, y_val=None,
                       epochs=50, batch_size=256, lr=0.001, 
                       task='regression', verbose=True, 
                       device: torch.device = None, use_gpu: bool = True,
                       patience: int = 15):
    """
    Train PyTorch model.
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        task: 'regression' or 'multitask'
        verbose: Logging verbosity
        device: Explicit torch.device (overrides use_gpu)
        use_gpu: Whether to use GPU if available
        patience: Early stopping patience
    
    Returns:
        Training history
    """
    if device is None:
        device = get_device(use_gpu)
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    
    # Early stopping state
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            if task == 'multitask':
                # Multi-task: y is (regression_target, class_label)
                reg_pred, cls_pred = model(X_batch)
                loss = mse_loss(reg_pred, y_batch[:, :10]) + ce_loss(cls_pred, y_batch[:, 10].long())
            else:
                # Regression only
                pred = model(X_batch)
                loss = mse_loss(pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(dataloader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                if task == 'multitask':
                    reg_pred, cls_pred = model(X_val_t)
                    val_loss = (mse_loss(reg_pred, y_val_t[:, :10]) + 
                               ce_loss(cls_pred, y_val_t[:, 10].long())).item()
                else:
                    val_pred = model(X_val_t)
                    val_loss = mse_loss(val_pred, y_val_t).item()
                
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Log every epoch for verbose training visibility
        if verbose:
            val_str = f", Val Loss: {val_loss:.6f}" if X_val is not None else ""
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}{val_str}")
    
    # Restore best model if we have validation data
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history
