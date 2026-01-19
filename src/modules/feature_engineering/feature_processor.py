"""
Feature Processor: Advanced feature compression and embedding with strict leakage prevention.

Supports:
- PCA compression (fit on train only)
- Denoising Autoencoder embeddings (fit on train only)
- Feature Gating (learnable weights, L1 regularization)
- Multiple processing modes: raw, raw+pca, raw+ae, raw+pca+ae

CRITICAL: All fitting happens ONLY on training data. Holdout/future use transform only.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder for feature compression."""
    
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super(DenoisingAutoencoder, self).__init__()
        
        hidden_dim = max(latent_dim * 2, input_dim // 2)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, noise_std: float = 0.0):
        """Forward pass with optional noise."""
        if noise_std > 0 and self.training:
            x = x + torch.randn_like(x) * noise_std
        
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """Encode to latent space."""
        with torch.no_grad():
            return self.encoder(x)


class FeatureGatingLayer(nn.Module):
    """Learnable feature gating with L1 regularization."""
    
    def __init__(self, n_features: int):
        super(FeatureGatingLayer, self).__init__()
        self.gates = nn.Parameter(torch.ones(n_features))
    
    def forward(self, x):
        """Apply learned gates to features."""
        return x * torch.sigmoid(self.gates)
    
    def get_gate_weights(self):
        """Get gate weights (0-1 range after sigmoid)."""
        return torch.sigmoid(self.gates).detach().cpu().numpy()
    
    def l1_loss(self):
        """L1 regularization loss."""
        return torch.abs(self.gates).sum()


class FeatureProcessor:
    """
    Process features with PCA, Autoencoder, or combined modes.
    
    CRITICAL: All fitting (PCA, AE, Scaler) happens ONLY on training data.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize feature processor.
        
        Args:
            config: Configuration dict with keys:
                - mode: "raw", "raw+pca", "raw+ae", "raw+pca+ae"
                - pca: {enabled, n_components}
                - autoencoder: {enabled, latent_dim, noise_std, epochs}
                - gating: {enabled, l1_lambda}
        """
        self.config = config or {}
        self.mode = self.config.get('mode', 'raw')
        
        # PCA config
        pca_config = self.config.get('pca', {})
        self.pca_enabled = pca_config.get('enabled', False)
        self.pca_n_components = pca_config.get('n_components', 20)
        
        # Autoencoder config
        ae_config = self.config.get('autoencoder', {})
        self.ae_enabled = ae_config.get('enabled', False)
        self.ae_latent_dim = ae_config.get('latent_dim', 16)
        self.ae_noise_std = ae_config.get('noise_std', 0.01)
        self.ae_epochs = ae_config.get('epochs', 50)
        self.ae_batch_size = ae_config.get('batch_size', 256)
        
        # Gating config
        gating_config = self.config.get('gating', {})
        self.gating_enabled = gating_config.get('enabled', False)
        self.gating_l1_lambda = gating_config.get('l1_lambda', 1e-4)
        
        # Fitted objects
        self.scaler = None
        self.pca = None
        self.autoencoder = None
        self.gating_layer = None
        self.fitted = False
        
        # Metadata
        self.input_dim = None
        self.output_dim = None
        self.pca_explained_variance = None
        self.ae_reconstruction_loss = None
        self.gating_weights = None
        
        logger.info(f"FeatureProcessor initialized with mode: {self.mode}")
        logger.info(f"  PCA: {self.pca_enabled} (n_components={self.pca_n_components})")
        logger.info(f"  Autoencoder: {self.ae_enabled} (latent_dim={self.ae_latent_dim})")
        logger.info(f"  Gating: {self.gating_enabled}")
    
    def fit(self, X_train: np.ndarray, verbose: bool = True) -> 'FeatureProcessor':
        """
        Fit processor on training data ONLY.
        
        Args:
            X_train: Training features (2D array)
            verbose: Logging verbosity
        
        Returns:
            self
        """
        if verbose:
            logger.info("=" * 80)
            logger.info("FITTING FEATURE PROCESSOR (TRAINING DATA ONLY)")
            logger.info("=" * 80)
        
        self.input_dim = X_train.shape[1]
        
        # 1. Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        if verbose:
            logger.info(f"Scaled {X_train.shape[0]} training samples, {self.input_dim} features")
        
        # 2. PCA compression (if enabled)
        if self.pca_enabled or 'pca' in self.mode:
            n_components = min(self.pca_n_components, X_scaled.shape[0], X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X_scaled)
            
            self.pca_explained_variance = np.sum(self.pca.explained_variance_ratio_)
            
            if verbose:
                logger.info(f"PCA: {n_components} components, explained variance: {self.pca_explained_variance:.2%}")
        
        # 3. Autoencoder embeddings (if enabled)
        if self.ae_enabled or 'ae' in self.mode:
            if verbose:
                logger.info(f"Training Denoising Autoencoder (latent_dim={self.ae_latent_dim}, epochs={self.ae_epochs})...")
            
            self.autoencoder = DenoisingAutoencoder(
                input_dim=X_scaled.shape[1],
                latent_dim=self.ae_latent_dim,
                dropout=0.1
            )
            
            # Train autoencoder
            self._train_autoencoder(X_scaled, verbose=verbose)
        
        # 4. Feature gating (will be trained later with model if enabled)
        if self.gating_enabled:
            self.gating_layer = FeatureGatingLayer(self.input_dim)
            if verbose:
                logger.info(f"Feature gating layer initialized ({self.input_dim} gates)")
        
        # Calculate output dimension
        self.output_dim = self._calculate_output_dim()
        
        self.fitted = True
        
        if verbose:
            logger.info(f"Feature processor fitted: {self.input_dim} → {self.output_dim} features")
            logger.info("=" * 80)
        
        return self
    
    def transform(self, X: np.ndarray, return_dict: bool = False) -> np.ndarray:
        """
        Transform features using fitted processor.
        
        Args:
            X: Input features (2D array)
            return_dict: If True, return dict with all feature types
        
        Returns:
            Transformed features (2D array) or dict
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor not fitted. Call fit() first.")
        
        # 1. Scale
        X_scaled = self.scaler.transform(X)
        
        features = {}
        
        # 2. Raw features (always included)
        if self.mode == 'raw' or self.mode.startswith('raw+'):
            features['raw'] = X_scaled
        
        # 3. PCA features
        if self.pca is not None and ('pca' in self.mode or self.pca_enabled):
            features['pca'] = self.pca.transform(X_scaled)
        
        # 4. Autoencoder latents
        if self.autoencoder is not None and ('ae' in self.mode or self.ae_enabled):
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                self.autoencoder.eval()
                latents = self.autoencoder.encode(X_tensor).numpy()
            features['ae'] = latents
        
        if return_dict:
            return features
        
        # Concatenate all feature types
        feature_list = []
        for key in sorted(features.keys()):
            feature_list.append(features[key])
        
        X_processed = np.hstack(feature_list) if len(feature_list) > 1 else feature_list[0]
        
        return X_processed
    
    def _train_autoencoder(self, X_train: np.ndarray, verbose: bool = True):
        """Train denoising autoencoder on training data."""
        X_tensor = torch.FloatTensor(X_train)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.ae_batch_size, 
            shuffle=True
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        self.autoencoder.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0
            for batch in dataloader:
                X_batch = batch[0]
                
                optimizer.zero_grad()
                reconstructed, latent = self.autoencoder(X_batch, noise_std=self.ae_noise_std)
                loss = criterion(reconstructed, X_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch [{epoch+1}/{self.ae_epochs}], Loss: {avg_loss:.6f}")
        
        self.ae_reconstruction_loss = avg_loss
        
        if verbose:
            logger.info(f"Autoencoder training complete. Final loss: {avg_loss:.6f}")
    
    def _calculate_output_dim(self) -> int:
        """Calculate output feature dimension based on mode."""
        dim = 0
        
        if self.mode == 'raw' or self.mode.startswith('raw+'):
            dim += self.input_dim
        
        if self.pca is not None and ('pca' in self.mode or self.pca_enabled):
            dim += self.pca.n_components_
        
        if self.autoencoder is not None and ('ae' in self.mode or self.ae_enabled):
            dim += self.ae_latent_dim
        
        return dim if dim > 0 else self.input_dim
    
    def get_summary(self) -> Dict:
        """Get summary of feature processing."""
        summary = {
            'mode': self.mode,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'fitted': self.fitted
        }
        
        if self.pca is not None:
            summary['pca'] = {
                'n_components': self.pca.n_components_,
                'explained_variance': float(self.pca_explained_variance) if self.pca_explained_variance else None
            }
        
        if self.autoencoder is not None:
            summary['autoencoder'] = {
                'latent_dim': self.ae_latent_dim,
                'reconstruction_loss': float(self.ae_reconstruction_loss) if self.ae_reconstruction_loss else None
            }
        
        if self.gating_layer is not None:
            summary['gating'] = {
                'enabled': True,
                'weights': self.gating_weights.tolist() if self.gating_weights is not None else None
            }
        
        return summary
    
    def get_metadata(self) -> Dict:
        """
        Get metadata for reporting in data_science module.
        
        Returns:
            Dict with keys like 'mode', 'pca_variance', 'ae_loss', etc.
        """
        metadata = {
            'mode': self.mode,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'pca_enabled': self.pca_enabled,
            'ae_enabled': self.ae_enabled,
            'gating_enabled': self.gating_enabled,
        }
        
        # PCA metadata
        if self.pca is not None:
            metadata['pca_n_components'] = self.pca.n_components_
            metadata['pca_variance'] = float(self.pca_explained_variance * 100) if self.pca_explained_variance else None
        
        # Autoencoder metadata
        if self.autoencoder is not None:
            metadata['ae_latent_dim'] = self.ae_latent_dim
            metadata['ae_loss'] = float(self.ae_reconstruction_loss) if self.ae_reconstruction_loss else None
        
        # Gating metadata
        if self.gating_layer is not None:
            metadata['gating_weights'] = self.gating_weights.tolist() if self.gating_weights is not None else None
        
        return metadata
    
    def save(self, filepath: str):
        """Save processor state."""
        import pickle
        state = {
            'config': self.config,
            'scaler': self.scaler,
            'pca': self.pca,
            'autoencoder': self.autoencoder.state_dict() if self.autoencoder else None,
            'gating_layer': self.gating_layer.state_dict() if self.gating_layer else None,
            'metadata': self.get_summary()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Feature processor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureProcessor':
        """Load processor state."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        processor = cls(state['config'])
        processor.scaler = state['scaler']
        processor.pca = state['pca']
        
        if state['autoencoder']:
            metadata = state['metadata']
            processor.autoencoder = DenoisingAutoencoder(
                input_dim=metadata['input_dim'],
                latent_dim=metadata['autoencoder']['latent_dim']
            )
            processor.autoencoder.load_state_dict(state['autoencoder'])
        
        if state['gating_layer']:
            processor.gating_layer = FeatureGatingLayer(metadata['input_dim'])
            processor.gating_layer.load_state_dict(state['gating_layer'])
        
        processor.fitted = True
        logger.info(f"Feature processor loaded from {filepath}")
        
        return processor


def create_feature_processor(config: dict = None) -> FeatureProcessor:
    """
    Convenience function to create feature processor.
    
    Args:
        config: Configuration dict
    
    Returns:
        FeatureProcessor instance
    """
    return FeatureProcessor(config)
