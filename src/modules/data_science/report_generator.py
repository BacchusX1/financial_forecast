"""
Advanced Report Generator for Data Science Models

Generates comprehensive HTML reports with:
- Confusion matrices and classification plots
- Regression performance metrics
- Training loss curves
- Forecast visualizations
- Model comparisons
"""

import os
import json
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
from typing import List, Dict
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ReportGenerator:
    """Generate comprehensive HTML reports for benchmark results."""
    
    TREND_LABELS = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
    TREND_COLORS = {0: '#FF6B6B', 1: '#808080', 2: '#51CF66'}
    
    def __init__(self, benchmark_results: Dict = None, config: Dict = None, output_dir: str = 'out'):
        """Initialize report generator."""
        # Handle case where first arg is output_dir (legacy support)
        if isinstance(benchmark_results, str):
            self.output_dir = benchmark_results
            self.benchmark_results = {}
            self.config = {}
        else:
            self.benchmark_results = benchmark_results or {}
            self.config = config or {}
            self.output_dir = output_dir
            
        self.figures = {}
    
    @staticmethod
    def figure_to_base64() -> str:
        """Convert current matplotlib figure to base64 string."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        return image_base64
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, class_names: list = None) -> str:
        """Generate confusion matrix heatmap."""
        if class_names is None:
            class_names = ['DOWN', 'SIDEWAYS', 'UP']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                   yticklabels=class_names, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_training_history(self, history, metric: str = 'loss', model_name: str = '') -> str:
        """Generate training history plot (loss/accuracy over epochs) with log scale for loss."""
        plt.figure(figsize=(12, 5))
        
        if hasattr(history, 'history'):  # Keras history
            train_metric = history.history.get(metric, [])
            val_metric = history.history.get(f'val_{metric}', [])
            
            # Only plot if we have training data
            if len(train_metric) > 0:
                epochs = range(1, len(train_metric) + 1)
                plt.plot(epochs, train_metric, 'b-', label=f'Training {metric}', linewidth=2)
                
                # Plot validation metric if available
                if len(val_metric) > 0:
                    plt.plot(epochs, val_metric, 'r-', label=f'Validation {metric}', linewidth=2)
                
                # Use logarithmic scale for loss metric
                if metric == 'loss':
                    plt.yscale('log')
                    plt.ylabel(f'{metric.capitalize()} (log scale)')
                else:
                    plt.ylabel(metric.capitalize())
        
        plt.xlabel('Epoch')
        plt.title(f'Training History - {model_name.upper()} ({metric})')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_regression_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str, metric_name: str = 'R²') -> str:
        """Generate regression scatter plot: actual vs predicted."""
        plt.figure(figsize=(8, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {model_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_timeseries_forecast(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str) -> str:
        """Generate time series forecast plot."""
        plt.figure(figsize=(14, 6))
        
        x = np.arange(len(y_true))
        
        plt.plot(x, y_true, 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
        plt.plot(x, y_pred, 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
        
        plt.xlabel('Time Step (Test Data)')
        plt.ylabel('Price')
        plt.title(f'Time Series Forecast - {model_name.upper()}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_classifier_comparison(self, results: dict) -> str:
        """Generate comparison bar chart for all classifiers."""
        plt.figure(figsize=(12, 6))
        
        names = list(results.keys())
        accuracies = [results[m]['accuracy'] * 100 for m in names]
        f1_scores = [results[m]['f1_score'] * 100 for m in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Score (%)')
        plt.title('Classifier Comparison - Accuracy vs F1-Score')
        plt.xticks(x, [n.upper() for n in names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_regressor_comparison(self, results: dict) -> str:
        """Generate comparison bar chart for all regressors."""
        plt.figure(figsize=(12, 6))
        
        # Sort by R² score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        names = [m[0] for m in sorted_results]
        r2_scores = [m[1]['r2'] for m in sorted_results]
        
        colors = ['#51CF66' if r2 > 0.8 else '#FFD93D' if r2 > 0 else '#FF6B6B' for r2 in r2_scores]
        
        plt.barh(names, r2_scores, color=colors, alpha=0.8)
        plt.xlabel('R² Score')
        plt.title('Regressor Comparison - R² Scores')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        """Generate residual plot."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {model_name.upper()}')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution - {model_name.upper()}')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        base64_img = self.figure_to_base64()
        return base64_img
    
    # =========================================================================
    # NEW ENHANCED VISUALIZATION METHODS
    # =========================================================================
    
    def plot_per_step_metrics(self, regressor_details: Dict, horizon: int = 10) -> str:
        """
        Generate per-step RMSE/MAE chart showing error accumulation over forecast horizon.
        
        Args:
            regressor_details: Dict with model_name -> {rmse_per_step, mae_per_step, etc.}
            horizon: Forecast horizon
            
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(14, 6))
        
        steps = np.arange(1, horizon + 1)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        
        for idx, (model_name, data) in enumerate(regressor_details.items()):
            rmse_per_step = data.get('rmse_per_step', [])
            if rmse_per_step and len(rmse_per_step) > 0:
                # Use available steps up to horizon
                n_points = min(len(rmse_per_step), horizon)
                data_to_plot = rmse_per_step[:n_points]
                current_steps = steps[:n_points]
                
                color = colors[idx % len(colors)]
                plt.plot(current_steps, data_to_plot, '-o', color=color, 
                        label=f'{model_name.upper()}', linewidth=2, markersize=6)
        
        plt.xlabel('Forecast Step', fontsize=12)
        plt.ylabel('RMSE ($)', fontsize=12)
        plt.title('Error Accumulation: RMSE by Forecast Step', fontsize=14, weight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, alpha=0.3)
        plt.xticks(steps)
        plt.tight_layout()
        
        return self.figure_to_base64()
    
    def plot_all_models_overlay(self, regressor_details: Dict, ticker: str = 'Asset') -> str:
        """
        Generate overlay plot showing all model predictions vs actual on the same chart.
        
        Args:
            regressor_details: Dict with model_name -> {y_val, y_pred}
            ticker: Asset name
            
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(18, 8))
        
        # Get actual values from first model
        actual_values = None
        for model_name, data in regressor_details.items():
            if 'y_val' in data:
                # For multi-step, take first step predictions
                y_val = np.array(data['y_val'])
                if y_val.ndim > 1:
                    actual_values = y_val[:, 0]  # First step actual
                else:
                    actual_values = y_val
                break
        
        if actual_values is None:
            plt.text(0.5, 0.5, 'No validation data available', ha='center', va='center')
            return self.figure_to_base64()
        
        x = np.arange(len(actual_values))
        
        # Plot actual values
        plt.plot(x, actual_values, 'k-', label='Actual', linewidth=3, alpha=0.8)
        
        # Plot each model's predictions
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        for idx, (model_name, data) in enumerate(regressor_details.items()):
            if 'y_pred' in data:
                y_pred = np.array(data['y_pred'])
                if y_pred.ndim > 1:
                    pred_values = y_pred[:, 0]  # First step predictions
                else:
                    pred_values = y_pred
                
                color = colors[idx % len(colors)]
                plt.plot(x, pred_values[:len(x)], '--', color=color, 
                        label=f'{model_name.upper()}', linewidth=1.5, alpha=0.8)
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'{ticker}: All Models vs Actual (Step 1 Predictions)', fontsize=14, weight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self.figure_to_base64()
    
    def plot_error_distribution_comparison(self, regressor_details: Dict) -> str:
        """
        Generate side-by-side error distribution plots for all models in LOG RETURN space.
        
        Args:
            regressor_details: Dict with model_name -> {y_val_transformed, y_pred_transformed}
            
        Returns:
            Base64 encoded image
        """
        n_models = len(regressor_details)
        if n_models == 0:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No models to display', ha='center', va='center')
            return self.figure_to_base64()
        
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4), squeeze=False)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c']
        
        # Get target transform type for labeling
        target_transform = 'log_return'
        for data in regressor_details.values():
            if 'target_transform' in data:
                target_transform = data['target_transform']
                break
        
        for idx, (model_name, data) in enumerate(regressor_details.items()):
            ax = axes[0, idx]
            
            # Prefer transformed space (log return), fall back to price space
            if 'y_val_transformed' in data and 'y_pred_transformed' in data:
                y_val = np.array(data['y_val_transformed']).flatten()
                y_pred = np.array(data['y_pred_transformed']).flatten()[:len(y_val)]
                space_label = target_transform.replace('_', ' ').title()
            elif 'y_val' in data and 'y_pred' in data:
                y_val = np.array(data['y_val']).flatten()
                y_pred = np.array(data['y_pred']).flatten()[:len(y_val)]
                space_label = 'Price'
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            errors = y_val - y_pred
            
            color = colors[idx % len(colors)]
            ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.4f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            ax.set_title(f'{model_name.upper()}\nσ={std_err:.4f}', fontsize=10, weight='bold')
            ax.set_xlabel(f'Prediction Error ({space_label})')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Error Distribution Comparison ({target_transform.replace("_", " ").title()} Space)', 
                    fontsize=12, weight='bold', y=1.02)
        plt.tight_layout()
        
        return self.figure_to_base64()
    
    def plot_model_ranking_bar(self, benchmark_results: Dict, metric: str = 'r2',
                                 regressor_details: Dict = None) -> str:
        """
        Generate horizontal bar chart ranking all models by R² in LOG RETURN space.
        
        Uses the transformed space (log return/pct change) R² which is the true
        measure of model accuracy - computed across ALL forecast steps.
        
        Args:
            benchmark_results: Full benchmark results dict
            metric: Metric to rank by ('r2', 'rmse', 'mae')
            regressor_details: Dict with model details including transformed predictions
            
        Returns:
            Base64 encoded image
        """
        from sklearn.metrics import r2_score
        
        # Compute R² in transformed space if regressor_details available
        model_r2_values = {}
        target_transform = 'log_return'
        
        if regressor_details:
            for model_name, data in regressor_details.items():
                if 'target_transform' in data:
                    target_transform = data['target_transform']
                
                if 'y_val_transformed' in data and 'y_pred_transformed' in data:
                    y_val = np.array(data['y_val_transformed'])
                    y_pred = np.array(data['y_pred_transformed'])
                    
                    # Compute R² across all steps (flattened)
                    try:
                        r2_all = r2_score(y_val.flatten(), y_pred.flatten())
                        model_r2_values[model_name] = r2_all
                    except:
                        model_r2_values[model_name] = 0.0
        
        # Fallback to benchmark_results if no regressor_details
        if not model_r2_values:
            regressors = benchmark_results.get('regressors', {})
            if not regressors:
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, 'No regressor results', ha='center', va='center')
                return self.figure_to_base64()
            
            for name, data in regressors.items():
                model_r2_values[name] = data.get(metric, 0)
        
        plt.figure(figsize=(12, max(6, len(model_r2_values) * 0.5)))
        
        # Sort by R² (higher is better)
        sorted_models = sorted(model_r2_values.items(), key=lambda x: x[1], reverse=True)
        
        names = [m[0].upper() for m in sorted_models]
        values = [m[1] for m in sorted_models]
        
        # Color by performance - adjusted thresholds for log return R²
        # In log return space, even small positive R² is meaningful
        colors = ['#2ecc71' if v > 0.1 else '#f1c40f' if v > 0 else '#e74c3c' for v in values]
        
        y_pos = np.arange(len(names))
        
        bars = plt.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            # Position label based on value sign
            x_pos = bar.get_width() + 0.01 if val >= 0 else bar.get_width() - 0.01
            ha = 'left' if val >= 0 else 'right'
            plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha=ha, fontsize=9)
        
        # Add reference lines
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Zero (random guess)')
        
        plt.yticks(y_pos, names)
        plt.xlabel(f'R² Score ({target_transform.replace("_", " ").title()} Space)', fontsize=12)
        plt.title(f'Model Ranking by R² ({target_transform.replace("_", " ").title()} Space - All Steps)', 
                 fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        return self.figure_to_base64()
    
    def plot_directional_accuracy(self, regressor_details: Dict) -> str:
        """
        Generate directional accuracy chart (correct up/down predictions).
        
        Args:
            regressor_details: Dict with model_name -> {y_val, y_pred}
            
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(12, 6))
        
        model_names = []
        dir_accuracies = []
        
        for model_name, data in regressor_details.items():
            if 'y_val' in data and 'y_pred' in data:
                y_val = np.array(data['y_val']).flatten()
                y_pred = np.array(data['y_pred']).flatten()[:len(y_val)]
                
                # Calculate directional changes
                if len(y_val) > 1:
                    actual_dir = np.sign(np.diff(y_val))
                    pred_dir = np.sign(np.diff(y_pred[:len(y_val)]))
                    
                    correct = np.sum(actual_dir == pred_dir)
                    total = len(actual_dir)
                    dir_acc = correct / total * 100 if total > 0 else 50
                    
                    model_names.append(model_name.upper())
                    dir_accuracies.append(dir_acc)
        
        if not model_names:
            plt.text(0.5, 0.5, 'No data for directional accuracy', ha='center', va='center')
            return self.figure_to_base64()
        
        # Sort by accuracy
        sorted_indices = np.argsort(dir_accuracies)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        dir_accuracies = [dir_accuracies[i] for i in sorted_indices]
        
        colors = ['#2ecc71' if acc > 55 else '#f1c40f' if acc > 50 else '#e74c3c' for acc in dir_accuracies]
        
        x_pos = np.arange(len(model_names))
        bars = plt.bar(x_pos, dir_accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, dir_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, weight='bold')
        
        # Reference lines
        plt.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Random Guess (50%)')
        
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.ylabel('Directional Accuracy (%)', fontsize=12)
        plt.title('Directional Accuracy: Correct Up/Down Predictions', fontsize=14, weight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 100)
        plt.tight_layout()
        
        return self.figure_to_base64()
    
    def plot_multistep_heatmap(self, regressor_details: Dict, horizon: int = 10) -> str:
        """
        Generate heatmap showing R² per model per forecast step in LOG RETURN space.
        
        Args:
            regressor_details: Dict with model_name -> {y_val_transformed, y_pred_transformed}
            horizon: Forecast horizon
            
        Returns:
            Base64 encoded image
        """
        from sklearn.metrics import r2_score
        
        # Collect data - compute R² per step in transformed space
        model_names = []
        r2_matrix = []
        
        # Get target transform type for labeling
        target_transform = 'log_return'
        for data in regressor_details.values():
            if 'target_transform' in data:
                target_transform = data['target_transform']
                break
        
        for model_name, data in regressor_details.items():
            # Prefer transformed space (log return)
            if 'y_val_transformed' in data and 'y_pred_transformed' in data:
                y_val = np.array(data['y_val_transformed'])
                y_pred = np.array(data['y_pred_transformed'])
            else:
                # Skip if no transformed data
                continue
            
            # Ensure 2D
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            actual_horizon = min(horizon, y_val.shape[1], y_pred.shape[1])
            
            # Compute R² per step
            r2_per_step = []
            for step in range(actual_horizon):
                try:
                    r2 = r2_score(y_val[:, step], y_pred[:, step])
                    r2_per_step.append(r2)
                except:
                    r2_per_step.append(0.0)
            
            # Pad to horizon if needed
            while len(r2_per_step) < horizon:
                r2_per_step.append(np.nan)
            
            model_names.append(model_name.upper())
            r2_matrix.append(r2_per_step[:horizon])
        
        if not model_names:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'No per-step data available in {target_transform} space', ha='center', va='center')
            return self.figure_to_base64()
        
        r2_matrix = np.array(r2_matrix)
        
        fig, ax = plt.subplots(figsize=(14, max(4, len(model_names) * 0.6)))
        
        # Use RdYlGn colormap (red=bad, green=good for R²)
        im = ax.imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('R² Score', fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(horizon))
        ax.set_xticklabels([f'Step {i+1}' for i in range(horizon)])
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_yticklabels(model_names)
        
        # Add value annotations
        for i in range(len(model_names)):
            for j in range(horizon):
                val = r2_matrix[i, j]
                if not np.isnan(val):
                    # Use white text on dark backgrounds
                    text_color = 'white' if val < 0.3 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=text_color, fontsize=8)
        
        ax.set_xlabel('Forecast Step', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'Multi-Step R² Heatmap ({target_transform.replace("_", " ").title()} Space)', 
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        return self.figure_to_base64()
    
    def plot_predicted_vs_actual_scatter_grid(self, regressor_details: Dict, 
                                               horizon: int = 10) -> List[str]:
        """
        Generate scatter plots showing predicted vs actual in transformed space (log_return/pct_change).
        
        Creates one figure per forecast step, with all models shown as subplots.
        This is the true visualization of what models learn - the relationship 
        between predicted returns and actual returns.
        
        Args:
            regressor_details: Dict with model_name -> {y_val_transformed, y_pred_transformed, target_transform}
            horizon: Forecast horizon (number of steps to plot)
            
        Returns:
            List of base64 encoded images (one per step)
        """
        images = []
        
        # Get list of valid models with transformed data
        valid_models = []
        for model_name, data in regressor_details.items():
            if 'y_val_transformed' in data and 'y_pred_transformed' in data:
                valid_models.append((model_name, data))
        
        if not valid_models:
            # Fallback: no transformed data available
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No transformed space data available\n(Requires target_transform != price)', 
                    ha='center', va='center')
            images.append(self.figure_to_base64())
            return images
        
        # Get target transform type for labeling
        target_transform = valid_models[0][1].get('target_transform', 'log_return')
        
        # Label based on transform type
        if target_transform == 'log_return':
            axis_label = 'Log Return'
        elif target_transform == 'pct_change':
            axis_label = 'Percentage Change'
        else:
            axis_label = 'Transformed Value'
        
        # Determine actual horizon from data
        sample_data = valid_models[0][1]['y_val_transformed']
        actual_horizon = min(horizon, sample_data.shape[1] if sample_data.ndim > 1 else 1)
        
        # Colors for models
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        
        # Create one figure per step
        for step_idx in range(actual_horizon):
            n_models = len(valid_models)
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
            
            for model_idx, (model_name, data) in enumerate(valid_models):
                row = model_idx // n_cols
                col = model_idx % n_cols
                ax = axes[row, col]
                
                y_val_trans = np.array(data['y_val_transformed'])
                y_pred_trans = np.array(data['y_pred_transformed'])
                
                # Extract the specific step
                if y_val_trans.ndim > 1 and y_val_trans.shape[1] > step_idx:
                    y_actual = y_val_trans[:, step_idx]
                    y_predicted = y_pred_trans[:, step_idx]
                else:
                    y_actual = y_val_trans.flatten()
                    y_predicted = y_pred_trans.flatten()
                
                # Calculate R²
                if len(y_actual) > 1:
                    ss_res = np.sum((y_actual - y_predicted) ** 2)
                    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                else:
                    r2 = 0
                
                # Plot scatter
                color = colors[model_idx % len(colors)]
                ax.scatter(y_predicted, y_actual, alpha=0.5, s=25, c=color, edgecolors='none')
                
                # Add 45° perfect fit line
                all_vals = np.concatenate([y_predicted, y_actual])
                min_val = np.min(all_vals)
                max_val = np.max(all_vals)
                margin = (max_val - min_val) * 0.1
                ax.plot([min_val - margin, max_val + margin], 
                       [min_val - margin, max_val + margin], 
                       'k--', linewidth=1.5, alpha=0.7, label='Perfect Fit')
                
                # Add zero reference lines
                ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
                
                # Labels and title
                ax.set_xlabel(f'Predicted {axis_label}', fontsize=10)
                ax.set_ylabel(f'Actual {axis_label}', fontsize=10)
                ax.set_title(f'{model_name.upper()}\nR² = {r2:.4f}', fontsize=11, weight='bold')
                ax.grid(True, alpha=0.3)
                
                # Set equal aspect ratio
                ax.set_aspect('equal', adjustable='box')
                
                # Set symmetric limits around 0
                max_abs = max(abs(min_val), abs(max_val)) + margin
                ax.set_xlim(-max_abs, max_abs)
                ax.set_ylim(-max_abs, max_abs)
            
            # Hide unused subplots
            for idx in range(n_models, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.suptitle(f'Predicted vs Actual {axis_label} (Step {step_idx + 1})\n'
                        f'Perfect model = all points on diagonal line (R²=1.0)', 
                        fontsize=12, weight='bold', y=1.02)
            plt.tight_layout()
            
            images.append(self.figure_to_base64())
        
        return images

    def plot_predicted_vs_actual_scaled_grid(self, regressor_details: Dict, 
                                              horizon: int = 10) -> List[str]:
        """
        Generate scatter plots showing predicted vs actual in SCALED space (StandardScaler output).
        
        This shows what the model ACTUALLY sees during training - the z-score normalized values.
        This is the truest representation of model performance since it's the exact input/output
        space the optimization happens in.
        
        Args:
            regressor_details: Dict with model_name -> {y_val_scaled, y_pred_scaled}
            horizon: Forecast horizon (number of steps to plot)
            
        Returns:
            List of base64 encoded images (one per step)
        """
        images = []
        
        # Get list of valid models with scaled data
        valid_models = []
        for model_name, data in regressor_details.items():
            if 'y_val_scaled' in data and 'y_pred_scaled' in data:
                valid_models.append((model_name, data))
        
        if not valid_models:
            # Fallback: no scaled data available
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No scaled space data available', 
                    ha='center', va='center')
            images.append(self.figure_to_base64())
            return images
        
        # Determine actual horizon from data
        sample_data = valid_models[0][1]['y_val_scaled']
        actual_horizon = min(horizon, sample_data.shape[1] if sample_data.ndim > 1 else 1)
        
        # Colors for models
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        
        # Create one figure per step
        for step_idx in range(actual_horizon):
            n_models = len(valid_models)
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
            
            for model_idx, (model_name, data) in enumerate(valid_models):
                row = model_idx // n_cols
                col = model_idx % n_cols
                ax = axes[row, col]
                
                y_val_sc = np.array(data['y_val_scaled'])
                y_pred_sc = np.array(data['y_pred_scaled'])
                
                # Extract the specific step
                if y_val_sc.ndim > 1 and y_val_sc.shape[1] > step_idx:
                    y_actual = y_val_sc[:, step_idx]
                    y_predicted = y_pred_sc[:, step_idx]
                else:
                    y_actual = y_val_sc.flatten()
                    y_predicted = y_pred_sc.flatten()
                
                # Calculate R²
                if len(y_actual) > 1:
                    ss_res = np.sum((y_actual - y_predicted) ** 2)
                    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                else:
                    r2 = 0
                
                # Plot scatter
                color = colors[model_idx % len(colors)]
                ax.scatter(y_predicted, y_actual, alpha=0.5, s=25, c=color, edgecolors='none')
                
                # Add 45° perfect fit line
                all_vals = np.concatenate([y_predicted, y_actual])
                min_val = np.min(all_vals)
                max_val = np.max(all_vals)
                margin = (max_val - min_val) * 0.1
                ax.plot([min_val - margin, max_val + margin], 
                       [min_val - margin, max_val + margin], 
                       'k--', linewidth=1.5, alpha=0.7, label='Perfect Fit')
                
                # Add zero reference lines (mean in scaled space)
                ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
                
                # Labels and title
                ax.set_xlabel('Predicted (z-score)', fontsize=10)
                ax.set_ylabel('Actual (z-score)', fontsize=10)
                ax.set_title(f'{model_name.upper()}\nR² = {r2:.4f}', fontsize=11, weight='bold')
                ax.grid(True, alpha=0.3)
                
                # Set equal aspect ratio
                ax.set_aspect('equal', adjustable='box')
                
                # Set symmetric limits around 0
                max_abs = max(abs(min_val), abs(max_val)) + margin
                ax.set_xlim(-max_abs, max_abs)
                ax.set_ylim(-max_abs, max_abs)
            
            # Hide unused subplots
            for idx in range(n_models, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.suptitle(f'Predicted vs Actual - Scaled Space (Step {step_idx + 1})\n'
                        f'StandardScaler output (z-score) - This is what models actually see', 
                        fontsize=12, weight='bold', y=1.02)
            plt.tight_layout()
            
            images.append(self.figure_to_base64())
        
        return images

    def plot_feature_importance(self, benchmark_results: Dict, top_k: int = 20) -> str:
        """
        Generate feature importance plot for tree-based models.
        
        Args:
            benchmark_results: Full results including feature importances
            top_k: Number of top features to show
            
        Returns:
            Base64 encoded image
        """
        feature_importance = benchmark_results.get('feature_importance', {})
        
        if not feature_importance:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No feature importance data available\n(Requires LightGBM or RandomForest)', 
                    ha='center', va='center')
            return self.figure_to_base64()
        
        # Take top_k features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No features to display', ha='center', va='center')
            return self.figure_to_base64()
        
        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(12, max(6, len(names) * 0.3)))
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values, color='#3498db', alpha=0.8, edgecolor='black')
        
        plt.yticks(y_pos, names, fontsize=9)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_k} Feature Importances', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().invert_yaxis()  # Highest at top
        
        plt.tight_layout()
        return self.figure_to_base64()

    def plot_regressor_test_cases(self, y_test: np.ndarray, y_pred: np.ndarray, 
                                 y_all: np.ndarray, forecast_horizon: int = 30, 
                                 model_name: str = '', num_plots: int = 5) -> List[str]:
        """
        Generate multiple test case plots showing last 90 timesteps + forecast.
        Each plot shows historical data (90 days) + model predictions vs actual values.
        
        Args:
            y_test: Test set actual prices (in original scale) - can be 1D or 2D (multi-step)
            y_pred: Predictions for test set (in original scale) - can be 1D or 2D (multi-step)
            y_all: Full timeseries data for historical context (in original scale)
            forecast_horizon: Number of days forecasted
            model_name: Model name for title
            num_plots: Number of test case plots to generate (default: 5)
        
        Returns:
            List of base64 encoded images
        """
        base64_images = []
        
        # Convert to numpy arrays if needed
        y_test = np.asarray(y_test)
        y_pred = np.asarray(y_pred)
        y_all = np.asarray(y_all)
        
        # Handle 2D arrays (multi-step forecasting): shape (n_samples, horizon)
        is_multistep = y_test.ndim == 2 and y_test.shape[1] > 1
        
        # Calculate plot interval to spread plots across test set
        step = max(1, len(y_test) // num_plots)
        indices = [i * step for i in range(num_plots) if i * step < len(y_test)]
        
        for plot_idx, test_idx in enumerate(indices):
            plt.figure(figsize=(14, 6))
            
            # Historical window: last 90 days before this test point
            history_window = 90
            history_start = max(0, test_idx - history_window)
            
            # Determine the corresponding position in y_all
            # This is approximate since y_test is a subset of y_all
            all_idx = test_idx + (len(y_all) - len(y_test))
            
            # Historical data (last 90 days)
            hist_start = max(0, all_idx - history_window)
            hist_data = y_all[hist_start:all_idx]
            hist_x = np.arange(-len(hist_data), 0)
            
            # Extract actual and predicted values, handling both 1D and 2D cases
            if is_multistep:
                # Multi-step: y_test[test_idx] is an array of shape (horizon,)
                actual_horizon = y_test[test_idx] if test_idx < len(y_test) else y_test[-1]
                pred_horizon = y_pred[test_idx] if test_idx < len(y_pred) else actual_horizon
                # Use first step value as the "current" point
                actual_value = float(actual_horizon[0]) if len(actual_horizon) > 0 else float(y_all[-1])
                pred_value = float(pred_horizon[0]) if len(pred_horizon) > 0 else actual_value
                # Forecast uses the full multi-step prediction
                forecast_actual = np.asarray(actual_horizon).flatten()
                forecast_pred = np.asarray(pred_horizon).flatten()
                actual_horizon_len = len(forecast_actual)
            else:
                # 1D: single-step prediction
                actual_value = float(y_test[test_idx]) if test_idx < len(y_test) else float(y_all[-1])
                pred_value = float(y_pred[test_idx]) if test_idx < len(y_pred) else actual_value
                forecast_actual = None
                forecast_pred = None
                actual_horizon_len = forecast_horizon
            
            # Plot historical data
            plt.plot(hist_x, hist_data, 'b-', label=f'Historical Data ({len(hist_data)} steps)', linewidth=2, marker='o', markersize=3)
            
            if is_multistep and forecast_actual is not None and forecast_pred is not None:
                # Multi-step: plot the full forecast horizon
                forecast_x = np.arange(1, len(forecast_pred) + 1)
                
                # Plot actual future values (ground truth for the horizon)
                plt.plot(forecast_x, forecast_actual, 'g-', label=f'Actual Future ({len(forecast_actual)} steps)', 
                        linewidth=2, marker='o', markersize=5, alpha=0.8)
                
                # Plot predicted future values
                plt.plot(forecast_x, forecast_pred, 'r--', label=f'Predicted Future ({len(forecast_pred)} steps)', 
                        linewidth=2, marker='x', markersize=5, alpha=0.8)
                
                # Mark the starting point (t=0) with the last historical value
                last_hist_value = float(hist_data[-1]) if len(hist_data) > 0 else actual_value
                plt.scatter([0], [last_hist_value], color='blue', s=150, marker='s', 
                           label='Current Point', zorder=5, edgecolors='darkblue', linewidth=2)
            else:
                # Single-step: original behavior
                pred_x = 0
                # Plot actual value at current point
                plt.scatter([pred_x], [actual_value], color='green', s=200, marker='o', 
                           label='Actual Value (Now)', zorder=5, edgecolors='darkgreen', linewidth=2)
                
                # Plot prediction at current point
                plt.scatter([pred_x], [pred_value], color='red', s=200, marker='X', 
                           label='Predicted Value (Now)', zorder=5, edgecolors='darkred', linewidth=2)
                
                # Forecast trend line
                forecast_x = np.arange(1, forecast_horizon + 1)
                forecast_trend = np.linspace(pred_value, actual_value, forecast_horizon)
                plt.plot(forecast_x, forecast_trend, 'r--', label=f'Forecast Trend ({forecast_horizon} steps)', 
                        linewidth=2, alpha=0.7)
                
                # Add text annotations
                plt.text(pred_x, actual_value * 1.02, f'${actual_value:.2f}', ha='center', fontsize=9, color='green', weight='bold')
                plt.text(pred_x, pred_value * 0.98, f'${pred_value:.2f}', ha='center', fontsize=9, color='red', weight='bold')
            
            # Add vertical line at current point
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Labels and formatting
            plt.xlabel('Time Step (Negative = Past, 0 = Now, Positive = Future)')
            plt.ylabel('Price')
            plt.title(f'{model_name.upper()} - Test Case {plot_idx + 1}: {len(hist_data)}-Step History + {actual_horizon_len}-Step Forecast')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            base64_img = self.figure_to_base64()
            base64_images.append(base64_img)
        
        return base64_images
    
    def plot_holdout_forecast(self, holdout_forecasts: Dict, holdout_days: int, 
                              ticker: str = 'Asset', period_unit: str = 'days') -> str:
        """
        Generate a single timeline plot showing forecast vs actual for holdout period.
        
        This is the key validation plot: shows how well each model predicted
        the last N periods that were held out from training.
        
        Args:
            holdout_forecasts: Dict with model_name -> {predictions, actual, metrics}
            holdout_days: Number of holdout periods
            ticker: Asset name
            period_unit: Unit label for display (days, hours, 5m candles, etc.)
        
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(16, 8))
        
        # Get actual values from first model's data
        actual_values = None
        for model_name, data in holdout_forecasts.items():
            if 'actual' in data and data['actual']:
                actual_values = np.array(data['actual'])
                break
        
        if actual_values is None:
            plt.text(0.5, 0.5, 'No holdout data available', ha='center', va='center')
            plt.tight_layout()
            return self.figure_to_base64()
        
        x_days = np.arange(1, len(actual_values) + 1)
        
        # Plot actual values (thick black line)
        plt.plot(x_days, actual_values, 'ko-', label='Actual Price', linewidth=3, markersize=8, zorder=10)
        
        # Color palette for models
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        
        # Plot each model's predictions
        for idx, (model_name, data) in enumerate(holdout_forecasts.items()):
            if model_name == 'naive_baseline':
                continue  # Plot baseline separately
            
            if 'predictions' in data and data['predictions']:
                predictions = np.array(data['predictions'])[:len(actual_values)]
                color = colors[idx % len(colors)]
                
                r2 = data.get('r2', 0)
                rmse = data.get('rmse', 0)
                beats_naive = data.get('beats_naive', False)
                
                # Use different style for models that beat naive
                marker = 's' if beats_naive else 'x'
                style = '-' if r2 > 0 else ':'
                label = f"{model_name.upper()} (R²={r2:.3f}, RMSE=${rmse:.2f})"
                if beats_naive:
                    label += " ✓"
                
                plt.plot(x_days, predictions, style, color=color, label=label, 
                        linewidth=2.5, marker=marker, markersize=5, alpha=0.85)
        
        # Plot naive baseline
        if 'naive_baseline' in holdout_forecasts:
            baseline_data = holdout_forecasts['naive_baseline']
            if 'predictions' in baseline_data:
                baseline_pred = np.array(baseline_data['predictions'])[:len(actual_values)]
                plt.plot(x_days, baseline_pred, 'gray', linestyle=':', 
                        label=f"Naive Baseline (RMSE=${baseline_data.get('rmse', 0):.2f})",
                        linewidth=2, alpha=0.7)
        
        plt.xlabel(f'Holdout {period_unit.title()}', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'{ticker}: Holdout Validation - Forecast vs Actual (Last {holdout_days} {period_unit})', fontsize=14, weight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for training cutoff
        plt.axvline(x=0.5, color='green', linestyle='-', linewidth=2, alpha=0.5)
        plt.text(1, plt.ylim()[1] * 0.98, 'Start of Holdout\n(Never Seen in Training)', fontsize=9, color='green', 
                verticalalignment='top')
        
        plt.tight_layout()
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_rolling_holdout_aggregation(self, holdout_rolling: Dict, 
                                         ticker: str = 'Asset',
                                         holdout_steps: int = 10) -> str:
        """
        Generate a plot showing rolling aggregated holdout predictions with uncertainty bands.
        
        This is the key visualization for the rolling aggregation approach:
        - Shows aggregated predictions from overlapping multi-step forecasts
        - Includes uncertainty bands (±1 std from prediction variance)
        - Compares all models to actual prices
        
        Args:
            holdout_rolling: Dict with model_name -> {daily_predictions, daily_std, daily_actual, metrics}
            ticker: Asset name
            holdout_steps: Number of holdout steps for title
        
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(16, 7))
        
        # Get actual values from first model
        actual_values = None
        for model_name, data in holdout_rolling.items():
            if 'error' in data:
                continue
            if 'daily_actual' in data and data['daily_actual']:
                actual_values = np.array(data['daily_actual'])
                break
        
        if actual_values is None:
            plt.text(0.5, 0.5, 'No rolling holdout data available', ha='center', va='center')
            plt.tight_layout()
            return self.figure_to_base64()
        
        n_steps = len(actual_values)
        x_steps = np.arange(1, n_steps + 1)
        
        # Plot actual values (thick black line)
        plt.plot(x_steps, actual_values, 'k-', label='Actual Price', linewidth=3, marker='o', 
                markersize=6, zorder=20)
        
        # Color palette
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c']
        
        # Plot each model with uncertainty band
        color_idx = 0
        for model_name, data in holdout_rolling.items():
            if 'error' in data or model_name == 'naive':
                continue
            
            if 'daily_predictions' in data:
                predictions = np.array(data['daily_predictions'])[:n_steps]
                std = np.array(data['daily_std'])[:n_steps] if 'daily_std' in data else np.zeros_like(predictions)
                
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                metrics = data.get('metrics', {})
                rmse = metrics.get('rmse', 0)
                r2 = metrics.get('r2', 0)
                
                # Plot prediction line
                label = f"{model_name.upper()} (RMSE=${rmse:.2f}, R²={r2:.3f})"
                plt.plot(x_steps, predictions, '-', color=color, label=label, 
                        linewidth=2.5, marker='s', markersize=4, alpha=0.9, zorder=10)
                
                # Plot uncertainty band (±1 std)
                if np.any(std > 0):
                    plt.fill_between(x_steps, predictions - std, predictions + std, 
                                   color=color, alpha=0.2, zorder=5)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'{ticker}: Rolling {holdout_steps}-Step Forecast Aggregation (Holdout Period)\n'
                 f'Each point = mean of overlapping predictions targeting that timestep', 
                 fontsize=14, weight='bold')
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Add shaded region at the start
        plt.axvspan(0, 1, alpha=0.1, color='green')
        plt.text(1.5, plt.ylim()[1] * 0.98, 'Holdout Start', fontsize=9, color='green', 
                verticalalignment='top')
        
        plt.tight_layout()
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_trend_assessment(self, trend_data: Dict, ticker: str = 'Asset', 
                               forecast_horizon: int = 10) -> str:
        """
        Generate a visual representation of trend assessment.
        
        Shows:
        - Trend predictions from each classifier with probabilities
        - Expected percentage change from best regressor
        
        Args:
            trend_data: Dict with model assessments
            ticker: Asset name
            forecast_horizon: Number of steps for forecast horizon
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Trend predictions per classifier
        ax1 = axes[0]
        
        models = []
        trend_values = []
        colors = []
        trend_map = {'DOWN': -1, 'SIDEWAYS': 0, 'HOLD': 0, 'UP': 1}
        color_map = {'DOWN': '#e74c3c', 'SIDEWAYS': '#f39c12', 'HOLD': '#f39c12', 'UP': '#27ae60'}
        
        for model_name, data in trend_data.items():
            if model_name == 'expected_pct_change':
                continue
            if 'error' in data:
                continue
            if 'trend_label' in data:
                models.append(model_name.upper())
                trend_label = data['trend_label']
                trend_values.append(trend_map.get(trend_label, 0))
                colors.append(color_map.get(trend_label, '#808080'))
        
        if models:
            y_pos = np.arange(len(models))
            ax1.barh(y_pos, trend_values, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(models)
            ax1.set_xlabel('Trend Direction')
            ax1.set_title(f'{forecast_horizon}-Step Trend Predictions by Classifier', fontsize=12, weight='bold')
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_xticks([-1, 0, 1])
            ax1.set_xticklabels(['DOWN', 'SIDEWAYS', 'UP'])
            ax1.grid(True, alpha=0.3, axis='x')
        else:
            ax1.text(0.5, 0.5, 'No trend data available', ha='center', va='center')
        
        # Right: Expected % change gauge
        ax2 = axes[1]
        
        if 'expected_pct_change' in trend_data:
            pct_data = trend_data['expected_pct_change']
            pct_value = pct_data.get('value', 0)
            source_model = pct_data.get('source_model', 'Unknown')
            current_price = pct_data.get('current_price', 0)
            predicted_price = pct_data.get('predicted_price_final', pct_data.get('predicted_price_day_10', 0))
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones(100)
            
            ax2.fill_between(theta[:33], 0, r[:33], alpha=0.3, color='#e74c3c', label='Down Zone')
            ax2.fill_between(theta[33:66], 0, r[33:66], alpha=0.3, color='#f39c12', label='Neutral')
            ax2.fill_between(theta[66:], 0, r[66:], alpha=0.3, color='#27ae60', label='Up Zone')
            
            # Calculate needle position (map pct_value to angle)
            # -10% → 0, 0% → π/2, +10% → π
            pct_clamped = max(-10, min(10, pct_value))
            needle_angle = (pct_clamped + 10) / 20 * np.pi
            
            ax2.arrow(0, 0, np.cos(needle_angle) * 0.8, np.sin(needle_angle) * 0.8, 
                     head_width=0.08, head_length=0.05, fc='black', ec='black', linewidth=2)
            ax2.plot([0], [0], 'ko', markersize=10)
            
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-0.1, 1.2)
            ax2.axis('off')
            ax2.set_title(f'Expected {forecast_horizon}-Step Change: {pct_value:+.2f}%\n'
                         f'(${current_price:.2f} → ${predicted_price:.2f}, via {source_model})', 
                         fontsize=12, weight='bold')
            
            # Add labels
            ax2.text(-1, 0.05, '-10%', ha='center', fontsize=10, color='#e74c3c')
            ax2.text(0, 1.1, '0%', ha='center', fontsize=10, color='#f39c12')
            ax2.text(1, 0.05, '+10%', ha='center', fontsize=10, color='#27ae60')
        else:
            ax2.text(0.5, 0.5, 'No expected % change data', ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        base64_img = self.figure_to_base64()
        return base64_img
    
    def plot_future_forecast_table(self, future_forecasts: Dict, horizon: int = 10,
                                   last_price: float = 0, history: List[float] = None) -> str:
        """
        Generate a visual table/chart of future forecast predictions with history context.
        
        Args:
            future_forecasts: Dict with model predictions
            horizon: Forecast horizon
            last_price: Current price for % change calculation
            history: List of recent historical prices (optional) to show continuity
        
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(16, 8))
        
        # Filter out baseline and error entries
        valid_models = {k: v for k, v in future_forecasts.items() 
                       if k != 'naive_baseline' and 'predictions' in v and 'error' not in v}
        
        if not valid_models:
            plt.text(0.5, 0.5, 'No future forecast data available', ha='center', va='center')
            plt.tight_layout()
            return self.figure_to_base64()
        
        # Setup time axis
        future_steps = np.arange(1, horizon + 1)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c']
        
        # Plot History if available
        if history and len(history) > 0:
            hist_len = min(20, len(history)) # Show last 20 points
            hist_data = history[-hist_len:]
            hist_steps = np.arange(-hist_len + 1, 1)
            
            plt.plot(hist_steps, hist_data, 'k-o', linewidth=2, label='History', alpha=0.7, markersize=4)
            
            # Connect history to forecast
            connect_x = [0, 1]
            last_hist = hist_data[-1]
            
            # Update last_price if history is more recent/accurate source
            if last_price == 0 or abs(last_price - last_hist) > abs(last_price) * 0.1:
                last_price = last_hist
        
        for idx, (model_name, data) in enumerate(valid_models.items()):
            predictions = np.array(data['predictions'])[:horizon]
            color = colors[idx % len(colors)]
            
            # Plot forecast
            plt.plot(future_steps, predictions, '-o', color=color, label=model_name.upper(), 
                    linewidth=2.5, markersize=6, alpha=0.9)
            
            # Draw connecting line from last price to first prediction
            if last_price > 0:
                plt.plot([0, 1], [last_price, predictions[0]], '--', color=color, alpha=0.5, linewidth=1)
            
            # Annotate last point with price and % change
            if last_price > 0:
                pct_change = ((predictions[-1] - last_price) / last_price) * 100
                plt.annotate(f'${predictions[-1]:.2f}\n({pct_change:+.1f}%)', 
                           (future_steps[-1], predictions[-1]),
                           textcoords="offset points", xytext=(10, 0),
                           fontsize=9, color=color, weight='bold')
        
        # Add horizontal line for current price
        if last_price > 0:
            plt.axhline(y=last_price, color='black', linestyle='--', linewidth=1.5, 
                       label=f'Current: ${last_price:.2f}')
        
        # Add naive baseline if available
        if 'naive_baseline' in future_forecasts:
            naive_preds = future_forecasts['naive_baseline'].get('predictions', [])[:horizon]
            if naive_preds:
                plt.plot(future_steps, naive_preds, ':', color='gray', linewidth=2, 
                        label='Naive (flat)', alpha=0.7)
        
        plt.xlabel('Step Ahead', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title(f'Future Price Forecast (Next {horizon} Steps)', fontsize=14, weight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(future_steps)
        
        plt.tight_layout()
        base64_img = self.figure_to_base64()
        return base64_img

    def plot_sample_forecasts(self, sample_forecasts: List[Dict]) -> List[str]:
        """
        Generate plots for random test samples (History + Forecast vs Actual).
        
        Args:
            sample_forecasts: List of dicts with history, actual_future, model_predictions
            
        Returns:
            List of base64 encoded images
        """
        images = []
        
        for i, sample in enumerate(sample_forecasts):
            plt.figure(figsize=(14, 6))
            
            history = np.array(sample['history'])
            actual_future = np.array(sample['actual_future'])
            
            # Time axis
            hist_len = len(history)
            fut_len = len(actual_future)
            
            t_hist = np.arange(-hist_len + 1, 1)
            t_fut = np.arange(1, fut_len + 1)
            
            # Plot history
            plt.plot(t_hist, history, 'k-', label='History', linewidth=2)
            
            # Plot actual future
            plt.plot(t_fut, actual_future, 'k--', label='Actual Future', linewidth=2, marker='o')
            
            # Plot model predictions
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c']
            
            for idx, (model_name, pred) in enumerate(sample['model_predictions'].items()):
                pred = np.array(pred)
                if len(pred) > fut_len:
                    pred = pred[:fut_len]
                
                color = colors[idx % len(colors)]
                plt.plot(t_fut[:len(pred)], pred, '-', color=color, label=model_name.upper(), 
                        linewidth=1.5, marker='x', markersize=4, alpha=0.8)
            
            plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)
            plt.title(f'Sample Forecast #{i+1} (Index: {sample.get("index", "?")})')
            plt.xlabel('Time Step (0 = Prediction Time)')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            images.append(self.figure_to_base64())
            
        return images

    def plot_recursive_holdout_forecast(self, recursive_forecasts: Dict, 
                                        holdout_actual: np.ndarray,
                                        ticker: str = 'Asset',
                                        period_unit: str = 'periods') -> str:
        """
        Generate plot showing TRUE RECURSIVE forecasts vs actual holdout prices.
        
        This shows how models perform in a real-world scenario where predictions
        are fed back to generate further forecasts beyond the native horizon.
        
        Args:
            recursive_forecasts: Dict with model_name -> {predictions, predictions_transformed, chunk_boundaries}
            holdout_actual: Actual holdout prices
            ticker: Asset name
            period_unit: Unit label for display
        
        Returns:
            Base64 encoded image
        """
        plt.figure(figsize=(18, 10))
        
        holdout_length = len(holdout_actual)
        x_steps = np.arange(1, holdout_length + 1)
        
        # Plot actual values
        plt.plot(x_steps, holdout_actual, 'ko-', label='Actual Price', linewidth=3, markersize=6, zorder=20)
        
        # Color palette
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#1abc9c', '#e67e22']
        
        # Plot each model's recursive forecast
        for idx, (model_name, data) in enumerate(recursive_forecasts.items()):
            if 'error' in data:
                continue
            
            predictions = np.array(data.get('predictions', []))[:holdout_length]
            chunk_boundaries = data.get('chunk_boundaries', [])
            
            if len(predictions) == 0:
                continue
            
            color = colors[idx % len(colors)]
            
            # Calculate metrics
            valid_len = min(len(predictions), len(holdout_actual))
            rmse = np.sqrt(np.mean((predictions[:valid_len] - holdout_actual[:valid_len])**2))
            r2 = 1 - np.sum((holdout_actual[:valid_len] - predictions[:valid_len])**2) / np.sum((holdout_actual[:valid_len] - np.mean(holdout_actual[:valid_len]))**2)
            
            label = f"{model_name.upper()} (RMSE=${rmse:.2f}, R²={r2:.3f})"
            
            # Plot predictions
            plt.plot(x_steps[:len(predictions)], predictions, '-', color=color, 
                    label=label, linewidth=2.5, marker='s', markersize=4, alpha=0.9, zorder=10)
            
            # Mark chunk boundaries with vertical lines
            for boundary in chunk_boundaries[1:-1]:  # Skip first (0) and last
                if boundary < holdout_length:
                    plt.axvline(x=boundary + 1, color=color, linestyle=':', alpha=0.4, linewidth=1)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'{ticker}: Recursive Holdout Forecast (True Autoregressive)\n'
                 f'Models predict native horizon, then feed predictions back to continue',
                 fontsize=14, weight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add annotation explaining the approach
        plt.annotate('← Recursive: predictions fed back as inputs →', 
                    xy=(holdout_length / 2, plt.ylim()[0]), 
                    fontsize=9, ha='center', color='gray', style='italic')
        
        plt.tight_layout()
        return self.figure_to_base64()

    def plot_dual_space_comparison(self, model_name: str,
                                   predictions_price: np.ndarray,
                                   actual_price: np.ndarray,
                                   predictions_transformed: np.ndarray,
                                   actual_transformed: np.ndarray,
                                   transform_name: str = 'log_return',
                                   ticker: str = 'Asset') -> str:
        """
        Generate dual-panel plot showing predictions in both price and transformed space.
        
        Args:
            model_name: Name of the model
            predictions_price: Predicted prices
            actual_price: Actual prices
            predictions_transformed: Predictions in training space (log_return, pct_change)
            actual_transformed: Actual values in training space
            transform_name: Name of the transform ('log_return', 'pct_change')
            ticker: Asset name
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        n_points = min(len(predictions_price), len(actual_price))
        x = np.arange(1, n_points + 1)
        
        # Left panel: Price space
        ax1 = axes[0]
        ax1.plot(x, actual_price[:n_points], 'k-', label='Actual', linewidth=2)
        ax1.plot(x, predictions_price[:n_points], 'r--', label='Predicted', linewidth=2)
        
        # Calculate metrics for price space
        rmse_price = np.sqrt(np.mean((predictions_price[:n_points] - actual_price[:n_points])**2))
        r2_price = 1 - np.sum((actual_price[:n_points] - predictions_price[:n_points])**2) / np.sum((actual_price[:n_points] - np.mean(actual_price[:n_points]))**2)
        
        ax1.set_xlabel('Period', fontsize=11)
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title(f'Price Space (RMSE=${rmse_price:.2f}, R²={r2_price:.3f})', fontsize=12, weight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Fill between for error visualization
        ax1.fill_between(x, actual_price[:n_points], predictions_price[:n_points], 
                        alpha=0.2, color='red', label='Error')
        
        # Right panel: Transformed space
        ax2 = axes[1]
        n_trans = min(len(predictions_transformed), len(actual_transformed))
        x_trans = np.arange(1, n_trans + 1)
        
        ax2.plot(x_trans, actual_transformed[:n_trans], 'k-', label='Actual', linewidth=2)
        ax2.plot(x_trans, predictions_transformed[:n_trans], 'r--', label='Predicted', linewidth=2)
        
        # Calculate metrics for transformed space
        rmse_trans = np.sqrt(np.mean((predictions_transformed[:n_trans] - actual_transformed[:n_trans])**2))
        
        # Transform-specific labels
        if transform_name == 'log_return':
            ylabel = 'Log Return'
            title_suffix = 'Training Space (Log Returns)'
        elif transform_name == 'pct_change':
            ylabel = 'Percent Change'
            title_suffix = 'Training Space (% Change)'
        else:
            ylabel = 'Value'
            title_suffix = 'Training Space'
        
        ax2.set_xlabel('Period', fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(f'{title_suffix} (RMSE={rmse_trans:.6f})', fontsize=12, weight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Fill between for error visualization
        ax2.fill_between(x_trans, actual_transformed[:n_trans], predictions_transformed[:n_trans],
                        alpha=0.2, color='red')
        
        plt.suptitle(f'{ticker}: {model_name.upper()} - Dual Space Comparison', 
                    fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        
        return self.figure_to_base64()

    def plot_all_models_dual_space(self, recursive_forecasts: Dict,
                                   holdout_actual_price: np.ndarray,
                                   holdout_actual_transformed: np.ndarray,
                                   transform_name: str = 'log_return',
                                   ticker: str = 'Asset') -> List[str]:
        """
        Generate dual-space plots for all models.
        
        Args:
            recursive_forecasts: Dict with model predictions
            holdout_actual_price: Actual holdout prices
            holdout_actual_transformed: Actual holdout values in training space
            transform_name: Name of the transform
            ticker: Asset name
        
        Returns:
            List of base64 encoded images (one per model)
        """
        images = []
        
        for model_name, data in recursive_forecasts.items():
            if 'error' in data:
                continue
            
            predictions_price = np.array(data.get('predictions', []))
            predictions_transformed = np.array(data.get('predictions_transformed', []))
            
            if len(predictions_price) == 0:
                continue
            
            img = self.plot_dual_space_comparison(
                model_name=model_name,
                predictions_price=predictions_price,
                actual_price=holdout_actual_price,
                predictions_transformed=predictions_transformed,
                actual_transformed=holdout_actual_transformed,
                transform_name=transform_name,
                ticker=ticker
            )
            images.append((model_name, img))
        
        return images

    def generate_pipeline_overview_html(self, pipeline_info: Dict) -> str:
        """
        Generate a comprehensive HTML section showing the data pipeline overview.
        
        This appears ONCE at the beginning of the report with detailed feature lists.
        
        Args:
            pipeline_info: Dict containing pipeline details from benchmark
        
        Returns:
            HTML string for the pipeline overview section
        """
        # Extract info
        feature_columns = pipeline_info.get('feature_columns', [])
        target = pipeline_info.get('target', 'Close')
        n_features = pipeline_info.get('n_features', len(feature_columns))
        training_samples = pipeline_info.get('training_samples', 0)
        holdout_samples = pipeline_info.get('holdout_samples', 0)
        total_samples = pipeline_info.get('total_samples', training_samples + holdout_samples)
        raw_data_shape = pipeline_info.get('raw_data_shape', (0, 0))
        model_input_shape = pipeline_info.get('model_input_shape', (0, 0))
        
        # Group features by type
        lag_features = [f for f in feature_columns if 'lag' in f.lower()]
        return_features = [f for f in feature_columns if 'return' in f.lower()]
        rolling_features = [f for f in feature_columns if 'rolling' in f.lower()]
        momentum_features = [f for f in feature_columns if 'momentum' in f.lower() or 'roc' in f.lower()]
        other_features = [f for f in feature_columns if f not in lag_features + return_features + rolling_features + momentum_features]
        
        # Build feature list HTML
        def format_features(features, emoji):
            if not features:
                return ""
            items = "".join([f"<li><code>{f}</code></li>" for f in features])
            return f"<ul style='margin: 5px 0 5px 20px; font-size: 13px;'>{items}</ul>"
        
        html = f"""
        <h2>🔄 Data Pipeline Overview</h2>
        <div class="summary">
            <strong>Pipeline Status:</strong> All data transformations applied once for the entire benchmark.
            <br><strong>Data Split Strategy:</strong> Strict temporal split with holdout validation (no data leakage).
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 25px 0;">
            
            <!-- Data Split Info -->
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #4CAF50;">
                <h3 style="color: #2e7d32; margin: 0 0 15px 0;">📊 Data Split Configuration</h3>
                <table style="width: 100%; font-size: 14px;">
                    <tr><td style="padding: 5px 0;"><strong>Raw Data Shape:</strong></td><td>{raw_data_shape}</td></tr>
                    <tr><td style="padding: 5px 0;"><strong>Model Input Shape:</strong></td><td>{model_input_shape}</td></tr>
                    <tr><td style="padding: 5px 0;"><strong>Training Samples:</strong></td><td style="color: #2e7d32; font-weight: bold;">{training_samples}</td></tr>
                    <tr><td style="padding: 5px 0;"><strong>Holdout Samples:</strong></td><td style="color: #c62828; font-weight: bold;">{holdout_samples}</td></tr>
                    <tr><td style="padding: 5px 0;"><strong>Total Samples:</strong></td><td>{total_samples}</td></tr>
                </table>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 8px; font-size: 12px;">
                    ⚠️ <strong>Holdout Guarantee:</strong> The last {holdout_samples} samples are NEVER seen during training. Models are validated on this true out-of-sample period.
                </p>
            </div>
            
            <!-- Target Transform & Scale-Free Config -->
            <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #9C27B0;">
                <h3 style="color: #6a1b9a; margin: 0 0 15px 0;">🔄 Target Transform & Scale-Free Config</h3>
                <table style="width: 100%; font-size: 14px;">
                    <tr>
                        <td style="padding: 5px 0;"><strong>Target Transform:</strong></td>
                        <td style="font-weight: bold; color: #6a1b9a;">{pipeline_info.get('target_transform', 'price').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;"><strong>Base Price Column:</strong></td>
                        <td><code>{pipeline_info.get('base_price_column', 'Close')}</code></td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;"><strong>Allow Additional Price Cols:</strong></td>
                        <td>{'✅ Yes' if pipeline_info.get('allow_additional_price_columns', False) else '❌ No'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;"><strong>Allow Absolute Scale Features:</strong></td>
                        <td>{'✅ Yes' if pipeline_info.get('allow_absolute_scale_features', False) else '❌ No'}</td>
                    </tr>
                </table>
                <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 8px; font-size: 12px;">
                    📝 <strong>Transform modes:</strong> <code>price</code> = raw values, <code>pct_change</code> = % returns, <code>log_return</code> = log returns (scale-free)
                </p>
            </div>
            
            <!-- Target Info -->
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #FF9800;">
                <h3 style="color: #e65100; margin: 0 0 15px 0;">🎯 Target Variable</h3>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                    <span style="font-size: 24px; font-weight: bold; color: #e65100;">{target}</span>
                    <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">Next timestep's closing price (t+1)</p>
                </div>
                <p style="margin: 15px 0 0 0; font-size: 13px;">
                    <strong>No Data Leakage:</strong> Only historical data (t-n) is used to predict future values (t+1).
                </p>
            </div>
        </div>
        
        <!-- Feature Details -->
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 25px; border-radius: 12px; margin: 25px 0; border-left: 5px solid #2196F3;">
            <h3 style="color: #1565c0; margin: 0 0 20px 0;">📋 Feature Columns ({n_features} Total)</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
"""
        
        if lag_features:
            html += f"""
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong style="color: #1565c0;">📈 Lag Features ({len(lag_features)})</strong>
                    {format_features(lag_features, '📈')}
                </div>
"""
        
        if return_features:
            html += f"""
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong style="color: #7b1fa2;">📊 Return Features ({len(return_features)})</strong>
                    {format_features(return_features, '📊')}
                </div>
"""
        
        if rolling_features:
            html += f"""
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong style="color: #00695c;">🔄 Rolling Features ({len(rolling_features)})</strong>
                    {format_features(rolling_features, '🔄')}
                </div>
"""
        
        if momentum_features:
            html += f"""
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong style="color: #c62828;">⚡ Momentum Features ({len(momentum_features)})</strong>
                    {format_features(momentum_features, '⚡')}
                </div>
"""
        
        if other_features:
            html += f"""
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong style="color: #455a64;">🔧 Other Features ({len(other_features)})</strong>
                    {format_features(other_features, '🔧')}
                </div>
"""
        
        html += """
            </div>
        </div>
        
        <!-- Pipeline Flow Diagram -->
        <div style="background: #f5f5f5; padding: 25px; border-radius: 12px; margin: 25px 0; text-align: center;">
            <h3 style="color: #37474f; margin: 0 0 20px 0;">🔄 Data Flow Pipeline</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 10px;">
                <div style="background: #e3f2fd; padding: 15px 25px; border-radius: 8px; border: 2px solid #2196F3;">
                    <strong>Raw Data</strong><br><span style="font-size: 12px; color: #666;">OHLCV</span>
                </div>
                <span style="font-size: 24px; color: #2196F3;">→</span>
                <div style="background: #e8f5e9; padding: 15px 25px; border-radius: 8px; border: 2px solid #4CAF50;">
                    <strong>Feature Engineering</strong><br><span style="font-size: 12px; color: #666;">Lag, Rolling, Momentum</span>
                </div>
                <span style="font-size: 24px; color: #4CAF50;">→</span>
                <div style="background: #fff3e0; padding: 15px 25px; border-radius: 8px; border: 2px solid #FF9800;">
                    <strong>Train/Holdout Split</strong><br><span style="font-size: 12px; color: #666;">Temporal Separation</span>
                </div>
                <span style="font-size: 24px; color: #FF9800;">→</span>
                <div style="background: #fce4ec; padding: 15px 25px; border-radius: 8px; border: 2px solid #E91E63;">
                    <strong>Scaling</strong><br><span style="font-size: 12px; color: #666;">StandardScaler</span>
                </div>
                <span style="font-size: 24px; color: #E91E63;">→</span>
                <div style="background: #ede7f6; padding: 15px 25px; border-radius: 8px; border: 2px solid #673AB7;">
                    <strong>Model Training</strong><br><span style="font-size: 12px; color: #666;">LSTM, DNN, etc.</span>
                </div>
            </div>
        </div>
"""
        
        # ========== FEATURE PRUNING SECTION ==========
        pruning_info = pipeline_info.get('pruning', {})
        if pruning_info:
            orig_count = pruning_info.get('original_count', 0)
            final_count = pruning_info.get('final_count', 0)
            removed = pruning_info.get('removed_features', [])
            removed_count = len(removed)
            removal_reasons = pruning_info.get('removal_reasons', {})
            
            html += f"""
        <!-- Feature Pruning Summary -->
        <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 25px; border-radius: 12px; margin: 25px 0; border-left: 5px solid #f44336;">
            <h3 style="color: #c62828; margin: 0 0 20px 0;">✂️ Feature Pruning (Train-Only Fit)</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 15px;">
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; color: #1565c0; font-weight: bold;">{orig_count}</div>
                    <div style="color: #666; font-size: 12px;">Original Features</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; color: #c62828; font-weight: bold;">{removed_count}</div>
                    <div style="color: #666; font-size: 12px;">Pruned</div>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; color: #2e7d32; font-weight: bold;">{final_count}</div>
                    <div style="color: #666; font-size: 12px;">Final Features</div>
                </div>
            </div>
"""
            if removed:
                html += """
            <details style="margin-top: 15px;">
                <summary style="cursor: pointer; color: #c62828; font-weight: bold; padding: 10px; background: white; border-radius: 5px;">
                    📋 Show Pruned Features ({} removed)
                </summary>
                <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 10px; max-height: 300px; overflow-y: auto;">
                    <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                        <tr style="background: #f5f5f5;">
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Feature</th>
                            <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Reason</th>
                        </tr>
""".format(removed_count)
                
                for feat in removed[:50]:  # Limit to first 50
                    reason = removal_reasons.get(feat, 'variance/correlation')
                    html += f"""
                        <tr>
                            <td style="padding: 6px 8px; border-bottom: 1px solid #eee;"><code>{feat}</code></td>
                            <td style="padding: 6px 8px; border-bottom: 1px solid #eee; color: #666;">{reason}</td>
                        </tr>
"""
                if len(removed) > 50:
                    html += f"""
                        <tr>
                            <td colspan="2" style="padding: 8px; text-align: center; color: #999;">
                                ... and {len(removed) - 50} more
                            </td>
                        </tr>
"""
                html += """
                    </table>
                </div>
            </details>
"""
            
            html += """
            <p style="margin: 15px 0 0 0; padding: 10px; background: white; border-radius: 8px; font-size: 12px;">
                ⚠️ <strong>Train-Only Pruning:</strong> Feature selection was performed using ONLY training data statistics to prevent holdout leakage.
            </p>
        </div>
"""
        
        return html
    
    def plot_data_pipeline(self, pipeline_info: Dict, model_name: str) -> str:
        """
        Generate data pipeline flow diagram showing data transformations.
        
        Args:
            pipeline_info: Dict with keys:
                - 'raw_data_shape': Original data shape
                - 'cleaned_shape': After cleaning (optional)
                - 'scaled_shape': After scaling
                - 'features_shape': After feature engineering
                - 'model_input_shape': Final model input shape
                - 'model_output_shape': Model output/prediction shape
                - 'model_type': 'classifier' or 'regressor'
            model_name: Name of the model
        
        Returns:
            Base64 encoded pipeline diagram image
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define pipeline steps
        steps = [
            {'title': 'Raw Data', 'shape': pipeline_info.get('raw_data_shape'), 'x': 1, 'y': 8.5, 'color': '#E8F4F8'},
            {'title': 'Data Cleaning', 'shape': pipeline_info.get('cleaned_shape'), 'x': 2.5, 'y': 8.5, 'color': '#D4E9F7'},
            {'title': 'Scaling /\nNormalization', 'shape': pipeline_info.get('scaled_shape'), 'x': 4, 'y': 8.5, 'color': '#C0DDF5'},
            {'title': 'Feature\nEngineering', 'shape': pipeline_info.get('features_shape'), 'x': 5.5, 'y': 8.5, 'color': '#ACCCF2'},
            {'title': 'Model Input', 'shape': pipeline_info.get('model_input_shape'), 'x': 7.5, 'y': 8.5, 'color': '#98B5EF', 'border': 2},
            {'title': 'Model\nProcessing', 'shape': None, 'x': 7.5, 'y': 5, 'color': '#667eea', 'text_color': 'white'},
            {'title': 'Model Output /\nPredictions', 'shape': pipeline_info.get('model_output_shape'), 'x': 7.5, 'y': 1.5, 'color': '#F1ABAB', 'border': 2},
        ]
        
        # Colors for different model types
        model_type = pipeline_info.get('model_type', 'regressor')
        if model_type == 'classifier':
            step_colors = ['#E8F4F8', '#D4E9F7', '#C0DDF5', '#ACCCF2', '#98B5EF', '#667eea', '#FFD166']
        else:
            step_colors = ['#E8F4F8', '#D4E9F7', '#C0DDF5', '#ACCCF2', '#98B5EF', '#667eea', '#A8D8EA']
        
        # Draw boxes and connections
        for i, step in enumerate(steps):
            x, y = step['x'], step['y']
            color = step.get('color', step_colors[min(i, len(step_colors)-1)])
            border_width = step.get('border', 1)
            
            # Draw rectangle
            rect = plt.Rectangle((x-0.4, y-0.6), 0.8, 1.2, 
                                 fill=True, facecolor=color, edgecolor='#333', linewidth=border_width)
            ax.add_patch(rect)
            
            # Add title
            text_color = step.get('text_color', '#333')
            ax.text(x, y + 0.25, step['title'], ha='center', va='center', 
                   fontsize=10, weight='bold', color=text_color)
            
            # Add shape information if available
            if step['shape']:
                shape_text = self._format_shape(step['shape'])
                ax.text(x, y - 0.25, shape_text, ha='center', va='center',
                       fontsize=8, color=text_color, family='monospace')
            
            # Draw arrow to next step (except last step)
            if i < len(steps) - 2:  # Don't draw arrow from model processing
                next_x = steps[i + 1]['x']
                next_y = steps[i + 1]['y']
                
                # Horizontal arrow
                if abs(next_y - y) < 0.5:  # Horizontal
                    ax.arrow(x + 0.45, y, next_x - x - 0.9, 0,
                            head_width=0.2, head_length=0.15, fc='#333', ec='#333', linewidth=1.5)
            
            # Special arrows for model processing
            if i == 4:  # From model input to model processing
                ax.arrow(x, y - 0.65, 0, -2.0,
                        head_width=0.2, head_length=0.15, fc='#333', ec='#333', linewidth=2)
            elif i == 5:  # From model processing to output
                ax.arrow(x, y - 0.65, 0, -2.0,
                        head_width=0.2, head_length=0.15, fc='#333', ec='#333', linewidth=2)
        
        # Add title
        title = f"Data Pipeline - {model_name.upper()}"
        ax.text(5, 9.7, title, ha='center', va='top', fontsize=14, weight='bold')
        
        # Add model type info
        model_info = f"Model Type: {model_type.capitalize()}"
        ax.text(5, 9.3, model_info, ha='center', va='top', fontsize=10, style='italic', color='#666')
        
        # Add legend for dimensions format
        legend_text = "Dimensions Format: (Samples × Features) or (Batch × Time × Features)"
        ax.text(5, 0.3, legend_text, ha='center', va='bottom', fontsize=8, style='italic', color='#888')
        
        plt.tight_layout()
        base64_img = self.figure_to_base64()
        return base64_img
    
    @staticmethod
    def _format_shape(shape) -> str:
        """Format shape tuple to readable string."""
        if shape is None:
            return ""
        if isinstance(shape, (tuple, list)):
            # Format as (dim1 × dim2 × dim3)
            return "(" + " × ".join(str(int(s)) for s in shape) + ")"
        return str(shape)
    
    def _generate_llm_review_section(self, benchmark_results: dict) -> str:
        """
        Generate LLM Model Review section with actual LLM response or prompt template.
        
        This section shows the AI-generated review if available, otherwise provides
        a formatted summary that can be fed to an LLM for analysis.
        """
        html = """
            <h2>🤖 LLM Model Review</h2>
"""
        
        # Check if we have an actual LLM response
        llm_review = benchmark_results.get('llm_review')
        
        if llm_review and llm_review.get('content'):
            # Show actual LLM response
            model_name = llm_review.get('model', 'Unknown Model')
            duration = llm_review.get('duration_s', 0)
            content = llm_review.get('content', '')
            
            # Escape HTML and convert markdown-ish formatting
            import html as html_module
            content_escaped = html_module.escape(content)
            # Basic markdown conversion
            content_html = content_escaped.replace('\n\n', '</p><p>').replace('\n', '<br>')
            content_html = f'<p>{content_html}</p>'
            
            html += f"""
            <div class="section" style="background: linear-gradient(135deg, #1e3a5f 0%, #2c3e50 100%); 
                        padding: 25px; border-radius: 12px; border-left: 5px solid #3498db;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="color: #3498db; margin: 0;">✅ AI Analysis Complete</h3>
                    <div style="background: rgba(52, 152, 219, 0.2); padding: 8px 15px; border-radius: 20px;">
                        <span style="color: #3498db; font-size: 12px;">
                            🤖 {model_name} • ⏱️ {duration:.1f}s
                        </span>
                    </div>
                </div>
                
                <div style="background: #0d1b2a; color: #e0e0e0; padding: 20px; 
                            border-radius: 8px; font-size: 14px; line-height: 1.6;">
                    {content_html}
                </div>
            </div>
"""
        elif llm_review and llm_review.get('error'):
            # Show error message
            error = llm_review.get('error', 'Unknown error')
            html += f"""
            <div class="section" style="background: linear-gradient(135deg, #4a1c1c 0%, #3d1515 100%); 
                        padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c;">
                <h3 style="color: #e74c3c; margin-top: 0;">⚠️ LLM Review Failed</h3>
                <p style="color: #f5a8a8;">Error: {error}</p>
                <p style="color: #999; font-size: 12px; margin-top: 10px;">
                    To enable LLM review, ensure Ollama is running: <code>ollama serve</code>
                </p>
            </div>
"""
        else:
            # Show prompt template for manual use
            html += """
            <div class="section">
                <p style="color: #666; margin-bottom: 20px;">
                    LLM review not enabled. To enable, set <code>llm.enabled: true</code> in configuration.yml
                    and ensure Ollama is running locally.
                </p>
                
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2c3e50 100%); 
                            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #3498db; margin-top: 0;">📊 Model Performance Summary</h3>
                    <pre style="background: #0d1b2a; color: #e0e0e0; padding: 15px; 
                                border-radius: 5px; overflow-x: auto; font-size: 12px;">
"""
        
            # Build classifier summary
            summary_lines = ["=== CLASSIFIER PERFORMANCE ==="]
            if 'classifiers' in benchmark_results:
                for name, metrics in benchmark_results['classifiers'].items():
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_score', 0)
                    summary_lines.append(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")
            
            summary_lines.append("\n=== REGRESSOR PERFORMANCE (Multi-Step) ===")
            if 'regressors' in benchmark_results:
                for name, metrics in benchmark_results['regressors'].items():
                    r2 = metrics.get('r2_avg', metrics.get('r2', 0))
                    rmse = metrics.get('rmse_avg', metrics.get('rmse', 0))
                    summary_lines.append(f"{name}: R²_avg={r2:.4f}, RMSE_avg=${rmse:.2f}")
            
            # Add holdout performance
            summary_lines.append("\n=== HOLDOUT VALIDATION (Direct Prediction) ===")
            if 'holdout_forecasts' in benchmark_results:
                for name, data in benchmark_results['holdout_forecasts'].items():
                    if 'metrics' in data:
                        metrics = data['metrics']
                        r2 = metrics.get('r2_avg', 0)
                        rmse = metrics.get('rmse_avg', 0)
                        beats = '✓' if data.get('beats_naive', False) else '✗'
                        summary_lines.append(f"{name}: R²_avg={r2:.4f}, RMSE_avg=${rmse:.2f}, Beats Naive={beats}")
            
            # Add baselines
            summary_lines.append("\n=== BASELINES ===")
            if 'baselines' in benchmark_results:
                for name, data in benchmark_results['baselines'].items():
                    rmse = data.get('metrics', {}).get('rmse_avg', 0)
                    summary_lines.append(f"{name}: RMSE_avg=${rmse:.2f}")
            
            html += "\n".join(summary_lines)
            html += """
                    </pre>
                </div>
                
                <div style="background: linear-gradient(135deg, #2d4a3e 0%, #1e3a2f 100%); 
                            padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #2ecc71; margin-top: 0;">💡 Analysis Prompt</h3>
                    <div style="background: #0d2818; color: #a8d5ba; padding: 15px; 
                                border-radius: 5px; font-family: monospace; font-size: 12px;">
                        <p><strong>Copy this prompt for manual LLM analysis:</strong></p>
                        <textarea readonly style="width: 100%; height: 150px; background: #1a3d2a; 
                                                 color: #a8d5ba; border: 1px solid #2ecc71; 
                                                 border-radius: 5px; padding: 10px; font-family: monospace;">
Analyze this multi-step financial forecasting model performance:

FORECAST MODE: Multi-output (10 steps ahead, direct prediction)
MODELS TESTED:
- Classification: DNN, SVC, Random Forest (trend direction)
- Regression: LSTM, DNN, KRR, Linear (multi-output price prediction)

KEY METRICS TO EVALUATE:
- Holdout R²: Should be > 0 to beat naive baseline
- RMSE per step: Error should grow with horizon
- Beats Naive: Must outperform last-price baseline

QUESTIONS TO ANSWER:
1. Which models generalize best on holdout data?
2. How does error accumulate across the forecast horizon?
3. Are there signs of overfitting (high train R², low holdout R²)?
4. What ensemble strategies might improve predictions?
5. Should we use direct multi-output or recursive single-step?
                        </textarea>
                    </div>
                </div>
            </div>
"""
        return html
    
    def _generate_artifacts_section(self, artifacts: dict) -> str:
        """
        Generate a section listing run artifacts (config, pruning decisions, etc.)
        
        Args:
            artifacts: Dict mapping artifact_name -> file_path
            
        Returns:
            HTML string for the artifacts section
        """
        if not artifacts:
            return ""
            
        html = """
            <div class="section" style="border-left: 5px solid #3498db;">
                <h2 class="section-title" style="color: #3498db;">📦 Run Artifacts</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Files generated during this run for reproducibility and debugging.
                </p>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
                        <tr style="background: #ecf0f1;">
                            <th style="text-align: left; padding: 12px; border-bottom: 2px solid #bdc3c7;">Artifact</th>
                            <th style="text-align: left; padding: 12px; border-bottom: 2px solid #bdc3c7;">Path</th>
                        </tr>
"""
        
        artifact_icons = {
            'config_snapshot': '⚙️',
            'feature_pruner': '✂️',
            'benchmark_results': '📊',
        }
        
        for name, path in artifacts.items():
            icon = artifact_icons.get(name, '📄')
            # Get just the filename for display
            import os
            filename = os.path.basename(str(path)) if path else 'N/A'
            html += f"""
                        <tr>
                            <td style="padding: 10px; border-bottom: 1px solid #ecf0f1;">
                                {icon} <strong>{name.replace('_', ' ').title()}</strong>
                            </td>
                            <td style="padding: 10px; border-bottom: 1px solid #ecf0f1;">
                                <code style="background: #e8e8e8; padding: 3px 8px; border-radius: 4px;">{filename}</code>
                            </td>
                        </tr>
"""
        
        html += """
                    </table>
                </div>
                <p style="margin-top: 15px; color: #7f8c8d; font-size: 12px;">
                    💡 <strong>Tip:</strong> These files can be found in the output directory alongside this report.
                </p>
            </div>
"""
        return html

    def _generate_model_errors_section(self, model_errors: dict) -> str:
        """
        Generate a section listing failed models with their error details.
        
        Args:
            model_errors: Dict mapping model_name -> {'exception': str, 'exception_type': str, 'traceback': str}
            
        Returns:
            HTML string for the errors section
        """
        if not model_errors:
            return ""
            
        html = """
            <div class="section" style="border-left: 5px solid #e74c3c;">
                <h2 class="section-title" style="color: #e74c3c;">⚠️ Failed Models</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    The following models encountered errors during training. 
                    Review the errors below for debugging.
                </p>
"""
        for model_name, error_info in model_errors.items():
            exception_type = error_info.get('exception_type', 'Exception')
            exception_msg = error_info.get('exception', 'Unknown error')
            traceback_str = error_info.get('traceback', '')
            
            # Escape HTML characters in traceback
            traceback_escaped = (traceback_str
                                .replace('&', '&amp;')
                                .replace('<', '&lt;')
                                .replace('>', '&gt;')
                                .replace('\n', '<br>'))
            
            html += f"""
                <div style="background: #fdf2f2; border: 1px solid #f8d7da; 
                            padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color: #c0392b; margin: 0 0 10px 0;">
                        ❌ {model_name.upper()}
                    </h4>
                    <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <strong style="color: #e74c3c;">{exception_type}:</strong> 
                        <span style="color: #333;">{exception_msg}</span>
                    </div>
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; color: #666; font-weight: bold;">
                            📜 Full Stack Trace
                        </summary>
                        <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; 
                                    border-radius: 5px; overflow-x: auto; font-size: 11px; 
                                    margin-top: 10px; white-space: pre-wrap; word-wrap: break-word;">
{traceback_escaped}
                        </pre>
                    </details>
                </div>
"""
        
        html += """
            </div>
"""
        return html
    
    def generate_comprehensive_report(self, benchmark_results: dict, classifier_details: dict, 
                                     regressor_details: dict, ticker: str = 'Asset', 
                                     forecast_horizon: int = 30, y_all: np.ndarray = None) -> str:
        """
        Generate comprehensive HTML report with all visualizations.
        
        Args:
            benchmark_results: Main benchmark results dict
            classifier_details: Dict with detailed classifier results (y_test, history, cm, etc.)
            regressor_details: Dict with detailed regressor results (y_test, history, etc.)
            ticker: Asset name
            forecast_horizon: Forecast horizon from config
            y_all: Full timeseries data for context plots
            ticker: Asset name
        
        Returns:
            Path to generated HTML report
        """
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Model Benchmark Report - {ticker}</title>
    <style>
        * {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 48px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .header-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }}
        
        .info-card {{
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .info-card p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        
        .info-label {{
            font-weight: 600;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-top: 40px;
            margin-bottom: 25px;
            font-size: 32px;
        }}
        
        h2:first-of-type {{
            margin-top: 0;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 22px;
        }}
        
        .section {{
            margin-bottom: 45px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 12px;
            border-left: 5px solid #667eea;
        }}
        
        /* Collapsible sections */
        details.model-details {{
            margin-bottom: 20px;
            border: 2px solid #ecf0f1;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        details.model-details summary {{
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ecf0f1 100%);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 18px;
            color: #2c3e50;
            transition: background 0.2s ease;
        }}
        
        details.model-details summary:hover {{
            background: linear-gradient(135deg, #e8f4f8 0%, #d5e8f0 100%);
        }}
        
        details.model-details[open] summary {{
            border-bottom: 2px solid #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        details.model-details .model-content {{
            padding: 25px;
            background: white;
        }}
        
        .model-summary-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge-excellent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-good {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .model-container {{
            margin-bottom: 50px;
            padding: 25px;
            background: white;
            border-radius: 12px;
            border: 2px solid #ecf0f1;
            transition: all 0.3s ease;
        }}
        
        .model-container:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            border-color: #667eea;
        }}
        
        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            margin-bottom: 15px;
        }}
        
        .model-name {{
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .model-badge {{
            display: inline-block;
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .metric.excellent {{
            background: linear-gradient(135deg, #51CF66 0%, #37B24D 100%);
        }}
        
        .metric.good {{
            background: linear-gradient(135deg, #FFD93D 0%, #FFC107 100%);
        }}
        
        .metric.poor {{
            background: linear-gradient(135deg, #FF6B6B 0%, #FA5252 100%);
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 12px;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .plot-container {{
            margin: 25px 0;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border: 1px solid #ecf0f1;
            text-align: center;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        
        .plot-title {{
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            text-align: left;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .summary {{
            background: linear-gradient(135deg, #ecf0f1 0%, #d5d8e0 100%);
            padding: 20px;
            border-left: 5px solid #667eea;
            margin: 20px 0;
            border-radius: 8px;
        }}
        
        .comparison-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 12px;
        }}
        
        .up {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .down {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .sideways {{
            color: #f39c12;
            font-weight: bold;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 28px;
            }}
            
            .header-info {{
                grid-template-columns: 1fr;
            }}
            
            .comparison-section {{
                grid-template-columns: 1fr;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Advanced Model Benchmark Report</h1>
            <div class="header-info">
                <div class="info-card">
                    <p><span class="info-label">📊 Asset:</span> {ticker}</p>
                </div>
                <div class="info-card">
                    <p><span class="info-label">📅 Generated:</span> {benchmark_results['timestamp']}</p>
                </div>
                <div class="info-card">
                    <p><span class="info-label">🔮 Forecast Horizon:</span> {benchmark_results['forecast_horizon']} steps</p>
                </div>
                <div class="info-card">
                    <p><span class="info-label">🤖 Total Models:</span> {len(benchmark_results.get('classifiers', {}))} classifiers + {len(benchmark_results.get('regressors', {}))} regressors</p>
                </div>
            </div>
        </div>
        
        <div class="content">
"""
        
        # ========== PIPELINE OVERVIEW SECTION (ONCE AT BEGINNING) ==========
        if 'pipeline_info' in benchmark_results:
            html_content += self.generate_pipeline_overview_html(benchmark_results['pipeline_info'])
        
        # ========== HOLDOUT VALIDATION SECTION ==========
        period_unit = benchmark_results.get('period_unit', 'steps')
        holdout_steps = benchmark_results.get('holdout_days', 10)
        
        # ========== ROLLING HOLDOUT AGGREGATION SECTION ==========
        if 'holdout_rolling' in benchmark_results and benchmark_results['holdout_rolling']:
            holdout_rolling = benchmark_results['holdout_rolling']
            if 'error' not in holdout_rolling:
                # Generate the rolling aggregation plot
                rolling_plot = self.plot_rolling_holdout_aggregation(holdout_rolling, ticker, holdout_steps)
                
                # Get aggregation method
                agg_method = 'mean'
                for m_name, m_data in holdout_rolling.items():
                    if 'agg_method' in m_data:
                        agg_method = m_data['agg_method']
                        break
                
                html_content += f"""
            <h2>🔬 Holdout Validation: Rolling Forecast Aggregation ({holdout_steps} Steps)</h2>
            <div class="summary">
                <strong>True Out-of-Sample Validation:</strong> Predictions on data NEVER seen during training.
                <br><strong>Aggregation Method:</strong> {agg_method.upper()} of overlapping multi-step predictions
                <br><strong>Purpose:</strong> Each timestep receives predictions from multiple origin points. We aggregate them to get a single forecast with uncertainty band.
                <br><strong>Advantage:</strong> Smoother predictions, reduced variance, natural uncertainty quantification.
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Rolling Aggregated Forecasts with Uncertainty Bands (±1 std)</div>
                <img src="data:image/png;base64,{rolling_plot}" alt="Rolling Holdout Aggregation">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Aggregated RMSE</th>
                        <th>Aggregated MAE</th>
                        <th>Aggregated R²</th>
                        <th>Avg Uncertainty (Std)</th>
                    </tr>
                </thead>
                <tbody>
"""
                # Add model rows
                for model_name, data in holdout_rolling.items():
                    if 'error' in data:
                        continue
                    metrics = data.get('metrics', {})
                    rmse = metrics.get('rmse', 0)
                    mae = metrics.get('mae', 0)
                    r2 = metrics.get('r2', 0)
                    # Try multiple sources for uncertainty: step_std, daily_std, or std from predictions
                    std_values = data.get('step_std', data.get('daily_std', []))
                    if len(std_values) > 0 and not np.all(np.isnan(std_values)):
                        avg_std = np.nanmean(std_values)
                    else:
                        # Fallback: compute from prediction variance if available
                        preds = data.get('step_predictions', data.get('daily_predictions', []))
                        if len(preds) > 0:
                            avg_std = np.std(preds)
                        else:
                            avg_std = 0.0
                    
                    # Format uncertainty display
                    uncertainty_str = f'±${avg_std:.2f}' if avg_std > 0 and not np.isnan(avg_std) else 'N/A'
                    
                    html_content += f"""
                    <tr>
                        <td><strong>{model_name.upper()}</strong></td>
                        <td>${rmse:.2f}</td>
                        <td>${mae:.2f}</td>
                        <td>{r2:.4f}</td>
                        <td>{uncertainty_str}</td>
                    </tr>
"""
                
                # Add rolling baselines if available
                if 'rolling_baselines' in benchmark_results:
                    for baseline_name, baseline_data in benchmark_results['rolling_baselines'].items():
                        if 'metrics' in baseline_data:
                            b_metrics = baseline_data['metrics']
                            html_content += f"""
                    <tr style="background: #f8f9fa;">
                        <td><em>{baseline_name}</em></td>
                        <td>${b_metrics.get('rmse', 0):.2f}</td>
                        <td>${b_metrics.get('mae', 0):.2f}</td>
                        <td>{b_metrics.get('r2', 0):.4f}</td>
                        <td>-</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
"""
        
        # ========== RECURSIVE HOLDOUT FORECAST SECTION (TRUE AUTOREGRESSIVE) ==========
        if 'recursive_forecasts' in benchmark_results and benchmark_results['recursive_forecasts']:
            recursive_forecasts = benchmark_results['recursive_forecasts']
            if 'error' not in recursive_forecasts:
                holdout_actual = np.array(benchmark_results.get('holdout_actual_prices', []))
                holdout_actual_transformed = benchmark_results.get('holdout_actual_transformed', [])
                target_transform = benchmark_results.get('target_transform', 'price')
                holdout_days = benchmark_results.get('holdout_days', len(holdout_actual))
                
                if len(holdout_actual) > 0:
                    # Generate the recursive holdout forecast plot
                    recursive_plot = self.plot_recursive_holdout_forecast(
                        recursive_forecasts, holdout_actual, ticker, period_unit
                    )
                    
                    # Get model horizon from first valid model
                    model_horizon = 5
                    for m_data in recursive_forecasts.values():
                        if 'model_horizon' in m_data:
                            model_horizon = m_data['model_horizon']
                            break
                    
                    html_content += f"""
            <h2>🔄 Recursive Holdout Forecast (True Autoregressive)</h2>
            <div class="summary">
                <strong>Real-World Scenario:</strong> Models predict {model_horizon} steps ahead, then predictions are fed back to forecast further.
                <br><strong>Method:</strong> Each model uses its native horizon, feeding predictions back to continue beyond.
                <br><strong>Key Insight:</strong> This shows how errors compound over time - the ultimate test of model stability.
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Recursive Forecast: Predictions Fed Back as Inputs</div>
                <img src="data:image/png;base64,{recursive_plot}" alt="Recursive Holdout Forecast">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>RMSE ($)</th>
                        <th>MAE ($)</th>
                        <th>R²</th>
                        <th>Chunks</th>
                        <th>Method</th>
                    </tr>
                </thead>
                <tbody>
"""
                    # Add model rows sorted by R²
                    sorted_models = sorted(
                        [(name, data) for name, data in recursive_forecasts.items() if 'metrics' in data],
                        key=lambda x: x[1].get('metrics', {}).get('r2', 0),
                        reverse=True
                    )
                    
                    for model_name, data in sorted_models:
                        metrics = data.get('metrics', {})
                        rmse = metrics.get('rmse', 0)
                        mae = metrics.get('mae', 0)
                        r2 = metrics.get('r2', 0)
                        n_chunks = data.get('n_chunks', 0)
                        method = data.get('method', 'RECURSIVE')
                        
                        # Color based on R²
                        if r2 >= 0.5:
                            r2_color = "#27ae60"
                        elif r2 >= 0:
                            r2_color = "#f39c12"
                        else:
                            r2_color = "#e74c3c"
                        
                        html_content += f"""
                    <tr>
                        <td><strong>{model_name.upper()}</strong></td>
                        <td>${rmse:.2f}</td>
                        <td>${mae:.2f}</td>
                        <td style="color: {r2_color}; font-weight: bold;">{r2:.4f}</td>
                        <td>{n_chunks}</td>
                        <td style="font-size: 11px;">{method}</td>
                    </tr>
"""
                    
                    html_content += """
                </tbody>
            </table>
"""
                    
                    # Generate dual-space plots for each model
                    if len(holdout_actual_transformed) > 0:
                        html_content += f"""
            <h3>📈 Dual-Space Analysis (Training Space vs Price Space)</h3>
            <div class="summary">
                <strong>Training Space:</strong> {target_transform.replace('_', ' ').title()} - what the model was trained to predict
                <br><strong>Price Space:</strong> Absolute prices - what we actually care about
            </div>
"""
                        holdout_actual_trans_arr = np.array(holdout_actual_transformed)
                        
                        for model_name, data in sorted_models[:5]:  # Top 5 models only
                            if 'error' in data:
                                continue
                            
                            predictions_price = np.array(data.get('predictions', []))
                            predictions_transformed = np.array(data.get('predictions_transformed', []))
                            
                            if len(predictions_price) > 0 and len(predictions_transformed) > 0:
                                try:
                                    dual_plot = self.plot_dual_space_comparison(
                                        model_name=model_name,
                                        predictions_price=predictions_price,
                                        actual_price=holdout_actual,
                                        predictions_transformed=predictions_transformed,
                                        actual_transformed=holdout_actual_trans_arr,
                                        transform_name=target_transform,
                                        ticker=ticker
                                    )
                                    html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{model_name.upper()}: Price Space vs Training Space ({target_transform})</div>
                <img src="data:image/png;base64,{dual_plot}" alt="{model_name} Dual Space">
            </div>
"""
                                except Exception as e:
                                    logger.warning(f"Could not generate dual-space plot for {model_name}: {e}")
        
        # ========== TREND ASSESSMENT SECTION ==========
        forecast_horizon = benchmark_results.get('forecast_horizon', 10)
        if 'trend_10day' in benchmark_results and benchmark_results['trend_10day']:
            trend_data = benchmark_results['trend_10day']
            
            # Generate the trend assessment plot
            trend_plot = self.plot_trend_assessment(trend_data, ticker, forecast_horizon)
            
            # Get expected % change info
            pct_change_info = trend_data.get('expected_pct_change', {})
            expected_pct = pct_change_info.get('value', 0)
            source_model = pct_change_info.get('source_model', 'N/A')
            current_price = pct_change_info.get('current_price', 0)
            predicted_price = pct_change_info.get('predicted_price_final', pct_change_info.get('predicted_price_day_10', 0))
            
            pct_color = '#27ae60' if expected_pct > 0 else '#e74c3c' if expected_pct < 0 else '#f39c12'
            pct_arrow = '↑' if expected_pct > 0 else '↓' if expected_pct < 0 else '→'
            
            html_content += f"""
            <h2>🎯 {forecast_horizon}-Step Trend Assessment</h2>
            <div class="summary">
                <strong>Combined Analysis:</strong> Classifier trend predictions + Regressor expected % change
                <br><strong>Horizon:</strong> {forecast_horizon} steps ahead
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                <div style="background: linear-gradient(135deg, {pct_color}22 0%, {pct_color}44 100%); padding: 25px; border-radius: 12px; border-left: 5px solid {pct_color};">
                    <h3 style="color: {pct_color}; margin: 0 0 15px 0;">Expected {forecast_horizon}-Step Change</h3>
                    <div style="font-size: 48px; font-weight: bold; color: {pct_color}; text-align: center;">
                        {pct_arrow} {expected_pct:+.2f}%
                    </div>
                    <div style="text-align: center; margin-top: 15px;">
                        <span style="font-size: 18px;">${current_price:.2f} → ${predicted_price:.2f}</span>
                        <br><span style="font-size: 12px; color: #666;">Source: {source_model}</span>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px;">
                    <h3 style="margin: 0 0 15px 0;">Classifier Consensus</h3>
                    <table style="margin: 0;">
                        <tr><th>Model</th><th>Trend</th><th>Confidence</th></tr>
"""
            # Add classifier rows
            for model_name, model_data in trend_data.items():
                if model_name == 'expected_pct_change' or 'error' in model_data:
                    continue
                trend_label = model_data.get('trend_label', 'HOLD')
                proba = model_data.get('probabilities', {})
                
                trend_color = '#27ae60' if trend_label == 'UP' else '#e74c3c' if trend_label == 'DOWN' else '#f39c12'
                max_proba = max(proba.values()) if proba else 0
                
                html_content += f"""
                        <tr>
                            <td>{model_name.upper()}</td>
                            <td style="color: {trend_color}; font-weight: bold;">{trend_label}</td>
                            <td>{max_proba*100:.1f}%</td>
                        </tr>
"""
            
            html_content += """
                    </table>
                </div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Trend Assessment Visualization</div>
                <img src="data:image/png;base64,""" + trend_plot + """" alt="Trend Assessment">
            </div>
"""
        
        # ========== MODEL COMPARISON OVERVIEW (NEW) ==========
        # Get reporting config
        reporting_config = benchmark_results.get('config', {}).get('reporting', {})
        
        # Add model comparison section if we have regressors
        if regressor_details and len(regressor_details) > 0:
            html_content += f"""
            <h2>📊 Model Comparison Overview</h2>
            <div class="summary">
                <strong>Quick Comparison:</strong> Side-by-side analysis of all trained models.
                <br><strong>Purpose:</strong> Identify best-performing models and understand their characteristics.
            </div>
"""
            
            # Show model ranking chart if configured
            if reporting_config.get('show_model_ranking_chart', True):
                try:
                    ranking_plot = self.plot_model_ranking_bar(benchmark_results, metric='r2', 
                                                                regressor_details=regressor_details)
                    # Get target transform for title
                    target_transform = 'log_return'
                    for data in regressor_details.values():
                        if 'target_transform' in data:
                            target_transform = data['target_transform']
                            break
                    transform_label = target_transform.replace('_', ' ').title()
                    html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Model Ranking by R² Score ({transform_label} Space - All Steps)</div>
                <img src="data:image/png;base64,{ranking_plot}" alt="Model Ranking">
            </div>
"""
                except Exception as e:
                    logger.warning(f"Could not generate ranking plot: {e}")
            
            # Show predicted vs actual scatter plots in transformed space (log_return/pct_change)
            # This shows what the models actually learn - the relationship between predicted and actual returns
            if reporting_config.get('show_predictions_vs_actual', True):
                try:
                    scatter_plots = self.plot_predicted_vs_actual_scatter_grid(regressor_details, forecast_horizon)
                    for step_idx, scatter_plot in enumerate(scatter_plots):
                        step_num = step_idx + 1
                        html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Predicted vs Actual (Step {step_num}) - Training Space (Log Return / Pct Change)</div>
                <img src="data:image/png;base64,{scatter_plot}" alt="Scatter Step {step_num}">
            </div>
"""
                except Exception as e:
                    logger.warning(f"Could not generate scatter plots: {e}")
            
            # Show predicted vs actual scatter plots in SCALED space (StandardScaler z-scores)
            # This is what models ACTUALLY see during training - the normalized values
            if reporting_config.get('show_predictions_vs_actual', True):
                try:
                    scaled_scatter_plots = self.plot_predicted_vs_actual_scaled_grid(regressor_details, forecast_horizon)
                    for step_idx, scatter_plot in enumerate(scaled_scatter_plots):
                        step_num = step_idx + 1
                        html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Predicted vs Actual (Step {step_num}) - Scaled Space (StandardScaler z-score)</div>
                <img src="data:image/png;base64,{scatter_plot}" alt="Scaled Scatter Step {step_num}">
            </div>
"""
                except Exception as e:
                    logger.warning(f"Could not generate scaled scatter plots: {e}")
            
            # Show error distribution comparison
            if reporting_config.get('show_error_distribution', True):
                try:
                    error_dist_plot = self.plot_error_distribution_comparison(regressor_details)
                    # Get target transform for title
                    target_transform = 'log_return'
                    for data in regressor_details.values():
                        if 'target_transform' in data:
                            target_transform = data['target_transform']
                            break
                    transform_label = target_transform.replace('_', ' ').title()
                    html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Prediction Error Distribution ({transform_label} Space)</div>
                <img src="data:image/png;base64,{error_dist_plot}" alt="Error Distribution">
            </div>
"""
                except Exception as e:
                    logger.warning(f"Could not generate error distribution plot: {e}")
            
            # Show per-step metrics
            if reporting_config.get('show_per_step_metrics', True):
                try:
                    per_step_plot = self.plot_per_step_metrics(regressor_details, forecast_horizon)
                    html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Error Accumulation: RMSE by Forecast Step</div>
                <img src="data:image/png;base64,{per_step_plot}" alt="Per-Step Metrics">
            </div>
"""
                except Exception as e:
                    logger.warning(f"Could not generate per-step metrics plot: {e}")
            
            # Show directional accuracy
            try:
                dir_acc_plot = self.plot_directional_accuracy(regressor_details)
                html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Directional Accuracy (Correct Up/Down Predictions)</div>
                <img src="data:image/png;base64,{dir_acc_plot}" alt="Directional Accuracy">
            </div>
"""
            except Exception as e:
                logger.warning(f"Could not generate directional accuracy plot: {e}")
            
            # Show multi-step heatmap
            try:
                heatmap_plot = self.plot_multistep_heatmap(regressor_details, forecast_horizon)
                # Get target transform for title
                target_transform = 'log_return'
                for data in regressor_details.values():
                    if 'target_transform' in data:
                        target_transform = data['target_transform']
                        break
                transform_label = target_transform.replace('_', ' ').title()
                html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Multi-Step R² Heatmap ({transform_label} Space)</div>
                <img src="data:image/png;base64,{heatmap_plot}" alt="Multi-Step Heatmap">
            </div>
"""
            except Exception as e:
                logger.warning(f"Could not generate multi-step heatmap: {e}")
            
            # Show feature importance if available
            if reporting_config.get('show_feature_importance', True):
                feature_importance = benchmark_results.get('feature_importance')
                if feature_importance:
                    try:
                        top_k = reporting_config.get('feature_importance_top_k', 20)
                        fi_plot = self.plot_feature_importance(benchmark_results, top_k)
                        html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Top {top_k} Feature Importances</div>
                <img src="data:image/png;base64,{fi_plot}" alt="Feature Importance">
            </div>
"""
                    except Exception as e:
                        logger.warning(f"Could not generate feature importance plot: {e}")
        
        # ========== CLASSIFIER SECTION ==========
        # Only show classifier section if there are classifiers
        if benchmark_results.get('classifiers'):
            html_content += f"""
            <h2>📊 Classifier Results - Trend Classification</h2>
            <div class="summary">
                <strong>Task:</strong> Predict price trend direction (UP/DOWN/SIDEWAYS) based on technical indicators.
                <br><strong>Models Trained:</strong> {len(benchmark_results['classifiers'])} classifiers
            </div>
"""
        
            # Classifier comparison chart
            clf_comparison = self.plot_classifier_comparison(benchmark_results['classifiers'])
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Model Performance Comparison</div>
                <img src="data:image/png;base64,{clf_comparison}" alt="Classifier Comparison">
            </div>
"""
        
            # Individual classifier results
            for clf_name, clf_data in benchmark_results['classifiers'].items():
                trend = clf_data['trend_prediction']
                trend_class = trend.lower()
                accuracy_pct = clf_data['accuracy'] * 100
                
                # Determine metric color
                if accuracy_pct >= 75:
                    metric_class = 'excellent'
                elif accuracy_pct >= 60:
                    metric_class = 'good'
                else:
                    metric_class = 'poor'
                
                html_content += f"""
            <div class="model-container">
                <div class="model-header">
                    <div>
                        <div class="model-name">{clf_name.upper()}</div>
                        <p style="color: #7f8c8d; margin-top: 5px;">Classifier</p>
                    </div>
                    <span class="model-badge">{trend}</span>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric {metric_class}">
                        <div class="metric-value">{accuracy_pct:.2f}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{clf_data['f1_score']:.4f}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>
"""
                
                # Add classifier-specific plots
                if clf_name in classifier_details:
                    details = classifier_details[clf_name]
                    
                    # Confusion matrix
                    if 'cm' in details and 'y_test' in details:
                        cm_plot = self.plot_confusion_matrix(details['cm'], clf_name)
                        html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">Confusion Matrix</div>
                    <img src="data:image/png;base64,{cm_plot}" alt="Confusion Matrix">
                </div>
"""
                    
                    # Training history (for neural networks)
                    if 'history' in details:
                        loss_plot = self.plot_training_history(details['history'], 'loss', clf_name)
                        accuracy_plot = self.plot_training_history(details['history'], 'accuracy', clf_name)
                        html_content += f"""
                <div class="comparison-section">
                    <div class="plot-container">
                        <div class="plot-title">Training Loss Curve</div>
                        <img src="data:image/png;base64,{loss_plot}" alt="Loss Curve">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">Training Accuracy Curve</div>
                        <img src="data:image/png;base64,{accuracy_plot}" alt="Accuracy Curve">
                    </div>
                </div>
"""
                    
                    # NOTE: Pipeline visualization moved to beginning of report (single instance)
                
                html_content += """
            </div>
"""
        
        # ========== REGRESSOR SECTION ==========
        period_unit = benchmark_results.get('period_unit', 'steps')
        html_content += f"""
            <h2>📈 Regressor Results - Price Forecasting</h2>
            <div class="summary">
                <strong>Task:</strong> Forecast future price values for the next {benchmark_results['forecast_horizon']} steps.
                <br><strong>Models Trained:</strong> {len(benchmark_results['regressors'])} regressors
                <br><strong>Forecasting Method:</strong> TRUE RECURSIVE (each prediction uses prior predictions)
            </div>
            
            <div class="section" style="background: #e8f5e9; border-left-color: #4CAF50;">
                <h3 style="color: #2e7d32; margin-top: 0;">🔬 Expert Pipeline Methodology</h3>
                <p style="margin-bottom: 15px;">This benchmark uses <strong>Close-only features</strong> and <strong>true recursive forecasting</strong>:</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
                        <strong>📊 Features (Input)</strong>
                        <ul style="margin: 10px 0 0 20px; font-size: 13px;">
                            <li>Close_lag_1, Close_lag_5, Close_lag_10, Close_lag_20</li>
                            <li>Close_return_1, Close_return_5, Close_return_10</li>
                            <li>Close_rolling_5_mean, Close_rolling_20_mean</li>
                            <li>Close_rolling_5_std, Close_rolling_20_std</li>
                            <li>Close_ROC_10, Close_momentum_10</li>
                        </ul>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800;">
                        <strong>🎯 Target (Output)</strong>
                        <p style="margin: 10px 0 0 0; font-size: 13px;">
                            Next timestep's Close price (Close_t+1)
                        </p>
                        <p style="margin: 10px 0 0 0; font-size: 13px;">
                            <strong>No data leakage:</strong> Only past Close values used in features
                        </p>
                    </div>
                </div>
                
                <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <strong>🔮 Recursive Multi-Step Forecasting</strong>
                    <ol style="margin: 10px 0 0 20px; font-size: 13px;">
                        <li><strong>Step 1:</strong> Model uses actual Close history → Predicts Close_t+1</li>
                        <li><strong>Step 2:</strong> Model uses history + Step 1 prediction → Predicts Close_t+2</li>
                        <li><strong>Step N:</strong> Model uses history + all prior predictions → Predicts Close_t+N</li>
                    </ol>
                    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
                        ✅ This is the correct approach for multi-step ahead forecasting where features depend on prior outputs.
                    </p>
                </div>
            </div>
"""
        
        # Regressor comparison chart
        if benchmark_results['regressors']:
            reg_comparison = self.plot_regressor_comparison(benchmark_results['regressors'])
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Model Performance Comparison (R² Score)</div>
                <img src="data:image/png;base64,{reg_comparison}" alt="Regressor Comparison">
            </div>
"""
        
        # Regressor ranking table
        html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>R² Score</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        sorted_regressors = sorted(benchmark_results['regressors'].items(), 
                                  key=lambda x: x[1]['r2'], reverse=True)
        for rank, (reg_name, reg_data) in enumerate(sorted_regressors, 1):
            r2 = reg_data['r2']
            status_emoji = '⭐' if r2 > 0.9 else '✓' if r2 > 0.7 else '⚠️'
            html_content += f"""
                    <tr>
                        <td><strong>#{rank} {status_emoji}</strong></td>
                        <td><strong>{reg_name.upper()}</strong></td>
                        <td>{r2:.4f}</td>
                        <td>{reg_data['rmse']:.6f}</td>
                        <td>{reg_data['mae']:.6f}</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
"""
        
        # Check if collapsible sections are enabled
        use_collapsible = reporting_config.get('use_collapsible_sections', True)
        
        # Individual regressor results
        html_content += """
            <h3>📋 Individual Model Details</h3>
            <p style="color: #666; margin-bottom: 20px;">Click on each model to expand and view detailed metrics and plots.</p>
"""
        
        for reg_name, reg_data in sorted_regressors:
            r2 = reg_data['r2']
            rmse = reg_data['rmse']
            
            if r2 > 0.9:
                metric_class = 'excellent'
                badge_class = 'badge-excellent'
                badge_text = '⭐ Excellent'
            elif r2 > 0.7:
                metric_class = 'good'
                badge_class = 'badge-good'
                badge_text = '✓ Good'
            else:
                metric_class = 'poor'
                badge_class = 'badge-poor'
                badge_text = '⚠️ Needs Work'
            
            if use_collapsible:
                html_content += f"""
            <details class="model-details">
                <summary>
                    <span>{reg_name.upper()} <span class="model-summary-badge {badge_class}">{badge_text}</span></span>
                    <span style="font-size: 14px; font-weight: normal;">R²={r2:.4f} | RMSE=${rmse:.2f}</span>
                </summary>
                <div class="model-content">
"""
            else:
                html_content += f"""
            <div class="model-container">
                <div class="model-header">
                    <div>
                        <div class="model-name">{reg_name.upper()}</div>
                        <p style="color: #7f8c8d; margin-top: 5px;">Regressor</p>
                    </div>
                </div>
"""
            
            html_content += f"""
                <div class="metrics-grid">
                    <div class="metric {metric_class}">
                        <div class="metric-value">{r2:.4f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{reg_data['rmse']:.6f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{reg_data['mae']:.6f}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                </div>
"""
            
            # Add regressor-specific plots
            if reg_name in regressor_details:
                details = regressor_details[reg_name]
                
                # Support both 'y_val' and 'y_test' keys for compatibility
                y_actual_key = 'y_val' if 'y_val' in details else 'y_test'
                
                # Actual vs Predicted scatter plot
                if 'y_pred' in details and y_actual_key in details:
                    scatter_plot = self.plot_regression_predictions(
                        details[y_actual_key], details['y_pred'], reg_name
                    )
                    html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">Actual vs Predicted Values (Original Price Range)</div>
                    <img src="data:image/png;base64,{scatter_plot}" alt="Actual vs Predicted">
                </div>
"""
                    
                    # Time series plot
                    ts_plot = self.plot_timeseries_forecast(
                        details[y_actual_key], details['y_pred'], reg_name
                    )
                    html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">Time Series Forecast (Test Data)</div>
                    <img src="data:image/png;base64,{ts_plot}" alt="Time Series Forecast">
                </div>
"""
                    
                    # Residual plot
                    residuals_plot = self.plot_residuals(
                        details[y_actual_key], details['y_pred'], reg_name
                    )
                    html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">Residual Analysis</div>
                    <img src="data:image/png;base64,{residuals_plot}" alt="Residuals">
                </div>
"""
                    
                    # Test case plots (5 samples showing 90-day history + forecast)
                    if y_all is not None:
                        num_samples = reporting_config.get('num_sample_forecasts', 5)
                        test_case_plots = self.plot_regressor_test_cases(
                            details[y_actual_key], details['y_pred'], y_all, 
                            forecast_horizon=forecast_horizon, model_name=reg_name,
                            num_plots=num_samples
                        )
                        html_content += f"""
                <div style="margin-top: 30px;">
                    <div style="font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 20px;">
                        📊 Test Cases: History + Forecast ({num_samples} Samples)
                    </div>
"""
                        for i, plot_img in enumerate(test_case_plots, 1):
                            html_content += f"""
                    <div class="plot-container">
                        <div class="plot-title">Test Case {i} - Historical + Future Forecast</div>
                        <img src="data:image/png;base64,{plot_img}" alt="Test Case {i}">
                    </div>
"""
                        html_content += """
                </div>
"""
                
                # NOTE: Pipeline visualization moved to beginning of report (single instance)
                
                # Training history (for neural networks)
                if 'history' in details:
                    loss_plot = self.plot_training_history(details['history'], 'loss', reg_name)
                    mae_plot = self.plot_training_history(details['history'], 'mae', reg_name)
                    html_content += f"""
                <div class="comparison-section">
                    <div class="plot-container">
                        <div class="plot-title">Training Loss Curve (Log Scale)</div>
                        <img src="data:image/png;base64,{loss_plot}" alt="Loss Curve">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">Training MAE Curve</div>
                        <img src="data:image/png;base64,{mae_plot}" alt="MAE Curve">
                    </div>
                </div>
"""
            
            # Close the model section (collapsible or regular)
            if use_collapsible:
                html_content += """
                </div>
            </details>
"""
            else:
                html_content += """
            </div>
"""
        
        # ========== SAMPLE FORECASTS SECTION ==========
        if benchmark_results.get('sample_forecasts'):
            sample_images = self.plot_sample_forecasts(benchmark_results['sample_forecasts'])
            
            html_content += f"""
            <h2>🔍 Sample Forecasts (Test Set)</h2>
            <div class="summary">
                <strong>Visualization:</strong> Random samples from the test set showing history and {forecast_horizon}-step forecast vs actual.
                <br><strong>Purpose:</strong> Visual inspection of model behavior on specific cases.
            </div>
            <div class="comparison-section">
"""
            for img in sample_images:
                html_content += f"""
                <div class="plot-container">
                    <img src="data:image/png;base64,{img}" alt="Sample Forecast">
                </div>
"""
            html_content += """
            </div>
"""

        # ========== FORECAST SECTION ==========
        # Generate future forecast chart
        if benchmark_results.get('future_forecasts'):
            last_price = benchmark_results.get('last_actual_price', 0)
            horizon = benchmark_results.get('forecast_horizon', 10)
            
            # Extract history from y_all if available
            history_data = None
            if y_all is not None and len(y_all) > 0:
                history_data = y_all[-30:].tolist() # Last 30 points
            
            future_chart = self.plot_future_forecast_table(
                benchmark_results['future_forecasts'], 
                horizon=horizon,
                last_price=last_price,
                history=history_data
            )
            
            html_content += f"""
            <h2>🔮 Price Forecasts (Next {benchmark_results['forecast_horizon']} Steps)</h2>
            <div class="summary">
                <strong>Forecasted Values:</strong> Predicted closing prices from all trained regressors for the next {benchmark_results['forecast_horizon']} steps.
                <br><strong>Method:</strong> Recursive multi-step forecasting (each prediction uses previous predictions as features)
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Future Price Forecast Comparison</div>
                <img src="data:image/png;base64,{future_chart}" alt="Future Forecast Chart">
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
"""
        else:
            html_content += f"""
            <h2>🔮 Price Forecasts (Next {benchmark_results['forecast_horizon']} Steps)</h2>
            <div class="summary">
                <strong>Forecasted Values:</strong> Predicted closing prices from all trained regressors for the next {benchmark_results['forecast_horizon']} steps.
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
"""
        
        for reg_name, _ in sorted_regressors:
            html_content += f"                        <th>{reg_name.upper()}</th>\n"
        
        html_content += """
                    </tr>
                </thead>
                <tbody>
"""
        
        for step in range(1, min(16, benchmark_results['forecast_horizon'] + 1)):
            html_content += f"                    <tr><td><strong>Step {step}</strong></td>"
            for reg_name, _ in sorted_regressors:
                if reg_name in benchmark_results['future_forecasts']:
                    pred = benchmark_results['future_forecasts'][reg_name]['predictions'][step-1]
                    last_price = benchmark_results['future_forecasts'][reg_name]['last_price']
                    pct_change = ((pred - last_price) / last_price * 100) if last_price != 0 else 0
                    
                    color_class = 'up' if pct_change > 0 else 'down' if pct_change < 0 else ''
                    html_content += f'<td><span class="{color_class}">${pred:.2f} ({pct_change:+.2f}%)</span></td>'
            html_content += "</tr>\n"
        
        html_content += """
                </tbody>
            </table>
"""
        
        # ========== LLM MODEL REVIEW SECTION ==========
        # NOTE: LLM review section disabled - may be re-enabled in future
        # html_content += self._generate_llm_review_section(benchmark_results)
        
        # ========== MODEL ERRORS SECTION ==========
        model_errors = benchmark_results.get('model_errors', {})
        if model_errors:
            html_content += self._generate_model_errors_section(model_errors)
        
        # ========== ARTIFACTS SECTION ==========
        artifacts = benchmark_results.get('artifacts', {})
        if artifacts:
            html_content += self._generate_artifacts_section(artifacts)
        
        
        html_content += """
        </div>
        
        <div class="footer">
            <p><strong>Advanced Model Benchmark Report</strong></p>
            <p>All models trained and evaluated on the latest available financial data.</p>
            <p>Results may vary with market conditions. This report is for informational purposes only.</p>
            <p style="margin-top: 15px; color: #95a5a6;">Generated: {timestamp}</p>
        </div>
    </div>
</body>
</html>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Save report
        report_path = os.path.join(self.output_dir, 'benchmark_report_advanced.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path

    def generate_advanced_report_html(self, ticker: str = 'Asset') -> str:
        """
        Generate advanced HTML report using stored benchmark results.
        Wrapper around generate_comprehensive_report.
        """
        if not self.benchmark_results:
            print("No benchmark results available for report generation.")
            return ""
            
        return self.generate_comprehensive_report(
            self.benchmark_results, 
            self.benchmark_results.get('classifiers', {}),
            self.benchmark_results.get('regressors', {}),
            ticker=ticker,
            forecast_horizon=self.benchmark_results.get('pipeline_info', {}).get('forecast_horizon', 30)
        )

    @staticmethod
    def generate_stub_report(output_dir: str, report_name: str, 
                            error_info: dict = None, 
                            benchmark_results: dict = None) -> str:
        """
        Generate a minimal stub report when normal generation fails.
        
        Args:
            output_dir: Directory to save the report
            report_name: Filename for the report (e.g., 'benchmark_report.html')
            error_info: Optional dict with 'exception' and 'traceback' keys
            benchmark_results: Optional partial results dict
            
        Returns:
            Path to generated stub report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        error_section = ""
        if error_info:
            exception_msg = error_info.get('exception', 'Unknown error')
            traceback_str = error_info.get('traceback', '')
            traceback_escaped = (traceback_str
                                .replace('&', '&amp;')
                                .replace('<', '&lt;')
                                .replace('>', '&gt;')
                                .replace('\n', '<br>'))
            error_section = f"""
                <div style="background: #fdf2f2; border: 2px solid #e74c3c; 
                            padding: 20px; border-radius: 10px; margin: 20px;">
                    <h3 style="color: #c0392b;">⚠️ Report Generation Error</h3>
                    <p><strong>Exception:</strong> {exception_msg}</p>
                    <details>
                        <summary style="cursor: pointer; color: #666;">Full Stack Trace</summary>
                        <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; 
                                    border-radius: 5px; overflow-x: auto; font-size: 11px; 
                                    margin-top: 10px;">{traceback_escaped}</pre>
                    </details>
                </div>
"""
        
        # Extract any available info from partial results
        info_section = ""
        if benchmark_results:
            ticker = benchmark_results.get('ticker', 'Unknown')
            horizon = benchmark_results.get('forecast_horizon', 'Unknown')
            model_errors = benchmark_results.get('model_errors', {})
            successful_regressors = list(benchmark_results.get('regressors', {}).keys())
            
            model_errors_html = ""
            if model_errors:
                model_errors_html = "<h4>Failed Models:</h4><ul>"
                for model_name, err in model_errors.items():
                    model_errors_html += f"<li><strong>{model_name}</strong>: {err.get('exception', 'Unknown')}</li>"
                model_errors_html += "</ul>"
            
            info_section = f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px;">
                    <h3>Partial Run Information</h3>
                    <p><strong>Ticker:</strong> {ticker}</p>
                    <p><strong>Forecast Horizon:</strong> {horizon}</p>
                    <p><strong>Successful Models:</strong> {', '.join(successful_regressors) if successful_regressors else 'None'}</p>
                    {model_errors_html}
                </div>
"""
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Benchmark Report (Stub) - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚠️ Benchmark Report (Incomplete)</h1>
        <p style="color: #666;">
            This is a stub report generated because the full report generation encountered errors.
            See below for available information.
        </p>
        {error_section}
        {info_section}
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #999;">
            <p>Generated: {timestamp}</p>
            <p>Report file: {report_name}</p>
        </div>
    </div>
</body>
</html>
"""
        
        report_path = os.path.join(output_dir, report_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path


def ensure_required_reports(output_dir: str, 
                           benchmark_results: dict = None,
                           required_reports: list = None,
                           generation_errors: dict = None) -> dict:
    """
    Ensure that required report files exist, generating stubs if needed.
    
    This is a safety-net function that should be called after the normal
    report generation to guarantee outputs exist.
    
    Args:
        output_dir: Directory where reports should exist
        benchmark_results: Optional results dict (for partial info in stubs)
        required_reports: List of report filenames to ensure exist
                         Defaults to ['benchmark_report.html', 'benchmark_report_advanced.html']
        generation_errors: Dict mapping report_name -> {'exception': str, 'traceback': str}
        
    Returns:
        Dict with 'created_stubs' list and 'existing_reports' list
    """
    if required_reports is None:
        required_reports = ['benchmark_report_advanced.html']
    
    if generation_errors is None:
        generation_errors = {}
    
    result = {
        'created_stubs': [],
        'existing_reports': []
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for report_name in required_reports:
        report_path = os.path.join(output_dir, report_name)
        if os.path.exists(report_path):
            result['existing_reports'].append(report_path)
        else:
            # Generate stub
            error_info = generation_errors.get(report_name)
            stub_path = ReportGenerator.generate_stub_report(
                output_dir, report_name, 
                error_info=error_info,
                benchmark_results=benchmark_results
            )
            result['created_stubs'].append(stub_path)
            logger.warning(
                f"Created stub report: {stub_path} (normal generation failed)"
            )
    
    return result
