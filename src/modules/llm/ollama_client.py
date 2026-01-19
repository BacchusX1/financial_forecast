"""
Ollama Client for Trading Forecast v2

Provides HTTP API integration with local Ollama server for LLM-based
model analysis and review.

Requirements:
- Ollama running locally (http://localhost:11434)
- Model installed (e.g., llama3.1:8b-instruct)

Usage:
    from ollama_client import OllamaClient
    
    client = OllamaClient(model="llama3.1:8b-instruct")
    response = client.chat(
        system_prompt="You are a quant analyst...",
        user_prompt="Analyze this model performance..."
    )
"""

import json
import logging
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import socket

logger = logging.getLogger(__name__)

# Check if requests is available (preferred)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.info("requests library not available, using urllib")

OLLAMA_AVAILABLE = True  # Will be updated after connection test


class OllamaClient:
    """
    Client for interacting with local Ollama API.
    
    Supports:
    - Chat completions via /api/chat
    - Connection testing
    - Configurable model, timeout, and base URL
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1:8b-instruct"
    DEFAULT_TIMEOUT = 120  # seconds
    
    # System prompt for model review
    QUANT_REVIEWER_SYSTEM_PROMPT = """You are a Senior Quantitative Analyst and ML Reviewer specializing in financial time series forecasting.

Your task is to review machine learning model performance reports and provide actionable insights.

FOCUS AREAS:
1. **Data Leakage Detection**: Check for signs of unrealistic metrics (R² > 0.99, suspiciously low errors)
2. **Overfitting Analysis**: Compare training vs validation metrics, check model complexity
3. **Baseline Comparison**: Evaluate if models beat naive baselines (essential for financial forecasting)
4. **Multi-Step Forecast Quality**: Assess error propagation across forecast horizons
5. **Feature Engineering**: Suggest improvements based on reported features

RESPONSE FORMAT:
Provide a structured review with:
1. **Leakage/Methodology Check** (1-2 sentences)
2. **Model Ranking Commentary** (which models show promise)
3. **Why Simple Models May Outperform** (if applicable)
4. **Next Experiments** (max 5 bullet points, specific and actionable)
5. **Risks & Gotchas** (max 4 bullet points)

RULES:
- Be concise and technical
- NO financial advice or trading recommendations
- Focus on model/pipeline improvements only
- If metrics look suspicious, flag them clearly"""

    def __init__(self, 
                 base_url: str = None, 
                 model: str = None, 
                 timeout_s: int = None):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model: Model name (default: llama3.1:8b-instruct)
            timeout_s: Request timeout in seconds (default: 120)
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout_s or self.DEFAULT_TIMEOUT
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        logger.info(f"OllamaClient initialized: {self.base_url}, model={self.model}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            url = f"{self.base_url}/api/version"
            
            if REQUESTS_AVAILABLE:
                response = requests.get(url, timeout=5)
                return response.status_code == 200
            else:
                request = Request(url, method='GET')
                with urlopen(request, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def list_models(self) -> list:
        """
        List available models on Ollama server.
        
        Returns:
            List of model names, or empty list if unavailable
        """
        try:
            url = f"{self.base_url}/api/tags"
            
            if REQUESTS_AVAILABLE:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [m.get('name', '') for m in data.get('models', [])]
            else:
                request = Request(url, method='GET')
                with urlopen(request, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    return [m.get('name', '') for m in data.get('models', [])]
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
        
        return []
    
    def chat(self, 
             system_prompt: str = None, 
             user_prompt: str = None,
             temperature: float = 0.7,
             max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Send chat request to Ollama.
        
        Args:
            system_prompt: System message (role: system)
            user_prompt: User message (role: user)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with keys:
                - success: bool
                - content: str (response text)
                - error: str (if failed)
                - model: str
                - duration_s: float (response time)
        """
        if system_prompt is None:
            system_prompt = self.QUANT_REVIEWER_SYSTEM_PROMPT
        
        if not user_prompt:
            return {
                'success': False,
                'content': '',
                'error': 'No user prompt provided',
                'model': self.model
            }
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        url = f"{self.base_url}/api/chat"
        
        try:
            import time
            start_time = time.time()
            
            if REQUESTS_AVAILABLE:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    return {
                        'success': False,
                        'content': '',
                        'error': f"HTTP {response.status_code}: {response.text[:200]}",
                        'model': self.model
                    }
                
                data = response.json()
            else:
                # urllib fallback
                request = Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                
                with urlopen(request, timeout=self.timeout) as response:
                    if response.status != 200:
                        return {
                            'success': False,
                            'content': '',
                            'error': f"HTTP {response.status}",
                            'model': self.model
                        }
                    
                    data = json.loads(response.read().decode())
            
            duration = time.time() - start_time
            
            # Extract content from response
            content = data.get('message', {}).get('content', '')
            
            if not content:
                return {
                    'success': False,
                    'content': '',
                    'error': 'Empty response from model',
                    'model': self.model,
                    'duration_s': duration
                }
            
            logger.info(f"Ollama response received in {duration:.1f}s ({len(content)} chars)")
            
            return {
                'success': True,
                'content': content,
                'error': None,
                'model': self.model,
                'duration_s': duration
            }
            
        except (URLError, HTTPError, socket.timeout) as e:
            error_msg = str(e)
            if 'Connection refused' in error_msg:
                error_msg = "Ollama server not running. Start with: ollama serve"
            elif 'timed out' in error_msg.lower():
                error_msg = f"Request timed out after {self.timeout}s. Try a smaller model."
            
            return {
                'success': False,
                'content': '',
                'error': error_msg,
                'model': self.model
            }
        
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return {
                'success': False,
                'content': '',
                'error': str(e),
                'model': self.model
            }
    
    def generate_review_prompt(self, benchmark_results: Dict) -> str:
        """
        Generate a structured user prompt from benchmark results.
        
        Args:
            benchmark_results: Dict from DataScientist.train_and_evaluate()
        
        Returns:
            Formatted prompt string for model review
        """
        prompt_parts = []
        
        # Header
        ticker = benchmark_results.get('ticker', 'Unknown')
        horizon = benchmark_results.get('forecast_horizon', 10)
        holdout = benchmark_results.get('holdout_days', 50)
        
        prompt_parts.append(f"## BENCHMARK REPORT: {ticker}")
        prompt_parts.append(f"- Forecast Horizon: {horizon} days (multi-step)")
        prompt_parts.append(f"- Holdout Period: {holdout} days")
        prompt_parts.append("")
        
        # Classifier Performance
        if benchmark_results.get('classifiers'):
            prompt_parts.append("### CLASSIFIER PERFORMANCE (Trend: UP/DOWN/SIDEWAYS)")
            for name, metrics in benchmark_results['classifiers'].items():
                acc = metrics.get('accuracy', 0) * 100
                f1 = metrics.get('f1_score', 0)
                trend = metrics.get('trend_prediction', 'N/A')
                prompt_parts.append(f"- {name.upper()}: Acc={acc:.1f}%, F1={f1:.3f}, Trend={trend}")
            prompt_parts.append("")
        
        # Regressor Performance
        if benchmark_results.get('regressors'):
            prompt_parts.append("### REGRESSOR PERFORMANCE (Price Prediction)")
            for name, metrics in benchmark_results['regressors'].items():
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                prompt_parts.append(f"- {name.upper()}: R²={r2:.4f}, RMSE=${rmse:.2f}")
            prompt_parts.append("")
        
        # Holdout Performance (critical for multi-step)
        if benchmark_results.get('holdout_forecasts'):
            prompt_parts.append("### HOLDOUT VALIDATION (True Out-of-Sample)")
            for name, data in benchmark_results['holdout_forecasts'].items():
                if name == 'naive_baseline':
                    continue
                metrics = data.get('metrics', data)
                if isinstance(metrics, dict):
                    rmse = metrics.get('rmse_avg', metrics.get('rmse', 0))
                    r2 = metrics.get('r2_avg', metrics.get('r2', 0))
                    beats = metrics.get('beats_naive', data.get('beats_naive', False))
                    prompt_parts.append(f"- {name.upper()}: RMSE_avg=${rmse:.2f}, R²_avg={r2:.4f}, Beats Naive={'✓' if beats else '✗'}")
            prompt_parts.append("")
        
        # Baselines
        if benchmark_results.get('baselines'):
            prompt_parts.append("### BASELINES (Multi-Step)")
            for name, data in benchmark_results['baselines'].items():
                metrics = data.get('metrics', data)
                if isinstance(metrics, dict):
                    rmse = metrics.get('rmse_avg', metrics.get('rmse', 0))
                    prompt_parts.append(f"- {name}: RMSE_avg=${rmse:.2f}")
            prompt_parts.append("")
        
        # Pipeline Info
        pipeline = benchmark_results.get('pipeline_info', {})
        if pipeline:
            features = pipeline.get('feature_columns', [])
            prompt_parts.append("### PIPELINE CONFIG")
            prompt_parts.append(f"- Features ({len(features)}): {', '.join(features[:8])}{'...' if len(features) > 8 else ''}")
            prompt_parts.append(f"- Scaling: {pipeline.get('scaling', 'StandardScaler')}")
            prompt_parts.append(f"- No Leakage Guarantee: {pipeline.get('no_leakage_guarantee', 'Unknown')}")
            prompt_parts.append("")
        
        # Warnings
        warnings = benchmark_results.get('warnings', [])
        if warnings:
            prompt_parts.append("### WARNINGS")
            for w in warnings[:5]:
                prompt_parts.append(f"- {w}")
            prompt_parts.append("")
        
        prompt_parts.append("### ANALYSIS REQUEST")
        prompt_parts.append("Please provide a structured review following the format specified in your system prompt.")
        
        return "\n".join(prompt_parts)


def test_ollama_connection(base_url: str = None) -> Dict[str, Any]:
    """
    Test connection to Ollama server.
    
    Args:
        base_url: Ollama API base URL
    
    Returns:
        Dict with connection status and available models
    """
    client = OllamaClient(base_url=base_url)
    
    result = {
        'available': False,
        'models': [],
        'error': None
    }
    
    if client.is_available():
        result['available'] = True
        result['models'] = client.list_models()
        logger.info(f"Ollama available with {len(result['models'])} models")
    else:
        result['error'] = "Ollama server not responding. Start with: ollama serve"
        logger.warning(result['error'])
    
    return result
