"""
LLM Module for Trading Forecast v2

Provides integration with local Ollama for model review and analysis.
"""

from .ollama_client import OllamaClient, OLLAMA_AVAILABLE

__all__ = ['OllamaClient', 'OLLAMA_AVAILABLE']
