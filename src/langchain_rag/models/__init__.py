"""
Data models and LLM interface abstractions.

This module contains base models, LLM proxies, and data structures.
"""

from .base_model import OpenAIProxy, OllamaProxy, LLMProxy

__all__ = [
    "OpenAIProxy",
    "OllamaProxy", 
    "LLMProxy",
] 