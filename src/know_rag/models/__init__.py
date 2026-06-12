"""
Data models and LLM interface abstractions.

This module contains base models, LLM proxies, and data structures.
"""

from .base_model import LLMProxy, OllamaProxy, OpenAIProxy

__all__ = [
    "OpenAIProxy",
    "OllamaProxy",
    "LLMProxy",
]
