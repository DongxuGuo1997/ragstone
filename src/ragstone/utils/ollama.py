"""Shared Ollama server helpers."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def list_installed_models(timeout: float = 5.0) -> List[str]:
    """Model names the local Ollama server reports, or [] when offline.

    Never raises: every caller (UI dropdowns, interactive prompts) treats
    "Ollama unreachable" as an empty menu, not an error.
    """
    try:
        import requests

        from ..config.settings import get_config

        base_url = get_config().api.ollama_base_url
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
        logger.debug(f"Ollama /api/tags returned {response.status_code}")
    except Exception as e:
        logger.debug(f"Could not list Ollama models: {e}")
    return []
