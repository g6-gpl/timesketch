"""LM Studio LLM provider for Timesketch.

This is a minimal provider implementation that calls an LM Studio HTTP
endpoint. It implements the same constructor signature as other providers
and a `generate` method returning text or parsed JSON when requested.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from timesketch.lib.llms.providers import interface
from timesketch.lib.llms.providers import manager

logger = logging.getLogger("timesketch.llm.lmstudio")


class LMStudioProvider(interface.LLMProvider):
    """Provider for LM Studio (minimal HTTP wrapper)."""

    NAME = "lmstudio"

    def __init__(self, config: dict):
        """Initialize provider.

        Expected config keys:
        - server_url: base URL for LM Studio HTTP API (required)
        - model: model identifier (optional)
        - api_key: optional API key to include as `Authorization: Bearer ...`
        """
        super().__init__(config)
        self._server_url = self.config.get("server_url")
        self._model = self.config.get("model")
        self._api_key = self.config.get("api_key")

        if not self._server_url:
            raise ValueError("LMStudio provider requires 'server_url' in config")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def generate(self, prompt: str, response_schema: Optional[dict] = None) -> str:
        """Generate text via LM Studio HTTP API.

        This sends a POST to `<server_url>/generate` with a simple JSON body.
        Adjust as needed for your LM Studio deployment API.
        """
        url = self._server_url.rstrip("/") + "/generate"
        payload = {"prompt": prompt}
        if self._model:
            payload["model"] = self._model

        resp = requests.post(url, json=payload, headers=self._headers(), timeout=60)
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.exception("LM Studio request failed: %s", e)
            raise

        # If response_schema requested, try to parse JSON
        if response_schema:
            try:
                return resp.json()
            except Exception as e:
                raise ValueError("Failed to parse JSON response from LM Studio") from e

        # Default: return plain text body
        return resp.text


# Register provider
try:
    manager.LLMManager.register_provider(LMStudioProvider)
except Exception:
    # Ignore registration errors during tests/imports where manager may be cleared.
    pass
