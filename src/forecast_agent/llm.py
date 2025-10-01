"""Utilities for interacting with a Hugging Face chat model."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B:hf-inference"


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    timeout: int = 20


class LLMClient:
    """Lightweight helper for classification and answer generation."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        token = os.getenv("HF_TOKEN", "").strip()
        self._session = requests.Session() if token else None
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    @property
    def available(self) -> bool:
        return self._session is not None

    def classify(self, question: str) -> Optional[Dict[str, Any]]:
        """Return classification dict with keys is_domain/metric, or None on failure."""
        if not self.available or self._session is None:
            return None

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You decide whether a user question belongs to SaaS forecasting. "
                        "Respond ONLY with JSON: {\"is_domain\": bool, \"metric\": string|null}. "
                        "Treat questions referencing SaaS metrics, revenue, customers, churn, "
                        "marketing, signups, forecasts, projections, or trends as in-domain. "
                        "Everything else is out-of-domain and should set is_domain to false."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "max_tokens": 120,
            "temperature": 0,
        }

        try:
            data = self._post(payload)
            message = data["choices"][0]["message"]["content"].strip()
            message = self._strip_backticks(message)
            return json.loads(message)
        except Exception:
            return None

    def answer(self, *, question: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate an answer using the provided context. Returns None on failure."""
        if not self.available or self._session is None:
            return None

        system_prompt = (
            "You are a SaaS forecasting analyst. Use the provided JSON context to answer the "
            "user's question. If the question is outside the SaaS forecasting context, reply "
            "exactly with: 'I can only answer questions about the SaaS metrics and forecasts in "
            "this dashboard. Try asking about revenue, customers, churn, or similar topics.' "
            "When the context lacks data, you may invent a plausible illustrative answer but you "
            "must preface it with 'Fictional for demo:' so the user knows. Stay within 3 concise "
            "sentences and reference concrete numbers from context when available."
        )

        context_json = json.dumps(context, ensure_ascii=False)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context JSON:\n{context_json}"
        )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.4,
        }

        try:
            data = self._post(payload)
            message = data["choices"][0]["message"]["content"].strip()
            return message
        except Exception:
            return None

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        response = self._session.post(
            HF_CHAT_URL,
            headers=self._headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _strip_backticks(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
        return stripped.strip()


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
