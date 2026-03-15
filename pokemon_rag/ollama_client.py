from __future__ import annotations

from typing import Any

import requests


class OllamaChatClient:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _content_to_text(self, content: Any) -> str:
        """Normalize Ollama response content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"]).strip()
            return str(content).strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "\n".join(part.strip() for part in parts if str(part).strip())
        return str(content).strip()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return self._content_to_text(data["message"].get("content"))