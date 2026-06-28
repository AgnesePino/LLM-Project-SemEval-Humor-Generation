from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from typing import Any


class LLMClient:
    def chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> str:
        messages = normalize_messages_for_model(self.model, messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if should_disable_reasoning(self.model):
            payload["reasoning_effort"] = "none"
        if seed is not None:
            payload["seed"] = seed
        data = post_json(f"{self.base_url}/chat/completions", payload, self.timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible response: {data}") from exc


def normalize_messages_for_model(model: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    model_lower = model.lower()
    if "mistral" not in model_lower or "ministral" in model_lower:
        return messages
    system_parts = [message["content"] for message in messages if message.get("role") == "system"]
    if not system_parts:
        return messages
    system_prefix = "\n\n".join(system_parts).strip()
    normalized: list[dict[str, str]] = []
    inserted = False
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            continue
        if role == "user" and not inserted:
            content = f"{system_prefix}\n\n{content}" if system_prefix else content
            inserted = True
        normalized.append({"role": role, "content": content})
    if not inserted and system_prefix:
        normalized.insert(0, {"role": "user", "content": system_prefix})
    return normalized


def should_disable_reasoning(model: str) -> bool:
    model_lower = model.lower()
    return "gemma" in model_lower


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> str:
        options: dict[str, Any] = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        data = post_json(f"{self.base_url}/api/chat", payload, self.timeout)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {data}") from exc


class MockClient(LLMClient):
    """Deterministic backend for smoke tests and pipeline validation."""

    def chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> str:
        prompt = messages[-1]["content"]
        if "Return JSON" in prompt or "Candidate A" in prompt or "candidates" in prompt.lower():
            return '{"winner": "A", "score": 8.0, "reason": "concise and constraint-compliant"}'
        words = re.findall(r"`([^`]+)`", prompt)
        if len(words) < 2:
            official_words = re.search(r"Words to include:\s*\{([^,{}]+),\s*([^{}]+)\}", prompt)
            if official_words:
                words = [official_words.group(1).strip(), official_words.group(2).strip()]
        if len(words) >= 2:
            return f"I brought {words[0]} to meet {words[1]}; now even the dictionary wants couples therapy."
        headline_match = re.search(r"headline:\s*(.+)", prompt, re.IGNORECASE | re.DOTALL)
        if headline_match:
            headline = headline_match.group(1).strip().splitlines()[0]
        else:
            quoted = re.search(r'"([^"]+)"', prompt)
            headline = quoted.group(1).strip() if quoted else "the news"
        return f"{headline} is the kind of headline that makes my coffee ask for a press secretary."


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local LLM server returned HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach local LLM server at {url}: {exc}") from exc


def make_client_from_config(backend: str, base_url: str, model: str, timeout: int) -> LLMClient:
    if backend == "openai":
        return OpenAICompatibleClient(base_url, model, timeout)
    if backend == "ollama":
        return OllamaClient(base_url, model, timeout)
    if backend == "mock":
        return MockClient()
    raise ValueError(f"Unsupported backend: {backend}")


def make_client(args: argparse.Namespace) -> LLMClient:
    return make_client_from_config(args.backend, args.base_url, args.model, args.timeout)
