# -*- coding: utf-8 -*-
"""Провайдеры LLM для генерации и проверки QA: GigaChat и OpenRouter.

Секреты берутся из переменных окружения:
    GIGACHAT_CREDENTIALS  — ключ авторизации GigaChat API (Base64 client_id:client_secret)
    GIGACHAT_CA_CERT      — путь к CA-сертификату Сберa (опционально; иначе verify=False)
    OPENROUTER_API_KEY    — ключ OpenRouter
    GIGACHAT_SCOPE        — скоуп (по умолчанию GIGACHAT_API_PERS)
"""
from __future__ import annotations

import abc
import base64
import json
import os
import time
import uuid
from typing import Literal

import requests

GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

Message = dict  # {"role": ..., "content": ...}


class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def chat(self, messages: list[Message], temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
        """Один чат-запрос, возвращает текст ответа."""


class OpenAICompatibleMixin:
    """Общая логика chat/completions для OpenAI-совместимых API."""

    api_url: str
    model: str
    session: requests.Session

    def chat(self, messages: list[Message], temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self.session.post(self.api_url, json=payload, timeout=120)
        if response.status_code == 429:  # лимит — подождём и повторим один раз
            time.sleep(10)
            response = self.session.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class GigaChatProvider(LLMProvider, OpenAICompatibleMixin):
    name = "gigachat"

    def __init__(self, model: str = "GigaChat-Max"):
        credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise RuntimeError("Не задан GIGACHAT_CREDENTIALS (ключ авторизации GigaChat API)")
        self.model = model
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.session = requests.Session()
        ca_cert = os.environ.get("GIGACHAT_CA_CERT")
        if ca_cert:
            self.session.verify = ca_cert
        else:
            # У Сбера свой CA; без его сертификата остаётся отключить проверку.
            # Для пет-проекта приемлемо, но предупреждаем громко.
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            self.session.verify = False
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        response = self.session.post(
            GIGACHAT_OAUTH_URL,
            headers={
                "Authorization": f"Basic {self.credentials}",
                "RqUID": uuid.uuid4().hex,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"scope": self.scope},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expires_at = data.get("expires_at", 0) / 1000  # мс -> сек
        return self._token

    @property
    def credentials(self) -> str:
        # Ключ уже приходит в Base64 — если это не так, закодируем.
        raw = os.environ["GIGACHAT_CREDENTIALS"]
        try:
            base64.b64decode(raw, validate=True)
            return raw
        except Exception:
            return base64.b64encode(raw.encode()).decode()

    def chat(self, messages: list[Message], temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
        token = self._get_token()
        self.session.headers["Authorization"] = f"Bearer {token}"
        try:
            return super().chat(messages, temperature, max_tokens)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                self._token = None  # токен протух — обновим при следующем вызове
            raise


class OpenRouterProvider(LLMProvider, OpenAICompatibleMixin):
    name = "openrouter"

    def __init__(self, model: str):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Не задан OPENROUTER_API_KEY")
        self.model = model
        self.api_url = OPENROUTER_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })


def make_provider(provider: Literal["gigachat", "openrouter"], model: str) -> LLMProvider:
    if provider == "gigachat":
        return GigaChatProvider(model=model)
    if provider == "openrouter":
        return OpenRouterProvider(model=model)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def extract_json_array(text: str) -> list:
    """Достаём JSON-массив из ответа LLM (терпим ```- fences и болтовню вокруг)."""
    text = text.strip()
    text = re_strip_fences(text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("в ответе нет JSON-массива")
    return json.loads(text[start:end + 1])


def re_strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()
