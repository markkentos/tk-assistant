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
        # Ретраи с экспоненциальным бэккоффом: 429 (лимиты), 5xx, таймауты, обрывы.
        delays = [15, 30, 60, 120, 300]
        response = None
        for attempt in range(len(delays) + 1):
            try:
                response = self.session.post(self.api_url, json=payload, timeout=120)
            except requests.RequestException as exc:  # таймауты, обрывы, DNS
                if attempt == len(delays):
                    raise RuntimeError(
                        f"LLM API недоступна после {attempt} ретраев: {exc!r}") from exc
                time.sleep(delays[attempt])
                continue
            if response.status_code == 200:
                break
            retriable = response.status_code == 429 or response.status_code >= 500
            if not retriable or attempt == len(delays):
                break
            pause = delays[attempt]
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                pause = max(pause, int(retry_after))
            time.sleep(pause)
        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else "нет ответа"
            body = response.text[:300] if response is not None else ""
            raise RuntimeError(f"LLM API error HTTP {status}: {body}")
        data = response.json()
        return data["choices"][0]["message"]["content"]


class GigaChatProvider(OpenAICompatibleMixin, LLMProvider):
    name = "gigachat"

    def __init__(self, model: str = "GigaChat-Max"):
        self.model = model
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.session = requests.Session()
        ca_cert = os.environ.get("GIGACHAT_CA_CERT")
        if ca_cert:
            self.session.verify = ca_cert
        else:
            # Сертификаты Сбера подписаны российским CA (Минцифры), которого нет
            # в certifi — без переданного CA-сертификата отключаем проверку.
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            self.session.verify = False

        # Режим 1 (новая платформа): статичный API-ключ -> Bearer на api.giga.chat,
        # OAuth не нужен. Режим 2 (классика): ключ авторизации -> OAuth на ngw.
        self.api_key = os.environ.get("GIGACHAT_API_KEY", "").strip()
        if self.api_key:
            self.mode = "bearer"
            self.api_url = "https://api.giga.chat/v1/chat/completions"
        else:
            self.mode = "oauth"
            self.api_url = GIGACHAT_API_URL
            self._token: str | None = None
            self._token_expires_at: float = 0.0
            _ = self.credentials  # провоцируем ошибку рано, если ключей нет вовсе

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        # ВАЖНО: OAuth-эндпоинт принимает application/x-www-form-urlencoded
        # (data=..., БЕЗ Content-Type: application/json — иначе шлюз отвечает 415).
        # Так же делает официальный SDK (gigachat/api/auth.py: data={"scope": ...}).
        response = self.session.post(
            GIGACHAT_OAUTH_URL,
            headers={
                "Authorization": f"Basic {self.credentials}",
                "RqUID": str(uuid.uuid4()),
            },
            data={"scope": self.scope},
            timeout=30,
        )
        if response.status_code != 200:
            # Тело ответа Сберa обычно объясняет причину (401 — неверный ключ,
            # 403 — не тот скоуп/нет доступа, 400 — «Can't decode Authorization» —
            # скопирован не тот/не весь ключ).
            raise RuntimeError(
                f"GigaChat OAuth failed: HTTP {response.status_code}, "
                f"ответ: {response.text[:300]}")
        data = response.json()
        self._token = data["access_token"]
        self._token_expires_at = data.get("expires_at", 0) / 1000  # мс -> сек
        return self._token

    @staticmethod
    def _clean_credentials(raw: str) -> str:
        """Ключ авторизации GigaChat уходит в Basic как есть: срезаем только мусор.

        Новые ключи — это JWE-токены (начинаются с eyJ..., содержат точки),
        поэтому «докодировать» их нельзя — только испортим.
        """
        return raw.strip().strip('"').strip("'").strip()

    @property
    def credentials(self) -> str:
        """Basic-ключ: готовый «Ключ авторизации» ИЛИ base64(client_id:client_secret)."""
        raw = self._clean_credentials(os.environ.get("GIGACHAT_CREDENTIALS", ""))
        if raw:
            return raw
        client_id = os.environ.get("GIGACHAT_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GIGACHAT_CLIENT_SECRET", "").strip()
        if client_id and client_secret:
            return base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        raise RuntimeError(
            "Не задан GIGACHAT_CREDENTIALS (ключ авторизации) либо пара "
            "GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET — заполни .env")

    def chat(self, messages: list[Message], temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
        if self.mode == "bearer":
            # Статичный API-ключ новой платформы api.giga.chat — OAuth не нужен
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
            return super().chat(messages, temperature, max_tokens)
        token = self._get_token()
        self.session.headers["Authorization"] = f"Bearer {token}"
        try:
            return super().chat(messages, temperature, max_tokens)
        except RuntimeError as exc:
            if "HTTP 401" in str(exc):
                self._token = None  # токен протух — обновим при следующем вызове
            raise


class OpenRouterProvider(OpenAICompatibleMixin, LLMProvider):
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
