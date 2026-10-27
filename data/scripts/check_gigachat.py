# -*- coding: utf-8 -*-
"""Диагностика ключа GigaChat: OAuth + один пробный чат-запрос.

Запуск:
    python data/scripts/check_gigachat.py

Печатает замаскированный ключ (чтобы видеть форму, но не секрет),
результат OAuth и ответ модели на тестовый вопрос.
"""
from __future__ import annotations

import os
import sys

from common import load_env, utf8_console
from providers import GigaChatProvider


def mask(value: str) -> str:
    if len(value) <= 12:
        return f"<короткий ключ: {len(value)} симв.>"
    return f"{value[:6]}...{value[-4:]} (длина {len(value)})"


def main() -> None:
    utf8_console()
    load_env()

    raw = os.environ.get("GIGACHAT_CREDENTIALS", "")
    pair = (os.environ.get("GIGACHAT_CLIENT_ID", ""), os.environ.get("GIGACHAT_CLIENT_SECRET", ""))
    if not raw and not all(pair):
        sys.exit("Нет ни GIGACHAT_CREDENTIALS, ни пары GIGACHAT_CLIENT_ID/CLIENT_SECRET — заполни .env")

    if raw:
        print(f"Ключ из .env в исходном виде : {mask(raw)}")
        if "." in raw:
            print("  ВНИМАНИЕ: в ключе есть точки (похоже на JWE/токен) — сервер такой не принимает.")
            print("  Нужен «Ключ авторизации»: одна base64-строка без точек. "
                  "Или заполни GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET.")
    else:
        print(f"Использую пару client_id ({len(pair[0])} симв.) / client_secret "
              f"({len(pair[1])} симв.) — base64 соберётся автоматически")

    provider = GigaChatProvider(model=os.environ.get("GIGACHAT_MODEL", "GigaChat-Max"))
    print(f"\nМодель: {provider.model} | режим: {provider.mode}")

    if provider.mode == "bearer":
        print("Статичный API-ключ: OAuth не требуется, проверяю чат сразу...")
    else:
        try:
            token = provider._get_token()
            print("OAuth: OK (токен получен)")
        except Exception as exc:
            print(f"OAuth: ОШИБКА — {exc}")
            print("\nПодсказки:")
            print("- 400 «Can't decode Authorization header»: в .env не тот ключ — нужен "
                  "«Ключ авторизации» (base64 без точек) или пара client_id/client_secret")
            print("- 401: неверный ключ")
            print("- 403: не выдан доступ к API или не тот скоуп (GIGACHAT_SCOPE в .env)")
            sys.exit(1)

    try:
        answer = provider.chat(
            [{"role": "user", "content": "Ответь одним словом: работает?"}],
            temperature=0.0, max_tokens=16)
        print(f"Чат: OK — модель ответила: {answer!r}")
        print("\nВсё работает. Можно запускать:")
        print("  python data/scripts/generate_qa.py --articles 80,81,99 --limit 3")
    except Exception as exc:
        print(f"Чат: ОШИБКА — {exc}")
        if "401" in str(exc) and provider.mode == "bearer":
            print("401 при валидном по виду ключе: ключ истёк или отозван — "
                  "создай новый в кабинете.")
        sys.exit(1)


if __name__ == "__main__":
    main()
