# -*- coding: utf-8 -*-
"""Диагностика: все заголовки ответа RAG-сервиса."""
import json
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

r = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    json={"messages": [{"role": "user",
                        "content": "Сколько дней отпуска положено работнику в возрасте до 18 лет?"}],
          "stream": False, "max_tokens": 60, "temperature": 0.2},
    timeout=600,
)
print("status:", r.status_code)
for k, v in r.headers.items():
    print(f"  {k}: {v[:80]}")
d = r.json()
print("answer:", d["choices"][0]["message"]["content"][:200])
