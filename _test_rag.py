# -*- coding: utf-8 -*-
"""Тест RAG-сервиса: несколько вопросов → проверка ретрива и ответов."""
import json
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8001"

TESTS = [
    ("Сколько дней отпуска положено работнику в возрасте до 18 лет?", "267"),
    ("Может ли работодатель взыскать ущерб больше среднего месячного заработка?", "241"),
    ("За сколько дней предупреждают об увольнении по собственному желанию?", "80"),
    ("Привет, как дела?", None),  # приветствие — статья искаться не должна
]

for question, expected in TESTS:
    print(f"Q: {question[:90]} (ожидалась ст. {expected})", flush=True)
    response = requests.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": question}],
              "stream": False, "max_tokens": 150, "temperature": 0.2},
        timeout=600,
    )
    retrieved = response.headers.get("x-retrieved-article", "-")
    ok = response.status_code == 200
    content = ""
    if ok:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    mark = "OK " if ok else "ERR"
    print(f"[{mark}] retrieved={retrieved:>6s}", flush=True)
    print(f"   A: {content[:220]}", flush=True)
    print(flush=True)
