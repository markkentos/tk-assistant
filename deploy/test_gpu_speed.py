# -*- coding: utf-8 -*-
"""Замер скорости и качества GPU-инференса через RAG на ПК."""
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://192.168.1.170:8001"

TESTS = [
    ("Сколько дней отпуска положено работнику в возрасте до 18 лет?", "267"),
    ("Может ли работодатель взыскать ущерб больше среднего месячного заработка?", "241"),
    ("За сколько дней предупреждают об увольнении по собственному желанию?", "80"),
]

for question, expected in TESTS:
    print(f"Q: {question[:90]} (ожидалась ст. {expected})", flush=True)
    t0 = time.time()
    try:
        response = requests.post(
            f"{BASE}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": question}],
                  "stream": False, "max_tokens": 200, "temperature": 0.2},
            timeout=300,
        )
        dt = time.time() - t0
        retrieved = response.headers.get("x-retrieved-article", "-")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  retrieved={retrieved} | {dt:.1f} сек | HTTP {response.status_code}",
              flush=True)
        print(f"  A: {content[:250]}", flush=True)
    except Exception as exc:
        print(f"  ОШИБКА: {exc!r}", flush=True)
    print(flush=True)
