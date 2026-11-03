# -*- coding: utf-8 -*-
"""Бенчмарк через RAG-сервис: 100 вопросов → answers_tuned_e6v2_rag.jsonl."""
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://192.168.1.170:8001"          # RAG-сервис на ПК
BENCHMARK = "eval/benchmark.jsonl"
OUTPUT = "eval/runs/answers_tuned_e6v2_rag.jsonl"

rows = [json.loads(l) for l in open(BENCHMARK, encoding="utf-8") if l.strip()]
out = []
t_start = time.time()

for i, row in enumerate(rows, 1):
    question = row["question"]
    t0 = time.time()
    answer = ""
    try:
        response = requests.post(
            f"{BASE}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": question}],
                  "stream": False, "max_tokens": 400, "temperature": 0.2},
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        answer = f"ОШИБКА: {exc!r}"
    dt = time.time() - t0
    retrieved = response.headers.get("x-retrieved-article", "-") if "response" in dir() else "-"
    out.append({
        "id": row["id"], "article": row["article"], "question": question,
        "answer": answer, "reference": row.get("reference", ""),
        "meta": {"mode": "tuned_e6v2_rag", "retrieved_article": retrieved,
                 "time_sec": round(dt, 1)},
    })
    if i % 10 == 0 or i == len(rows):
        elapsed = time.time() - t_start
        print(f"[{i}/{len(rows)}] {elapsed:.0f} сек всего", flush=True)

with open(OUTPUT, "w", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Готово: {OUTPUT}", flush=True)
