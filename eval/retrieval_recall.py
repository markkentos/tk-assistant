# -*- coding: utf-8 -*-
"""Recall@k ретрива на бенчмарке (без LLM): вопрос → топ-3 статей сервиса."""
import json
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.environ.get("BASE", "http://192.168.1.170:8001")
BENCHMARK = "eval/benchmark.jsonl"
OUT = "eval/runs/retrieval_recall.jsonl"

rows = [json.loads(l) for l in open(BENCHMARK, encoding="utf-8") if l.strip()]
hits1 = hits3 = 0
in_context = 0
rr_sum = 0.0
misses = []
results = []

for i, row in enumerate(rows, 1):
    expected = str(row["article"])
    r = requests.post(f"{BASE}/retrieve",
                      json={"query": row["question"]}, timeout=60)
    r.raise_for_status()
    data = r.json()
    numbers = [a["number"] for a in data["articles"]]
    hit1 = bool(numbers) and numbers[0] == expected
    hit3 = expected in numbers[:3]
    inctx = expected in numbers
    rank = numbers.index(expected) + 1 if expected in numbers else 0
    hits1 += hit1
    hits3 += hit3
    in_context += inctx
    rr_sum += 1.0 / rank if rank else 0.0
    if not inctx:
        misses.append({"id": row["id"], "expected": expected,
                       "retrieved": numbers,
                       "question": row["question"][:120]})
    results.append({"id": row["id"], "expected": expected,
                    "retrieved": numbers, "route": data["route"],
                    "ms": data["retrieval_ms"], "hit@1": hit1,
                    "hit@3": hit3, "in_context": inctx, "rank": rank})
    if i % 20 == 0 or i == len(rows):
        print(f"[{i}/{len(rows)}] in_context = {in_context}/{i}", flush=True)

n = len(rows)
print(f"\nRecall@1: {hits1}/{n} = {hits1 / n:.1%}")
print(f"Recall@3: {hits3}/{n} = {hits3 / n:.1%}")
print(f"В контексте (с соседями): {in_context}/{n} = {in_context / n:.1%}")
print(f"MRR: {rr_sum / n:.3f}")

with open(OUT, "w", encoding="utf-8") as fh:
    for r in results:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nПромахи топ-3 ({len(misses)}):")
for m in misses:
    print(f"  {m['id']}: ожидалась {m['expected']}, "
          f"получено {m['retrieved']} — {m['question']}")
print(f"\nДетали: {OUT}")
