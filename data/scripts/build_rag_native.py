# -*- coding: utf-8 -*-
"""RAG-native переформатирование train/val: в каждый пример подмешивается
контекст (3 статьи) в формате продового rag_service.

- expected article ГАРАНТИРОВАННО в кандидатах (правильная supervising-сигнал),
  позиция случайна (0/1/2);
- отвлекающие — топ ретрива без expected (hard negatives, как в бою);
- каждый 7-й пример (~14%) остаётся чистым — модель не разучивается
  работать без RAG (одна модель на оба режима).
"""
import json
import os
import random
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.environ.get("BASE", "http://192.168.1.170:8001")
ARTICLES_PATH = "data/processed/tk_articles.json"
PLAIN_EVERY = 7
rng = random.Random(42)

articles = {a["number"]: a for a in
            json.load(open(ARTICLES_PATH, encoding="utf-8"))["articles"]
            if a.get("status") == "active"}
all_numbers = sorted(articles)


def wait_ready() -> None:
    for _ in range(60):
        try:
            h = requests.get(f"{BASE}/health", timeout=8).json()
            if h.get("index_ready"):
                print(f"ретривер готов: {h['articles']} статей", flush=True)
                return
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("ретривер так и не поднялся")


def build_row(row: dict) -> dict:
    expected = str(row["article"])
    r = requests.post(f"{BASE}/retrieve",
                      json={"query": row["question"]}, timeout=60)
    r.raise_for_status()
    retrieved = [a["number"] for a in r.json()["articles"]]
    in_retrieval = expected in retrieved
    distractors = [n for n in retrieved if n != expected][:2]
    while len(distractors) < 2:
        cand = rng.choice(all_numbers)
        if cand != expected and cand not in distractors:
            distractors.append(cand)
    pos = rng.randrange(3)
    cands = distractors[:pos] + [expected] + distractors[pos:]
    row = dict(row)
    row["candidates"] = [dict(number=n, title=articles[n]["title"],
                              text=articles[n]["text"]) for n in cands]
    row["meta_rag"] = {"in_retrieval": in_retrieval,
                       "retrieved": retrieved, "pos": pos}
    return row


wait_ready()

for name in ("train", "val"):
    rows = [json.loads(l) for l in
            open(f"data/processed/{name}.jsonl", encoding="utf-8") if l.strip()]
    out, in_retr, t0 = [], 0, time.time()
    for i, row in enumerate(rows, 1):
        if i % PLAIN_EVERY == 0:
            out.append(dict(row))               # чистый пример, без контекста
        else:
            new_row = build_row(row)
            in_retr += new_row["meta_rag"]["in_retrieval"]
            out.append(new_row)
        if i % 100 == 0 or i == len(rows):
            print(f"[{name} {i}/{len(rows)}] {time.time() - t0:.0f} c | "
                  f"expected в ретриве: {in_retr}/{i - i // PLAIN_EVERY}",
                  flush=True)
    path = f"data/processed/{name}_ragnative.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} строк → {path}", flush=True)

print("ГОТОВО", flush=True)
