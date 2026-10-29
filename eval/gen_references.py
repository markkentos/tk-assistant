# -*- coding: utf-8 -*-
"""Этап 2: генерация эталонных ответов бенчмарка LLM по тексту статьи.

Эталоны создаются заземлённо (статья дана модели в промпте) и верифицируются
вручную: человек проверяет CSV и либо ставит OK, либо исправляет ответ.
В README это честно описано как «LLM-генерация + ручная верификация».

Запуск:
    python eval/gen_references.py          # после eval/build_benchmark.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time

import yaml

sys.path.insert(0, "data/scripts")
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from common import load_env, load_json, load_jsonl, utf8_console  # noqa: E402
from providers import make_provider  # noqa: E402

PROMPT = """\
Ты юрист-эксперт по трудовому праву. Ответь на вопрос СТРОГО по тексту статьи —
не используй знания вне этого текста. Если ответа в статье нет, так и напиши.

Статья {num} ТК РФ:
---
{article}
---
Вопрос: {question}

Формат ответа — строго две части:
<Краткий ответ одним предложением.>
<Обоснование: 1–3 предложения со ссылкой «ст. {num} ТК РФ».>"""


def main() -> None:
    utf8_console()
    load_env()
    with open("configs/judge.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    judge = config["judge"]
    provider = make_provider(judge["provider"], judge["model"])

    articles = {a["number"]: a for a in load_json(
        "data/processed/tk_articles.json")["articles"]}
    benchmark = load_jsonl("eval/benchmark.jsonl")
    missing_ref = [b for b in benchmark if b.get("reference")]
    if missing_ref:
        print(f"У {len(missing_ref)} вопросов уже есть reference — пропускаю их")

    cache_path = "eval/runs/references_cache.jsonl"
    cache: dict[str, str] = {}
    try:
        for row in load_jsonl(cache_path):
            cache[row["id"]] = row["reference"]
    except FileNotFoundError:
        pass

    todo = [b for b in benchmark if not b.get("reference") and b["id"] not in cache]
    print(f"К генерации эталонов: {len(todo)} из {len(benchmark)}")

    for i, item in enumerate(todo, 1):
        article = articles.get(item["article"])
        prompt = PROMPT.format(num=item["article"],
                               article=article["text"] if article else "(статья не найдена)",
                               question=item["question"])
        reference = None
        for attempt in range(1, 4):
            try:
                reference = provider.chat([{"role": "user", "content": prompt}],
                                          temperature=0.2, max_tokens=512).strip()
                break
            except Exception as exc:
                print(f"\n[{item['id']}] попытка {attempt}: {exc}")
                time.sleep(20 * attempt)
        if reference is None:
            reference = "(ОШИБКА ГЕНЕРАЦИИ — перезапусти скрипт)"
        cache[item["id"]] = reference
        os.makedirs("eval/runs", exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": item["id"], "reference": reference},
                                ensure_ascii=False) + "\n")
        if i % 10 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] готово")
        time.sleep(2.5)

    stamp = dt.datetime.now().strftime("%Y%m%d")
    path = f"eval/benchmark_references_to_verify_{stamp}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "id", "article", "chapter", "question_type", "question",
            "llm_reference", "verdict", "corrected_reference", "note"])
        writer.writeheader()
        for item in benchmark:
            writer.writerow({
                "id": item["id"], "article": item["article"],
                "chapter": item["chapter"], "question_type": item["question_type"],
                "question": item["question"],
                "llm_reference": item.get("reference") or cache.get(item["id"], ""),
                "verdict": "", "corrected_reference": "", "note": "",
            })
    print(f"\nCSV для верификации: {path}")
    print("Проверь каждый эталон по статье, заполни verdict (OK/BAD). "
          "При BAD — исправь в corrected_reference.")


if __name__ == "__main__":
    main()
