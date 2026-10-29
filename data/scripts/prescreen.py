# -*- coding: utf-8 -*-
"""LLM-предскрин (протокол качества, вариант B): третий независимый прогон.

Другой промпт, чем у заземления: ищем утверждения, которых нет в статье,
противоречия ей и неверные цифры/сроки. Флагнутые пары идут на ручную проверку,
из чистых берётся случайная выборка (оценка остаточного брака).

Вердикты кэшируются (prescreen_cache.jsonl) — перезапуск продолжает с места
остановки, API не тратится заново.

Запуск:
    python data/scripts/prescreen.py                # полный прогон
    python data/scripts/prescreen.py --limit 5      # смоук-тест
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import random
import sys
import time

import yaml

from common import append_jsonl, load_env, load_json, load_jsonl, save_jsonl, utf8_console
from providers import make_provider

PROMPT = """\
Тебе даны текст статьи Трудового кодекса РФ, вопрос и ответ ассистента.
Проверь ответ СТРОГО по тексту статьи. Ищи:
1) утверждения, которых НЕТ в тексте статьи (в том числе из других статей);
2) противоречия тексту статьи;
3) неверные цифры, сроки, размеры.

Текст статьи:
---
{article}
---
Вопрос: {question}
Ответ: {answer}

Ответь строго в формате (две строки):
Проблема: да|нет
Что: <если да — что именно неверно, одним предложением>"""


def uid_of(pair: dict) -> str:
    return hashlib.sha1(f"{pair['article']}|{pair['question']}".encode("utf-8")).hexdigest()[:12]


def write_review_csv(path: str, rows: list[dict], extra_cols: list[str] | None = None) -> None:
    columns = ["uid", "article", "chapter", "question_type", "question", "answer",
               "verdict", "note"] + (extra_cols or [])
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    utf8_console()
    load_env()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/qa_dedup.jsonl")
    parser.add_argument("--config", default="configs/filters.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    checker = config["checker"]
    provider = make_provider(checker["provider"], checker["model"])

    articles = {a["number"]: a for a in load_json(
        config.get("articles", "data/processed/tk_articles.json"))["articles"]}
    pairs = load_jsonl(args.input)
    if args.limit:
        pairs = pairs[:args.limit]
    for pair in pairs:
        pair.setdefault("uid", uid_of(pair))

    cache_path = "data/processed/prescreen_cache.jsonl"
    cache: dict[str, dict] = {}
    try:
        for row in load_jsonl(cache_path):
            cache[row["uid"]] = row
    except FileNotFoundError:
        pass
    todo = [p for p in pairs if p["uid"] not in cache]
    print(f"Пар: {len(pairs)}, в кэше: {len(cache)}, к проверке: {len(todo)}")

    for i, pair in enumerate(todo, 1):
        article = articles.get(pair["article"])
        article_text = article["text"] if article else ""
        prompt = PROMPT.format(article=article_text, question=pair["question"],
                               answer=pair["answer"])
        verdict = None
        for attempt in range(1, 4):
            try:
                raw = provider.chat([{"role": "user", "content": prompt}],
                                    temperature=checker.get("temperature", 0.2),
                                    max_tokens=checker.get("max_tokens", 512)).strip()
                low = raw.lower()
                flagged = "проблема: да" in low
                what = ""
                for line in raw.splitlines():
                    if line.lower().startswith("что:"):
                        what = line.split(":", 1)[1].strip()
                        break
                verdict = {"flagged": flagged, "what": what, "raw": raw[:200]}
                break
            except Exception as exc:
                print(f"\n[ст. {pair['article']}] попытка {attempt}: {exc}")
                time.sleep(30 * attempt)
        if verdict is None:
            verdict = {"flagged": None, "what": "не проверено (ошибки API)", "raw": ""}
        cache[pair["uid"]] = verdict
        append_jsonl(cache_path, {"uid": pair["uid"], **verdict})
        if i % 25 == 0 or i == len(todo):
            flagged_n = sum(1 for v in cache.values() if v.get("flagged"))
            print(f"[{i}/{len(todo)}] флагов: {flagged_n}")
        time.sleep(checker.get("request_delay_sec", 2.5))

    # Собираем результаты по всем парам
    stamp = dt.datetime.now().strftime("%Y%m%d")
    flagged, clean = [], []
    for pair in pairs:
        verdict = cache.get(pair["uid"], {"flagged": None, "what": ""})
        row = {**pair, "prescreen_what": verdict.get("what", "")}
        if verdict.get("flagged"):
            flagged.append(row)
        elif verdict.get("flagged") is False:
            clean.append(row)

    save_jsonl("data/processed/qa_prescreen.jsonl", [
        {**p, "prescreen_flagged": cache.get(p["uid"], {}).get("flagged"),
         "prescreen_what": cache.get(p["uid"], {}).get("what", "")}
        for p in pairs
    ])

    write_review_csv(f"data/review/review_flagged_{len(flagged)}_{stamp}.csv", flagged,
                     extra_cols=["prescreen_what"])
    rng = random.Random(42)
    sample_n = min(100, len(clean))
    clean_sample = rng.sample(clean, sample_n)
    write_review_csv(f"data/review/review_clean_sample_{sample_n}_{stamp}.csv", clean_sample)

    print(f"\nИтог: флагов {len(flagged)}, чистых {len(clean)}, "
          f"не проверено {sum(1 for p in pairs if cache.get(p['uid'], {}).get('flagged') is None)}")
    print(f"Ручная проверка: {len(flagged)} флагнутых + {sample_n} случайных чистых "
          f"(~1 час работы)")
    print("CSV в data/review/ — заполни verdict (OK/BAD_ANSWER/BAD_QUESTION/BAD_REF/OTHER)")


if __name__ == "__main__":
    main()
