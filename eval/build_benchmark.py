# -*- coding: utf-8 -*-
"""Этап 2, шаг 1: собрать мини-бенчмарк из test-сплита.

Берём 100 вопросов стратифицированно по типам (детерминированно, seed).
ЭТАЛОННЫЕ ОТВЕТЫ ПИШЕТСЯ ВРУЧНУЮ по тексту ТК — не LLM (иначе судья оценивает
подгонку). Для этого создаётся CSV-шаблон: заполни колонку reference и пересохрани.

Запуск:
    python eval/build_benchmark.py --n 100
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import random
import sys

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from common import load_jsonl, save_jsonl, utf8_console, PROCESSED_DIR


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default=str(PROCESSED_DIR / "test.jsonl"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/benchmark.jsonl")
    args = parser.parse_args()

    test = load_jsonl(args.test)
    if args.n > len(test):
        sys.exit(f"В test только {len(test)} пар")

    rng = random.Random(args.seed)
    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in test:
        strata[pair["question_type"]].append(pair)
    per_type = {t: max(1, round(args.n * len(rows) / len(test))) for t, rows in strata.items()}

    benchmark = []
    for qtype, rows in sorted(strata.items()):
        rng.shuffle(rows)
        benchmark += rows[:per_type[qtype]]
    benchmark = sorted(benchmark[:args.n], key=lambda r: r["uid"])
    for i, pair in enumerate(benchmark, 1):
        pair["benchmark_id"] = f"bm_{i:03d}"

    save_jsonl(args.out, [{
        "id": p["benchmark_id"], "uid": p["uid"], "article": p["article"],
        "chapter": p["chapter"], "question_type": p["question_type"],
        "question": p["question"], "reference": "",  # <-- заполняется вручную
    } for p in benchmark])

    stamp = dt.datetime.now().strftime("%Y%m%d")
    csv_path = f"eval/benchmark_references_{stamp}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "id", "article", "chapter", "question_type", "question", "reference", "note"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark)

    print(f"Бенчмарк: {args.out} ({len(benchmark)} вопросов)")
    print(f"Шаблон эталонов: {csv_path}")
    print("\nДАЛЬШЕ ВРУЧНУЮ:")
    print("1. Для каждого вопроса открой статью в data/processed/tk_articles.json")
    print("2. Напиши эталонный ответ в формате «краткий ответ + обоснование со ссылкой»")
    print("3. Перенеси эталоны в " + args.out + " (поле reference)")
    print("4. Только после этого — python src/generate_answers.py --mode base ...")


if __name__ == "__main__":
    main()
