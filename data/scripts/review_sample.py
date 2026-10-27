# -*- coding: utf-8 -*-
"""Этап 1: выборка для ручной вычитки (план: 500 случайных + 50 отфильтрованных).

Вычитка — главная защита от «garbage in, garbage out» и главный кейс для интервью
(«как обеспечивал качество данных?»). Отмечай вердикты в колонке verdict прямо в CSV:

    OK            — вопрос и ответ корректны, ссылка верна
    BAD_ANSWER    — ответ неточный/неполный/не следует из статьи
    BAD_QUESTION  — вопрос некорректен или неоднозначен
    BAD_REF       — ссылка не на ту статью / неверный формат ссылки
    OTHER         — прочее (опиши в note)

Запуск:
    python data/scripts/review_sample.py --n 500
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import random
import sys

from common import load_jsonl, utf8_console

COLUMNS = ["uid", "article", "chapter", "question_type", "question", "answer",
           "verdict", "note"]


def sample_stratified(pairs: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in pairs:
        strata[pair["question_type"]].append(pair)
    per_type = {t: max(1, round(n * len(rows) / len(pairs))) for t, rows in strata.items()}
    sampled: list[dict] = []
    for qtype, rows in sorted(strata.items()):
        rng.shuffle(rows)
        sampled += rows[:per_type[qtype]]
    return sampled[:n]


def write_csv(path: str, rows: list[dict], extra_reason: bool = False) -> None:
    columns = COLUMNS + (["reject_reason"] if extra_reason else [])
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # BOM — для Excel
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/qa_dedup.jsonl")
    parser.add_argument("--rejected-input", default="data/processed/qa_rejected.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n-rejected", type=int, default=50,
                        help="сколько отфильтрованных пар взять для проверки фильтров")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_jsonl(args.input)
    if args.n > len(pairs):
        sys.exit(f"В датасете только {len(pairs)} пар — меньше запрошенных {args.n}")
    sampled = sample_stratified(pairs, args.n, args.seed)

    stamp = dt.datetime.now().strftime("%Y%m%d")
    review_path = f"data/review/review_{args.n}_{stamp}.csv"
    write_csv(review_path, sampled)
    print(f"Выборка на вычитку: {review_path} ({len(sampled)} пар, "
          f"стратифицировано по типам)")

    try:
        rejected = load_jsonl(args.rejected_input)
        rng = random.Random(args.seed + 1)
        rng.shuffle(rejected)
        rejected_path = f"data/review/review_filtered_out_{args.n_rejected}_{stamp}.csv"
        write_csv(rejected_path, rejected[:args.n_rejected], extra_reason=True)
        print(f"Проверка фильтров: {rejected_path} ({min(args.n_rejected, len(rejected))} пар)")
    except FileNotFoundError:
        print("qa_rejected.jsonl не найден — сначала запусти filters.py")

    print("\nИнструкция: data/review/README.md")
    print("После вычитки посчитай долю брака — она пойдёт в README и карточку датасета.")


if __name__ == "__main__":
    main()
