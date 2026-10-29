# -*- coding: utf-8 -*-
"""Калибровка судьи: 30 ответов для СЛЕПОЙ ручной оценки.

Судейские баллы сознательно НЕ показываются — сначала свои оценки, потом сравнение.

Запуск:
    python eval/make_calibration.py --answers eval/runs/answers_base.jsonl --n 30
Разметка: заполни колонки accuracy / completeness / reference / format (1-5).
Сравнение с судьёй:
    python eval/judge.py --name base --human eval/human_scores_*.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from common import load_jsonl, utf8_console  # noqa: E402


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="eval/runs/answers_base.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.answers)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    stamp = dt.datetime.now().strftime("%Y%m%d")
    path = f"eval/human_scores_{args.n}_{stamp}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "id", "article", "question", "answer",
            "accuracy", "completeness", "reference", "format", "note"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample)

    print(f"CSV для ручной оценки: {path} ({len(sample)} ответов)")
    print("Шкала 1-5 по каждому критерию:")
    print("  accuracy     — фактическая точность")
    print("  completeness — полнота ответа")
    print("  reference    — корректность ссылки на статью")
    print("  format       — соблюдение формата (краткий ответ + обоснование)")


if __name__ == "__main__":
    main()
