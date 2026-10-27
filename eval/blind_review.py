# -*- coding: utf-8 -*-
"""Этап 4: подготовка слепой ручной оценки (30 пар).

Перемешивает ответы base и tuned, НЕ подписывая, какая модель где. Размечаешь
CSV колонкой better (A/B/tie/both_bad), потом --reveal показывает, кто есть кто.

Запуск:
    python eval/blind_review.py --n 30                 # создать CSV для разметки
    python eval/blind_review.py --reveal --key ...json # расшифровать после разметки
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random

from common import load_json, load_jsonl, save_json, utf8_console


def build(args) -> None:
    base = {r["id"]: r for r in load_jsonl(args.base_answers)}
    tuned = {r["id"]: r for r in load_jsonl(args.tuned_answers)}
    common_ids = sorted(set(base) & set(tuned))
    if args.n > len(common_ids):
        raise SystemExit(f"Пересечение id: {len(common_ids)} < {args.n}")

    rng = random.Random(args.seed)
    rng.shuffle(common_ids)
    selected = common_ids[:args.n]

    rows, key = [], {}
    for pair_id in selected:
        assignment = {"A": "base", "B": "tuned"}
        if rng.random() < 0.5:
            assignment = {"A": "tuned", "B": "base"}
        key[pair_id] = assignment
        rows.append({
            "pair_id": pair_id,
            "article": base[pair_id]["article"],
            "question": base[pair_id]["question"],
            "answer_A": base[pair_id]["answer"] if assignment["A"] == "base"
            else tuned[pair_id]["answer"],
            "answer_B": tuned[pair_id]["answer"] if assignment["B"] == "tuned"
            else base[pair_id]["answer"],
            "better": "",   # A | B | tie | both_bad
            "note": "",
        })

    stamp = dt.datetime.now().strftime("%Y%m%d")
    csv_path = f"eval/blind_review_{args.n}_{stamp}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    key_path = f"eval/blind_review_key_{args.n}_{stamp}.json"
    save_json(key_path, key)
    print(f"Слепая разметка: {csv_path} ({len(rows)} пар)")
    print(f"Ключ (не подглядывать до разметки!): {key_path}")
    print("После разметки: python eval/blind_review.py --reveal --csv ... --key ...")


def reveal(args) -> None:
    key = load_json(args.key)
    with open(args.csv, "r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    wins = {"base": 0, "tuned": 0, "tie": 0, "both_bad": 0}
    for row in rows:
        choice = row["better"].strip().lower()
        if choice in ("a", "b"):
            winner_model = key[row["pair_id"]][choice.upper()]
            wins[winner_model] += 1
        else:
            wins[choice or "tie"] += 1
    total = len(rows)
    print(f"Слепая оценка, {total} пар:")
    for name, count in wins.items():
        print(f"  {name:9s}: {count} ({count / max(total, 1):.0%})")
    print("\nЭто независимый «второй взгляд» на метрики судьи — расхождение с ним"
          " само по себе интересный результат для error analysis.")


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-answers", default="eval/runs/answers_base.jsonl")
    parser.add_argument("--tuned-answers", default="eval/runs/answers_tuned.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reveal", action="store_true")
    parser.add_argument("--csv", default=None, help="размеченный CSV (для --reveal)")
    parser.add_argument("--key", default=None, help="ключ (для --reveal)")
    args = parser.parse_args()

    if args.reveal:
        if not (args.csv and args.key):
            raise SystemExit("--reveal требует --csv и --key")
        reveal(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
