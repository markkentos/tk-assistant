# -*- coding: utf-8 -*-
"""Этап 1, финал: стратифицированный сплит train/val/test + финальные id.

Стратификация — по question_type, чтобы все сплиты видели все типы вопросов
одинаково. Сплит детерминирован (seed в CLI).

ВАЖНО: после этого скрипта test-сплит считается ЗАКРЫТЫМ до этапа 4.
Эталоны бенчмарка пишутся вручную по тексту ТК, а не LLM.

Запуск:
    python data/scripts/split_dataset.py [--train 0.78 --val 0.11 --test 0.11]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import random
import sys

from common import load_jsonl, save_json, save_jsonl, utf8_console

INPUT = "data/processed/qa_dedup.jsonl"


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--train", type=float, default=0.78)
    parser.add_argument("--val", type=float, default=0.11)
    parser.add_argument("--test", type=float, default=0.11)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"Доли должны давать 1.0, сейчас {total}")

    pairs = load_jsonl(args.input)
    rng = random.Random(args.seed)

    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in pairs:
        strata[pair["question_type"]].append(pair)

    train, val, test = [], [], []
    for qtype in sorted(strata):
        group = sorted(strata[qtype], key=lambda p: p["uid"])
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * args.train)
        n_val = round(n * args.val)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]

    for name, split in (("train", train), ("val", val), ("test", test)):
        rng.shuffle(split)
        save_jsonl(f"data/processed/{name}.jsonl", split)

    # Финальные id назначаем после сплита, только в train/val (test — при сборке бенчмарка)
    for i, pair in enumerate(train + val, 1):
        pair["id"] = f"qa_{i:06d}"
    save_jsonl("data/processed/train.jsonl", train)
    save_jsonl("data/processed/val.jsonl", val)

    def dist(rows):
        return dict(collections.Counter(r["question_type"] for r in rows))

    save_json("data/processed/split_stats.json", {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "fractions": {"train": args.train, "val": args.val,
                                         "test": args.test},
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "question_type_dist": {"train": dist(train), "val": dist(val), "test": dist(test)},
    })
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("Распределение по типам:", dist(train))
    print("\n*** TEST-СПЛИТ ЗАКРЫТ до этапа 4. Не смотри, не тюнь по нему. ***")
    print("Дальше (этап 2): python eval/build_benchmark.py --n 100")


if __name__ == "__main__":
    main()
