# -*- coding: utf-8 -*-
"""Финализация датасета по итогам ручной разметки (вариант B).

Читает размеченные CSV (флагнутые + случайная сотня чистых), удаляет
подтверждённо-плохие пары, считает честные метрики качества.
"""
import csv
import sys

sys.path.insert(0, "data/scripts")
from common import load_jsonl, save_json, save_jsonl, utf8_console  # noqa: E402

utf8_console()

BAD_VERDICTS = {"BAD_ANSWER", "BAD_QUESTION", "BAD_REF", "OTHER"}


def read_marked(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    empty = [r for r in rows if not r["verdict"].strip()]
    if empty:
        sys.exit(f"В {path} не размечено {len(empty)} пар — заполни verdict у всех.")
    return rows


flagged = read_marked("data/review/review_flagged_14_20261029.csv")
clean_sample = read_marked("data/review/review_clean_sample_100_20261029.csv")

flag_confirmed = [r for r in flagged if r["verdict"] in BAD_VERDICTS]
flag_false = [r for r in flagged if r["verdict"] == "OK"]
clean_bad = [r for r in clean_sample if r["verdict"] in BAD_VERDICTS]

print("=== Результаты разметки ===")
print(f"Флагнутые ({len(flagged)}): подтверждено {len(flag_confirmed)}, "
      f"ложных срабатываний {len(flag_false)}")
flag_types = {}
for r in flag_confirmed:
    flag_types[r["verdict"]] = flag_types.get(r["verdict"], 0) + 1
print(f"  по типам: {flag_types}")
print(f"Чистая сотня ({len(clean_sample)}): брак {len(clean_bad)} "
      f"({len(clean_bad) / len(clean_sample):.0%})")
for r in clean_bad:
    print(f"  - {r['verdict']} ст. {r['article']}: {r['note'] or r['question'][:80]}")

bad_uids = {r["uid"] for r in flag_confirmed} | {r["uid"] for r in clean_bad}

pairs = load_jsonl("data/processed/qa_dedup.jsonl")
final = [p for p in pairs if p["uid"] not in bad_uids]
save_jsonl("data/processed/qa_final.jsonl", final)
print(f"\nДатасет: {len(pairs)} → {len(final)} (удалено {len(pairs) - len(final)})")

residual_note = (
    f"остаточный брак по случайной сотне: {len(clean_bad) / len(clean_sample):.0%}"
    if clean_sample else "")
save_json("data/processed/quality_report.json", {
    "protocol": "format → grounding → prescreen → manual (flagged + random 100)",
    "flagged_total": len(flagged),
    "flag_confirmed": len(flag_confirmed),
    "prescreen_false_positives": len(flag_false),
    "clean_sample_size": len(clean_sample),
    "clean_sample_defects": len(clean_bad),
    "removed_total": len(pairs) - len(final),
    "final_pairs": len(final),
    "residual": residual_note,
})
print("Метрики: data/processed/quality_report.json")
print("Дальше: python data/scripts/split_dataset.py --input data/processed/qa_final.jsonl")
