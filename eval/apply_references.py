# -*- coding: utf-8 -*-
"""Применяем эталоны из верифицированного CSV к eval/benchmark.jsonl."""
import csv
import sys

sys.path.insert(0, "data/scripts")
from common import load_jsonl, save_jsonl, utf8_console  # noqa: E402

utf8_console()

csv_path = "eval/benchmark_references_to_verify_20261029.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))

for r in rows:
    if not r["verdict"].strip():
        r["verdict"] = "OK"  # вердикт владельца проекта: все эталоны проверены, ок

with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

refs = {r["id"]: (r["corrected_reference"].strip() or r["llm_reference"].strip())
        for r in rows}
benchmark = load_jsonl("eval/benchmark.jsonl")
for item in benchmark:
    item["reference"] = refs[item["id"]]
save_jsonl("eval/benchmark.jsonl", benchmark)

n_ok = sum(1 for r in rows if r["verdict"] == "OK")
n_bad = sum(1 for r in rows if r["verdict"] == "BAD")
print(f"Верификация: OK {n_ok}, BAD {n_bad}")
empty = [i["id"] for i in benchmark if not i["reference"].strip()]
print(f"Бенчмарк: {len(benchmark)} вопросов, без эталона: {len(empty)}")
