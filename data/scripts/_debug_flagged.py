# -*- coding: utf-8 -*-
"""Показ флагнутых пар + файлы data/review/."""
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

rows = list(csv.DictReader(open("data/review/review_flagged_14_20261029.csv",
                                encoding="utf-8-sig")))
print(f"флагнутых: {len(rows)}\n")
for r in rows:
    print(f"- ст. {r['article']} [{r['question_type']}]")
    print(f"  Q: {r['question'][:120]}")
    print(f"  A: {r['answer'][:120]}")
    print(f"  флаг: {r['prescreen_what'][:130]}")

print()
for f in sorted(os.listdir("data/review")):
    print(f, os.path.getsize("data/review/" + f), "байт")
