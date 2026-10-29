# -*- coding: utf-8 -*-
"""Error analysis: куда указывают ссылки tuned-модели? Тематические соседи или случайные?"""
import sys
from collections import Counter

sys.path.insert(0, "data/scripts")
from common import load_json, load_jsonl, utf8_console  # noqa: E402

utf8_console()

REF_RE = re.compile(r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", re.IGNORECASE) \
    if False else __import__("re").compile(
        r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", 2 | 64)

import re
REF_RE = re.compile(r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", re.IGNORECASE)

answers = load_jsonl("eval/runs/answers_tuned.jsonl")
articles = {a["number"]: a for a in load_json("data/processed/tk_articles.json")["articles"]}

gaps = []
cited_counter = Counter()
for a in answers:
    refs = REF_RE.findall(a["answer"])
    if not refs or a["article"] in refs:
        continue
    cited = refs[0]
    cited_counter[cited] += 1
    try:
        gap = abs(int(a["article"].split(".")[0]) - int(cited.split(".")[0]))
        gaps.append(gap)
    except ValueError:
        gaps.append(999)

print(f"Ошибочных ссылок: {len(gaps)}")
if gaps:
    gaps_sorted = sorted(gaps)
    print(f"Медианный отступ от правильной статьи: {gaps_sorted[len(gaps_sorted)//2]} номеров")
    print(f"В пределах ±5 статей: {sum(1 for g in gaps if g <= 5)} "
          f"({sum(1 for g in gaps if g <= 5) / len(gaps):.0%})")
print("\nТоп-10 ошибочно цитируемых статей:")
for num, count in cited_counter.most_common(10):
    title = articles.get(num, {}).get("title", "?")[:60]
    print(f"  ст. {num}: {count} раз — {title}")

print("\n--- 3 примера промахов ---")
shown = 0
for a in answers:
    refs = REF_RE.findall(a["answer"])
    if not refs or a["article"] in refs or shown >= 3:
        continue
    print(f"\n[правильная ст. {a['article']}: {articles[a['article']]['title'][:60]}]")
    print(f"Q: {a['question'][:120]}")
    print(f"A: {a['answer'][:200]}")
    shown += 1
