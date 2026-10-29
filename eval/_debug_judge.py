# -*- coding: utf-8 -*-
"""Объективная проверка: судья сломан или base-модель так плоха?

Считаем без LLM: есть ли в ответе ссылка на правильную статью (regex),
и fuzzy-сходство ответа с эталоном.
"""
import re
import sys

from rapidfuzz import fuzz

sys.path.insert(0, "data/scripts")
from common import load_jsonl, utf8_console  # noqa: E402

utf8_console()

REF_RE = re.compile(r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", re.IGNORECASE)
answers = load_jsonl("eval/runs/answers_base.jsonl")

no_ref, correct_ref, wrong_ref = 0, 0, 0
similarities = []
for a in answers:
    refs = REF_RE.findall(a["answer"])
    if not refs:
        no_ref += 1
    elif a["article"] in refs:
        correct_ref += 1
    else:
        wrong_ref += 1
    similarities.append(fuzz.partial_ratio(a["reference"].lower(), a["answer"].lower()))

n = len(answers)
print(f"Ответов: {n}")
print(f"  без ссылки вовсе:     {no_ref} ({no_ref / n:.0%})")
print(f"  ссылка на ПРАВИЛЬНУЮ статью: {correct_ref} ({correct_ref / n:.0%})")
print(f"  ссылка на чужую:      {wrong_ref} ({wrong_ref / n:.0%})")
print(f"  медиана fuzzy-сходства с эталоном: {sorted(similarities)[n // 2]:.0f}")

print("\n--- 3 примера ответов base (с эталоном для сравнения) ---")
for a in answers[:3]:
    print(f"\n[ст. {a['article']}] {a['question'][:110]}")
    print(f"BASE: {a['answer'][:250]}")
    print(f"ЭТАЛОН: {a['reference'][:250]}")

scores = load_jsonl("eval/runs/scores_base.jsonl")
print("\n--- Судейские вердикты для тех же пар ---")
by_id = {s["id"]: s for s in scores}
for a in answers[:3]:
    s = by_id.get(a["id"])
    if s:
        print(f"[{a['id']}] acc={s['accuracy']} ref={s['reference']} "
              f"halluc={s['hallucination']} wrong_ref={s['wrong_reference']} "
              f"| {s['comment'][:100]}")
