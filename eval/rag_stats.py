# -*- coding: utf-8 -*-
"""Статистика e6v2+RAG только по валидным ответам (без ОШИБКА-строк)."""
import json
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")

scores = {r["id"]: r for r in
          (json.loads(l) for l in open("eval/runs/scores_e6v2_rag.jsonl", encoding="utf-8"))}
answers = {r["id"]: r for r in
           (json.loads(l) for l in open("eval/runs/answers_e6v2_rag.jsonl", encoding="utf-8"))}

rows = [scores[u] for u, a in answers.items()
        if u in scores and not a["answer"].startswith("ОШИБКА")]
n = len(rows)
overall = statistics.mean(
    (r["accuracy"] + r["completeness"] + r["reference"] + r["format"]) / 4
    for r in rows)


def m(key):
    return round(statistics.mean(r[key] for r in rows), 2)


print(f"валидных: {n}")
print(f"overall: {overall:.2f} | accuracy: {m('accuracy')} | "
      f"completeness: {m('completeness')} | reference: {m('reference')} | "
      f"format: {m('format')}")
print(f"галлюцинации: {round(100 * sum(r['hallucination'] for r in rows) / n)}% | "
      f"неверные ссылки: {round(100 * sum(r['wrong_reference'] for r in rows) / n)}%")
