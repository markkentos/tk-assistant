# -*- coding: utf-8 -*-
"""Этап 4: сравнение прогонов (base / tuned-e1 / tuned-e3 / ...) — итоговая таблица.

Сводит судейские оценки (eval/runs/judge_summary_*.json, scores_*.jsonl) и
ROUGE-L против ручных эталонов в одну таблицу → eval/results.md.

Запуск:
    python eval/compare.py --names base tuned tuned_e3
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from common import load_json, load_jsonl, save_text, utf8_console


def rouge_l(answers_path: str) -> float | None:
    """Средний ROUGE-L (F1) ответов модели против ручных эталонов."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for row in load_jsonl(answers_path):
        if not row.get("reference"):
            continue
        scores.append(scorer.score(row["reference"], row["answer"])["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else None


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", default="base tuned",
                        help="метки прогонов через пробел ( answers_{name}.jsonl )")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()
    names = args.names.split()

    rows = []
    for name in names:
        try:
            summary = load_json(f"eval/runs/judge_summary_{name}.json")
        except FileNotFoundError:
            print(f"Нет judge_summary_{name}.json — прогони judge.py --name {name}")
            continue
        rouge = rouge_l(f"eval/runs/answers_{name}.jsonl")
        rows.append({
            "name": name,
            "judge": summary["mean_overall"],
            "accuracy": summary["mean_accuracy"],
            "completeness": summary["mean_completeness"],
            "reference": summary["mean_reference"],
            "format": summary["mean_format"],
            "hallucinations": f"{summary['hallucination_rate']:.0%}",
            "wrong_refs": f"{summary['wrong_reference_rate']:.0%}",
            "rouge_l": f"{rouge:.3f}" if rouge is not None else "—",
        })
    if not rows:
        return

    header = list(rows[0])
    lines = [
        "# Результаты: сравнение прогонов",
        f"\nОбновлено: {dt.datetime.now().isoformat(timespec='seconds')}",
        "\n| Метрика | " + " | ".join(r["name"] for r in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    labels = {
        "judge": "Judge score (1–5), mean",
        "accuracy": "— фактическая точность",
        "completeness": "— полнота",
        "reference": "— корректность ссылки",
        "format": "— формат",
        "hallucinations": "Галлюцинации, %",
        "wrong_refs": "Неверные ссылки, %",
        "rouge_l": "ROUGE-L vs эталон",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | " + " | ".join(str(r[key]) for r in rows) + " |")

    best = max(rows, key=lambda r: r["judge"])
    worst = min(rows, key=lambda r: r["judge"])
    lines.append(f"\n**Лучший прогон: {best['name']} (judge {best['judge']}).**")
    if len(rows) >= 2:
        lines.append(f"Δ к base: {best['judge'] - worst['judge']:+.2f} … "
                     f"{best['judge'] - rows[0]['judge']:+.2f}")
    save_text(args.out, "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
