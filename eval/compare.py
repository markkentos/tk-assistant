# -*- coding: utf-8 -*-
"""Этап 4: сравнение base vs tuned — итоговая таблица.

Сводит судейские оценки (eval/runs/scores_*.jsonl, judge_summary_*.json) и
ROUGE-L против ручных эталонов в одну таблицу → eval/results.md.

Запуск:
    python eval/compare.py --base-base answers_base.jsonl --tuned-answers answers_tuned.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt

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
    parser.add_argument("--base-answers", default="eval/runs/answers_base.jsonl")
    parser.add_argument("--tuned-answers", default="eval/runs/answers_tuned.jsonl")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    rows = []
    for name, answers_path in (("base", args.base_answers),
                               ("tuned", args.tuned_answers)):
        try:
            summary = load_json(f"eval/runs/judge_summary_{name}.json")
        except FileNotFoundError:
            print(f"Нет judge_summary_{name}.json — прогони judge.py --name {name}")
            continue
        rouge = rouge_l(answers_path)
        rows.append({
            "name": name,
            "judge": summary["mean_overall"],
            "accuracy": summary["mean_accuracy"],
            "completeness": summary["mean_completeness"],
            "reference": summary["mean_reference"],
            "format": summary["mean_format"],
            "hallucinations": f"{summary['hallucination_rate']:.1%}",
            "wrong_refs": f"{summary['wrong_reference_rate']:.1%}",
            "rouge_l": f"{rouge:.3f}" if rouge is not None else "—",
        })

    if not rows:
        return

    lines = [
        "# Результаты: base vs tuned",
        f"\nОбновлено: {dt.datetime.now().isoformat(timespec='seconds')}",
        "\n| Метрика | " + " | ".join(r["name"] for r in rows) + " |",
        "|---|" + "---|" * len(rows),
        f"| Judge score (1–5), mean | " + " | ".join(str(r["judge"]) for r in rows) + " |",
        f"| — фактическая точность | " + " | ".join(str(r["accuracy"]) for r in rows) + " |",
        f"| — полнота | " + " | ".join(str(r["completeness"]) for r in rows) + " |",
        f"| — корректность ссылки | " + " | ".join(str(r["reference"]) for r in rows) + " |",
        f"| — формат | " + " | ".join(str(r["format"]) for r in rows) + " |",
        f"| Галлюцинации, % | " + " | ".join(r["hallucinations"] for r in rows) + " |",
        f"| Неверные ссылки, % | " + " | ".join(r["wrong_refs"] for r in rows) + " |",
        f"| ROUGE-L vs эталон | " + " | ".join(r["rouge_l"] for r in rows) + " |",
    ]
    if len(rows) == 2:
        delta = rows[1]["judge"] - rows[0]["judge"]
        lines.append(f"\n**Δ judge score (tuned − base): {delta:+.2f}**")
    save_text(args.out, "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {args.out} — перенеси таблицу в README.")


if __name__ == "__main__":
    main()
