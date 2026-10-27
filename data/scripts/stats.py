# -*- coding: utf-8 -*-
"""Сводная статистика датасета → data/processed/dataset_report.md

Прогоняй после каждого шага пайплайна: raw → filtered → dedup → splits.
Отчёт идёт в README и в карточку датасета на HF.

Запуск:
    python data/scripts/stats.py
"""
from __future__ import annotations

import collections
import datetime as dt

from common import load_json, load_jsonl, save_text, utf8_console


def dist(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(r[key] for r in rows).items(),
                       key=lambda t: -t[1]))


def try_load_jsonl(path: str) -> list[dict]:
    try:
        return load_jsonl(path)
    except FileNotFoundError:
        return []


def main() -> None:
    utf8_console()
    tk = load_json("data/processed/tk_articles.json")
    active = [a for a in tk["articles"] if a["status"] == "active"]

    qa_raw = try_load_jsonl("data/processed/qa_raw.jsonl")
    qa_filtered = try_load_jsonl("data/processed/qa_filtered.jsonl")
    qa_dedup = try_load_jsonl("data/processed/qa_dedup.jsonl")
    splits = {name: try_load_jsonl(f"data/processed/{name}.jsonl")
              for name in ("train", "val", "test")}

    filter_stats = {}
    try:
        filter_stats = load_json("data/processed/filter_stats.json")
    except FileNotFoundError:
        pass

    lines = [
        "# Отчёт по датасету",
        f"\nОбновлён: {dt.datetime.now().isoformat(timespec='seconds')}",
        "\n## Источник",
        f"- Статей в ТК РФ: {len(tk['articles'])} (активных: {len(active)}, "
        f"утратили силу: {len(tk['articles']) - len(active)})",
        f"- Символов текста: {sum(len(a['text']) for a in tk['articles']):,}",
        "\n## Пайплайн",
        f"| Стадия | Пар |",
        f"|---|---|",
        f"| qa_raw (генерация) | {len(qa_raw)} |",
        f"| qa_filtered (фильтры) | {len(qa_filtered)} |",
        f"| qa_dedup (дедупликация) | {len(qa_dedup)} |",
    ]
    if filter_stats:
        lines.append(f"\nФильтры ({filter_stats.get('at', '')}): "
                     f"вход {filter_stats.get('input')}, прошло {filter_stats.get('kept')}, "
                     f"отсеяно по стадиям: {filter_stats.get('rejected_by_stage')}")
    if qa_raw:
        lines += [
            "\n## Типы вопросов (raw)",
            *[f"- {k}: {v}" for k, v in dist(qa_raw, "question_type").items()],
            f"\n- Статей покрыто: {len({r['article'] for r in qa_raw})} из {len(active)}",
            f"- Пар на статью: {len(qa_raw) / max(len({r['article'] for r in qa_raw}), 1):.2f}",
        ]
    if qa_dedup:
        lens = sorted(len(r["answer"]) for r in qa_dedup)
        med = lens[len(lens) // 2]
        lines += [
            "\n## Длины ответов (после дедупа)",
            f"- медиана: {med} симв., мин: {lens[0]}, макс: {lens[-1]}",
        ]
    lines.append("\n## Сплиты")
    lines.append("| Сплит | Пар |")
    lines.append("|---|---|")
    for name, rows in splits.items():
        lines.append(f"| {name} | {len(rows)} |")
    for name, rows in splits.items():
        if rows:
            lines.append(f"\nТипы вопросов в {name}: {dist(rows, 'question_type')}")

    report = "\n".join(lines) + "\n"
    save_text("data/processed/dataset_report.md", report)
    print(report)


if __name__ == "__main__":
    main()
