# -*- coding: utf-8 -*-
"""Этап 1, шаг 5: дедупликация QA-датасета (убивает самоповторы генератора).

По умолчанию — fuzzy-сравнение вопросов (rapidfuzz token_set_ratio).
Опционально --embeddings — дедуп по эмбеддингам (sentence-transformers, точнее
ловит перефразированные повторы, но требует модель).

Запуск:
    python data/scripts/dedup.py
    python data/scripts/dedup.py --embeddings   # если установлен sentence-transformers
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from rapidfuzz import fuzz

from common import load_jsonl, save_json, save_jsonl, utf8_console

INPUT = "data/processed/qa_filtered.jsonl"
OUTPUT = "data/processed/qa_dedup.jsonl"
FUZZY_THRESHOLD = 90        # token_set_ratio, 0-100
EMBED_THRESHOLD = 0.92      # cosine similarity


def dedup_fuzzy(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Жадная дедуп: сортируем детерминированно, оставляем первого, хвост — дубли."""
    kept, removed = [], []
    kept_questions: list[str] = []
    for pair in sorted(pairs, key=lambda p: (p["article"], p["question"])):
        duplicate = False
        # сравниваем в первую очередь с вопросами той же статьи (частый кейс),
        # но ловим и межстатейные повторы — так честнее
        for other in reversed(kept_questions[-400:]):
            if fuzz.token_set_ratio(pair["question"].lower(), other.lower()) >= FUZZY_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            removed.append(pair)
        else:
            kept.append(pair)
            kept_questions.append(pair["question"])
    return kept, removed


def dedup_embeddings(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Нужен sentence-transformers: pip install sentence-transformers")
    import numpy as np

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode([p["question"] for p in pairs], normalize_embeddings=True,
                              show_progress_bar=True)
    kept, removed = [], []
    kept_embs: list = []
    for pair, emb in sorted(zip(pairs, embeddings), key=lambda t: (t[0]["article"], t[0]["question"])):
        if kept_embs and max(float(np.dot(kept_embs, emb))) >= EMBED_THRESHOLD:
            removed.append(pair)
        else:
            kept.append(pair)
            kept_embs.append(emb)
    return kept, removed


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--embeddings", action="store_true",
                        help="дедуп по эмбеддингам вместо fuzzy")
    args = parser.parse_args()

    pairs = load_jsonl(args.input)
    print(f"Вход: {len(pairs)} пар")
    kept, removed = dedup_embeddings(pairs) if args.embeddings else dedup_fuzzy(pairs)
    save_jsonl(args.output, kept)

    removed_per_article: dict[str, int] = {}
    for pair in removed:
        removed_per_article[pair["article"]] = removed_per_article.get(pair["article"], 0) + 1
    top_removed = sorted(removed_per_article.items(), key=lambda t: -t[1])[:10]
    save_json("data/processed/dedup_stats.json", {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "embeddings" if args.embeddings else "fuzzy",
        "input": len(pairs), "kept": len(kept), "removed": len(removed),
        "top_articles_with_duplicates": dict(top_removed),
    })
    print(f"Удалено дубликатов: {len(removed)} → осталось {len(kept)}")
    print("Статьи-лидеры по повторам:", dict(top_removed))
    print("Дальше: python data/scripts/review_sample.py --n 500")


if __name__ == "__main__":
    main()
