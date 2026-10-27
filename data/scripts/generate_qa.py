# -*- coding: utf-8 -*-
"""Этап 1, шаг 3: генерация QA-пар по статьям ТК РФ через LLM API.

Для каждой активной статьи просим сильную модель сгенерировать 2–3 вопроса разных
типов (определения, сроки, размеры, порядок действий, ситуационные) и ответ в
едином формате: краткий ответ → обоснование со ссылкой «ст. N ТК РФ».

Скрипт умеет докачиваться: уже сгенерированные статьи пропускаются по output-файлу.

Запуск:
    python data/scripts/generate_qa.py --config configs/generate_qa.yaml
    python data/scripts/generate_qa.py --config ... --limit 3 --dry-run   # показать промпты
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import yaml

from common import PROCESSED_DIR, append_jsonl, load_json, load_jsonl, utf8_console
from providers import extract_json_array, make_provider

SYSTEM_PROMPT = """\
Ты — методист-юрист, готовишь учебный датасет по трудовому праву РФ.
По фрагменту Трудового кодекса РФ нужно составить вопросы и эталонные ответы.

Требования к вопросам:
1. Вопрос должен однозначно отвечаться ТОЛЬКО по тексту данной статьи. Не используй
   знания, которых в статье нет. Если информации для вопроса конкретного типа мало —
   сформулируй вопрос другого типа из разрешённых.
2. Не копируй формулировки статьи дословно — перефразируй, как спросил бы человек.
3. Ситуационные вопросы формулируй от лица работника или работодателя
   («Может ли работодатель…», «Что делать, если…»).
4. Вопросы должны различаться по формулировкам, не повторять друг друга.

Требования к ответам — СТРОГО единый формат из двух строк:
<Краткий ответ одним предложением.>
<Обоснование: 1–3 предложения со ссылкой на статью в формате «ст. N ТК РФ».>

Ссылка на статью обязательна и должна указывать именно на эту статью.
Не выдумывай номера статей, не ссылайся на статьи вне данного фрагмента.

Верни результат строго в формате JSON-массива без пояснений:
[
  {"type": "<тип>", "question": "...", "answer": "..."}
]
"""

USER_PROMPT = """\
Тип статьи: ст. {num} ТК РФ — «{title}»{hierarchy}
Обязательные типы вопросов для этой статьи:
{types}

(В поле type ответа верни слаг из скобок, напр. "{type_example}".)

Текст статьи:
---
{text}
---

Составь {n} вопросов и эталонных ответов. Верни только JSON-массив."""

# Слаги типов и их описания для промпта генератора
TYPE_LABELS = {
    "definition": "definition — что это такое: определения, понятия",
    "deadline": "deadline — сроки: за сколько, в какой срок, когда",
    "amount": "amount — размеры: ставки, компенсации, минимальные значения",
    "procedure": "procedure — порядок действий: как оформить, что сделать",
    "situational": "situational — ситуационный: может ли работодатель/работник…",
}


def build_prompt(article: dict, types: list[str], n: int) -> str:
    hierarchy = ""
    parts = [p for p in (article.get("part"), article.get("section"),
                         article.get("chapter_title")) if p]
    if parts:
        hierarchy = f"\nРасположение: {' / '.join(parts)}"
    type_lines = "\n".join(f"- {TYPE_LABELS.get(t, t)}" for t in types)
    return USER_PROMPT.format(
        num=article["number"], title=article["title"], hierarchy=hierarchy,
        types=type_lines, type_example=types[0] if types else "definition",
        n=n, text=article["text"],
    )


def parse_pairs(raw_answer: str, article: dict, allowed_types: set[str]) -> list[dict]:
    """Парсим и первично валидируем ответ генератора."""
    items = extract_json_array(raw_answer)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pairs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        qtype = str(item.get("type", "")).strip().lower()
        if not question or not answer:
            continue
        pairs.append({
            "id": None,  # назначим после фильтрации, при финальной сборке
            "article": article["number"],
            "chapter": article.get("chapter"),
            "question_type": qtype if qtype in allowed_types else "other",
            "question": question,
            "answer": answer,
            "source_article_url": article["url"],
            "meta": {
                "generator": f"{ARTICLE_GENERATOR}",
                "generated_at": now,
                "raw_type": qtype,
            },
        })
    return pairs


ARTICLE_GENERATOR = ""  # заполнится в run()


def run(config_path: str, limit: int | None, only_articles: list[str] | None,
        dry_run: bool) -> None:
    global ARTICLE_GENERATOR

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    provider = None
    if not dry_run:
        provider = make_provider(config["provider"], config["model"])
    ARTICLE_GENERATOR = f"{config['provider']}:{config['model']}"

    articles = load_json(config["input_articles"])["articles"]
    allowed_types = set(config["question_types"])

    done = set()
    try:
        done = {r["article"] for r in load_jsonl(config["output_raw"])}
        print(f"Докачка: в {config['output_raw']} уже есть статьи: {len(done)}")
    except FileNotFoundError:
        pass

    todo = []
    for article in articles:
        if article["status"] != "active" or article["number"] in done:
            continue
        if len(article["text"]) < 100:  # слишком короткие не информативны
            continue
        if only_articles and article["number"] not in only_articles:
            continue
        todo.append(article)
    if limit:
        todo = todo[:limit]
    print(f"К генерации: {len(todo)} статей "
          f"(ожидается ~{len(todo) * config['n_questions_per_article']} пар)")

    n_questions = config["n_questions_per_article"]
    all_types = config["question_types"]
    stats = {"articles": 0, "pairs": 0, "rejected_articles": 0}

    for index, article in enumerate(todo):
        # типы вопросов по кругу, чтобы распределение было равномерным
        offset = index * n_questions
        types = [all_types[(offset + j) % len(all_types)] for j in range(n_questions)]
        prompt = build_prompt(article, sorted(set(types)), n_questions)

        if dry_run:
            print("=" * 80)
            print(f"[DRY RUN] ст. {article['number']}")
            print(SYSTEM_PROMPT)
            print(prompt)
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        raw_answer = None
        for attempt in range(1, config["max_retries"] + 1):
            try:
                raw_answer = provider.chat(
                    messages, temperature=config["temperature"],
                    max_tokens=config["max_tokens"])
                pairs = parse_pairs(raw_answer, article, allowed_types)
                break
            except Exception as exc:  # сеть / JSON / лимиты
                print(f"\n[ст. {article['number']}] попытка {attempt} не удалась: {exc}")
                time.sleep(5 * attempt)
        else:
            append_jsonl(config["output_rejects"], {
                "article": article["number"], "error": "max_retries_exceeded",
                "raw": raw_answer, "at": dt.datetime.now().isoformat(timespec="seconds"),
            })
            stats["rejected_articles"] += 1
            continue

        for pair in pairs:
            append_jsonl(config["output_raw"], pair)
        stats["articles"] += 1
        stats["pairs"] += len(pairs)
        if (index + 1) % 10 == 0 or index + 1 == len(todo):
            print(f"[{index + 1}/{len(todo)}] ст. {article['number']}: +{len(pairs)} пар "
                  f"(всего {stats['pairs']})")
        time.sleep(config["request_delay_sec"])

    print(f"\nИтог: статей {stats['articles']}, пар {stats['pairs']}, "
          f"статей с ошибками {stats['rejected_articles']}")
    if not dry_run and stats["pairs"]:
        print(f"Сырой датасет: {config['output_raw']}\n"
              f"Дальше: python data/scripts/filters.py --config {config_path}")


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generate_qa.yaml")
    parser.add_argument("--limit", type=int, default=None, help="только N статей (тест)")
    parser.add_argument("--articles", default=None,
                        help="конкретные статьи через запятую, напр. 80,81,99")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать промпты без обращения к API")
    args = parser.parse_args()

    only = args.articles.split(",") if args.articles else None
    run(args.config, args.limit, only, args.dry_run)


if __name__ == "__main__":
    main()
