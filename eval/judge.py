# -*- coding: utf-8 -*-
"""Оценка ответов LLM-as-judge (этапы 2 и 4).

Судья — ДРУГАЯ сильная модель (GigaChat), не та, что обучается. Оценивает по 4
критериям 1–5 (фактическая точность, полнота, корректность ссылки на статью,
формат) + флаги галлюцинаций и неверной ссылки. Судье даётся текст статьи —
оценка заземлена, а не «на глаз».

Калибровка (обязательно по плану!): размrkи 30 ответов вручную (CSV), затем
    python eval/judge.py --answers ... --name base --human eval/human_scores_*.csv

Запуск:
    python eval/judge.py --answers eval/runs/answers_base.jsonl --name base
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics
import sys

import yaml

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from common import append_jsonl, load_env, load_json, load_jsonl, save_json, save_jsonl, utf8_console
from providers import make_provider

JUDGE_SYSTEM = """\
Ты — строгий эксперт по трудовому праву РФ. Оцениваешь ответ ассистента на вопрос.
Эталонный ответ и текст статьи даны для сверки. Оценивай независимо и честно.

Критерии (каждый 1–5):
- accuracy:      фактическая точность (нет ли утверждений, противоречащих статье/эталону)
- completeness:  полнота (все существенные части ответа из эталона/статьи раскрыты)
- reference:     корректность ссылки на статью (нужная статья, формат «ст. N ТК РФ»)
- format:        соблюдение формата: краткий ответ, затем обоснование со ссылкой

Флаги:
- hallucination: «да», если ассистент утверждает то, чего нет в статье и эталоне
- wrong_reference: «да», если ссылка ведёт не на ту статью или её нет

Верни строго JSON без пояснений:
{"accuracy": 1-5, "completeness": 1-5, "reference": 1-5, "format": 1-5,
 "hallucination": "да|нет", "wrong_reference": "да|нет",
 "comment": "<одно предложение>"}"""

JUDGE_USER = """\
Вопрос: {question}

Текст статьи {article} ТК РФ:
---
{article_text}
---
Эталонный ответ (написан человеком):
---
{reference}
---
Ответ ассистента:
---
{answer}
---"""


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"нет JSON в ответе судьи: {text[:120]}")
    data = json.loads(text[start:end + 1])
    scores = {}
    for key in ("accuracy", "completeness", "reference", "format"):
        scores[key] = int(re.search(r"\d+", str(data[key])).group())
        if not 1 <= scores[key] <= 5:
            raise ValueError(f"оценка вне шкалы: {key}={scores[key]}")
    scores["hallucination"] = str(data["hallucination"]).lower().startswith("да")
    scores["wrong_reference"] = str(data["wrong_reference"]).lower().startswith("да")
    scores["comment"] = str(data.get("comment", "")).strip()
    return scores


def uid_of(answer_row: dict) -> str:
    raw = f"{answer_row['id']}|{answer_row['answer']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def judge_all(args, config: dict) -> None:
    answers = load_jsonl(args.answers)
    missing_ref = [a for a in answers if not a.get("reference")]
    if missing_ref:
        sys.exit(f"У {len(missing_ref)} ответов пустой reference — сначала заполни эталоны "
                 f"(eval/build_benchmark.py), судье нужен эталон.")

    articles = {a["number"]: a for a in load_json(
        "data/processed/tk_articles.json")["articles"]}
    provider = make_provider(config["judge"]["provider"], config["judge"]["model"])

    cache_path = "eval/runs/judge_cache.jsonl"
    cache: dict[str, dict] = {}
    try:
        for row in load_jsonl(cache_path):
            cache[row["uid"]] = row
    except FileNotFoundError:
        pass

    results, failures = [], 0
    for i, row in enumerate(answers, 1):
        uid = uid_of(row)
        article = articles.get(row.get("article"))
        article_text = article["text"] if article else "(статья не найдена)"

        if uid in cache:
            scores = cache[uid]["scores"]
        else:
            prompt = JUDGE_USER.format(question=row["question"], article=row.get("article"),
                                       article_text=article_text,
                                       reference=row["reference"], answer=row["answer"])
            scores = None
            for attempt in range(1, config["judge"].get("max_retries", 3) + 1):
                try:
                    raw = provider.chat(
                        [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": prompt}],
                        temperature=config["judge"].get("temperature", 0.0),
                        max_tokens=config["judge"].get("max_tokens", 512))
                    scores = parse_judge_json(raw)
                    break
                except Exception as exc:
                    print(f"\n[{row['id']}] попытка {attempt}: {exc}")
            if scores is None:
                failures += 1
                continue
            append_jsonl(cache_path, {"uid": uid, "scores": scores})
        results.append({**scores, "id": row["id"], "uid": uid})
        if i % 10 == 0 or i == len(answers):
            print(f"[{i}/{len(answers)}] оценено")

    out_path = f"eval/runs/scores_{args.name}.jsonl"
    save_jsonl(out_path, results)

    mean = lambda key: statistics.mean(r[key] for r in results)
    summary = {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "name": args.name,
        "judge": f"{config['judge']['provider']}:{config['judge']['model']}",
        "n_judged": len(results), "n_failed": failures,
        "mean_accuracy": round(mean("accuracy"), 2),
        "mean_completeness": round(mean("completeness"), 2),
        "mean_reference": round(mean("reference"), 2),
        "mean_format": round(mean("format"), 2),
        "mean_overall": round(statistics.mean(
            [mean(k) for k in ("accuracy", "completeness", "reference", "format")]), 2),
        "hallucination_rate": round(statistics.mean(
            [r["hallucination"] for r in results]), 3),
        "wrong_reference_rate": round(statistics.mean(
            [r["wrong_reference"] for r in results]), 3),
    }
    save_json(f"eval/runs/judge_summary_{args.name}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print(f"Не оценено {failures} ответов — перезапусти, кэш докачает.")
    print(f"\nДальше: python eval/compare.py (после обоих прогонов)")


def calibrate(args) -> None:
    """Сравнение с ручными оценками: совпадает ли «4» судьи с твоей «4»?"""
    human: dict[str, dict] = {}
    with open(args.human, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            human[row["id"]] = {k: int(row[k]) for k in
                                ("accuracy", "completeness", "reference", "format")}

    scores = {r["id"]: r for r in load_jsonl(f"eval/runs/scores_{args.name}.jsonl")}
    exact, within1, deltas = 0, 0, []
    compared = 0
    for qid, h_scores in human.items():
        if qid not in scores:
            continue
        compared += 1
        j_scores = scores[qid]
        for key in h_scores:
            delta = j_scores[key] - h_scores[key]
            deltas.append(abs(delta))
            exact += (delta == 0)
            within1 += (abs(delta) <= 1)
    if not compared:
        sys.exit("Нет пересечений id — проверь CSV ручных оценок.")
    n = compared * 4
    print(f"Сравнено {compared} ответов × 4 критерия = {n} оценок")
    print(f"Точное совпадение: {exact / n:.0%}")
    print(f"Совпадение ±1:     {within1 / n:.0%}")
    print(f"Средняя |Δ|:       {statistics.mean(deltas):.2f}")
    print("Ориентир: точных ≥50% и ±1 ≥90% — судью можно доверять; иначе правь промпт судьи.")


def main() -> None:
    utf8_console()
    load_env()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, help="jsonl из generate_answers.py")
    parser.add_argument("--name", required=True, help="метка прогона: base / tuned")
    parser.add_argument("--config", default="configs/generate_qa.yaml",
                        help="судья берётся из этого конфига (клч judge) либо configs/judge.yaml")
    parser.add_argument("--human", default=None,
                        help="CSV ручных оценок → режим калибровки вместо оценки")
    args = parser.parse_args()

    if args.human:
        calibrate(args)
        return

    config = None
    for path in (args.config, "configs/judge.yaml"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
            if "judge" in config:
                break
        except FileNotFoundError:
            continue
    if config is None or "judge" not in config:
        sys.exit("В конфиге нет секции judge — добавь configs/judge.yaml")

    judge_all(args, config)


if __name__ == "__main__":
    main()
