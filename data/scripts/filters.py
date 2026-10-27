# -*- coding: utf-8 -*-
"""Этап 1, шаг 4: автоматические фильтры качества QA-датасета.

Стадии (по возрастанию цены):
  format       — без API: длины, «?», наличие ссылки «ст. N ТК РФ», совпадение номера
  grounding    — API: вторая модель проверяет, следует ли ответ из текста статьи
  consistency  — API: два независимых ответа на вопрос по статье, fuzzy-сравнение
                 с эталонным ответом пары (расходятся — выбрасываем)

Вердикты API-стадий кэшируются (filters_cache.jsonl) — перезапуск не жжёт API заново.
Каждая пара получает uid (sha1) — стабильный ключ для вычитки и сплитов.

Запуск:
    python data/scripts/filters.py --config configs/filters.yaml
    python data/scripts/filters.py --config configs/filters.yaml --stages format   # только дёшево
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import time

import yaml

from common import append_jsonl, load_json, load_jsonl, save_json, save_jsonl, utf8_console
from providers import make_provider

REF_RE = re.compile(r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", re.IGNORECASE)

GROUNDING_PROMPT = """\
Тебе даны текст статьи Трудового кодекса РФ, вопрос и ответ.
Проверь: следует ли ответ ПОЛНОСТЬЮ из текста этой статьи (без привлечения внешних знаний)?

Текст статьи:
---
{article}
---
Вопрос: {question}
Ответ: {answer}

Ответь строго в формате:
Итог: да|нет
Причина: <одно предложение>"""

CONSISTENCY_PROMPT = """\
Прочитай статью Трудового кодекса РФ и ответь на вопрос СТРОГО по её тексту.

Текст статьи:
---
{article}
---
Вопрос: {question}

Формат ответа — две строки:
<краткий ответ одним предложением>
<обоснование со ссылкой «ст. N ТК РФ»>"""


def uid_of(pair: dict) -> str:
    raw = f"{pair['article']}|{pair['question']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class Cache:
    """Докачка вердиктов API-проверок: uid -> {stage: verdict}."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, dict] = {}
        try:
            for row in load_jsonl(path):
                self.data[row["uid"]] = row
        except FileNotFoundError:
            pass
        self.new_rows: list[dict] = []

    def get(self, uid: str, stage: str):
        return self.data.get(uid, {}).get(stage)

    def put(self, uid: str, stage: str, verdict) -> None:
        self.data.setdefault(uid, {})[stage] = verdict
        row = self.data[uid]
        row["uid"] = uid
        self.new_rows.append(dict(row))
        append_jsonl(self.path, row)

    def flush(self) -> None:
        # append_jsonl пишет построчно; flush нужен только для статистики
        self.new_rows.clear()


def check_format(pair: dict, cfg: dict) -> tuple[bool, str]:
    q, a = pair["question"], pair["answer"]
    if not (cfg["question_min_chars"] <= len(q) <= cfg["question_max_chars"]):
        return False, f"format: длина вопроса {len(q)}"
    if not (cfg["answer_min_chars"] <= len(a) <= cfg["answer_max_chars"]):
        return False, f"format: длина ответа {len(a)}"
    if cfg["require_question_mark"] and "?" not in q:
        return False, "format: нет знака вопроса"
    if cfg["require_article_reference"]:
        refs = REF_RE.findall(a)
        if not refs:
            return False, "format: нет ссылки «ст. N ТК РФ»"
        if pair["article"] not in refs:
            return False, f"format: ссылка на чужую статью {refs}, ожидалась {pair['article']}"
    return True, "ok"


class ApiChecker:
    def __init__(self, cfg: dict):
        self.provider = make_provider(cfg["checker"]["provider"], cfg["checker"]["model"])
        self.cfg = cfg["checker"]
        self.calls = 0

    def ask(self, prompt: str) -> str:
        self.calls += 1
        result = self.provider.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"])
        time.sleep(self.cfg["request_delay_sec"])
        return result.strip()

    def grounding(self, pair: dict, article_text: str, pass_value: str) -> tuple[bool, str]:
        prompt = GROUNDING_PROMPT.format(article=article_text, question=pair["question"],
                                         answer=pair["answer"])
        answer = self.ask(prompt).lower()
        ok = f"итог: {pass_value}" in answer
        return ok, f"grounding: {answer.splitlines()[0][:80] if answer else 'пусто'}"

    def consistency(self, pair: dict, article_text: str, n_samples: int,
                    min_similarity: int) -> tuple[bool, str]:
        from rapidfuzz import fuzz
        prompt = CONSISTENCY_PROMPT.format(article=article_text, question=pair["question"])
        for _ in range(n_samples):
            answer = self.ask(prompt)
            similarity = fuzz.partial_ratio(pair["answer"].lower(), answer.lower())
            if similarity < min_similarity:
                return False, f"consistency: сходство {similarity:.0f} < {min_similarity}"
        return True, "ok"


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/filters.yaml")
    parser.add_argument("--stages", default=None,
                        help="через запятую, напр. format,grounding (по умолчанию из конфига)")
    parser.add_argument("--limit", type=int, default=None, help="обработать N пар (тест)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    stages = (args.stages.split(",") if args.stages else config["stages"])
    articles = {a["number"]: a for a in load_json(
        config.get("articles", "data/processed/tk_articles.json"))["articles"]}

    pairs = load_jsonl(config["input_raw"])
    if args.limit:
        pairs = pairs[:args.limit]
    for pair in pairs:
        pair.setdefault("uid", uid_of(pair))

    cache = Cache(config["output_cache"])
    api = None
    if any(s in stages for s in ("grounding", "consistency")):
        api = ApiChecker(config)

    kept, rejected = [], []
    stats = {s: 0 for s in stages}
    stats["passed_all"] = 0

    for i, pair in enumerate(pairs):
        article = articles.get(pair["article"])
        article_text = article["text"] if article else ""
        reasons = []

        for stage in stages:
            if stage == "format":
                ok, reason = check_format(pair, config["format"])
            else:
                verdict = cache.get(pair["uid"], stage)
                if verdict is None:
                    if stage == "grounding":
                        ok, reason = api.grounding(pair, article_text,
                                                   config["grounding"]["pass_value"])
                    elif stage == "consistency":
                        ok, reason = api.consistency(
                            pair, article_text, config["consistency"]["n_samples"],
                            config["consistency"]["min_similarity"])
                    else:
                        raise ValueError(f"неизвестная стадия: {stage}")
                    cache.put(pair["uid"], stage, {"ok": ok, "reason": reason})
                else:
                    ok, reason = verdict["ok"], verdict["reason"]
            if not ok:
                reasons.append(reason)
                break
        if reasons:
            stats[stage] += 1
            rejected.append({**pair, "reject_reason": reasons[0]})
        else:
            pair["passes_filters"] = True
            pair["meta"]["filter_stages"] = stages
            kept.append(pair)
            stats["passed_all"] += 1
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(pairs)}] прошло {stats['passed_all']}, "
                  f"отсеяно {len(rejected)}")

    stamp = dt.datetime.now().isoformat(timespec="seconds")
    save_jsonl(config["output_filtered"], kept)
    save_jsonl(config["output_rejected"], rejected)
    save_json("data/processed/filter_stats.json", {
        "at": stamp, "stages": stages, "input": len(pairs),
        "kept": len(kept), "rejected": len(rejected),
        "rejected_by_stage": {s: stats[s] for s in stages},
        "api_calls": api.calls if api else 0,
    })
    print(f"\nИтог: прошло {len(kept)} из {len(pairs)} "
          f"({len(kept) / max(len(pairs), 1):.0%}), API-вызовов: {api.calls if api else 0}")
    for stage in stages:
        print(f"  отсеяно на {stage}: {stats[stage]}")
    print(f"Файлы: {config['output_filtered']} / {config['output_rejected']}")
    if not kept:
        sys.exit("Ни одна пара не прошла — проверь конфиг/промпты.")
    print("Дальше: python data/scripts/dedup.py")


if __name__ == "__main__":
    main()
