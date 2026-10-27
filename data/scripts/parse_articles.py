# -*- coding: utf-8 -*-
"""Этап 1, шаг 2: разобрать HTML-кэш ТК РФ в структурированные статьи.

Вход:  data/raw/tk/toc.html, data/raw/tk/articles/*.html   (download_tk.py)
Выход: data/processed/tk_articles.json                     (статьи + иерархия)
       data/processed/tk_parse_report.json                 (статистика парсинга)

Каждая статья: {number, title, text, chapter, chapter_title, section, part,
status: active|repealed, url}. Комментарии сайта (lawcomment-box) НЕ включаются —
в датасет идёт только официальный текст кодекса.

Запуск:
    python data/scripts/parse_articles.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import statistics
import sys

from bs4 import BeautifulSoup

from common import PROCESSED_DIR, RAW_DIR, load_json, save_json, utf8_console

TK_URL = "https://www.zakonrf.info/tk/{num}/"

# Токены оглавления в порядке следования в HTML: часть → раздел → глава → статья.
TOC_TOKEN_RE = re.compile(
    r'<a[^>]+href="/tk/gl(?P<chapter_num>\d+)/"[^>]*>(?P<chapter_text>.*?)</a>'
    r'|<a[^>]+href="/tk/(?P<article_num>\d+(?:\.\d+)?)/"[^>]*>(?P<article_text>.*?)</a>'
    r'|<a[^>]+href="/tk/(?P<section_slug>[a-z\-]+)/"[^>]*>(?P<section_text>.*?)</a>'
    r'|>\s*(?P<part_text>ЧАСТЬ\s+[А-ЯЁ]+(?:\s+[А-ЯЁ]+)*)\s*<'
    r'|>\s*(?P<razdel_text>(?:РАЗДЕЛ|Раздел)\s+[IVXL]+[\s.][^<]{0,300}?)\s*<',
    re.S,
)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean(fragment: str) -> str:
    return WS_RE.sub(" ", TAG_RE.sub("", fragment)).strip()


def parse_toc(toc_html: str) -> tuple[dict[str, dict], list[str]]:
    """Идём по оглавлению в порядке документа, накапливаем иерархию.

    Возвращает (статьи: number -> метаданные, ненайденные номера).
    """
    soup = BeautifulSoup(toc_html, "lxml")
    main = soup.find("main") or soup
    html = str(main)

    chapters: dict[int, dict] = {}
    articles: dict[str, dict] = {}
    unmatched_articles: list[str] = []
    current_part, current_section = "", ""

    for match in TOC_TOKEN_RE.finditer(html):
        if match.group("chapter_num"):
            num = int(match.group("chapter_num"))
            text = clean(match.group("chapter_text"))
            chapters[num] = {"title": text, "part": "", "section": ""}
        elif match.group("article_num"):
            num = match.group("article_num")
            title = clean(match.group("article_text"))
            title = re.sub(r"^Статья\s+[\d.]+\.\s*", "", title)
            chapter = max(chapters) if chapters else None
            if num in articles:
                continue
            articles[num] = {
                "number": num,
                "title": title,
                "chapter": chapter,
                "chapter_title": chapters.get(chapter, {}).get("title", "") if chapter else "",
                "section": current_section,
                "part": current_part,
                "status": "active",
                "url": TK_URL.format(num=num),
            }
        elif match.group("section_slug"):
            current_section = clean(match.group("section_text"))
            for meta in chapters.values():  # раздел действует до следующего
                if not meta["section"]:
                    meta["section"] = current_section
        elif match.group("part_text"):
            current_part = clean(match.group("part_text"))
            for meta in chapters.values():
                if not meta["part"]:
                    meta["part"] = current_part
        elif match.group("razdel_text"):
            current_section = clean(match.group("razdel_text"))
            for meta in chapters.values():
                if not meta["section"]:
                    meta["section"] = current_section

    return articles, unmatched_articles


def parse_article_page(html: str, num: str) -> dict:
    """Достаём заголовок и текст статьи из страницы статьи."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_=re.compile(r"law-element"))
    if article is None:
        raise ValueError("не найден <article class='law-element'>")

    title_raw = article.find("h1").get_text(" ", strip=True)
    title = re.sub(r"^Статья\s+[\d.]+\s*ТК\s*РФ\.\s*", "", title_raw)
    title = re.sub(r"\s*\(действующая редакция\)\s*$", "", title).strip()

    body = article.find("div", class_=re.compile(r"law-element__body"))
    if body is None:
        raise ValueError("не найден div.law-element__body")
    paragraphs = [WS_RE.sub(" ", p.get_text(" ", strip=True)) for p in body.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)

    status = "active"
    if re.search(r"[Уу]тратила силу", title) or re.search(r"[Уу]тратила силу", text[:300]):
        status = "repealed"

    chapter = None
    for a in soup.select(".breadcrumbs a[href^='/tk/gl']"):
        m = re.search(r"/tk/gl(\d+)/", a["href"])
        if m:
            chapter = int(m.group(1))

    return {"title": title, "text": text, "status": status, "chapter": chapter}


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(RAW_DIR / "tk"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "tk_articles.json"))
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw)
    if not (raw_dir / "toc.html").exists():
        sys.exit("Нет data/raw/tk/toc.html — сначала запусти download_tk.py")

    toc_meta, unmatched = parse_toc((raw_dir / "toc.html").read_text(encoding="utf-8"))
    print(f"Оглавление: распознано статей {len(toc_meta)}")

    articles_dir = raw_dir / "articles"
    cached = {p.stem for p in articles_dir.glob("*.html")}
    missing = sorted(set(toc_meta) - cached, key=lambda n: tuple(int(x) for x in n.split(".")))
    if missing:
        sys.exit(f"Нет HTML для {len(missing)} статей: {missing[:10]}... "
                 "Докачай через download_tk.py.")

    records, errors = [], []
    for num, meta in toc_meta.items():
        html = (articles_dir / f"{num}.html").read_text(encoding="utf-8")
        try:
            page = parse_article_page(html, num)
        except ValueError as exc:
            errors.append({"number": num, "error": str(exc)})
            continue
        record = {**meta, "title": page["title"], "text": page["text"],
                  "status": page["status"]}
        # глава из оглавления надёжнее, но сверяемся с хлебными крошками страницы
        if record["chapter"] is None:
            record["chapter"] = page["chapter"]
        records.append(record)

    records.sort(key=lambda r: tuple(int(x) for x in r["number"].split(".")))
    for i, record in enumerate(records, 1):
        record["order"] = i

    active = [r for r in records if r["status"] == "active"]
    lengths = sorted(len(r["text"]) for r in records)
    report = {
        "parsed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_in_toc": len(toc_meta),
        "n_parsed": len(records),
        "n_active": len(active),
        "n_repealed": len(records) - len(active),
        "n_errors": len(errors),
        "errors": errors,
        "text_chars": {"min": lengths[0], "median": int(statistics.median(lengths)),
                       "mean": int(statistics.mean(lengths)), "max": lengths[-1]},
        "n_text_below_50_chars": sum(1 for x in lengths if x < 50),
        "n_no_chapter": sum(1 for r in records if r["chapter"] is None),
        "total_chars": sum(lengths),
        "chapters": sorted({r["chapter"] for r in records if r["chapter"] is not None}),
    }
    save_json(args.out, {
        "source": "zakonrf.info",
        "law": "Трудовой кодекс РФ",
        "parsed_at": report["parsed_at"],
        "n_articles": len(records),
        "articles": records,
    })
    save_json(PROCESSED_DIR / "tk_parse_report.json", report)

    print(f"Распарсено: {len(records)} (активных {len(active)}, "
          f"утративших силу {report['n_repealed']}, ошибок {len(errors)})")
    print(f"Символов текста всего: {report['total_chars']:,} "
          f"(медиана на статью {report['text_chars']['median']})")
    if errors:
        print("Ошибки разбора:", errors)
    print(f"Файлы: {args.out}")
    if unmatched:
        print("Ссылки вне паттернов:", unmatched[:10])


if __name__ == "__main__":
    main()
