# -*- coding: utf-8 -*-
"""Этап 1, шаг 1: скачать оглавление и все статьи ТК РФ (zakonrf.info) в data/raw/tk/.

Качаем вежливо: пауза между запросами, ретраи, докачка (существующие файлы не трогаем).
Результат — HTML-кэш; структурный разбор делает parse_articles.py.

Запуск:
    python data/scripts/download_tk.py [--limit 5]   # --limit для смоук-теста
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
import time

from common import PROCESSED_DIR, RAW_DIR, fetch, make_session, save_json, utf8_console

TOC_URL = "https://www.zakonrf.info/tk/"
ARTICLE_URL = "https://www.zakonrf.info/tk/{num}/"
ARTICLE_HREF_RE = re.compile(r'href="/tk/(\d+(?:\.\d+)?)/"')


def download_tk(out_dir: pathlib.Path, delay: float, limit: int | None,
                redownload: bool) -> None:
    session = make_session()
    articles_dir = out_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    print(f"Скачиваю оглавление: {TOC_URL}")
    toc_html = fetch(session, TOC_URL)
    (out_dir / "toc.html").write_text(toc_html, encoding="utf-8")

    numbers = list(dict.fromkeys(ARTICLE_HREF_RE.findall(toc_html)))
    # Номер статьи 351.1 и т.п. содержат точку; сортируем как (int, int)
    numbers.sort(key=lambda n: tuple(int(p) for p in n.split(".")))
    print(f"Найдено статей в оглавлении: {len(numbers)}")

    if limit:
        numbers = numbers[:limit]
        print(f"--limit {limit}: качаю только первые {len(numbers)} статей (смоук-тест)")

    fetched, skipped, failed = 0, 0, []
    started = time.time()
    try:
        from tqdm import tqdm
        iterator = tqdm(numbers, unit="ст")
    except ImportError:
        iterator = numbers

    for num in iterator:
        target = articles_dir / f"{num}.html"
        if target.exists() and target.stat().st_size > 1000 and not redownload:
            skipped += 1
            continue
        try:
            html = fetch(session, ARTICLE_URL.format(num=num))
            target.write_text(html, encoding="utf-8")
            fetched += 1
        except RuntimeError as exc:
            failed.append({"number": num, "error": str(exc)})
            print(f"\n[ОШИБКА] ст. {num}: {exc}")
        time.sleep(delay)

    manifest = {
        "source": "zakonrf.info",
        "law": "Трудовой кодекс РФ",
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "toc_url": TOC_URL,
        "n_articles_in_toc": len(numbers),
        "fetched_this_run": fetched,
        "skipped_existing": skipped,
        "failed": failed,
    }
    save_json(out_dir / "manifest.json", manifest)

    minutes = (time.time() - started) / 60
    print(f"\nГотово за {minutes:.1f} мин: скачано {fetched}, уже было {skipped}, "
          f"ошибок {len(failed)}")
    if failed:
        print("Перезапусти скрипт — он докачает упавшие статьи.")
        sys.exit(1)

    # Файл-маркер завершённости для parse_articles.py
    (out_dir / "_download_complete").write_text(
        dt.datetime.now().isoformat(), encoding="utf-8")
    print(f"HTML-кэш: {articles_dir}")


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(RAW_DIR / "tk"), help="каталог для HTML-кэша")
    parser.add_argument("--delay", type=float, default=0.6, help="пауза между запросами, сек")
    parser.add_argument("--limit", type=int, default=None,
                        help="скачать только N первых статей (смоук-тест)")
    parser.add_argument("--redownload", action="store_true", help="перекачать всё")
    args = parser.parse_args()

    download_tk(pathlib.Path(args.out), args.delay, args.limit, args.redownload)
    (PROCESSED_DIR).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
