# -*- coding: utf-8 -*-
"""Общие утилиты пайплайна данных: HTTP с ретраями, JSON/JSONL I/O, пути."""
from __future__ import annotations

import json
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

USER_AGENT = (
    "tk-assistant-dataset-bot/0.1 (educational pet project; "
    "source: https://github.com/markkentos/tk-assistant)"
)


def utf8_console() -> None:
    """UTF-8 в консоли Windows, чтобы кириллица не превращалась в кракозябры."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch(session: requests.Session, url: str, retries: int = 3, backoff: float = 2.0,
          timeout: int = 25) -> str:
    """GET с ретраями и экспоненциальной паузой. Возвращает текст (utf-8)."""
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.content.decode("utf-8", errors="replace")
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:  # сеть, таймауты
            last_error = repr(exc)
        time.sleep(backoff * attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {url} ({last_error})")


def load_json(path: str | pathlib.Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str | pathlib.Path, obj) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str | pathlib.Path, records: list[dict]) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_text(path: str | pathlib.Path, text: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(text, encoding="utf-8")


def append_jsonl(path: str | pathlib.Path, record: dict) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
