# -*- coding: utf-8 -*-
"""Калибровка гейта: best_cos на офтопике vs обычных вопросах ТК."""
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://192.168.1.170:8001"

OFFTOPIC = [
    "Сколько стоит автомобиль в Китае",
    "Как приготовить борщ",
    "Кто выиграл последний чемпионат мира по футболу",
    "Лучшие рецептыMarkdown выпечки с яблоком",
]
ONTOPIC = [
    "Сколько дней отпуска положено работнику в возрасте до 18 лет?",
    "За сколько дней предупреждают об увольнении по собственному желанию?",
    "Может ли работодатель взыскать ущерб больше среднего месячного заработка?",
    "Какова максимальная продолжительность рабочего времени в неделю?",
]

print("--- OFFTOPIC ---")
for q in OFFTOPIC:
    r = httpx.post(f"{BASE}/retrieve", json={"query": q}, timeout=60)
    d = r.json()
    print(f"cos={d.get('best_cos')} route={d.get('route')} | {q[:50]}")

print("--- ONTOPIC ---")
for q in ONTOPIC:
    r = httpx.post(f"{BASE}/retrieve", json={"query": q}, timeout=60)
    d = r.json()
    top = [a["number"] for a in d.get("articles", [])][:3]
    print(f"cos={d.get('best_cos')} route={d.get('route')} top={top} | {q[:50]}")
