# -*- coding: utf-8 -*-
"""Gradio-демо tk-assistant: чат с tuned-моделью через llama-server (OpenAI API)."""
import inspect
import os
import time
from collections import defaultdict, deque

import gradio as gr
from openai import OpenAI

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="none")

# Примитивный rate limiting: публичное демо на домашнем железе
RATE_LIMIT = 5          # максимум запросов
RATE_WINDOW = 60        # за скользящее окно, сек
_clients: dict[str, deque] = defaultdict(lambda: deque(maxlen=RATE_LIMIT))

SYSTEM_PROMPT = (
    "Ты — ассистент по трудовому праву РФ. Отвечай на вопросы, опираясь на "
    "Трудовой кодекс РФ. Формат ответа — строго две части:\n"
    "1) краткий ответ одним предложением;\n"
    "2) обоснование из 1–3 предложений со ссылкой на статью в формате «ст. N ТК РФ».\n"
    "Не выдумывай номера статей. Если не уверен — так и скажи."
)

EXAMPLES = [
    "За сколько дней работник должен предупредить работодателя об увольнении по собственному желанию?",
    "Может ли работодатель уволить беременную сотрудницу?",
    "Какова нормальная продолжительность рабочего времени в неделю?",
    "Сколько дней отпуска положено работнику в возрасте до 18 лет?",
]

DESCRIPTION = (
    "Демо модели Qwen2.5-7B-Instruct, дообученной методом QLoRA на 1615 парах "
    "вопрос–ответ по Трудовому кодексу РФ. "
    "[Код проекта](https://github.com/markkentos/tk-assistant).\n\n"
    "⚠️ Демонстрация, не юридическая консультация."
)


def _client_ip(request) -> str:
    """IP посетителя: Caddy пробрасывает X-Forwarded-For."""
    if request is None:
        return ""
    xff = request.headers.get("x-forwarded-for") or ""
    return xff.split(",")[0].strip() or (request.client.host if request.client else "")


def respond(message: str, history: list, request: gr.Request = None):
    ip = _client_ip(request)
    if ip:
        now = time.time()
        bucket = _clients[ip]
        while bucket and now - bucket[0] > RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            yield "⏳ Слишком много запросов с этого адреса — попробуй через минуту."
            return
        bucket.append(now)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model="local", messages=messages, stream=True,
        temperature=0.2, max_tokens=400,
    )
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


def build_demo():
    """ChatInterface с адаптацией к версии gradio (сигнатура меняется между мажорами)."""
    print("gradio version:", gr.__version__)
    kwargs = dict(
        title="tk-assistant · трудовое право РФ",
        description=DESCRIPTION,
        examples=EXAMPLES,
        cache_examples=False,
    )
    params = inspect.signature(gr.ChatInterface.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    all_keys = ("title", "description", "examples", "cache_examples")
    dropped = [k for k in all_keys if k not in params]
    if dropped:
        print("Gradio: параметры не поддерживаются этой версией, пропущены:", dropped)
    if "type" in params:
        return gr.ChatInterface(respond, type="messages", **kwargs)
    return gr.ChatInterface(respond, **kwargs)


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
