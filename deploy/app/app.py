# -*- coding: utf-8 -*-
"""Gradio-демо tk-assistant: чат с tuned-моделью через llama-server (OpenAI API)."""
import inspect
import os
import re
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
    "**Ассистент по трудовому праву РФ** · QLoRA fine-tune Qwen2.5-7B · "
    "[код проекта](https://github.com/markkentos/tk-assistant)\n\n"
    "⚠️ Демонстрация, не юридическая консультация"
)

TITLE = "tk-assistant — трудовое право РФ"

# Ссылки на статьи «ст. N ТК РФ» → кликабельные (zakonrf.info)
ARTICLE_LINK_RE = re.compile(
    r"ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ", re.IGNORECASE)


def linkify(text: str) -> str:
    def _link(m: re.Match) -> str:
        return f"[{m.group(0)}](https://www.zakonrf.info/tk/{m.group(1)}/)"

    return ARTICLE_LINK_RE.sub(_link, text)


# Примитивный rate limiting: публичное демо на домашнем железе
def _client_ip(request) -> str:
    if request is None:
        return ""
    xff = request.headers.get("x-forwarded-for") or ""
    return xff.split(",")[0].strip() or (request.client.host if request.client else "")


def _rate_ok(ip: str) -> bool:
    now = time.time()
    bucket = _clients[ip]
    while bucket and now - bucket[0] > RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        return False
    bucket.append(now)
    return True


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
        if delta:
            partial += delta
            yield linkify(partial)


# ── Визуал: тёплая бумага + янтарный акцент, узкая колонка ──────────────────
custom_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#faf9f7",
    body_background_fill_dark="#1c1917",
    block_background_fill="#ffffff",
    block_background_fill_dark="#292524",
    block_border_width="0px",
    block_title_text_color="#78716c",
    button_primary_background_fill="#d97706",
    button_primary_background_fill_hover="#b45309",
    button_primary_text_color="#ffffff",
    slider_color="#d97706",
)

custom_css = """
.gradio-container {max-width: 840px !important; margin: 0 auto !important;}
footer {visibility: hidden !important; display: none !important;}
#chatbot {border: none !important; box-shadow: none !important; background: transparent !important;}
h1 {font-weight: 700 !important; letter-spacing: -0.02em !important;}
.prose p {margin-bottom: 0.4rem !important;}
a {color: #b45309 !important; text-decoration: none !important;}
a:hover {text-decoration: underline !important;}
"""


def build_demo():
    """ChatInterface с адаптацией к версии gradio (сигнатура меняется между мажорами)."""
    print("gradio version:", gr.__version__)
    kwargs = dict(
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
        cache_examples=False,
        theme=custom_theme,
        css=custom_css,
    )
    params = inspect.signature(gr.ChatInterface.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    all_keys = ("title", "description", "examples", "cache_examples",
                "theme", "css")
    dropped = [k for k in all_keys if k not in params]
    if dropped:
        print("Gradio: параметры не поддерживаются этой версией, пропущены:", dropped)
    if "type" in params:
        return gr.ChatInterface(respond, type="messages", **kwargs)
    return gr.ChatInterface(respond, **kwargs)


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
