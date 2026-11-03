# -*- coding: utf-8 -*-
"""RAG-сервис tk-assistant: семантический поиск статьи + прокси в llama-server.

Retrieval: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) —
векторизация «заголовок×2 + текст» всех активных статей; запрос → косинусное
сходство. Модель эмбеддера предзагружается в образ при сборке (работа офлайн).

POST /v1/chat/completions: инъекция статьи в system + прозрачный прокси
(стриминг и не-стриминг). Остальные /v1/* — прозрачный прокси.
"""
import json
import os
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
from starlette.background import BackgroundTask

ARTICLES_PATH = os.environ.get("TK_ARTICLES_PATH", "/app/data/tk_articles.json")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
MODEL_NAME = os.environ.get("EMBEDDER_MODEL",
                            "paraphrase-multilingual-MiniLM-L12-v2")
MIN_ARTICLE_CHARS = 100

articles: list[dict] = []
article_vecs = None
embedder = None
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))


def load_articles(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return [a for a in data.get("articles", [])
            if a.get("status") == "active"
            and len(a.get("text", "")) >= MIN_ARTICLE_CHARS]


def retrieve(query: str) -> dict:
    qv = embedder.encode([query], normalize_embeddings=True)
    sims = (article_vecs @ qv.T).ravel()
    best = int(np.argmax(sims))
    article = articles[best]
    print(f"RAG DEBUG: retrieve → ст. {article['number']} "
          f"(cos={float(sims[best]):.3f})", flush=True)
    return article


def retrieve_query(messages: list[dict]) -> str | None:
    """Последнее сообщение пользователя; короткое уточнение склеиваем
    с предыдущим сообщением (для поиска)."""
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None
    query = user_msgs[-1]
    if len(query) < 60 and len(user_msgs) >= 2:
        query = user_msgs[-2] + " " + query
    return query


def inject_article(messages: list[dict], article: dict) -> list[dict]:
    block = (
        f"\n\nРелевантная статья Трудового кодекса РФ — ст. {article['number']}: "
        f"«{article['title']}».\nТекст статьи:\n{article['text']}\n"
        f"Отвечай СТРОГО по тексту этой статьи и обязательно ссылайся на неё "
        f"в формате «ст. {article['number']} ТК РФ»."
    )
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = msgs[0].get("content", "") + block
    else:
        msgs.insert(0, {"role": "system",
                        "content": "Ты — ассистент по трудовому праву РФ." + block})
    return msgs


async def relay(upstream_response: httpx.Response,
                extra_headers: dict | None = None) -> StreamingResponse:
    """Прозрачная ретрансляция ответа llama-server + доп. заголовки RAG."""
    headers = dict(upstream_response.headers)
    headers.pop("content-length", None)
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=headers,
        media_type=upstream_response.headers.get("content-type"),
        background=BackgroundTask(upstream_response.aclose),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedder, article_vecs, articles
    articles = load_articles(ARTICLES_PATH)
    print(f"RAG: загружаю эмбеддер... ({len(articles)} статей)", flush=True)
    embedder = SentenceTransformer(MODEL_NAME)
    corpus = [f"{a['title']}. {a['title']}. {a['text']}" for a in articles]
    article_vecs = embedder.encode(corpus, normalize_embeddings=True,
                                   show_progress_bar=False)
    print(f"RAG: {len(articles)} статей проиндексировано", flush=True)
    yield


app = FastAPI(title="tk-assistant RAG", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    messages = body.get("messages") or []

    query = retrieve_query(messages)
    article_num = None
    if query and len(query) >= 12:
        article = retrieve(query)
        article_num = article["number"]
        body["messages"] = inject_article(messages, article)

    upstream_request = client.build_request(
        "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions",
        content=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
    )
    upstream_response = await client.send(upstream_request, stream=True)

    extra = {"x-retrieved-article": article_num} if article_num else None
    return await relay(upstream_response, extra)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_other(request: Request, path: str):
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    upstream_request = client.build_request(
        request.method, f"{LLAMA_SERVER_URL}/v1/{path}",
        content=await request.body(), headers=headers,
    )
    upstream_response = await client.send(upstream_request, stream=True)
    return await relay(upstream_response)


@app.get("/health")
async def health():
    return {"status": "ok", "articles": len(articles),
            "index_ready": article_vecs is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
