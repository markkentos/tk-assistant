# -*- coding: utf-8 -*-
"""RAG-сервис tk-assistant: multi-query retrieval + гибрид + переранкинг + прокси.

Пайплайн на вопрос пользователя:
  1. LLM генерирует 3 переформулировки вопроса (расширение запроса)
  2. Для каждого из 4 запросов (оригинал + 3): BM25 (лексика) и bi-encoder
     (семантика) ищут топ-15 статей
  3. Голосование: статья набирает голос за каждый список, где она в топ-15
     → топ-5 кандидатов
  4. Cross-encoder переранкинг топ-5 → топ-3 статьи
  5. Топ-3 инъектируется в system как справочный материал

POST /v1/chat/completions — OpenAI-совместимо (стриминг и не-стриминг).
Остальные /v1/* — прозрачный прокси в llama-server.
"""
import json
import os
import re
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.background import BackgroundTask

ARTICLES_PATH = os.environ.get("TK_ARTICLES_PATH", "/app/data/tk_articles.json")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL",
                                "paraphrase-multilingual-MiniLM-L12-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATES_PER_SOURCE = 15   # кандидатов от каждого поисковика на запрос-вариант
VOTE_TOP_K = 5               # сколько статей после голосования
FINAL_TOP_K = 3              # сколько статей уходит модели
MIN_ARTICLE_CHARS = 100
MIN_QUERY_CHARS = 12

articles: list[dict] = []
bm25 = None
embedder = None
reranker = None
article_vecs = None
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))

stemmer = SnowballStemmer("russian")
_token_re = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

_RU_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "the", "a", "an", "of", "to", "is", "it", "in", "on", "for",
}


def tokens(text: str) -> list[str]:
    stems = [stemmer.stem(t) for t in _token_re.findall(text.lower())]
    return [s for s in stems if s not in _RU_STOP and len(s) > 1]


def load_articles(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return [a for a in data.get("articles", [])
            if a.get("status") == "active"
            and len(a.get("text", "")) >= MIN_ARTICLE_CHARS]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion списков индексов статей."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda t: -t[1])]


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


ARTICLE_QUERY_RE = re.compile(
    r"ст(?:ать(?:и|я|ю|е|ей)?)?\.?\s*(?:№\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)


def detect_article_reference(query: str) -> str | None:
    """Явная ссылка на статью в запросе («ст. 134», «статье 81.1») —
    прямой путь: достаём статью по метаданным, минуя векторный поиск."""
    m = ARTICLE_QUERY_RE.search(query)
    return m.group(1) if m else None


REPHRASE_PROMPT = """Переформулируй вопрос пользователя тремя разными способами
для поисковой системы по Трудовому кодексу РФ:
- используй синонимы и юридическую терминологию;
- в одном варианте раскрой сокращения цифрами или словами;
- каждый вариант должен искать статью через СВОИ ключевые слова;
- вывод — ровно 3 строки, без нумерации и пояснений.

Вопрос: {question}"""


async def expand_query(query: str) -> list[str]:
    """LLM порождает 3 переформулировки вопроса (запрос-адаптивный поиск)."""
    body = {
        "model": "local",
        "messages": [
            {"role": "system",
             "content": "Ты переформулируешь вопросы для поиска по нормативным "
                        "документам. Выводи только сами переформулировки, "
                        "по одной на строку, без нумерации и пояснений."},
            {"role": "user", "content": REPHRASE_PROMPT.format(question=query)},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "stream": False,
    }
    response = await client.post(f"{LLAMA_SERVER_URL}/v1/chat/completions",
                                 json=body)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    variants = [line.strip() for line in text.splitlines() if line.strip()]
    return variants[:3]


def vote_aggregate(variants: list[str], top_k: int) -> list[dict]:
    """Голосование: для каждого запроса BM25 + эмбеддер ищут топ-15;
    статья набирает голос за каждый список, где она в топе."""
    votes: dict[int, int] = {}
    for variant in variants:
        q_tokens = tokens(variant)
        bm_scores = bm25.get_scores(q_tokens)
        for idx in np.argsort(bm_scores)[::-1][:CANDIDATES_PER_SOURCE]:
            votes[idx] = votes.get(idx, 0) + 1
        qv = embedder.encode([f"query: {variant}"], normalize_embeddings=True)
        sims = (article_vecs @ qv.T).ravel()
        for idx in np.argsort(sims)[::-1][:CANDIDATES_PER_SOURCE]:
            votes[idx] = votes.get(idx, 0) + 1

    top = sorted(votes.items(), key=lambda kv: -kv[1])[:top_k]
    return [articles[idx] for idx, _ in top]


async def retrieve_candidates(query: str) -> list[dict]:
    """Multi-query: оригинал + 3 LLM-переформулировки → голосование."""
    variants = [query]
    try:
        variants += await expand_query(query)
    except Exception as exc:
        print(f"RAG: расширение запроса не удалось ({exc!r}) — "
              f"поиск по оригиналу", flush=True)
    return vote_aggregate(variants, VOTE_TOP_K)


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:FINAL_TOP_K]


def inject_candidates(messages: list[dict], candidates: list[dict]) -> list[dict]:
    parts = []
    for i, article in enumerate(candidates, 1):
        parts.append(
            f"Кандидат {i} — ст. {article['number']} ТК РФ: «{article['title']}».\n"
            f"Текст:\n{article['text']}"
        )
    block = (
        "\n\nНиже — статьи Трудового кодекса РФ, найденные по вопросу "
        "(в порядке убывания релевантности):\n\n"
        + "\n\n".join(parts)
        + "\n\nОпредели, какая из них РЕАЛЬНО отвечает на вопрос пользователя "
          "(если ни одна — так и скажи, не выдумывай номера статей). "
          "Отвечай СТРОГО по тексту выбранной статьи и обязательно ссылайся "
          f"на неё в формате «ст. N ТК РФ»."
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
    global articles, bm25, embedder, reranker, article_vecs
    articles = load_articles(ARTICLES_PATH)
    print(f"RAG: {len(articles)} активных статей", flush=True)

    print("RAG: BM25-индекс...", flush=True)
    corpus = [tokens(f"{a['title']}. {a['title']}. {a['text']}") for a in articles]
    bm25 = BM25Okapi(corpus)

    print("RAG: загружаю bi-encoder...", flush=True)
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    article_vecs = embedder.encode(
        [f"passage: {a['title']}. {a['text']}" for a in articles],
        normalize_embeddings=True, show_progress_bar=False)

    print(f"RAG: загружаю reranker ({RERANKER_MODEL})...", flush=True)
    reranker = CrossEncoder(RERANKER_MODEL)
    print("RAG: индексация завершена", flush=True)
    yield


app = FastAPI(title="tk-assistant RAG", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    messages = body.get("messages") or []

    query = retrieve_query(messages)
    candidates: list[dict] = []
    route = "none"

    # Прямой путь: в запросе явная ссылка на статью («ст. 134») —
    # достаём её по метаданным + соседей по кодексу, минуя поиск
    explicit_ref = detect_article_reference(query) if query else None
    if explicit_ref:
        exact = next((a for a in articles if a["number"] == explicit_ref), None)
        if exact:
            idx = articles.index(exact)
            candidates = [exact] + articles[idx + 1:idx + 2]
            route = f"direct ст. {explicit_ref}"

    if not candidates and query and len(query) >= MIN_QUERY_CHARS:
        # Multi-query: оригинал + 3 переформулировки → голосование → топ-5
        candidates = await retrieve_candidates(query)
        # Переранкинг cross-encoder → топ-3 для модели
        ranked = rerank_candidates(query, candidates)
        candidates = ranked
        route = "hybrid+rerank"

    retrieved_list = [a["number"] for a in candidates]
    body["messages"] = inject_candidates(messages, candidates)

    upstream_request = client.build_request(
        "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions",
        content=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
    )
    upstream_response = await client.send(upstream_request, stream=True)

    extra = {"x-retrieved-articles": ",".join(retrieved_list),
             "x-retrieval-route": route} if retrieved_list else \
            {"x-retrieval-route": route}
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
