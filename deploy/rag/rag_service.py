# -*- coding: utf-8 -*-
"""RAG-сервис tk-assistant: гибридный поиск + переранкинг + прокси.

Пайплайн на вопрос пользователя:
  1. BM25 (лексика) и bi-encoder (семантика) ищут топ-15 статей
  2. Голосование: статья набирает голос за каждый список, где она в топ-15
     → топ-5 кандидатов
  3. Cross-encoder переранкинг топ-5 → топ-3 статьи (TK_RERANK=1 — вкл;
     на CPU стоит 4-7 с, для демо по умолчанию выкл)
  4. Топ-3 инъектируется в system как справочный материал
  [опция] TK_MULTI_QUERY=1: перед поиском LLM порождает 3 переформулировки
  запроса (+2-6 с) — точнее на неявных вопросах, медленнее.

POST /v1/chat/completions — OpenAI-совместимо (стриминг и не-стриминг).
Остальные /v1/* — прозрачный прокси в llama-server.
"""
import json
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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

# Системный промпт — зеркальная копия SYSTEM_PROMPT из src/prompts.py.
# Тренировка (в т.ч. RAG-native) идёт с этим текстом; прод обязан совпадать.
SYSTEM_PROMPT = (
    "Ты — ассистент по трудовому праву РФ. Отвечай на вопросы, опираясь на "
    "Трудовой кодекс РФ. Формат ответа — строго две части:\n"
    "1) краткий ответ одним предложением;\n"
    "2) обоснование из 1–3 предложений со ссылкой на статью в формате «ст. N ТК РФ».\n"
    "Не выдумывай номера статей. Если не уверен — так и скажи."
)
# Тяжёлые этапы вынесены в env (замеры на ПК, Ryzen + 1070 Ti-хост, CPU-инференс):
# - cross-encoder rerank: 3.8-6.9 с на запрос (5 пар по ~700 символов) — слишком
#   дорого для демо, по умолчанию ВЫКЛ; TK_RERANK=1 для экспериментов качества
# - multi-query (LLM-переформулировки): +2-6 с на запрос — TK_MULTI_QUERY=1
# retrieve (голосование BM25+эмбеддер): 32-48 мс
MULTI_QUERY = os.environ.get("TK_MULTI_QUERY", "0") == "1"
USE_RERANK = os.environ.get("TK_RERANK", "0") == "1"
RERANK_CHARS = int(os.environ.get("TK_RERANK_CHARS", "700"))
NEIGHBORS = os.environ.get("TK_NEIGHBORS", "1") == "1"
# Гейт релевантности: если лучший cosine (эмбеддер) ниже порога — вопрос
# считается вне темы ТК, контекст не инъекцируется (маршрут offtopic).
# Калибровка (e5): офтопик 0.72–0.76, тематические 0.84–0.89 → порог 0.80
MIN_QUERY_COS = float(os.environ.get("TK_MIN_COS", "0.80"))

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


def vote_aggregate(variants: list[str], top_k: int) -> tuple[list[dict], float]:
    """Голосование: для каждого запроса BM25 + эмбеддер ищут топ-15;
    статья набирает голос за каждый список, где она в топе.
    Возвращает (кандидаты, лучший cosine по всем вариантам) — cosine
    используется гейтом релевантности (TK_MIN_COS)."""
    votes: dict[int, int] = {}
    best_cos = -1.0
    guaranteed: list[int] = []
    seen_g: set[int] = set()
    for variant in variants:
        q_tokens = tokens(variant)
        bm_scores = bm25.get_scores(q_tokens)
        for idx in np.argsort(bm_scores)[::-1][:CANDIDATES_PER_SOURCE]:
            votes[idx] = votes.get(idx, 0) + 1
        qv = embedder.encode([f"query: {variant}"], normalize_embeddings=True)
        sims = (article_vecs @ qv.T).ravel()
        best_cos = max(best_cos, float(sims.max()))
        for idx in np.argsort(sims)[::-1][:CANDIDATES_PER_SOURCE]:
            votes[idx] = votes.get(idx, 0) + 1
        # Гарантированные слоты: топ-2 по cosine каждого варианта — темы
        # мульти-вопроса не вытесняют друг друга из контекста
        for idx in np.argsort(sims)[::-1][:2]:
            if int(idx) not in seen_g:
                seen_g.add(int(idx))
                guaranteed.append(int(idx))
                if len(guaranteed) >= top_k:
                    break
        if len(guaranteed) >= top_k:
            break

    top = [idx for idx, _ in
           sorted(votes.items(), key=lambda kv: -kv[1])[:top_k]]
    merged: list[int] = []
    seen_m: set[int] = set()
    for idx in guaranteed + top:
        if idx not in seen_m:
            merged.append(idx)
            seen_m.add(idx)
        if len(merged) >= top_k:
            break
    return [articles[idx] for idx in merged], best_cos


def query_variants(message: str) -> list[str]:
    """Мульти-вопрос: каждое содержательное предложение — отдельный вариант
    поиска (иначе эмбеддер усредняет темы и в топ доезжает одна)."""
    parts = []
    for raw in re.split(r"[?\n]+", message):
        p = raw.strip()
        if len(p) >= 10:
            parts.append(p)
    if not parts:
        parts = [message]
    return parts[:4]  # не даём голосам размазаться по десяткам кусков


async def retrieve_candidates(query: str) -> tuple[list[dict], float]:
    """Голосование BM25 + эмбеддер. С TK_MULTI_QUERY=1 — ещё и 3 LLM-
    переформулировки запроса (медленнее на 2-6 с, точнее на неявных вопросах).
    Возвращает (кандидаты, лучший cosine)."""
    variants = query_variants(query)
    if MULTI_QUERY:
        try:
            variants += await expand_query(query)
        except Exception as exc:
            print(f"RAG: расширение запроса не удалось ({exc!r}) — "
                  f"поиск по оригиналу", flush=True)
    return vote_aggregate(variants, VOTE_TOP_K)


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    # Обрезаем текст статьи: начало статьи несёт основную релевантность,
    # а полные 512 токенов на пару на CPU — лишние секунды
    pairs = [[query, c["text"][:RERANK_CHARS]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:FINAL_TOP_K]


def neighbor_numbers(number: str) -> list[str]:
    """Номера соседних статей: предыдущая и следующая (±1 целой части)."""
    base = number.split(".")[0]
    try:
        b = int(base)
        return [str(b - 1), str(b + 1)]
    except ValueError:
        return []


def expand_with_neighbors(candidates: list[dict]) -> list[dict]:
    """Расширяем подборку соседями: подпункты той же статьи (x.N, по
    близости к найденному) и статьи ±1. Частый промах ретрива —
    правильная статья стоит соседом найденной (замер: 8 из 32 промахов).
    Лимит 5 итоговых статей: при 8 длинные статьи переполняли даже
    контекст 6144 (3 из 100 бенчмарк-запросов падали с переполнением)."""
    out = list(candidates)
    seen = {c["number"] for c in out}

    # 1) Подпункты той же статьи, ближайшие по номеру первыми
    for c in candidates:
        parts = c["number"].split(".", 1)
        if len(parts) != 2:
            continue
        try:
            sub = float(parts[1])
        except ValueError:
            continue
        sibs = [a for a in articles
                if a["number"] != c["number"]
                and a["number"].split(".", 1)[0] == parts[0]
                and "." in a["number"]]
        sibs.sort(key=lambda a: abs(float(a["number"].split(".", 1)[1]) - sub))
        for a in sibs:
            if a["number"] not in seen:
                out.append(a)
                seen.add(a["number"])
                if len(out) >= 5:
                    return out

    # 2) Соседние статьи ±1
    for c in candidates:
        for num in neighbor_numbers(c["number"]):
            if num in seen:
                continue
            a = next((x for x in articles if x["number"] == num), None)
            if a:
                out.append(a)
                seen.add(num)
                if len(out) >= 5:
                    return out
    return out


OFFTOPIC_NOTE = (
    "\n\nВажно: вопрос пользователя может не относиться к трудовому праву РФ. "
    "Если он вне твоей компетенции, ответь РОВНО в таком стиле: «Этот вопрос "
    "не относится к трудовому праву — я ассистент по Трудовому кодексу РФ и "
    "отвечаю только на вопросы о труде. Могу помочь, если вопрос про работу, "
    "отпуск, увольнение, зарплату и т.п.» Никаких статей и ссылок в этом "
    "случае не указывай. Если же вопрос всё же связан с трудовыми "
    "отношениями — отвечай в обычном формате по своим знаниям кодекса."
)


def inject_offtopic(messages: list[dict]) -> list[dict]:
    """Офториковый маршрут: контекст не инъекцируем, разрешаем честный отказ."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = msgs[0].get("content", "") + OFFTOPIC_NOTE
    else:
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT + OFFTOPIC_NOTE})
    return msgs


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
        + "\n\nОпредели, какие из них РЕАЛЬНО отвечают на вопрос пользователя. "
          "Если в сообщении несколько вопросов — ответь на каждый отдельным "
          "абзацем, разделяя ответы пустой строкой, и приведи ссылку для "
          "каждого. Если релевантная статья есть — отвечай СТРОГО по её "
          "тексту и ссылайся в формате «ст. N ТК РФ». Если же ни одна не "
          "относится к вопросу — так и скажи прямо и НЕ указывай никакую "
          "статью."
    )
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = msgs[0].get("content", "") + block
    else:
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT + block})
    return msgs


CITATION_RE = re.compile(
    r"\s*ст\.\s*\d+(?:\.\d+)?(?:\s*ТК\s*РФ)?\.?", re.IGNORECASE)


def strip_citations(text: str) -> str:
    """Маршрут offtopic: рефлекс файнтюна дорисовывает «ст. N» даже к отказам —
    вычищаем сами паттерны цитат (иногда они в той же строке, что и отказ,
    поэтому строки целиком удалять нельзя)."""
    cleaned = CITATION_RE.sub("", text)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def offtopic_sse_response(data: dict, extra_headers: dict | None = None):
    """Отказ в SSE-формате (фронт просит stream=true): один чанк + [DONE]."""
    try:
        content = strip_citations(data["choices"][0]["message"].get("content", ""))
    except (KeyError, IndexError, TypeError):
        content = ""
    chunk = {"id": data.get("id", "offtopic"),
             "object": "chat.completion.chunk",
             "model": data.get("model", "local"),
             "choices": [{"index": 0, "delta": {"content": content},
                          "finish_reason": "stop"}]}
    payload = (f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
               f"data: [DONE]\n\n")

    async def gen():
        yield payload

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers=extra_headers or {})


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
    if USE_RERANK:
        reranker = CrossEncoder(RERANKER_MODEL)
        print("RAG: reranker загружен", flush=True)
    else:
        reranker = None
        print("RAG: reranker выключен (TK_RERANK=0) — в память не грузим "
              "(экономия ~2,3 ГБ RAM)", flush=True)
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
    retrieval_ms = 0

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
        t_retr = time.perf_counter()
        # Голосование BM25 + эмбеддер → топ-5 (multi-query опционально)
        candidates, best_cos = await retrieve_candidates(query)
        t_rerank = time.perf_counter()
        if best_cos < MIN_QUERY_COS:
            # Гейт релевантности: контекст не инъекцируем, маршрут offtopic
            candidates = []
            route = "offtopic"
        elif USE_RERANK and candidates:
            # Переранкинг cross-encoder → топ-3 для модели
            candidates = rerank_candidates(query, candidates)
            route = "hybrid+rerank"
        else:
            candidates = candidates[:FINAL_TOP_K]
            route = "hybrid"
        if candidates and NEIGHBORS:
            candidates = expand_with_neighbors(candidates)
        print(f"RAG: retrieve {1000 * (t_rerank - t_retr):.0f} мс | "
              f"rerank {1000 * (time.perf_counter() - t_rerank):.0f} мс | "
              f"best_cos={best_cos:.2f} | multi_query={MULTI_QUERY} "
              f"rerank={USE_RERANK} | route={route}",
              flush=True)
        retrieval_ms = int(1000 * (time.perf_counter() - t_retr))

    retrieved_list = [a["number"] for a in candidates]
    client_wants_stream = bool(body.get("stream"))
    if route == "offtopic":
        # Отказ буферизуем целиком (вычищаем цитаты) — llama зовём без стрима
        body["stream"] = False
        body["messages"] = inject_offtopic(messages)
    else:
        body["messages"] = inject_candidates(messages, candidates)

    upstream_request = client.build_request(
        "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions",
        content=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
    )
    upstream_response = await client.send(upstream_request, stream=True)

    extra = {"x-retrieved-articles": ",".join(retrieved_list),
             "x-retrieval-route": route,
             "x-retrieval-ms": str(retrieval_ms)} if retrieved_list else \
            {"x-retrieval-route": route, "x-retrieval-ms": str(retrieval_ms)}

    if route == "offtopic":
        # Отказ: буферизуем и вычищаем случайные «ст. N» — рефлекс файнтюна
        # дорисовывает цитаты даже к отказам
        await upstream_response.aread()
        data = json.loads(upstream_response.content)
        await upstream_response.aclose()
        try:
            msg = data["choices"][0]["message"]
            msg["content"] = strip_citations(msg.get("content", ""))
        except (KeyError, IndexError, TypeError):
            pass
        if client_wants_stream:
            return offtopic_sse_response(data, extra)
        return JSONResponse(data, headers=extra)

    return await relay(upstream_response, extra)


@app.post("/retrieve")
async def retrieve(request: Request):
    """Только поиск, без LLM: бенчмарк ретрива (recall@k) и отладка."""
    body = await request.json()
    if isinstance(body, dict) and body.get("messages"):
        messages = body["messages"]
    else:
        messages = [{"role": "user",
                     "content": str((body or {}).get("query", ""))}]
    query = retrieve_query(messages)
    if not query:
        return {"query": None, "route": "none", "articles": []}

    candidates: list[dict] = []
    route = "none"
    explicit_ref = detect_article_reference(query)
    if explicit_ref:
        exact = next((a for a in articles if a["number"] == explicit_ref), None)
        if exact:
            idx = articles.index(exact)
            candidates = [exact] + articles[idx + 1:idx + 2]
            route = f"direct ст. {explicit_ref}"

    retrieval_ms = 0
    best_cos = -1.0
    if not candidates and query and len(query) >= MIN_QUERY_CHARS:
        t0 = time.perf_counter()
        candidates, best_cos = await retrieve_candidates(query)
        if best_cos < MIN_QUERY_COS:
            candidates = []
            route = "offtopic"
        elif USE_RERANK and candidates:
            candidates = rerank_candidates(query, candidates)
            route = "hybrid+rerank"
        else:
            candidates = candidates[:FINAL_TOP_K]
            route = "hybrid"
        if candidates and NEIGHBORS:
            candidates = expand_with_neighbors(candidates)
        retrieval_ms = int(1000 * (time.perf_counter() - t0))

    return {"query": query, "route": route, "retrieval_ms": retrieval_ms,
            "best_cos": round(best_cos, 3),
            "articles": [{"number": a["number"], "title": a["title"]}
                         for a in candidates]}


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
