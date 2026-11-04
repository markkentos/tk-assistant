# -*- coding: utf-8 -*-
"""Скачивание моделей RAG в HF-кэш с докачкой (запуск: docker run ... python /dl.py)."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # классический HTTP: докачка через Range

from huggingface_hub import snapshot_download

# Только то, что реально читают SentenceTransformer/CrossEncoder:
# без onnx/openvino-копий (~1 ГБ балласта)
PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]

for repo in (
    "BAAI/bge-reranker-v2-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
):
    print(f"=== {repo}", flush=True)
    path = snapshot_download(repo, allow_patterns=PATTERNS, max_workers=2)
    print(f"OK: {path}", flush=True)

print("ALL MODELS DONE", flush=True)
