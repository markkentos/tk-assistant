# -*- coding: utf-8 -*-
"""Скачивание моделей RAG на хост: эмбеддер + reranker (с докачкой)."""
import sys

from huggingface_hub import snapshot_download

sys.stdout.reconfigure(encoding="utf-8")

for repo in ("BAAI/bge-reranker-v2-m3",
             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    print(f"=== {repo}", flush=True)
    path = snapshot_download(repo)
    print(f"    готово: {path}", flush=True)
print("Все модели скачаны")
