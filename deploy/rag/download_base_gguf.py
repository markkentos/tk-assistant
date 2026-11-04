import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
SHARDS = (
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
)

for fname in SHARDS:
    p = hf_hub_download(repo_id=REPO, filename=fname, local_dir="/out")
    print("OK:", p, flush=True)
print("ALL SHARDS DONE", flush=True)
