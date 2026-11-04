# -*- coding: utf-8 -*-
"""Метаданные чистого бейса для convert_lora_to_gguf (--base): только
config + токенизатор, без весов (веса не нужны для конвертации LoRA)."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download

os.makedirs("outputs/_base_config", exist_ok=True)
for f in ("config.json", "generation_config.json",
          "tokenizer.json", "tokenizer_config.json"):
    p = hf_hub_download(repo_id="unsloth/Qwen2.5-7B-Instruct",
                        filename=f, local_dir="outputs/_base_config")
    print("OK:", p, flush=True)
