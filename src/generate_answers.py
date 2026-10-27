# -*- coding: utf-8 -*-
"""Генерация ответов модели на бенчмарке (этапы 2 и 4).

Режимы:
  --mode base    — base-модель, few-shot промпт (точка отсчёта ДО обучения)
  --mode tuned   — base + QLoRA-адаптер, zero-shot (формат выучен обучением)

Запуск на GPU (Colab/Kaggle):
    python src/generate_answers.py --mode base \
        --input eval/benchmark.jsonl --output eval/runs/answers_base.jsonl
    python src/generate_answers.py --mode tuned --adapter outputs/adapter_e1_lr2e-4 \
        --input eval/benchmark.jsonl --output eval/runs/answers_tuned.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import inference_prompt  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["base", "tuned"], required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default=None,
                        help="путь к адаптеру (для --mode tuned)")
    parser.add_argument("--input", default="eval/benchmark.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fewshot", action="store_true", default=None,
                        help="few-shot промпт (по умолчанию: base=да, tuned=нет)")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    fewshot = args.fewshot if args.fewshot is not None else (args.mode == "base")

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as exc:
        sys.exit(f"Нужна GPU-среда с unsloth: {exc}")

    source = args.adapter if (args.mode == "tuned" and args.adapter) else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"Модель: {source} | режим: {args.mode} | few-shot: {fewshot}")

    items = load_jsonl(args.input)
    results = []
    for i, item in enumerate(items, 1):
        prompt = inference_prompt(tokenizer, item["question"], fewshot=fewshot)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, temperature=None, top_p=None,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
        results.append({
            "id": item["id"], "article": item.get("article"),
            "question": item["question"], "reference": item.get("reference", ""),
            "answer": answer,
            "meta": {"mode": args.mode, "fewshot": fewshot, "model_source": source,
                     "generated_at": dt.datetime.now().isoformat(timespec="seconds")},
        })
        if i % 10 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] готовы")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Ответы: {out} — Дальше: python eval/judge.py --answers {out} --name {args.mode}")


if __name__ == "__main__":
    main()
