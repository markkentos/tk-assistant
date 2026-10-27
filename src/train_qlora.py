# -*- coding: utf-8 -*-
"""Этап 3: QLoRA-обучение через Unsloth + TRL SFTTrainer.

Запускается на GPU (Colab T4 / Kaggle). Локально не запускать — нужен unsloth+torch+CUDA.

Особенности T4: нет bf16 → обучение в fp16 (Unsloth выберет dtype автоматически,
в конфиге это зафиксировано явно).

Запуск (в Colab):
    python src/train_qlora.py --config configs/train_qlora.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import inference_prompt, training_text  # noqa: E402


def load_pairs(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train_qlora.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        sys.exit(f"Нужна GPU-среда с unsloth (см. requirements-colab.txt): {exc}")

    wandb_enabled = cfg["wandb"].get("enabled", False)
    if wandb_enabled:
        import wandb

    # ---------- Модель (4-bit NF4, dtype подберётся: на T4 это fp16) ----------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["train"]["seed"],
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Обучаемых параметров: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ---------- Данные: строго apply_chat_template ----------
    train_records = load_pairs(cfg["train"]["dataset"])
    train_dataset = train_records
    from datasets import Dataset

    def to_text(batch):
        return {"text": [training_text(tokenizer, r) for r in batch["records"]]}

    train_dataset = Dataset.from_dict({"records": train_records}).map(
        to_text, batched=True, remove_columns=["records"])
    eval_dataset = None
    if cfg["train"].get("val_dataset"):
        val_records = load_pairs(cfg["train"]["val_dataset"])
        eval_dataset = Dataset.from_dict({"records": val_records}).map(
            to_text, batched=True, remove_columns=["records"])

    sample = train_dataset[0]["text"]
    print("Пример обучающего текста:\n", sample[:600], "...")

    # ---------- Тренер ----------
    run_name = pathlib.Path(cfg["output"]["adapter_dir"]).name
    try:  # TRL >= 0.13
        train_args = SFTConfig(
            output_dir=str(pathlib.Path(cfg["output"]["adapter_dir"]) / "checkpoints"),
            dataset_text_field="text",
            max_seq_length=cfg["model"]["max_seq_length"],
            packing=False,
            num_train_epochs=cfg["train"]["epochs"],
            per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
            gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
            learning_rate=cfg["train"]["lr"],
            lr_scheduler_type=cfg["train"]["scheduler"],
            warmup_ratio=cfg["train"]["warmup_ratio"],
            weight_decay=cfg["train"]["weight_decay"],
            fp16=cfg["train"]["fp16"],
            logging_steps=cfg["train"]["logging_steps"],
            save_steps=cfg["train"]["save_steps"],
            seed=cfg["train"]["seed"],
            optim="adamw_8bit",
            report_to="wandb" if wandb_enabled else "none",
            run_name=run_name,
        )
    except TypeError:  # старый TRL: TrainingArguments + отдельные поля SFTTrainer
        from transformers import TrainingArguments
        train_args = TrainingArguments(
            output_dir=str(pathlib.Path(cfg["output"]["adapter_dir"]) / "checkpoints"),
            num_train_epochs=cfg["train"]["epochs"],
            per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
            gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
            learning_rate=cfg["train"]["lr"],
            lr_scheduler_type=cfg["train"]["scheduler"],
            warmup_ratio=cfg["train"]["warmup_ratio"],
            weight_decay=cfg["train"]["weight_decay"],
            fp16=cfg["train"]["fp16"],
            logging_steps=cfg["train"]["logging_steps"],
            save_steps=cfg["train"]["save_steps"],
            seed=cfg["train"]["seed"],
            optim="adamw_8bit",
            report_to="wandb" if wandb_enabled else "none",
            run_name=run_name,
        )

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
    )
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    trainer.train()

    # ---------- Сохранение адаптера ----------
    adapter_dir = pathlib.Path(cfg["output"]["adapter_dir"])
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Адаптер сохранён: {adapter_dir}")

    if cfg["output"].get("push_to_hub") and cfg["output"].get("hf_repo"):
        model.push_to_hub(cfg["output"]["hf_repo"], token=True)
        tokenizer.push_to_hub(cfg["output"]["hf_repo"], token=True)
        print(f"Адаптер запушен: https://huggingface.co/{cfg['output']['hf_repo']}")

    # ---------- Быстрый санити-сэмпл на val (10 ответов глазами) ----------
    n_sample = cfg["output"].get("sample_n", 10)
    if eval_dataset is None or n_sample == 0:
        return

    FastLanguageModel.for_inference(model)
    import torch

    samples = []
    for record in val_records[:n_sample]:
        prompt = inference_prompt(tokenizer, record["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                                    temperature=None, top_p=None,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        samples.append({"id": record.get("id"), "article": record["article"],
                        "question": record["question"],
                        "gold_answer": record["answer"], "model_answer": answer})
        print("-" * 60)
        print("Q:", record["question"])
        print("MODEL:", answer[:400])
        print("GOLD :", record["answer"][:200])

    sample_path = adapter_dir / f"val_samples_{dt.datetime.now():%Y%m%d_%H%M}.jsonl"
    with open(sample_path, "w", encoding="utf-8") as fh:
        for row in samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nСэмпл сохранён: {sample_path} — просмотри глазами перед экспериментом!")


if __name__ == "__main__":
    main()
