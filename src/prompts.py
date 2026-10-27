# -*- coding: utf-8 -*-
"""Промпты и форматирование данных для обучения и инференса.

Единый формат ответа — тот же, что задан генератору датасета и судье:
  <краткий ответ одним предложением>
  <обоснование со ссылкой «ст. N ТК РФ»>

Всё форматирование — строго через tokenizer.apply_chat_template.
Никакой ручной склейки строк с ролями/спец-токенами.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "Ты — ассистент по трудовому праву РФ. Отвечай на вопросы, опираясь на "
    "Трудовой кодекс РФ. Формат ответа — строго две части:\n"
    "1) краткий ответ одним предложением;\n"
    "2) обоснование из 1–3 предложений со ссылкой на статью в формате «ст. N ТК РФ».\n"
    "Не выдумывай номера статей. Если не уверен — так и скажи."
)

# Few-shot для base-модели на бенчмарке (этап 2): base без дообучения плохо держит
# наш формат, поэтому показываем 3 примера. Примеры — из train-сплита, не из test!
FEWSHOT_EXAMPLES = [
    {
        "question": "Какая максимальная продолжительность рабочего времени в неделю?",
        "answer": "Нормальная продолжительность рабочего времени не может превышать "
                  "40 часов в неделю.\n"
                  "Обоснование: это общее правило, установленное кодексом "
                  "(ст. 91 ТК РФ).",
    },
    {
        "question": "За сколько дней работник должен предупредить работодателя об "
                    "увольнении по собственному желанию?",
        "answer": "Работник обязан письменно предупредить работодателя не позднее чем "
                  "за две недели.\nОбоснование: течение срока начинается на следующий "
                  "день после получения заявления работодателем (ст. 80 ТК РФ).",
    },
    {
        "question": "Может ли работодатель уволить работника в период его болезни?",
        "answer": "Нет, расторжение трудового договора по инициативе работодателя в "
                  "период временной нетрудоспособности работника не допускается.\n"
                  "Обоснование: исключение — ликвидация организации или прекращение "
                  "деятельности работодателем — физическим лицом (ст. 81 ТК РФ).",
    },
]


def build_messages(question: str, fewshot: bool = False) -> list[dict]:
    """Сообщения для чат-шаблона: [system, (few-shot примеры), user]."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fewshot:
        for example in FEWSHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def training_text(tokenizer, record: dict) -> str:
    """Полный текст примера для SFT: system + user + assistant, один промпт-шаблон."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
        {"role": "assistant", "content": record["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def inference_prompt(tokenizer, question: str, fewshot: bool = False) -> str:
    """Промпт для генерации: chat-шаблон с add_generation_prompt=True."""
    return tokenizer.apply_chat_template(
        build_messages(question, fewshot=fewshot),
        tokenize=False, add_generation_prompt=True)
