# Деплой tk-assistant (GGUF Q4_K_M) — Docker-стек

Состав: `llama.cpp server` (инференс) + Gradio UI (чат со стримингом).

## Запуск на своём компьютере (Windows/Linux/macOS)

1. Положи GGUF-файл (`*.gguf`, ~4.4 ГБ из Colab-ноутбука `03_gguf_export_colab.ipynb`)
   в папку `deploy/models/`:

   ```
   deploy/
   ├── docker-compose.yml
   ├── models/
   │   └── qwen2.5-7b-tk-q4_k_m.gguf
   └── app/
   ```

2. Из папки `deploy/`:

   ```bash
   docker compose up --build -d
   ```

3. Открой http://localhost:7860 — чат-интерфейс.

Первый старт: llama-server загрузит модель в память (~1–2 минуты),
последующие ответы — стримингом по ~10–30 секунд (CPU).

## Полезное

- Логи инференса: `docker compose logs -f llama-server`
- Остановить: `docker compose down`
- Потоки CPU: параметр `-t 0` в `docker-compose.yml` (0 = авто)

## Публичный доступ (этап «домен на рег.ру»)

- Вариант РФ (рекомендуется): белый IP от провайдера → проброс портов 80/443
  на комп → Caddy-контейнер с авто-HTTPS (добавим по запросу) → A-запись домена.
- Вариант Cloudflare Tunnel — в РФ нестабилен (РКН замедляет CF), не рекомендуем.
- Вариант HF Spaces — без своего железа, см. notebooks/03_gguf_export_colab.ipynb (пуш на Hub).

llama-server и Gradio слушают только 127.0.0.1 — наружу ничего не торчит,
пока не настроен обратный прокси.
