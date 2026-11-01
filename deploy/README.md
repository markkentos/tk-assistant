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

## Публичный доступ (домен на рег.ру + белый IP)

1. **Белый IP** у провайдера (~200 ₽/мес). Статический — идеально; динамический —
   нужен DDNS-скрипт (обновляет A-запись при смене IP, напишем по запросу).
2. **DNS в рег.ру**: домен → DNS → создать запись `A` с твоим белым IP
   (для корня — `@`, для поддомена — `demo`). Распространение: минуты — 24 часа.
3. **Роутер**: проброс портов 80 и 443 → локальный IP компа (закрепи его в DHCP).
4. **Файрвол Windows** (PowerShell от админа):

   ```powershell
   New-NetFirewallRule -DisplayName "HTTP/HTTPS tk-assistant" -Direction Inbound -LocalPort 80,443 -Protocol TCP -Action Allow
   ```

5. **Caddyfile**: впиши свой домен вместо `your-domain.ru`.
6. `docker compose up -d caddy gradio llama-server` → Caddy сам получит
   HTTPS-сертификат Let's Encrypt (порты 80+443 должны быть доступны из интернета).

Проверка: `nslookup твой-домен` (должен вернуть белый IP) → открой `https://твой-домен`.

Безопасность: наружу открыты только 80/443 (Caddy), llama-server и Gradio слушают
внутри сети compose. Любой в интернете сможет пользоваться демо — это осознанный
выбор для портфолио; при желании Caddy умеет basic_auth (добавим по запросу).
