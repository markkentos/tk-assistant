# Деплой tk-assistant (GGUF Q4_K_M) — Docker-стек

Состав: **llama.cpp server** (инференс, OpenAI-совместимый API со стримингом) +
**Caddy** (кастомный тёмный фронтенд + reverse-proxy `/v1`).

## Структура

```
deploy/
├── docker-compose.yml   # llama-server + caddy
├── Caddyfile            # статика из web/ + прокси /v1/* → llama-server
├── models/              # qwen2.5-7b-tk-q4_k_m.gguf (~4.4 ГБ, в .gitignore)
└── web/                 # фронтенд: index.html, style.css, app.js (ванильный JS)
```

Фронтенд — ванильные HTML/CSS/JS без сборки: правки `web/*` подхватываются
обновлением страницы (F5). Контейнер пересобирать не нужно.

## Запуск

```bash
cd deploy
docker compose up -d
```

- Локально: `https://tk-assistant.ru` (через hairpin; «Обход NAT» должен быть ON)
- Снаружи: то же — при условии настроенного проброса и белого IP

API llama-server также доступен локально: `http://localhost:8080/v1` (127.0.0.1).

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
6. `docker compose up -d caddy` → Caddy сам получит
   HTTPS-сертификат Let's Encrypt (порты 80+443 должны быть доступны из интернета).

Проверка: `nslookup твой-домен` (должен вернуть белый IP) → открой `https://твой-домен`.

Безопасность: наружу открыты только 80/443 (Caddy), llama-server слушает
127.0.0.1 внутри хоста. Любой в интернете сможет пользоваться демо — это осознанный
выбор для портфолио; при желании Caddy умеет basic_auth (добавим по запросу).

## Безопасность (из реального опыта деплоя)

- Билайн/SmartBox GIGA дропает входящий 443 даже при настроенном пробросе —
  рабочее решение: внешний порт 8443 → внутренний 443 (см. роутер),
  ссылка `https://tk-assistant.ru:8443`
- Rate limiting на клиента — в `web/app.js` (5 запросов/мин на IP)
