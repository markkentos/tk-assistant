/* tk-assistant · логика чата: SSE-стриминг из llama-server (OpenAI API) */
"use strict";

const API = "/v1/chat/completions";
const MAX_HISTORY = 12;            // сообщений в контексте (без system)
const MAX_TOKENS = 400;
const TEMPERATURE = 0.2;

const SYSTEM_PROMPT =
  "Ты — ассистент по трудовому праву РФ. Отвечай на вопросы, опираясь на " +
  "Трудовой кодекс РФ. Формат ответа — строго две части:\n" +
  "1) краткий ответ одним предложением;\n" +
  "2) обоснование из 1–3 предложений со ссылкой на статью в формате «ст. N ТК РФ».\n" +
  "Не выдумывай номера статей. Если не уверен — так и скажи.";

const EXAMPLES = [
  "За сколько дней работник должен предупредить работодателя об увольнении по собственному желанию?",
  "Может ли работодатель уволить беременную сотрудницу?",
  "Какова нормальная продолжительность рабочего времени в неделю?",
  "Сколько дней отпуска положено работнику в возрасте до 18 лет?",
];

/* ── Мини-маркдаун: escape → ссылки → жирный → абзацы ─────────────── */
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const ARTICLE_LINK_RE = /ст(?:ать[а-яё]*)?\.?\s*№?\s*(\d+(?:\.\d+)?)\s*ТК\s*РФ/gi;

function linkifyArticles(escaped) {
  return escaped.replace(ARTICLE_LINK_RE, (m, num) =>
    `[${m}](https://www.zakonrf.info/tk/${num}/)`);
}

function mdRender(text) {
  let html = escapeHtml(text);
  html = linkifyArticles(html);
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // абзацы и переносы
  html = html.split(/\n{2,}/).map(p =>
    "<p>" + p.replace(/\n/g, "<br>") + "</p>").join("");
  return html;
}

/* ── DOM ──────────────────────────────────────────────────────────── */
const els = {
  empty: document.getElementById("empty"),
  chat: document.getElementById("chat"),
  chips: document.getElementById("chips"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
};

const history = [];          // {role, content} — без system
let streaming = false;
let controller = null;
let session = 0;             // номер сессии чата: newchat инвалидирует текущий стрим

/* ── UI-хелперы ───────────────────────────────────────────────────── */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToEnd() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addUserMessage(text) {
  const msg = el("div", "msg msg-user");
  const bubble = el("div", "bubble", text);
  msg.append(bubble);
  els.chat.append(msg);
  els.empty.hidden = true;
  els.chat.hidden = false;
  scrollToEnd();
}

function addAssistantMessage() {
  const msg = el("div", "msg msg-assistant");
  msg.append(el("div", "who", "tk-assistant"));
  const bubble = el("div", "bubble");
  const span = el("span");
  span.className = "caret";
  bubble.append(span);
  msg.append(bubble);
  els.chat.append(msg);
  scrollToEnd();
  return bubble;
}

function setComposerDisabled(state) {
  els.send.disabled = state;
  els.input.disabled = state;
  els.send.textContent = state ? "■" : "↑";
}

/* ── Отправка ─────────────────────────────────────────────────────── */
async function sendMessage(text) {
  text = text.trim();
  if (!text || streaming) return;

  els.empty.hidden = true;
  els.chat.hidden = false;
  addUserMessage(text);
  history.push({ role: "user", content: text });

  const bubble = addAssistantMessage();
  streaming = true;
  setComposerDisabled(true);
  controller = new AbortController();
  const sid = session;

  let acc = "";
  try {
    const messages = [{ role: "system", content: SYSTEM_PROMPT },
      ...history.slice(-MAX_HISTORY)];
    const response = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "local",
        messages,
        stream: true,
        temperature: TEMPERATURE,
        max_tokens: MAX_TOKENS,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") { buffer = ""; break; }
        try {
          const delta = JSON.parse(payload).choices?.[0]?.delta?.content;
          if (delta) {
            acc += delta;
            bubble.innerHTML = mdRender(acc) + '<span class="caret"></span>';
            scrollToEnd();
          }
        } catch (_) { /* частичный JSON — ждём следующий чанк */ }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      acc += acc ? "\n\n" : "";
      acc += "Ошибка соединения с моделью. Попробуйте ещё раз.";
    }
  } finally {
    if (sid !== session) return;   // чат сброшен кнопкой «Новый чат»
    streaming = false;
    controller = null;
    setComposerDisabled(false);
    if (!acc) acc = "Пустой ответ модели.";
    bubble.innerHTML = mdRender(acc);
    history.push({ role: "assistant", content: acc });
    scrollToEnd();
  }
}

/* ── Примеры-подсказки ────────────────────────────────────────────── */
EXAMPLES.forEach(text => {
  const chip = el("button", "chip", text);
  chip.addEventListener("click", () => sendMessage(text));
  els.chips.append(chip);
});

/* ── Ввод ─────────────────────────────────────────────────────────── */
function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
}

els.input.addEventListener("input", autosize);
els.input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = els.input.value;
    els.input.value = "";
    autosize();
    sendMessage(text);
  }
});
els.send.addEventListener("click", () => {
  const text = els.input.value;
  els.input.value = "";
  autosize();
  sendMessage(text);
});

/* ── Новый чат ────────────────────────────────────────────────────── */
document.getElementById("newchat").addEventListener("click", () => {
  session += 1;                       // активный стрим станет неактуальным
  if (controller) controller.abort();
  streaming = false;
  controller = null;
  history.length = 0;
  els.chat.innerHTML = "";
  els.chat.hidden = true;
  els.empty.hidden = false;
  els.input.value = "";
  setComposerDisabled(false);
  autosize();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
autosize();
