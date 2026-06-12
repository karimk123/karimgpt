/* =====================================================================
   KarimGPT frontend
   - Talks to the FastAPI backend (/api/chat, /api/models) over SSE.
   - Supports all three KarimGPT models via the model picker.
   - Stores conversations in localStorage (sidebar history, per-chat model).
   - When served by the backend itself, API_BASE = "" works out of the box.
   - When deployed to Vercel (static), point it at a backend with
     ?api=https://your-backend (remembered in localStorage).
   ===================================================================== */

const API_BASE = (() => {
  const fromQuery = new URLSearchParams(location.search).get("api");
  if (fromQuery) {
    localStorage.setItem("karimgpt_api", fromQuery);
    return fromQuery.replace(/\/$/, "");
  }
  if (window.KARIMGPT_API) return window.KARIMGPT_API.replace(/\/$/, "");
  const saved = localStorage.getItem("karimgpt_api");
  if (saved) return saved.replace(/\/$/, "");
  return ""; // same-origin (backend serves the page locally)
})();

/* ---------- Model catalog (server is source of truth; this is the fallback) ---------- */
const FALLBACK_MODELS = [
  { id: "v1", name: "KarimGPT 1", params: "13M", description: "Character-level · writes Shakespeare", style: "completion" },
  { id: "v2", name: "KarimGPT 2", params: "33M", description: "BPE · trained on 50k conversations", style: "chat" },
  { id: "v3", name: "KarimGPT 3", params: "91M", description: "GPT-2 style · largest and most capable", style: "chat" },
];

const SUGGESTIONS = {
  v1: ["ROMEO:", "To be, or not to be", "Write a soliloquy about the moon", "O gentle night,"],
  v2: ["Tell me about yourself", "What should I cook tonight?", "Write me a short story", "What is the meaning of life?"],
  v3: ["What is a neural network?", "How do I make pasta?", "Write a short poem about the ocean", "What are the benefits of exercise?"],
};

const PLACEHOLDERS = { completion: "Write the opening of a scene…", chat: "Ask anything" };

/* ---------- DOM ---------- */
const $ = (s) => document.querySelector(s);
const app = $("#app");
const sidebarEl = $("#sidebar");
const historyEl = $("#history");
const searchInput = $("#search-input");
const threadEl = $("#thread");
const messagesEl = $("#messages");
const welcomeEl = $("#welcome");
const welcomeSlot = $("#welcome-slot");
const suggestionsEl = $("#suggestions");
const bottomArea = $("#bottom-area");
const bottomSlot = $("#bottom-slot");
const composerEl = $("#composer");
const inputEl = $("#input");
const sendBtn = $("#send-btn");
const hintEl = $("#composer-hint");
const scrollBtn = $("#scroll-btn");
const modelBtn = $("#model-btn");
const modelNameEl = $("#model-name");
const modelMenu = $("#model-menu");
const chatMenu = $("#chat-menu");
const userMenu = $("#user-menu");
const connDot = $("#conn-dot");

const ICONS = {
  send: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>`,
  stop: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>`,
  copy: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>`,
  check: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`,
  retry: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/></svg>`,
  pencil: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`,
  trash: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>`,
  dots: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>`,
  warn: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>`,
  globe: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18"/></svg>`,
};

/* ---------- State ---------- */
let models = FALLBACK_MODELS;
let currentModel = localStorage.getItem("karimgpt_model") || "v3";
let chats = loadChats(); // [{id, title, model, messages:[{role,content}], updatedAt}]
let currentId = null;
let streaming = false;
let abortCtrl = null;
let stickToBottom = true;

/* ---------- Persistence ---------- */
function loadChats() {
  try {
    const v2 = JSON.parse(localStorage.getItem("karimgpt_chats_v2"));
    if (Array.isArray(v2)) return v2;
  } catch {}
  // migrate from the original single-model app
  try {
    const v1 = JSON.parse(localStorage.getItem("karimgpt_chats"));
    if (Array.isArray(v1)) {
      const migrated = v1.map((c, i) => ({
        ...c, model: "v2", updatedAt: Date.now() - i,
      }));
      localStorage.setItem("karimgpt_chats_v2", JSON.stringify(migrated));
      return migrated;
    }
  } catch {}
  return [];
}
function saveChats() {
  localStorage.setItem("karimgpt_chats_v2", JSON.stringify(chats));
}
function currentChat() {
  return chats.find((c) => c.id === currentId);
}
function modelById(id) {
  return models.find((m) => m.id === id) || models[models.length - 1];
}

/* =====================================================================
   MARKDOWN — minimal, XSS-safe renderer (DOM building, no innerHTML of
   model output). Covers what ChatGPT-style replies need: paragraphs,
   headings, lists, blockquotes, hr, fenced code, inline code/bold/italic,
   and links.
   ===================================================================== */
function renderMarkdown(target, text) {
  target.textContent = "";
  const lines = text.split("\n");
  let i = 0;
  let para = [];

  const flushPara = () => {
    if (!para.length) return;
    const p = document.createElement("p");
    renderInline(p, para.join("\n"));
    target.appendChild(p);
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code block
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      flushPara();
      const lang = fence[1] || "";
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++; // skip closing fence (or EOF)
      target.appendChild(codeCard(lang, buf.join("\n")));
      continue;
    }

    // heading
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      flushPara();
      const el = document.createElement(`h${h[1].length}`);
      renderInline(el, h[2]);
      target.appendChild(el);
      i++;
      continue;
    }

    // horizontal rule
    if (/^(---+|\*\*\*+)\s*$/.test(line)) {
      flushPara();
      target.appendChild(document.createElement("hr"));
      i++;
      continue;
    }

    // blockquote
    if (/^>\s?/.test(line)) {
      flushPara();
      const bq = document.createElement("blockquote");
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      renderInline(bq, buf.join("\n"));
      target.appendChild(bq);
      continue;
    }

    // lists
    const ul = /^\s*[-*]\s+/.test(line);
    const ol = /^\s*\d+[.)]\s+/.test(line);
    if (ul || ol) {
      flushPara();
      const list = document.createElement(ol ? "ol" : "ul");
      const re = ol ? /^\s*\d+[.)]\s+/ : /^\s*[-*]\s+/;
      while (i < lines.length && re.test(lines[i])) {
        const li = document.createElement("li");
        renderInline(li, lines[i].replace(re, ""));
        list.appendChild(li);
        i++;
      }
      target.appendChild(list);
      continue;
    }

    // blank line ends a paragraph
    if (line.trim() === "") {
      flushPara();
      i++;
      continue;
    }

    para.push(line);
    i++;
  }
  flushPara();
}

function renderInline(target, text) {
  // tokens: `code`  **bold**  *italic*  [text](url)
  const re = /(`[^`\n]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(\[[^\]]+\]\((https?:\/\/[^\s)]+)\))/g;
  let last = 0;
  let m;
  const emit = (s) => {
    if (!s) return;
    const parts = s.split("\n");
    parts.forEach((p, idx) => {
      target.appendChild(document.createTextNode(p));
      if (idx < parts.length - 1) target.appendChild(document.createElement("br"));
    });
  };
  while ((m = re.exec(text)) !== null) {
    emit(text.slice(last, m.index));
    if (m[1]) {
      const c = document.createElement("code");
      c.textContent = m[1].slice(1, -1);
      target.appendChild(c);
    } else if (m[2]) {
      const b = document.createElement("strong");
      renderInline(b, m[2].slice(2, -2));
      target.appendChild(b);
    } else if (m[3]) {
      const em = document.createElement("em");
      renderInline(em, m[3].slice(1, -1));
      target.appendChild(em);
    } else if (m[4]) {
      const a = document.createElement("a");
      a.href = m[5];
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = m[4].slice(1, m[4].indexOf("]"));
      target.appendChild(a);
    }
    last = re.lastIndex;
  }
  emit(text.slice(last));
}

function codeCard(lang, code) {
  const card = document.createElement("div");
  card.className = "code-card";
  const head = document.createElement("div");
  head.className = "code-head";
  const label = document.createElement("span");
  label.textContent = lang || "code";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "code-copy";
  copy.innerHTML = `${ICONS.copy}<span>Copy code</span>`;
  copy.addEventListener("click", () => copyFeedback(copy, code, "Copy code"));
  head.append(label, copy);
  const pre = document.createElement("pre");
  const c = document.createElement("code");
  c.className = "block";
  c.textContent = code;
  pre.appendChild(c);
  card.append(head, pre);
  return card;
}

function copyFeedback(btn, text, restoreLabel) {
  navigator.clipboard?.writeText(text).then(() => {
    const html = btn.innerHTML;
    btn.innerHTML = restoreLabel ? `${ICONS.check}<span>Copied!</span>` : ICONS.check;
    btn.classList.add("ok");
    setTimeout(() => {
      btn.innerHTML = restoreLabel ? `${ICONS.copy}<span>${restoreLabel}</span>` : html;
      btn.classList.remove("ok");
    }, 1500);
  });
}

/* =====================================================================
   RENDERING
   ===================================================================== */
function setEmptyState(empty) {
  welcomeEl.hidden = !empty;
  threadEl.hidden = empty;
  bottomArea.hidden = empty;
  (empty ? welcomeSlot : bottomSlot).appendChild(composerEl);
  if (empty) renderSuggestions();
}

function renderSuggestions() {
  suggestionsEl.textContent = "";
  for (const s of SUGGESTIONS[currentModel] || SUGGESTIONS.v3) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "suggestion";
    b.textContent = s;
    b.addEventListener("click", () => sendMessage(s));
    suggestionsEl.appendChild(b);
  }
  const m = modelById(currentModel);
  inputEl.placeholder = PLACEHOLDERS[m.style] || PLACEHOLDERS.chat;
}

/* ----- sidebar history, grouped by date ----- */
function groupLabel(ts) {
  const now = new Date();
  const d = new Date(ts || 0);
  const startOfDay = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.floor((startOfDay(now) - startOfDay(d)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 Days";
  if (days < 30) return "Previous 30 Days";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function renderHistory() {
  historyEl.textContent = "";
  const q = (searchInput.value || "").trim().toLowerCase();
  const list = chats.filter((c) => !q || c.title.toLowerCase().includes(q));

  if (!list.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = q ? "No chats found" : "No chats yet";
    historyEl.appendChild(empty);
    return;
  }

  let lastLabel = null;
  for (const chat of list) {
    const label = groupLabel(chat.updatedAt);
    if (label !== lastLabel) {
      const el = document.createElement("div");
      el.className = "history-label";
      el.textContent = label;
      historyEl.appendChild(el);
      lastLabel = label;
    }

    const item = document.createElement("div");
    item.className = "history-item" + (chat.id === currentId ? " active" : "");

    const titleBtn = document.createElement("button");
    titleBtn.type = "button";
    titleBtn.className = "title-btn";
    const fade = document.createElement("span");
    fade.className = "fade";
    fade.textContent = chat.title;
    titleBtn.appendChild(fade);
    titleBtn.addEventListener("click", () => openChat(chat.id));

    const moreBtn = document.createElement("button");
    moreBtn.type = "button";
    moreBtn.className = "more-btn";
    moreBtn.title = "Options";
    moreBtn.innerHTML = ICONS.dots;
    moreBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openChatMenu(item, moreBtn, chat);
    });

    item.append(titleBtn, moreBtn);
    historyEl.appendChild(item);
  }
}

function startRename(item, chat) {
  const input = document.createElement("input");
  input.className = "rename-input";
  input.value = chat.title;
  item.textContent = "";
  item.appendChild(input);
  input.focus();
  input.select();
  const commit = () => {
    const v = input.value.trim();
    if (v) {
      chat.title = v;
      saveChats();
    }
    renderHistory();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") renderHistory();
  });
  input.addEventListener("blur", commit);
}

/* ----- messages ----- */
function renderMessages() {
  const chat = currentChat();
  if (!chat || chat.messages.length === 0) {
    messagesEl.textContent = "";
    setEmptyState(true);
    return;
  }
  setEmptyState(false);
  messagesEl.textContent = "";
  chat.messages.forEach((m, idx) => {
    messagesEl.appendChild(
      m.role === "user" ? userRow(m.content) : assistantRow(m.content, idx === chat.messages.length - 1)
    );
  });
  markLastAssistant();
  jumpToBottom(false);
}

function userRow(content) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  row.appendChild(bubble);
  return row;
}

function assistantRow(content, withActions = true) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  const body = document.createElement("div");
  body.className = "assistant-body";
  if (content) renderMarkdown(body, content);
  row.appendChild(body);
  if (withActions) row.appendChild(actionsRow(row));
  return row;
}

function actionsRow(row) {
  const bar = document.createElement("div");
  bar.className = "msg-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "action-btn";
  copyBtn.title = "Copy";
  copyBtn.innerHTML = ICONS.copy;
  copyBtn.addEventListener("click", () => {
    const body = row.querySelector(".assistant-body");
    copyFeedback(copyBtn, body?.innerText || "");
  });

  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className = "action-btn";
  retryBtn.title = "Regenerate";
  retryBtn.innerHTML = ICONS.retry;
  retryBtn.addEventListener("click", regenerate);

  bar.append(copyBtn, retryBtn);
  return bar;
}

function markLastAssistant() {
  messagesEl.querySelectorAll(".msg-row.assistant").forEach((r) => r.classList.remove("last"));
  const rows = messagesEl.querySelectorAll(".msg-row.assistant");
  if (rows.length) rows[rows.length - 1].classList.add("last");
}

/* ----- scrolling ----- */
function jumpToBottom(smooth = true) {
  threadEl.scrollTo({ top: threadEl.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}
function maybeAutoScroll() {
  if (stickToBottom) threadEl.scrollTop = threadEl.scrollHeight;
}
threadEl.addEventListener("scroll", () => {
  const dist = threadEl.scrollHeight - threadEl.scrollTop - threadEl.clientHeight;
  stickToBottom = dist < 60;
  scrollBtn.hidden = stickToBottom;
});
scrollBtn.addEventListener("click", () => {
  stickToBottom = true;
  scrollBtn.hidden = true;
  jumpToBottom();
});

/* =====================================================================
   CHAT ACTIONS
   ===================================================================== */
function newChat() {
  if (streaming) stopStreaming();
  currentId = null;
  renderHistory();
  renderMessages();
  closeMenus();
  if (window.innerWidth <= 768) app.classList.remove("sidebar-open");
  inputEl.focus();
}

function openChat(id) {
  if (streaming) stopStreaming();
  currentId = id;
  const chat = currentChat();
  if (chat?.model && models.some((m) => m.id === chat.model)) {
    currentModel = chat.model;
    updateModelLabel();
  }
  renderHistory();
  renderMessages();
  closeMenus();
  if (window.innerWidth <= 768) app.classList.remove("sidebar-open");
}

function deleteChat(id) {
  chats = chats.filter((c) => c.id !== id);
  if (currentId === id) currentId = null;
  saveChats();
  renderHistory();
  renderMessages();
}

function ensureChat(firstMsg) {
  if (currentChat()) return currentChat();
  const chat = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: firstMsg.slice(0, 42) || "New chat",
    model: currentModel,
    messages: [],
    updatedAt: Date.now(),
  };
  chats.unshift(chat);
  currentId = chat.id;
  return chat;
}

function regenerate() {
  const chat = currentChat();
  if (!chat || streaming) return;
  // drop the trailing assistant reply, then re-send from history
  if (chat.messages[chat.messages.length - 1]?.role === "assistant") chat.messages.pop();
  saveChats();
  renderMessages();
  streamReply(chat);
}

/* =====================================================================
   SEND / STREAM
   ===================================================================== */
function sendMessage(text) {
  text = text.trim();
  if (!text || streaming) return;

  const chat = ensureChat(text);
  chat.messages.push({ role: "user", content: text });
  chat.updatedAt = Date.now();
  saveChats();
  renderHistory();
  renderMessages();

  inputEl.value = "";
  autoGrow();
  updateSendState();
  streamReply(chat);
}

async function streamReply(chat) {
  const modelId = chat.model || currentModel;
  const model = modelById(modelId);

  // assistant placeholder
  const row = assistantRow("", true);
  const body = row.querySelector(".assistant-body");
  const pending = document.createElement("span");
  pending.className = "pending";
  pending.innerHTML = `<span class="orb"></span>`;
  body.appendChild(pending);
  messagesEl.appendChild(row);
  markLastAssistant();
  stickToBottom = true;
  jumpToBottom(false);

  setStreaming(true);
  abortCtrl = new AbortController();

  let acc = "";
  let errored = false;

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelId,
        messages: chat.messages.map(({ role, content }) => ({ role, content })),
      }),
      signal: abortCtrl.signal,
    });

    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    setConn(true);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop(); // keep incomplete frame

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data:")) continue;
        let payload;
        try { payload = JSON.parse(line.slice(5).trim()); } catch { continue; }

        if (payload.status === "loading") {
          // payload.model is server-supplied — build the label via textContent.
          pending.textContent = "";
          const orb = document.createElement("span");
          orb.className = "orb";
          const label = document.createElement("span");
          label.className = "label";
          label.textContent = `Loading ${payload.model || model.name}…`;
          pending.append(orb, label);
          continue;
        }
        if (payload.error) {
          errored = true;
          showError(body, payload.error);
          continue;
        }
        if (payload.delta) {
          acc += payload.delta;
          paintStreaming(body, acc);
          maybeAutoScroll();
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      errored = true;
      setConn(false);
      showError(
        body,
        `Could not reach the model backend${API_BASE ? ` at ${API_BASE}` : ""}. ` +
        `Make sure server.py is running, then try again.`
      );
    }
  } finally {
    if (!errored) {
      renderMarkdown(body, acc || "…");
      chat.messages.push({ role: "assistant", content: acc || "…" });
      chat.updatedAt = Date.now();
      saveChats();
      renderHistory();
    }
    setStreaming(false);
    abortCtrl = null;
    maybeAutoScroll();
  }
}

function paintStreaming(body, text) {
  renderMarkdown(body, text);
  const cursor = document.createElement("span");
  cursor.className = "stream-cursor";
  (body.lastElementChild && /^(P|H1|H2|H3|LI|BLOCKQUOTE)$/.test(body.lastElementChild.tagName)
    ? body.lastElementChild
    : body
  ).appendChild(cursor);
}

function showError(body, message) {
  body.textContent = "";
  const card = document.createElement("div");
  card.className = "error-card";
  card.innerHTML = ICONS.warn;
  const span = document.createElement("span");
  span.textContent = message;
  card.appendChild(span);
  body.appendChild(card);
}

function stopStreaming() {
  if (abortCtrl) abortCtrl.abort();
}

function setStreaming(on) {
  streaming = on;
  if (on) {
    sendBtn.innerHTML = ICONS.stop;
    sendBtn.classList.add("streaming");
    sendBtn.title = "Stop generating";
    sendBtn.disabled = false;
  } else {
    sendBtn.innerHTML = ICONS.send;
    sendBtn.classList.remove("streaming");
    sendBtn.title = "Send";
    updateSendState();
  }
}

/* =====================================================================
   MODEL PICKER + MENUS
   ===================================================================== */
function updateModelLabel() {
  modelNameEl.textContent = modelById(currentModel).name;
}

function openModelMenu() {
  if (!modelMenu.hidden) { closeMenus(); return; }
  closeMenus();
  modelMenu.textContent = "";

  const header = document.createElement("div");
  header.className = "menu-header";
  header.textContent = "Models — trained from scratch by Karim";
  modelMenu.appendChild(header);

  for (const m of models) {
    const opt = document.createElement("button");
    opt.type = "button";
    opt.className = "model-option" + (m.id === currentModel ? " selected" : "");

    const meta = document.createElement("span");
    meta.className = "opt-meta";
    const name = document.createElement("span");
    name.className = "opt-name";
    name.textContent = m.name;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = m.params;
    name.appendChild(badge);
    const desc = document.createElement("span");
    desc.className = "opt-desc";
    desc.textContent = m.description;
    meta.append(name, desc);

    const check = document.createElement("span");
    check.className = "check";
    check.innerHTML = ICONS.check;

    opt.append(meta, check);
    opt.addEventListener("click", () => {
      currentModel = m.id;
      localStorage.setItem("karimgpt_model", currentModel);
      const chat = currentChat();
      if (chat) {
        chat.model = currentModel;
        saveChats();
      }
      updateModelLabel();
      renderSuggestions();
      closeMenus();
    });
    modelMenu.appendChild(opt);
  }

  positionMenu(modelMenu, modelBtn);
}

function openChatMenu(item, anchor, chat) {
  closeMenus();
  item.classList.add("menu-open");
  chatMenu.textContent = "";

  const rename = document.createElement("button");
  rename.type = "button";
  rename.className = "menu-item";
  rename.innerHTML = `${ICONS.pencil}<span>Rename</span>`;
  rename.addEventListener("click", () => {
    closeMenus();
    const fresh = [...historyEl.querySelectorAll(".history-item")].find(
      (el) => el.querySelector(".fade")?.textContent === chat.title
    );
    if (fresh) startRename(fresh, chat);
  });

  const del = document.createElement("button");
  del.type = "button";
  del.className = "menu-item danger";
  del.innerHTML = `${ICONS.trash}<span>Delete</span>`;
  del.addEventListener("click", () => {
    closeMenus();
    deleteChat(chat.id);
  });

  chatMenu.append(rename, del);
  positionMenu(chatMenu, anchor);
}

function openUserMenu() {
  if (!userMenu.hidden) { closeMenus(); return; }
  closeMenus();
  userMenu.textContent = "";

  const setApi = document.createElement("button");
  setApi.type = "button";
  setApi.className = "menu-item";
  setApi.innerHTML = `${ICONS.globe}<span>Set backend URL</span>`;
  setApi.addEventListener("click", () => {
    closeMenus();
    const url = prompt(
      "Backend URL (where server.py is running):",
      localStorage.getItem("karimgpt_api") || "http://localhost:8000"
    );
    if (url !== null) {
      if (url.trim()) localStorage.setItem("karimgpt_api", url.trim());
      else localStorage.removeItem("karimgpt_api");
      location.reload();
    }
  });

  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "menu-item danger";
  clear.innerHTML = `${ICONS.trash}<span>Clear all chats</span>`;
  clear.addEventListener("click", () => {
    closeMenus();
    if (confirm("Delete all chats? This cannot be undone.")) {
      chats = [];
      currentId = null;
      saveChats();
      renderHistory();
      renderMessages();
    }
  });

  userMenu.append(setApi, clear);
  positionMenu(userMenu, $("#user-btn"), "above");
}

function positionMenu(menu, anchor, placement = "below") {
  menu.hidden = false;
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth;
  const mh = menu.offsetHeight;
  let left = Math.min(r.left, window.innerWidth - mw - 12);
  let top = placement === "above" ? r.top - mh - 8 : r.bottom + 6;
  if (top + mh > window.innerHeight - 8) top = window.innerHeight - mh - 8;
  if (top < 8) top = 8;
  menu.style.left = `${Math.max(8, left)}px`;
  menu.style.top = `${top}px`;
}

function closeMenus() {
  [modelMenu, chatMenu, userMenu].forEach((m) => (m.hidden = true));
  historyEl.querySelectorAll(".menu-open").forEach((el) => el.classList.remove("menu-open"));
}

document.addEventListener("click", (e) => {
  if (
    !e.target.closest(".menu") &&
    !e.target.closest("#model-btn") &&
    !e.target.closest(".more-btn") &&
    !e.target.closest("#user-btn")
  ) {
    closeMenus();
  }
});
window.addEventListener("resize", closeMenus);

/* =====================================================================
   INPUT UX
   ===================================================================== */
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 220) + "px";
}
function updateSendState() {
  sendBtn.disabled = !streaming && inputEl.value.trim().length === 0;
}

inputEl.addEventListener("input", () => { autoGrow(); updateSendState(); });
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled && !streaming) sendMessage(inputEl.value);
  }
});
composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!streaming) sendMessage(inputEl.value);
});
sendBtn.addEventListener("click", () => {
  if (streaming) stopStreaming();
  else sendMessage(inputEl.value);
});

searchInput.addEventListener("input", renderHistory);

$("#new-chat-btn").addEventListener("click", newChat);
$("#topbar-new-chat").addEventListener("click", newChat);
$("#collapse-btn").addEventListener("click", () => app.classList.remove("sidebar-open"));
$("#open-sidebar-btn").addEventListener("click", () => app.classList.add("sidebar-open"));
$("#scrim").addEventListener("click", () => app.classList.remove("sidebar-open"));
modelBtn.addEventListener("click", openModelMenu);
$("#user-btn").addEventListener("click", openUserMenu);

/* =====================================================================
   BACKEND DISCOVERY
   ===================================================================== */
function setConn(online) {
  connDot.classList.toggle("online", online === true);
  connDot.classList.toggle("offline", online === false);
  connDot.title = online ? "Backend connected" : "Backend unreachable — click the avatar to set the URL";
}

async function discoverModels() {
  try {
    const resp = await fetch(`${API_BASE}/api/models`, { signal: AbortSignal.timeout(6000) });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    if (Array.isArray(data.models) && data.models.length) {
      models = data.models;
      if (!models.some((m) => m.id === currentModel)) currentModel = data.default || models[0].id;
    }
    setConn(true);
  } catch {
    setConn(false);
  }
  updateModelLabel();
  renderSuggestions();
}

/* =====================================================================
   BOOT
   ===================================================================== */
sendBtn.innerHTML = ICONS.send;
if (window.innerWidth <= 768) app.classList.remove("sidebar-open");
updateModelLabel();
renderHistory();
renderMessages();
updateSendState();
discoverModels();
inputEl.focus();
