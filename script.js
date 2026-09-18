// SimpleAI — 100% client-side AI chat.
// Uses WebLLM (https://github.com/mlc-ai/web-llm) loaded from the jsdelivr CDN
// as an ES module. WebLLM runs a real language model inside the browser tab
// using WebGPU, and does not talk to any server after the model file itself
// has been downloaded once (and cached by the browser).
//
// Model: Llama-3.2-1B-Instruct-q4f16_1-MLC (~880 MB download, 4-bit quantized).
// This is one of the smallest instruction-tuned chat models WebLLM ships
// prebuilt support for. Smaller "0.5B" class models exist but are
// noticeably worse at following instructions, so 1B is the practical floor
// for a chatbot that should give coherent answers.

import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/+esm";

const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

const statusBar = document.getElementById("status-bar");
const chatWindow = document.getElementById("chat-window");
const loadingIndicator = document.getElementById("loading-indicator");
const loadingText = document.getElementById("loading-text");
const inputForm = document.getElementById("input-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

let engine = null;
let modelReady = false;
let isGenerating = false;

// Chat history sent to the model each turn, OpenAI-style message objects.
let history = [
  {
    role: "system",
    content: "You are SimpleAI, a helpful, concise assistant running entirely inside the user's web browser.",
  },
];

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = "message " + role;
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function setStatus(text, kind) {
  statusBar.textContent = text;
  statusBar.classList.remove("ready", "error");
  if (kind) statusBar.classList.add(kind);
}

function setInputEnabled(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  userInput.placeholder = enabled ? "Type a message..." : "Waiting for AI model to load...";
}

async function initModel() {
  // WebGPU is required for WebLLM to run a model of this size at usable
  // speed. Plain WebAssembly-only fallback for a full instruction-tuned
  // LLM is not practical in-browser (it would be extremely slow, often
  // multiple seconds per word, and can stall mobile browsers). So if
  // WebGPU isn't available, we show a clear error instead of pretending
  // it works.
  if (!("gpu" in navigator)) {
    setStatus(
      "Your browser does not support WebGPU, so the AI model cannot run here. Try a recent version of Chrome or Edge on desktop, or Chrome on Android.",
      "error"
    );
    addMessage(
      "system",
      "WebGPU is not available in this browser. SimpleAI needs WebGPU to run the AI model locally. Please try desktop Chrome/Edge (or Chrome on a modern Android phone) and reload this page."
    );
    return;
  }

  try {
    const initProgressCallback = (report) => {
      loadingText.textContent = report.text || "Loading AI model...";
      setStatus(report.text || "Loading AI model...");
    };

    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback,
    });

    modelReady = true;
    loadingIndicator.classList.add("hidden");
    setStatus("AI is ready.", "ready");
    addMessage("system", "AI is ready. Ask it anything.");
    setInputEnabled(true);
    userInput.focus();
  } catch (err) {
    console.error(err);
    loadingIndicator.classList.add("hidden");
    setStatus("The AI model failed to load. See the message below.", "error");
    addMessage(
      "system",
      "Error: the AI model could not be loaded (" +
        (err && err.message ? err.message : "unknown error") +
        "). This can happen if your device is low on memory/VRAM, your connection dropped mid-download, or your browser blocked the download. Try reloading the page."
    );
  }
}

async function sendMessage(text) {
  if (!modelReady || isGenerating) return;

  addMessage("user", text);
  history.push({ role: "user", content: text });

  const aiEl = addMessage("ai", "…");
  isGenerating = true;
  setInputEnabled(false);

  try {
    const stream = await engine.chat.completions.create({
      messages: history,
      stream: true,
      temperature: 0.7,
    });

    let fullReply = "";
    aiEl.textContent = "";

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";
      fullReply += delta;
      aiEl.textContent = fullReply;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    if (!fullReply.trim()) {
      fullReply = "(no response generated)";
      aiEl.textContent = fullReply;
    }

    history.push({ role: "assistant", content: fullReply });
  } catch (err) {
    console.error(err);
    aiEl.textContent = "Error generating a response: " + (err && err.message ? err.message : "unknown error");
  } finally {
    isGenerating = false;
    setInputEnabled(true);
    userInput.focus();
  }
}

inputForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  userInput.value = "";
  sendMessage(text);
});

clearBtn.addEventListener("click", () => {
  chatWindow.innerHTML = "";
  history = [history[0]]; // keep the system prompt only
  addMessage("system", modelReady ? "Chat cleared. AI is ready." : "Chat cleared.");
});

setInputEnabled(false);
initModel();
