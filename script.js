import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Force CPU/WASM only — never touch WebGPU
env.backends.onnx.wasm.proxy = false;
env.allowLocalModels = false;

const statusEl = document.getElementById('status');
const chatBox = document.getElementById('chat-box');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');

let generator = null;

function addMessage(text, sender) {
  const div = document.createElement('div');
  div.className = 'msg ' + (sender === 'user' ? 'user-msg' : 'ai-msg');
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function loadModel() {
  try {
    generator = await pipeline(
      'text2text-generation',
      'Xenova/LaMini-Flan-T5-77M',
      {
        quantized: true,
        progress_callback: (data) => {
          if (data.status === 'progress' && data.progress) {
            statusEl.textContent = `Loading AI... ${Math.round(data.progress)}%`;
          }
        }
      }
    );
    statusEl.textContent = 'AI Ready';
    statusEl.className = 'ready';
    input.disabled = false;
    sendBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = 'Error: AI model failed to load.';
    statusEl.className = 'error';
    console.error(err);
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || !generator) return;

  addMessage(text, 'user');
  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;
  statusEl.textContent = 'Thinking...';

  try {
    const output = await generator(text, { max_new_tokens: 100 });
    const reply = output[0].generated_text.trim() || "I'm not sure.";
    addMessage(reply, 'ai');
  } catch (err) {
    addMessage('Sorry, something went wrong generating a reply.', 'ai');
    console.error(err);
  }

  statusEl.textContent = 'AI Ready';
  statusEl.className = 'ready';
  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});
clearBtn.addEventListener('click', () => {
  chatBox.innerHTML = '';
});

loadModel();
