const chatWindow = document.getElementById('chat-window');
const form = document.getElementById('chat-form');
const messageInput = document.getElementById('message');

const STORAGE_KEY = 'celine.conversation_id';
const conversationId = localStorage.getItem(STORAGE_KEY) || crypto.randomUUID();
localStorage.setItem(STORAGE_KEY, conversationId);
const seenMessages = new Set();

function appendMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message-row ${role}`;

  const bubble = document.createElement('div');
  bubble.className = `message ${role}`;

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'assistant' ? 'CELINE AI' : role === 'human' ? 'HUMAN CLINICIAN' : 'YOU';

  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = text;

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  bubble.appendChild(label);
  bubble.appendChild(content);
  bubble.appendChild(meta);
  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function markSeen(role, text) {
  const marker = `${role}|${text}`;
  seenMessages.add(marker);
}

function rememberMessage(message) {
  const marker = `${message.role}|${message.content}`;
  if (seenMessages.has(marker)) {
    return false;
  }
  seenMessages.add(marker);
  return true;
}

const initialGreeting = 'Hi, I am Celine. This is a triage support tool and not a medical diagnosis.';
appendMessage('assistant', initialGreeting);
markSeen('assistant', initialGreeting);

async function refreshConversation() {
  const response = await fetch(`/chat/history/${conversationId}`);
  if (!response.ok) {
    return;
  }

  const payload = await response.json();
  payload.messages.forEach((message) => {
    if (rememberMessage(message)) {
      appendMessage(message.role, message.content);
    }
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  appendMessage('user', message);
  markSeen('user', message);
  messageInput.value = '';

  const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });

  if (!response.ok) {
    appendMessage('assistant', 'Sorry, something went wrong.');
    return;
  }

  const payload = await response.json();
  if (payload.response) {
    appendMessage('assistant', payload.response);
    markSeen('assistant', payload.response);
  }

  if (payload.requires_handoff) {
    const handoffMessage = `⚠️ Human handoff triggered: ${payload.handoff_reason}`;
    appendMessage('assistant', handoffMessage);
    markSeen('assistant', handoffMessage);
  }

  await refreshConversation();
});

refreshConversation();
setInterval(refreshConversation, 3000);
