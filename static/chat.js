const chatWindow = document.getElementById('chat-window');
const form = document.getElementById('chat-form');
const messageInput = document.getElementById('message');
const conversationId = crypto.randomUUID();
const seenMessages = new Set();

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = role === 'human' ? `HUMAN CLINICIAN: ${text}` : text;
  chatWindow.appendChild(div);
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

const initialGreeting = 'Hi, I am Celine. I can help with initial clinical triage and care guidance.';
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
  appendMessage('assistant', payload.response);
  markSeen('assistant', payload.response);

  if (payload.requires_handoff) {
    const handoffMessage = `⚠️ Human handoff triggered: ${payload.handoff_reason}`;
    appendMessage('assistant', handoffMessage);
    markSeen('assistant', handoffMessage);
  }

  await refreshConversation();
});

setInterval(refreshConversation, 3000);
