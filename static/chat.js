const chatWindow = document.getElementById('chat-window');
const form = document.getElementById('chat-form');
const messageInput = document.getElementById('message');
const conversationId = crypto.randomUUID();

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

appendMessage('assistant', 'Hi, I am Celine. I can help with initial clinical triage and care guidance.');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  appendMessage('user', message);
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

  if (payload.requires_handoff) {
    appendMessage('assistant', `⚠️ Human handoff triggered: ${payload.handoff_reason}`);
  }
});
