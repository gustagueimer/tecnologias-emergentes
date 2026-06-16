async function refreshEvents() {
  const tableBody = document.getElementById("events-body");
  if (!tableBody) {
    return;
  }

  try {
    const response = await fetch("/events");
    const events = await response.json();
    tableBody.innerHTML = "";

    if (!events.length) {
      tableBody.innerHTML = '<tr><td colspan="3">Nenhum evento registrado.</td></tr>';
      return;
    }

    for (const event of events.slice(0, 20)) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${event.event_time}</td>
        <td>${event.label}</td>
        <td>${Number(event.confidence).toFixed(2)}</td>
      `;
      tableBody.appendChild(row);
    }
  } catch (error) {
    tableBody.innerHTML = '<tr><td colspan="3">Falha ao carregar eventos.</td></tr>';
  }
}

async function sendChat() {
  const messageInput = document.getElementById("message");
  const analysis = document.getElementById("analysis");
  const message = messageInput.value.trim();

  if (!message) {
    analysis.textContent = "Digite uma pergunta operacional para o agente.";
    return;
  }

  analysis.textContent = "Gerando analise...";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, limit: 10 }),
    });
    const data = await response.json();
    analysis.textContent = data.summary || "Sem resposta do agente.";
  } catch (error) {
    analysis.textContent = "Nao foi possivel consultar o agente.";
  }
}

async function sendChatStream() {
  const messageInput = document.getElementById("message");
  const analysis = document.getElementById("analysis");
  const message = messageInput.value.trim();

  if (!message) {
    analysis.textContent = "Digite uma pergunta operacional para o agente.";
    return;
  }

  analysis.textContent = "";

  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, limit: 10 }),
  });

  if (!response.body) {
    analysis.textContent = "Streaming indisponivel.";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split("\n\n")) {
      if (line.startsWith("data: ")) {
        analysis.textContent += line.replace("data: ", "");
      }
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("send-chat").addEventListener("click", sendChat);
  document.getElementById("send-chat-stream").addEventListener("click", () => {
    sendChatStream().catch(() => {
      document.getElementById("analysis").textContent = "Falha no streaming da resposta.";
    });
  });

  refreshEvents();
  setInterval(refreshEvents, 8000);
});