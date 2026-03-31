const itemsEl = document.getElementById("items");
const logsEl = document.getElementById("logs");
const statusTextEl = document.getElementById("status-text");
const activeItemEl = document.getElementById("active-item");
const recordingPathEl = document.getElementById("recording-path");
const viewerLinkEl = document.getElementById("viewer-link");
const viewerFrameEl = document.getElementById("viewer-frame");

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderItems(items) {
  itemsEl.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = `item-card${item.valid ? "" : " invalid"}`;
    card.innerHTML = `
      <header>
        <div>
          <strong>${item.label}</strong>
          <div class="pill">${item.dataset}</div>
        </div>
        <button type="button" ${item.valid ? "" : "disabled"}>${item.active ? "Running" : "Open"}</button>
      </header>
      <div>${item.description}</div>
      <code>${item.path}</code>
      ${item.error ? `<div>${item.error}</div>` : ""}
    `;
    const button = card.querySelector("button");
    if (item.valid) {
      button.addEventListener("click", () => openItem(item.id));
    }
    itemsEl.appendChild(card);
  }
}

function renderStatus(status) {
  statusTextEl.textContent = status.status;
  activeItemEl.textContent = status.current_item_id || "none";
  recordingPathEl.textContent = status.recording_path || "none";
  if (status.viewer_url) {
    viewerLinkEl.href = status.viewer_url;
    viewerFrameEl.src = status.viewer_url;
  }
}

function renderLogs(logs) {
  logsEl.textContent = logs.length ? logs.join("\n") : "No logs yet.";
  logsEl.scrollTop = logsEl.scrollHeight;
}

async function refresh() {
  const [itemsPayload, statusPayload, logsPayload] = await Promise.all([
    fetchJson("/api/items"),
    fetchJson("/api/status"),
    fetchJson("/api/logs"),
  ]);
  renderItems(itemsPayload.items);
  renderStatus(statusPayload);
  renderLogs(logsPayload.logs);
}

async function openItem(itemId) {
  try {
    await fetchJson("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: itemId }),
    });
    await refresh();
  } catch (error) {
    alert(error.message);
  }
}

async function stopCurrent() {
  try {
    await fetchJson("/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    await refresh();
  } catch (error) {
    alert(error.message);
  }
}

document.getElementById("refresh-btn").addEventListener("click", refresh);
document.getElementById("stop-btn").addEventListener("click", stopCurrent);

refresh();
setInterval(refresh, 2000);
