const state = {
  items: [],
  selectedItemId: null,
  status: "idle",
  activeItem: null,
  viewerUrl: "http://localhost:9090",
  recordingPath: "none",
  logs: [],
  saveRecording: false,
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

function renderItems() {
  els.items.innerHTML = "";
  state.items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `card ${state.selectedItemId === item.id ? "selected" : ""}`;
    button.innerHTML = `
      <div class="card-top">
        <div>
          <h3>${item.label}</h3>
          <p>${item.description}</p>
        </div>
        <span class="meta">${item.dataset}</span>
      </div>
      <p class="meta">${item.path}</p>
      ${item.valid ? "" : `<p class="meta error-text">${item.error || "invalid config"}</p>`}
      <div class="tag-row">
        <span class="tag">${item.input || "no input"}</span>
      </div>
    `;
    button.addEventListener("click", () => {
      if (!item.valid) {
        appendLog(`Cannot select invalid preset ${item.label}: ${item.error || "unknown error"}`);
        return;
      }
      state.selectedItemId = item.id;
      renderItems();
    });
    els.items.appendChild(button);
  });
}

function renderLogs() {
  els.logs.textContent = state.logs.length > 0 ? state.logs.join("\n") : "No logs yet.";
}

function renderStatus() {
  els.statusText.textContent = state.status;
  els.activeItem.textContent = state.activeItem || "none";
  els.viewerLink.href = state.viewerUrl;
  const viewerSessionKey = [
    state.viewerUrl,
    state.activeItem || "none",
    state.status,
    state.recordingPath || "none",
  ].join("|");
  if (els.viewerFrame.dataset.currentSessionKey !== viewerSessionKey) {
    els.viewerFrame.src = state.viewerUrl;
    els.viewerFrame.dataset.currentSessionKey = viewerSessionKey;
  }
  els.recordingPath.textContent = state.recordingPath || "none";
}

function appendLog(message) {
  state.logs.push(message);
  if (state.logs.length > 200) {
    state.logs = state.logs.slice(-200);
  }
  renderLogs();
}

async function refreshAll() {
  const [itemsPayload, statusPayload, logsPayload] = await Promise.all([
    fetchJson("/api/items"),
    fetchJson("/api/status"),
    fetchJson("/api/logs"),
  ]);

  state.items = itemsPayload.items || [];
  state.status = statusPayload.status || "idle";
  state.activeItem = statusPayload.current_item_id;
  state.viewerUrl = statusPayload.viewer_url || state.viewerUrl;
  state.recordingPath = statusPayload.recording_path || "none";
  state.logs = logsPayload.logs || [];

  if (!state.selectedItemId) {
    const firstValid = state.items.find((item) => item.valid);
    state.selectedItemId = firstValid ? firstValid.id : null;
  }

  renderItems();
  renderStatus();
  renderLogs();
}

async function openSelected() {
  const selected = state.items.find((item) => item.id === state.selectedItemId);
  if (!selected) {
    appendLog("No preset selected.");
    return;
  }
  try {
    await fetchJson("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_id: selected.id,
        save_recording: state.saveRecording,
      }),
    });
    appendLog(
      state.saveRecording
        ? `Started preset ${selected.label} with .rrd recording enabled`
        : `Started preset ${selected.label} in live viewer mode`,
    );
    await refreshAll();
  } catch (error) {
    appendLog(`Start failed: ${error.message}`);
  }
}

async function stopCurrent() {
  try {
    await fetchJson("/api/stop", { method: "POST" });
    appendLog("Stopped current session.");
    await refreshAll();
  } catch (error) {
    appendLog(`Stop failed: ${error.message}`);
  }
}

function bindEvents() {
  els.refreshBtn.addEventListener("click", () => {
    refreshAll().catch((error) => appendLog(`Refresh failed: ${error.message}`));
  });
  els.stopBtn.addEventListener("click", () => {
    stopCurrent().catch((error) => appendLog(`Stop failed: ${error.message}`));
  });
  els.saveRecordingToggle.addEventListener("change", (event) => {
    state.saveRecording = event.target.checked;
    appendLog(
      state.saveRecording
        ? "Saving .rrd files is enabled for the next launch."
        : "Launching without saving .rrd files.",
    );
  });
  els.viewerLink.addEventListener("click", () => {
    appendLog(`Opening viewer at ${state.viewerUrl}`);
  });

  const launchButton = document.createElement("button");
  launchButton.type = "button";
  launchButton.textContent = "Launch Selected";
  launchButton.className = "primary";
  launchButton.addEventListener("click", () => {
    openSelected().catch((error) => appendLog(`Launch failed: ${error.message}`));
  });
  els.actions.prepend(launchButton);
}

function init() {
  els.statusText = $("status-text");
  els.activeItem = $("active-item");
  els.viewerLink = $("viewer-link");
  els.recordingPath = $("recording-path");
  els.refreshBtn = $("refresh-btn");
  els.stopBtn = $("stop-btn");
  els.items = $("items");
  els.logs = $("logs");
  els.viewerFrame = $("viewer-frame");
  els.actions = document.querySelector(".actions");
  els.saveRecordingToggle = $("save-recording-toggle");

  bindEvents();
  refreshAll().catch((error) => appendLog(`Initial load failed: ${error.message}`));
  setInterval(() => {
    refreshAll().catch(() => {});
  }, 15000);
}

init();
