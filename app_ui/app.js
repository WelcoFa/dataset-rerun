const state = {
  items: [],
  selectedItemId: null,
  status: "idle",
  activeItem: null,
  viewerUrl: "http://localhost:9090",
  grpcUrl: "rerun+http://localhost:9876/proxy",
  recordingPath: "none",
  startedAt: null,
  processRunning: false,
  lastError: null,
  logs: [],
  saveRecording: false,
  presetSearch: "",
  datasetFilter: "all",
  logSearch: "",
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDateTime(value) {
  if (!value) {
    return "not started";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function basename(path) {
  if (!path || path === "none") {
    return "none";
  }
  return String(path).split(/[\\/]/).pop() || path;
}

function toneForStatus(status) {
  if (status === "failed") {
    return "failed";
  }
  if (status === "running") {
    return "running";
  }
  if (status === "starting") {
    return "starting";
  }
  if (status === "stopped") {
    return "stopped";
  }
  return "idle";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

function getVisibleItems() {
  return state.items.filter((item) => {
    const datasetMatch = state.datasetFilter === "all" || item.dataset === state.datasetFilter;
    if (!datasetMatch) {
      return false;
    }

    const query = state.presetSearch.trim().toLowerCase();
    if (!query) {
      return true;
    }

    const haystack = [item.label, item.dataset, item.path, item.input, item.description, item.error]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function renderDatasetFilter() {
  const previous = state.datasetFilter;
  const datasets = [...new Set(state.items.map((item) => item.dataset).filter(Boolean))].sort();
  els.datasetFilter.innerHTML = '<option value="all">All datasets</option>';
  datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset;
    option.textContent = dataset;
    els.datasetFilter.appendChild(option);
  });
  els.datasetFilter.value = datasets.includes(previous) || previous === "all" ? previous : "all";
  state.datasetFilter = els.datasetFilter.value;
}

function renderItems() {
  const visibleItems = getVisibleItems();
  els.items.innerHTML = "";
  els.presetCount.textContent = `${visibleItems.length} preset${visibleItems.length === 1 ? "" : "s"}`;

  if (visibleItems.length === 0) {
    els.items.innerHTML = '<div class="empty-state">No presets match the current search or filter.</div>';
    return;
  }

  visibleItems.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `preset-card ${state.selectedItemId === item.id ? "selected" : ""} ${item.valid ? "" : "invalid"}`;
    const statePill = item.valid
      ? item.active
        ? '<span class="pill">running</span>'
        : ""
      : '<span class="pill danger">invalid</span>';
    button.innerHTML = `
      <div class="preset-header">
        <div class="preset-title-group">
          <h3>${escapeHtml(item.label)}</h3>
          <p class="preset-desc">${escapeHtml(item.description || "Ready-to-play preset")}</p>
        </div>
        <span class="pill">${escapeHtml(item.dataset || "auto")}</span>
      </div>
      <p class="preset-path">${escapeHtml(item.path)}</p>
      ${item.valid ? "" : `<p class="preset-desc danger-copy">${escapeHtml(item.error || "Config parse failed")}</p>`}
      <div class="preset-footer">
        <div class="tag-row">
          <span class="tag">${escapeHtml(item.input || "no input")}</span>
        </div>
        ${statePill}
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
  const query = state.logSearch.trim().toLowerCase();
  const visibleLogs = query
    ? state.logs.filter((line) => String(line).toLowerCase().includes(query))
    : state.logs;

  if (visibleLogs.length === 0) {
    els.logs.innerHTML = `<div class="empty-state">${query ? "No log lines match the current search." : "No logs yet."}</div>`;
    return;
  }

  els.logs.innerHTML = visibleLogs
    .map((line) => {
      const text = String(line);
      const isError = /(error|failed|traceback|exception)/i.test(text);
      const matches = query && text.toLowerCase().includes(query);
      return `<div class="log-entry ${isError ? "error" : ""} ${matches ? "match" : ""}">${escapeHtml(text)}</div>`;
    })
    .join("");
}

function renderStatus() {
  const tone = toneForStatus(state.status);
  els.statusText.textContent = state.status;
  els.statusPill.textContent = state.status;
  els.statusPill.dataset.tone = tone;
  els.activeItem.textContent = state.activeItem || "none";
  els.viewerLink.href = state.viewerUrl;
  els.grpcUrl.textContent = state.grpcUrl || "rerun+http://localhost:9876/proxy";
  els.startedAt.textContent = formatDateTime(state.startedAt);
  els.recordingName.textContent = basename(state.recordingPath);
  els.recordingPath.textContent =
    state.recordingPath && state.recordingPath !== "none"
      ? state.recordingPath
      : "No `.rrd` file attached";

  const viewerPortMatch = String(state.viewerUrl).match(/:(\d+)/);
  els.viewerPort.textContent = viewerPortMatch ? viewerPortMatch[1] : "custom";

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

  if (state.lastError) {
    els.lastError.textContent = state.lastError;
    els.lastError.classList.remove("hidden");
  } else {
    els.lastError.textContent = "";
    els.lastError.classList.add("hidden");
  }
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
  state.grpcUrl = statusPayload.grpc_url || state.grpcUrl;
  state.recordingPath = statusPayload.recording_path || "none";
  state.startedAt = statusPayload.started_at || null;
  state.processRunning = Boolean(statusPayload.process_running);
  state.lastError = statusPayload.last_error || null;
  state.logs = logsPayload.logs || [];

  if (!state.selectedItemId || !state.items.some((item) => item.id === state.selectedItemId && item.valid)) {
    const firstValid = state.items.find((item) => item.valid);
    state.selectedItemId = firstValid ? firstValid.id : null;
  }

  renderDatasetFilter();
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
  els.launchBtn.addEventListener("click", () => {
    openSelected().catch((error) => appendLog(`Launch failed: ${error.message}`));
  });
  els.refreshBtn.addEventListener("click", () => {
    refreshAll().catch((error) => appendLog(`Refresh failed: ${error.message}`));
  });
  els.stopBtn.addEventListener("click", () => {
    stopCurrent().catch((error) => appendLog(`Stop failed: ${error.message}`));
  });
  els.saveRecordingToggle.addEventListener("change", (event) => {
    state.saveRecording = event.target.checked;
    renderStatus();
    appendLog(
      state.saveRecording
        ? "Saving .rrd files is enabled for the next launch."
        : "Launching without saving .rrd files.",
    );
  });
  els.viewerLink.addEventListener("click", () => {
    appendLog(`Opening viewer at ${state.viewerUrl}`);
  });
  els.presetSearch.addEventListener("input", (event) => {
    state.presetSearch = event.target.value;
    renderItems();
  });
  els.datasetFilter.addEventListener("change", (event) => {
    state.datasetFilter = event.target.value;
    renderItems();
  });
  els.logSearch.addEventListener("input", (event) => {
    state.logSearch = event.target.value;
    renderLogs();
  });

}

function init() {
  els.statusText = $("status-text");
  els.statusPill = $("status-pill");
  els.activeItem = $("active-item");
  els.lastError = $("last-error");
  els.viewerLink = $("viewer-link");
  els.viewerPort = $("viewer-port");
  els.grpcUrl = $("grpc-url");
  els.recordingName = $("recording-name");
  els.recordingPath = $("recording-path");
  els.startedAt = $("started-at");
  els.refreshBtn = $("refresh-btn");
  els.stopBtn = $("stop-btn");
  els.launchBtn = $("launch-btn");
  els.items = $("items");
  els.logs = $("logs");
  els.viewerFrame = $("viewer-frame");
  els.saveRecordingToggle = $("save-recording-toggle");
  els.presetSearch = $("preset-search");
  els.datasetFilter = $("dataset-filter");
  els.logSearch = $("log-search");
  els.presetCount = $("preset-count");

  bindEvents();
  renderStatus();
  renderLogs();
  refreshAll().catch((error) => appendLog(`Initial load failed: ${error.message}`));
  setInterval(() => {
    refreshAll().catch(() => {});
  }, 15000);
}

init();
