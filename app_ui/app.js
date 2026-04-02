const state = {
  items: [],
  selectedItemId: null,
  selectedSceneId: null,
  status: "idle",
  activeItem: null,
  activeSceneId: null,
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
  launchPending: false,
  layout: {
    slots: {
      library: "slot-a",
      viewer: "slot-b",
      logs: "slot-c",
    },
    leftWidth: 420,
    topHeight: 620,
  },
  dragPanelId: null,
};

const els = {};
const LAYOUT_STORAGE_KEY = "rerun-dashboard-layout-v1";

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

function isDesktopLayout() {
  return window.matchMedia("(min-width: 1181px)").matches;
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

function getSelectedItem() {
  return state.items.find((item) => item.id === state.selectedItemId) || null;
}

function chooseSceneId(item, preferredSceneId = null) {
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];
  if (scenes.length === 0) {
    return null;
  }
  if (preferredSceneId && scenes.some((scene) => scene.id === preferredSceneId)) {
    return preferredSceneId;
  }
  if (item.active_scene_id && scenes.some((scene) => scene.id === item.active_scene_id)) {
    return item.active_scene_id;
  }
  if (item.default_scene_id && scenes.some((scene) => scene.id === item.default_scene_id)) {
    return item.default_scene_id;
  }
  return scenes[0].id;
}

function setSelectedItem(itemId, preferredSceneId = null) {
  state.selectedItemId = itemId;
  const item = getSelectedItem();
  state.selectedSceneId = item ? chooseSceneId(item, preferredSceneId) : null;
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
      setSelectedItem(item.id);
      renderItems();
      renderSceneSelector();
    });
    els.items.appendChild(button);
  });
}

function renderSceneSelector() {
  const item = getSelectedItem();
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];
  if (!item || scenes.length <= 1) {
    els.sceneSelector.classList.add("hidden");
    els.sceneSelect.innerHTML = "";
    els.sceneDescription.textContent = "";
    els.sceneCount.textContent = "";
    return;
  }

  state.selectedSceneId = chooseSceneId(item, state.selectedSceneId);
  els.sceneSelector.classList.remove("hidden");
  els.sceneSelect.innerHTML = "";
  scenes.forEach((scene) => {
    const option = document.createElement("option");
    option.value = scene.id;
    option.textContent = scene.label || scene.id;
    els.sceneSelect.appendChild(option);
  });
  els.sceneSelect.value = state.selectedSceneId || scenes[0].id;
  const activeScene = scenes.find((scene) => scene.id === els.sceneSelect.value) || scenes[0];
  els.sceneDescription.textContent = activeScene.description || "Choose which scene to launch for this dataset preset.";
  els.sceneCount.textContent = `${scenes.length} scenes`;
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
  els.activeScene.textContent = state.activeSceneId || "default";
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

  const controlsDisabled = state.launchPending || state.status === "starting";
  els.launchBtn.disabled = controlsDisabled;
  els.stopBtn.disabled = state.launchPending;
  els.refreshBtn.disabled = state.launchPending;

  const viewerSessionKey = [
    state.viewerUrl,
    state.activeItem || "none",
    state.activeSceneId || "default",
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function saveLayout() {
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(state.layout));
  } catch (_) {}
}

function loadLayout() {
  try {
    const raw = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (parsed && parsed.slots && parsed.leftWidth && parsed.topHeight) {
      state.layout = {
        slots: {
          library: parsed.slots.library || "slot-a",
          viewer: parsed.slots.viewer || "slot-b",
          logs: parsed.slots.logs || "slot-c",
        },
        leftWidth: Number(parsed.leftWidth) || 420,
        topHeight: Number(parsed.topHeight) || 620,
      };
    }
  } catch (_) {}
}

function applyLayout() {
  if (!isDesktopLayout()) {
    els.contentGrid.dataset.layoutReady = "false";
    els.contentGrid.style.gridTemplateColumns = "";
    els.contentGrid.style.gridTemplateRows = "";
    els.columnResizer.style.left = "";
    els.rowResizer.style.left = "";
    els.rowResizer.style.right = "";
    els.rowResizer.style.top = "";
    return;
  }

  const gridWidth = els.contentGrid.clientWidth;
  const gridHeight = Math.max(els.contentGrid.clientHeight, 720);
  const gap = 18;
  const leftWidth = clamp(state.layout.leftWidth, 320, Math.max(360, gridWidth - 360));
  const topHeight = clamp(state.layout.topHeight, 260, Math.max(320, gridHeight - 220));
  state.layout.leftWidth = leftWidth;
  state.layout.topHeight = topHeight;

  els.contentGrid.dataset.layoutReady = "true";
  els.contentGrid.style.gridTemplateColumns = `${leftWidth}px minmax(0, 1fr)`;
  els.contentGrid.style.gridTemplateRows = `${topHeight}px minmax(220px, 1fr)`;

  Object.entries(state.layout.slots).forEach(([panelId, slot]) => {
    const panel = document.querySelector(`[data-panel-id="${panelId}"]`);
    if (panel) {
      panel.style.setProperty("--panel-slot", slot);
    }
  });

  els.columnResizer.style.left = `${leftWidth + gap / 2}px`;
  els.rowResizer.style.left = `${leftWidth + gap}px`;
  els.rowResizer.style.right = "0";
  els.rowResizer.style.top = `${topHeight + gap / 2}px`;
}

function swapPanelSlots(sourcePanelId, targetPanelId) {
  const nextSlots = { ...state.layout.slots };
  const sourceSlot = nextSlots[sourcePanelId];
  const targetSlot = nextSlots[targetPanelId];
  if (!sourceSlot || !targetSlot || sourceSlot === targetSlot) {
    return;
  }
  nextSlots[sourcePanelId] = targetSlot;
  nextSlots[targetPanelId] = sourceSlot;
  state.layout.slots = nextSlots;
  saveLayout();
  applyLayout();
}

function bindLayoutInteractions() {
  const panels = [...document.querySelectorAll(".dashboard-panel")];
  panels.forEach((panel) => {
    const handle = panel.querySelector(".panel-drag-handle");
    const panelId = panel.dataset.panelId;
    if (!handle || !panelId) {
      return;
    }

    handle.addEventListener("dragstart", (event) => {
      if (!isDesktopLayout()) {
        event.preventDefault();
        return;
      }
      state.dragPanelId = panelId;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", panelId);
    });

    handle.addEventListener("dragend", () => {
      state.dragPanelId = null;
      panels.forEach((item) => item.classList.remove("drop-target"));
    });

    panel.addEventListener("dragover", (event) => {
      if (!isDesktopLayout() || !state.dragPanelId || state.dragPanelId === panelId) {
        return;
      }
      event.preventDefault();
      panel.classList.add("drop-target");
    });

    panel.addEventListener("dragleave", () => {
      panel.classList.remove("drop-target");
    });

    panel.addEventListener("drop", (event) => {
      if (!isDesktopLayout()) {
        return;
      }
      event.preventDefault();
      panel.classList.remove("drop-target");
      const sourcePanelId = state.dragPanelId || event.dataTransfer.getData("text/plain");
      if (!sourcePanelId || sourcePanelId === panelId) {
        return;
      }
      swapPanelSlots(sourcePanelId, panelId);
    });
  });

  const startColumnResize = (event) => {
    if (!isDesktopLayout()) {
      return;
    }
    event.preventDefault();
    els.columnResizer.classList.add("active");
    const onMove = (moveEvent) => {
      const bounds = els.contentGrid.getBoundingClientRect();
      state.layout.leftWidth = clamp(moveEvent.clientX - bounds.left, 320, bounds.width - 360);
      applyLayout();
    };
    const onUp = () => {
      els.columnResizer.classList.remove("active");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      saveLayout();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const startRowResize = (event) => {
    if (!isDesktopLayout()) {
      return;
    }
    event.preventDefault();
    els.rowResizer.classList.add("active");
    const onMove = (moveEvent) => {
      const bounds = els.contentGrid.getBoundingClientRect();
      state.layout.topHeight = clamp(moveEvent.clientY - bounds.top, 260, bounds.height - 220);
      applyLayout();
    };
    const onUp = () => {
      els.rowResizer.classList.remove("active");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      saveLayout();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  els.columnResizer.addEventListener("pointerdown", startColumnResize);
  els.rowResizer.addEventListener("pointerdown", startRowResize);
  window.addEventListener("resize", () => applyLayout());
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
  state.activeSceneId = statusPayload.current_scene_id || null;
  state.viewerUrl = statusPayload.viewer_url || state.viewerUrl;
  state.grpcUrl = statusPayload.grpc_url || state.grpcUrl;
  state.recordingPath = statusPayload.recording_path || "none";
  state.startedAt = statusPayload.started_at || null;
  state.processRunning = Boolean(statusPayload.process_running);
  state.lastError = statusPayload.last_error || null;
  state.logs = logsPayload.logs || [];

  if (!state.selectedItemId || !state.items.some((item) => item.id === state.selectedItemId && item.valid)) {
    const firstValid = state.items.find((item) => item.valid);
    setSelectedItem(firstValid ? firstValid.id : null);
  } else {
    const selected = getSelectedItem();
    state.selectedSceneId = selected ? chooseSceneId(selected, state.selectedSceneId) : null;
  }

  renderDatasetFilter();
  renderItems();
  renderSceneSelector();
  renderStatus();
  renderLogs();
}

async function openSelected() {
  if (state.launchPending) {
    return;
  }
  const selected = state.items.find((item) => item.id === state.selectedItemId);
  if (!selected) {
    appendLog("No preset selected.");
    return;
  }

  try {
    state.launchPending = true;
    renderStatus();
    await fetchJson("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_id: selected.id,
        scene_id: state.selectedSceneId,
        save_recording: state.saveRecording,
      }),
    });
    const selectedScene =
      Array.isArray(selected.scenes) && state.selectedSceneId
        ? selected.scenes.find((scene) => scene.id === state.selectedSceneId)
        : null;
    appendLog(
      state.saveRecording
        ? `Started preset ${selected.label}${selectedScene ? ` (${selectedScene.label})` : ""} with .rrd recording enabled`
        : `Started preset ${selected.label}${selectedScene ? ` (${selectedScene.label})` : ""} in live viewer mode`,
    );
    await refreshAll();
  } catch (error) {
    appendLog(`Start failed: ${error.message}`);
  } finally {
    state.launchPending = false;
    renderStatus();
  }
}

async function stopCurrent() {
  if (state.launchPending) {
    return;
  }
  try {
    state.launchPending = true;
    renderStatus();
    await fetchJson("/api/stop", { method: "POST" });
    appendLog("Stopped current session.");
    await refreshAll();
  } catch (error) {
    appendLog(`Stop failed: ${error.message}`);
  } finally {
    state.launchPending = false;
    renderStatus();
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
  els.sceneSelect.addEventListener("change", (event) => {
    state.selectedSceneId = event.target.value || null;
    renderSceneSelector();
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
  els.activeScene = $("active-scene");
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
  els.sceneSelector = $("scene-selector");
  els.sceneSelect = $("scene-select");
  els.sceneDescription = $("scene-description");
  els.sceneCount = $("scene-count");
  els.contentGrid = $("content-grid");
  els.columnResizer = $("column-resizer");
  els.rowResizer = $("row-resizer");

  loadLayout();
  bindEvents();
  bindLayoutInteractions();
  applyLayout();
  renderSceneSelector();
  renderStatus();
  renderLogs();
  refreshAll().catch((error) => appendLog(`Initial load failed: ${error.message}`));
  setInterval(() => {
    refreshAll().catch(() => {});
  }, 15000);
}

init();
