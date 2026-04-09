const state = {
  route: "library",
  items: [],
  selectedItemId: null,
  selectedSceneId: null,
  status: "idle",
  activeItem: null,
  activeSceneId: null,
  viewerSceneId: null,
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
};

const els = {};
const ROUTES = {
  library: {
    title: "Session Explorer",
    description: "Browse and launch recorded sessions, choose a scene, and inspect runtime state.",
  },
  viewer: {
    title: "Viewer",
    description: "Inspect the live viewer and the full console together on one page.",
  },
};

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

function formatDuration(value) {
  if (!value) {
    return "0h 00m";
  }
  const started = new Date(value);
  if (Number.isNaN(started.getTime())) {
    return "0h 00m";
  }
  const elapsedMs = Math.max(0, Date.now() - started.getTime());
  const totalMinutes = Math.floor(elapsedMs / 60000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}

function basename(path) {
  if (!path || path === "none") {
    return "none";
  }
  return String(path).split(/[\\/]/).pop() || path;
}

function titleCase(value) {
  return String(value || "")
    .replaceAll(/[-_]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function trimSceneSuffix(value) {
  return String(value || "")
    .replace(/\s*\|\s*\d+\s+scenes?$/i, "")
    .trim();
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

function getRouteFromHash() {
  const raw = window.location.hash.replace(/^#\/?/, "").trim().toLowerCase();
  return ROUTES[raw] ? raw : "library";
}

function setRoute(route, replace = false) {
  const target = ROUTES[route] ? route : "library";
  const hash = `#/${target}`;
  if (replace) {
    history.replaceState(null, "", hash);
  } else {
    window.location.hash = hash;
  }
  state.route = target;
  renderRoute();
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

function getSelectedScene(item = getSelectedItem()) {
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];
  return scenes.find((scene) => scene.id === state.selectedSceneId) || null;
}

function getActiveItem() {
  return state.items.find((item) => item.id === state.activeItem) || null;
}

function getViewerScene(item = getActiveItem()) {
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];
  return scenes.find((scene) => scene.id === state.viewerSceneId) || null;
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
  renderLibrarySummary();
}

function renderRoute() {
  const route = ROUTES[state.route] ? state.route : "library";
  els.routeTitle.textContent = ROUTES[route].title;
  els.routeDescription.textContent = ROUTES[route].description;

  Object.entries(els.pages).forEach(([pageRoute, element]) => {
    element.classList.toggle("hidden", pageRoute !== route);
  });

  els.routeLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.routeLink === route);
  });
}

function renderItems() {
  const visibleItems = getVisibleItems();
  els.items.innerHTML = "";
  els.presetCount.textContent = `${visibleItems.length} session${visibleItems.length === 1 ? "" : "s"}`;

  if (visibleItems.length === 0) {
    els.items.innerHTML = '<div class="empty-state">No presets match the current search or filter.</div>';
    return;
  }

  visibleItems.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `preset-card ${state.selectedItemId === item.id ? "selected" : ""} ${item.valid ? "" : "invalid"}`;
    const sceneCount = Array.isArray(item.scenes) ? item.scenes.length : 0;
    const sceneSummary = `${sceneCount} scene${sceneCount === 1 ? "" : "s"}`;
    const defaultSceneLabel =
      item.default_scene_id && Array.isArray(item.scenes)
        ? item.scenes.find((scene) => scene.id === item.default_scene_id)?.label || item.default_scene_id
        : sceneCount > 0
          ? item.scenes[0]?.label || item.scenes[0]?.id || "default"
          : "default";
    const statusLabel = !item.valid ? "Error" : item.active ? "Running" : "Valid";
    const statusClass = !item.valid ? "error" : item.active ? "running" : "valid";
    const datasetLabel = titleCase(item.dataset || "auto");
    const inputLabel = basename(item.input || "none");
    const configLabel = basename(item.path || "none");
    const displayDescription = trimSceneSuffix(item.description || "Ready-to-play preset");
    const statePill = item.valid
      ? item.active
        ? '<span class="pill">running</span>'
        : ""
      : '<span class="pill danger">invalid</span>';
    button.innerHTML = `
      <div class="session-card-main">
        <div class="session-card-copy">
          <div class="preset-header">
            <div class="preset-title-group">
              <h3>${escapeHtml(item.label)}</h3>
              <p class="preset-desc">${escapeHtml(displayDescription || sceneSummary)}</p>
            </div>
            <div class="session-status session-status-${statusClass}">
              <span class="session-status-dot"></span>
              <span>${escapeHtml(statusLabel)}</span>
            </div>
          </div>
          <div class="session-meta-row">
            <span class="session-meta-item">${escapeHtml(sceneSummary)}</span>
          </div>
        </div>
        <div class="session-card-side">
          <div class="session-chip-row">
            <span class="session-chip">${escapeHtml(datasetLabel)}</span>
            <span class="session-chip">${escapeHtml(sceneSummary)}</span>
            ${statePill}
          </div>
          <div class="session-detail-grid">
            <div class="session-detail-box">
              <span>Default Scene</span>
              <strong>${escapeHtml(defaultSceneLabel || "default")}</strong>
            </div>
            <div class="session-detail-box">
              <span>Config</span>
              <strong>${escapeHtml(configLabel)}</strong>
            </div>
            <div class="session-detail-box">
              <span>Input</span>
              <strong>${escapeHtml(inputLabel)}</strong>
            </div>
          </div>
        </div>
      </div>
      ${item.valid ? "" : `<p class="preset-desc danger-copy">${escapeHtml(item.error || "Config parse failed")}</p>`}
    `;
    button.addEventListener("click", () => {
      if (!item.valid) {
        appendLog(`Cannot select invalid preset ${item.label}: ${item.error || "unknown error"}`);
        return;
      }
      setSelectedItem(item.id);
      renderItems();
      renderSceneSelector();
      renderLibrarySummary();
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
  const activeScene = getSelectedScene(item) || scenes[0];
  els.sceneDescription.textContent = activeScene.description || "Choose which scene to launch for this dataset preset.";
  els.sceneCount.textContent = `${scenes.length} scenes`;
}

function renderLibrarySummary() {
  const item = getSelectedItem();
  const scene = getSelectedScene(item);
  els.selectedItemName.textContent = item?.label || "none";
  els.selectedItemDataset.textContent = item?.dataset || "none";
  els.selectedItemInput.textContent = item?.input || "none";
  els.selectedSceneName.textContent = scene?.label || item?.default_scene_id || "default";
  els.selectedSceneSummary.textContent = scene?.description || item?.description || "Pick a preset to see what will be launched.";
}

function renderViewerSceneSelector() {
  const item = getActiveItem();
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];

  if (!item || scenes.length <= 1 || !["starting", "running"].includes(state.status)) {
    els.viewerScenePanel.classList.add("hidden");
    els.viewerSceneSelect.innerHTML = "";
    els.viewerSceneDescription.textContent = "Choose a scene from the active dataset.";
    els.viewerSceneCount.textContent = "";
    els.viewerSceneApplyBtn.disabled = true;
    return;
  }

  state.viewerSceneId = chooseSceneId(item, state.viewerSceneId || state.activeSceneId);
  const activeViewerScene = getViewerScene(item) || scenes[0];

  els.viewerScenePanel.classList.remove("hidden");
  els.viewerSceneSelect.innerHTML = "";
  scenes.forEach((scene) => {
    const option = document.createElement("option");
    option.value = scene.id;
    option.textContent = scene.label || scene.id;
    els.viewerSceneSelect.appendChild(option);
  });
  els.viewerSceneSelect.value = state.viewerSceneId || scenes[0].id;
  els.viewerSceneDescription.textContent =
    activeViewerScene.description || "Choose a different scene and relaunch the active dataset into it.";
  els.viewerSceneCount.textContent = `${scenes.length} scenes`;
  els.viewerSceneApplyBtn.disabled = state.launchPending;
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
  const totalSessions = state.items.length;
  const totalScenes = state.items.reduce((count, item) => count + (Array.isArray(item.scenes) ? item.scenes.length : 0), 0);
  const validSessions = state.items.filter((item) => item.valid).length;
  const viewerIsReady = Boolean(state.activeItem) && state.status === "running";
  const viewerIsStarting = Boolean(state.activeItem) && state.status === "starting";

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
  els.viewerEmptyState.classList.toggle("hidden", viewerIsReady);
  if (viewerIsStarting) {
    els.viewerEmptyTitle.textContent = "Starting Viewer";
    els.viewerEmptyCopy.textContent = "The dataset is launching now. The embedded viewer will appear here as soon as the session is ready.";
  } else {
    els.viewerEmptyTitle.textContent = "No Active Session";
    els.viewerEmptyCopy.textContent = "Launch a dataset from the Library page to open a live Rerun viewer here.";
  }

  els.viewerSessionItem.textContent = state.activeItem || "none";
  els.viewerSessionScene.textContent = state.activeSceneId || "default";
  els.viewerSessionStatus.textContent = state.status;
  els.viewerSessionUrl.textContent = state.viewerUrl;

  els.totalSessions.textContent = String(totalSessions);
  els.recordingDuration.textContent = formatDuration(state.startedAt);
  els.totalScenes.textContent = String(totalScenes);
  els.validatedCount.textContent = `${validSessions}/${totalSessions}`;

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

function renderAll() {
  renderItems();
  renderSceneSelector();
  renderLibrarySummary();
  renderStatus();
  renderViewerSceneSelector();
  renderLogs();
  renderRoute();
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
  state.viewerSceneId = state.activeSceneId || null;
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

  renderAll();
}

async function openSelected() {
  if (state.launchPending) {
    return;
  }
  const selected = getSelectedItem();
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
    setRoute("viewer");
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
    setRoute("library");
  } catch (error) {
    appendLog(`Stop failed: ${error.message}`);
  } finally {
    state.launchPending = false;
    renderStatus();
  }
}

async function switchViewerScene() {
  if (state.launchPending) {
    return;
  }

  const activeItem = getActiveItem();
  if (!activeItem) {
    appendLog("No active dataset is available to switch scenes.");
    return;
  }
  if (!state.viewerSceneId) {
    appendLog("Choose a scene before switching.");
    return;
  }

  const nextScene = getViewerScene(activeItem);
  if (state.viewerSceneId === state.activeSceneId) {
    appendLog(`Scene ${nextScene?.label || state.viewerSceneId} is already active.`);
    return;
  }

  try {
    state.launchPending = true;
    renderStatus();
    renderViewerSceneSelector();
    await fetchJson("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_id: activeItem.id,
        scene_id: state.viewerSceneId,
        save_recording: state.saveRecording,
      }),
    });
    appendLog(`Switched ${activeItem.label} to scene ${nextScene?.label || state.viewerSceneId}.`);
    await refreshAll();
  } catch (error) {
    appendLog(`Scene switch failed: ${error.message}`);
  } finally {
    state.launchPending = false;
    renderStatus();
    renderViewerSceneSelector();
  }
}

function bindEvents() {
  window.addEventListener("hashchange", () => {
    state.route = getRouteFromHash();
    renderRoute();
  });

  els.routeLinks.forEach((link) => {
    link.addEventListener("click", () => {
      state.route = link.dataset.routeLink || "library";
      renderRoute();
    });
  });

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
  els.sceneSelect.addEventListener("change", (event) => {
    state.selectedSceneId = event.target.value || null;
    renderSceneSelector();
    renderLibrarySummary();
  });
  els.viewerSceneSelect.addEventListener("change", (event) => {
    state.viewerSceneId = event.target.value || null;
    renderViewerSceneSelector();
  });
  els.viewerSceneApplyBtn.addEventListener("click", () => {
    switchViewerScene().catch((error) => appendLog(`Scene switch failed: ${error.message}`));
  });
  els.logSearch.addEventListener("input", (event) => {
    state.logSearch = event.target.value;
    renderLogs();
  });
  els.viewerFullscreenBtn.addEventListener("click", async () => {
    const viewerContainer = els.viewerFrame.closest(".panel");
    if (!viewerContainer) {
      return;
    }

    try {
      if (document.fullscreenElement === viewerContainer) {
        await document.exitFullscreen();
        return;
      }
      await viewerContainer.requestFullscreen();
    } catch (error) {
      appendLog(`Fullscreen failed: ${error.message}`);
    }
  });
}

function init() {
  els.routeTitle = $("route-title");
  els.routeDescription = $("route-description");
  els.routeLinks = [...document.querySelectorAll("[data-route-link]")];
  els.pages = {
    library: $("page-library"),
    viewer: $("page-viewer"),
  };

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
  els.totalSessions = $("total-sessions");
  els.recordingDuration = $("recording-duration");
  els.totalScenes = $("total-scenes");
  els.validatedCount = $("validated-count");
  els.refreshBtn = $("refresh-btn");
  els.stopBtn = $("stop-btn");
  els.launchBtn = $("launch-btn");
  els.items = $("items");
  els.logs = $("logs");
  els.viewerFrame = $("viewer-frame");
  els.viewerEmptyState = $("viewer-empty-state");
  els.viewerEmptyTitle = $("viewer-empty-title");
  els.viewerEmptyCopy = $("viewer-empty-copy");
  els.viewerFullscreenBtn = $("viewer-fullscreen-btn");
  els.viewerScenePanel = $("viewer-scene-panel");
  els.viewerSceneSelect = $("viewer-scene-select");
  els.viewerSceneDescription = $("viewer-scene-description");
  els.viewerSceneCount = $("viewer-scene-count");
  els.viewerSceneApplyBtn = $("viewer-scene-apply-btn");
  els.saveRecordingToggle = $("save-recording-toggle");
  els.presetSearch = $("preset-search");
  els.logSearch = $("log-search");
  els.presetCount = $("preset-count");
  els.sceneSelector = $("scene-selector");
  els.sceneSelect = $("scene-select");
  els.sceneDescription = $("scene-description");
  els.sceneCount = $("scene-count");

  els.selectedItemName = $("selected-item-name");
  els.selectedItemDataset = $("selected-item-dataset");
  els.selectedItemInput = $("selected-item-input");
  els.selectedSceneName = $("selected-scene-name");
  els.selectedSceneSummary = $("selected-scene-summary");

  els.viewerSessionItem = $("viewer-session-item");
  els.viewerSessionScene = $("viewer-session-scene");
  els.viewerSessionStatus = $("viewer-session-status");
  els.viewerSessionUrl = $("viewer-session-url");

  state.route = getRouteFromHash();
  if (!window.location.hash) {
    setRoute("library", true);
  }

  bindEvents();
  renderAll();
  refreshAll().catch((error) => appendLog(`Initial load failed: ${error.message}`));
  setInterval(() => {
    refreshAll().catch(() => {});
  }, 15000);
}

init();
