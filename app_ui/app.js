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
  annotationStatus: "idle",
  annotationRunner: null,
  annotationStartedAt: null,
  annotationLogs: [],
  annotationData: null,
  annotationError: null,
  annotationLoading: false,
};

const els = {};
const ROUTES = {
  home: {
    title: "Home",
    description: "See what the dashboard can do before you start browsing datasets or opening the viewer.",
  },
  library: {
    title: "Session Explorer",
    description: "Browse and launch recorded sessions, choose a scene, and inspect runtime state.",
  },
  viewer: {
    title: "Viewer",
    description: "Inspect the live viewer and the full console together on one page.",
  },
  annotation: {
    title: "Annotation Lab",
    description: "Run automated annotation, evaluate the universal label library, and compare Qwen versus Gemma metrics.",
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

function renderSceneDetails(details = []) {
  const normalized = Array.isArray(details) ? details.filter((detail) => detail?.label && detail?.value) : [];
  if (!normalized.length) {
    return '<div class="empty-state">No extra scene details available.</div>';
  }
  return normalized
    .map(
      (detail) => `
        <div class="scene-detail-item">
          <span>${escapeHtml(detail.label)}</span>
          <strong>${escapeHtml(detail.value)}</strong>
        </div>
      `,
    )
    .join("");
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

function focusLibrarySelection() {
  if (state.route !== "library" || !els.items) {
    return;
  }

  const scrollTarget = els.items.closest(".session-list-panel") || els.items.closest(".panel");
  if (!scrollTarget) {
    return;
  }

  const targetTop = scrollTarget.getBoundingClientRect().top + window.scrollY - 55;
  window.scrollTo({
    top: Math.max(0, targetTop),
    behavior: "smooth",
  });
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
  els.presetCount.textContent = `${visibleItems.length} dataset${visibleItems.length === 1 ? "" : "s"}`;

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
      refreshAnnotationData()
        .catch((error) => appendLog(`Annotation refresh failed: ${error.message}`))
        .finally(() => renderAll());
      focusLibrarySelection();
    });
    els.items.appendChild(button);
  });
}

function renderSceneSelector() {
  const item = getSelectedItem();
  const scenes = Array.isArray(item?.scenes) ? item.scenes : [];
  if (!item || scenes.length === 0) {
    els.sceneSelector.classList.add("hidden");
    els.sceneSelect.innerHTML = "";
    els.sceneDescription.textContent = "";
    els.sceneCount.textContent = "";
    els.sceneDetailList.innerHTML = "";
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
  els.sceneSelect.disabled = scenes.length === 1;
  const activeScene = getSelectedScene(item) || scenes[0];
  els.sceneDescription.textContent = activeScene.description || "Choose which scene to launch for this dataset preset.";
  els.sceneCount.textContent = `${scenes.length} scene${scenes.length === 1 ? "" : "s"}`;
  els.sceneDetailList.innerHTML = renderSceneDetails(activeScene.details);
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
    els.viewerSceneDetailList.innerHTML = "";
    els.viewerSceneApplyBtn.disabled = true;
    return;
  }

  state.viewerSceneId = chooseSceneId(item, state.viewerSceneId || state.activeSceneId);
  const activeViewerScene = getViewerScene(item) || scenes[0];
  const hasSceneChange = Boolean(state.viewerSceneId) && state.viewerSceneId !== state.activeSceneId;

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
  els.viewerSceneDetailList.innerHTML = renderSceneDetails(activeViewerScene.details);
  els.viewerSceneApplyBtn.disabled = state.launchPending || !hasSceneChange;
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
  els.totalScenes.textContent = String(totalScenes);
  els.activeSessionSummary.textContent =
    state.status === "running"
      ? "Running"
      : state.status === "starting"
        ? "Starting"
        : state.status === "failed"
          ? "Failed"
          : "Idle";
  els.validatedCount.textContent = `${validSessions}/${totalSessions}`;
  els.datasetsReadyCopy.textContent =
    totalSessions === 1 ? "1 config available to launch" : `${totalSessions} configs available to launch`;
  els.scenesAvailableCopy.textContent =
    totalScenes === 1 ? "1 scene across all datasets" : `${totalScenes} scenes across all datasets`;
  if (state.status === "running" || state.status === "starting") {
    const sessionParts = [state.activeItem || "unknown preset", state.activeSceneId || "default"];
    els.activeSessionCopy.textContent =
      state.status === "starting"
        ? `${sessionParts.join(" • ")} is preparing in the viewer`
        : `${sessionParts.join(" • ")} is live in the viewer`;
  } else if (state.status === "failed") {
    els.activeSessionCopy.textContent = state.lastError || "The last launch failed. Check the logs for details.";
  } else {
    els.activeSessionCopy.textContent = "No session is running right now";
  }
  els.validatedCopy.textContent =
    validSessions === totalSessions
      ? "All configs are launchable"
      : `${totalSessions - validSessions} config${totalSessions - validSessions === 1 ? "" : "s"} need attention`;

  if (state.lastError) {
    els.lastError.textContent = state.lastError;
    els.lastError.classList.remove("hidden");
  } else {
    els.lastError.textContent = "";
    els.lastError.classList.add("hidden");
  }
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (Number.isNaN(number)) {
    return String(value);
  }
  return number.toFixed(digits);
}

function modelDisplayName(runner) {
  if (runner === "qwen") {
    return "Qwen";
  }
  if (runner === "gemma4") {
    return "Gemma 4";
  }
  return titleCase(runner || "unknown");
}

function annotationCanRunAutomated(item = getSelectedItem()) {
  return ["gigahands", "hot3d", "being-h0", "dexwild", "thermohands", "wiyh"].includes(item?.dataset);
}

function currentAnnotationContext() {
  const item = getSelectedItem();
  const scene = getSelectedScene(item);
  if (!item || !scene) {
    return null;
  }
  return { item, scene };
}

function renderMetricBars(metrics, leftLabel, rightLabel) {
  if (!metrics.length) {
    return '<div class="empty-state">Run both Qwen and Gemma to compare metrics here.</div>';
  }
  return metrics
    .map((metric) => {
      const left = Number(metric.left || 0);
      const right = Number(metric.right || 0);
      const max = Math.max(left, right, metric.max || 1);
      const leftWidth = max > 0 ? (left / max) * 100 : 0;
      const rightWidth = max > 0 ? (right / max) * 100 : 0;
      return `
        <div class="annotation-chart-card">
          <div class="annotation-chart-header">
            <strong>${escapeHtml(metric.label)}</strong>
            <span>${escapeHtml(metric.hint || "")}</span>
          </div>
          <div class="annotation-bar-row">
            <span>${escapeHtml(leftLabel)}</span>
            <div class="annotation-bar-track"><div class="annotation-bar-fill annotation-bar-fill-left" style="width:${leftWidth}%"></div></div>
            <strong>${escapeHtml(metric.format(left))}</strong>
          </div>
          <div class="annotation-bar-row">
            <span>${escapeHtml(rightLabel)}</span>
            <div class="annotation-bar-track"><div class="annotation-bar-fill annotation-bar-fill-right" style="width:${rightWidth}%"></div></div>
            <strong>${escapeHtml(metric.format(right))}</strong>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderLibraryRows(rows) {
  if (!rows.length) {
    return '<div class="empty-state">The label library has no evaluation rows yet.</div>';
  }
  return rows
    .map((row) => {
      const usage = Number(row.usage_count || 0);
      const quality = Number(row.label_quality_score || 0);
      const usageWidth = Math.min(100, usage * 12);
      const qualityWidth = Math.max(0, Math.min(100, quality * 100));
      return `
        <div class="annotation-library-row ${row.needs_review ? "needs-review" : ""}">
          <div class="annotation-library-copy">
            <strong>${escapeHtml(row.name)}</strong>
            <span>${escapeHtml(row.status || "seeded")} • raw variants ${escapeHtml(String(row.unique_raw_variant_count || 0))}</span>
          </div>
          <div class="annotation-library-metrics">
            <div class="annotation-library-meter">
              <span>Usage</span>
              <div class="annotation-bar-track"><div class="annotation-bar-fill annotation-bar-fill-left" style="width:${usageWidth}%"></div></div>
              <strong>${escapeHtml(String(usage))}</strong>
            </div>
            <div class="annotation-library-meter">
              <span>Quality</span>
              <div class="annotation-bar-track"><div class="annotation-bar-fill annotation-bar-fill-right" style="width:${qualityWidth}%"></div></div>
              <strong>${escapeHtml(formatNumber(row.label_quality_score, 2))}</strong>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderAnnotationPage() {
  const context = currentAnnotationContext();
  const item = context?.item || null;
  const scene = context?.scene || null;
  const data = state.annotationData;
  const status = state.annotationStatus || "idle";
  const canRunAutomated = annotationCanRunAutomated(item);
  const canInspect = Boolean(item && scene);

  els.annotationSelectedDataset.textContent = item?.label || "none";
  els.annotationSelectedScene.textContent = scene?.label || "none";
  els.annotationSelectionSummary.textContent = scene
    ? JSON.stringify(scene.selection || {})
    : "Select a scene first.";
  els.annotationSceneName.textContent = scene?.label || "none";
  els.annotationSceneCopy.textContent = scene
    ? (scene.description || "Selected scene for annotation review.")
    : "Select a scene from Library to inspect its annotations.";
  els.annotationJobStatus.textContent = titleCase(status);
  els.annotationJobCopy.textContent =
    status === "running"
      ? `${modelDisplayName(state.annotationRunner)} annotation is running for ${scene?.label || "the selected scene"}.`
      : status === "completed"
        ? "The last annotation job completed. Reload or inspect the metrics below."
        : state.annotationError || "No annotation task is running right now.";
  els.annotationSupportCopy.textContent = canRunAutomated
    ? "This page uses the current Library selection. Run Qwen or Gemma for the selected dataset, refresh library evaluation, and inspect the resulting metrics."
    : canInspect
      ? "Saved annotations can be inspected for this dataset. Automated runs are only enabled when this dataset has a supported VLM pipeline."
      : "Select a scene to inspect saved annotations. Automated runs are only enabled when the selected dataset has a supported VLM pipeline.";

  const runButtonsDisabled = !canRunAutomated || !scene || state.annotationLoading || status === "running";
  els.annotationRunQwenBtn.disabled = runButtonsDisabled;
  els.annotationRunGemmaBtn.disabled = runButtonsDisabled;
  els.annotationStopBtn.disabled = status !== "running";
  els.annotationEvaluateBtn.disabled = state.annotationLoading;
  els.annotationRefreshBtn.disabled = state.annotationLoading;

  if (!canInspect || !data) {
    els.annotationRunsReady.textContent = "0";
    els.annotationRunsCopy.textContent = canRunAutomated
      ? "Run Qwen or Gemma to generate annotation outputs."
      : "Select a scene to load saved annotation outputs.";
    els.annotationLibraryCount.textContent = "0";
    els.annotationLibraryCopy.textContent = "Library evaluation will appear after loading annotation data.";
    els.annotationRunsGrid.innerHTML = '<div class="empty-state">No annotation artifacts loaded yet.</div>';
    els.annotationComparison.innerHTML = '<div class="empty-state">Run both Qwen and Gemma to compare metrics here.</div>';
    els.annotationLibraryReview.innerHTML = '<div class="empty-state">Library evaluation will appear here after loading annotation data.</div>';
    els.annotationLogs.innerHTML = state.annotationLogs.length
      ? state.annotationLogs.map((line) => `<div class="log-entry">${escapeHtml(line)}</div>`).join("")
      : '<div class="empty-state">No annotation logs yet.</div>';
    return;
  }

  const qwenRun = data.runs?.qwen || { available: false };
  const gemmaRun = data.runs?.gemma4 || { available: false };
  const availableRuns = [qwenRun, gemmaRun].filter((run) => run.available).length;
  els.annotationRunsReady.textContent = String(availableRuns);
  els.annotationRunsCopy.textContent =
    availableRuns === 2
      ? "Qwen and Gemma outputs are both available for this scene."
      : availableRuns === 1
        ? "One model output is available for this scene."
        : "No saved model outputs found for this scene yet.";

  const librarySummary = data.library_summary || {};
  const libraryEvaluation = data.library_evaluation || {};
  els.annotationLibraryCount.textContent = String(librarySummary.total_labels || 0);
  const reviewCount = Array.isArray(librarySummary.labels_needing_review) ? librarySummary.labels_needing_review.length : 0;
  els.annotationLibraryCopy.textContent =
    reviewCount > 0
      ? `${reviewCount} label${reviewCount === 1 ? "" : "s"} currently need review.`
      : "No label review flags are active right now.";

  els.annotationRunsGrid.innerHTML = ["qwen", "gemma4"]
    .map((runnerKey) => {
      const run = data.runs?.[runnerKey];
      if (!run) {
        return "";
      }
      const summary = run.summary || {};
      return `
        <article class="annotation-run-card ${run.available ? "" : "annotation-run-card-missing"}">
          <div class="annotation-run-header">
            <h4>${escapeHtml(modelDisplayName(runnerKey))}</h4>
            <span class="session-chip">${run.available ? "ready" : "missing"}</span>
          </div>
          ${
            run.available
              ? `
                <div class="annotation-run-stats">
                  <div><span>Raw clips</span><strong>${escapeHtml(String(run.raw_clip_count || 0))}</strong></div>
                  <div><span>Steps</span><strong>${escapeHtml(String(run.step_count || 0))}</strong></div>
                  <div><span>Parse</span><strong>${escapeHtml(formatNumber(summary.parse_success_rate, 2))}</strong></div>
                  <div><span>Other rate</span><strong>${escapeHtml(formatNumber(summary.other_rate, 2))}</strong></div>
                  <div><span>Sec / clip</span><strong>${escapeHtml(formatNumber(summary.sec_per_clip, 2))}</strong></div>
                  <div><span>Peak VRAM</span><strong>${escapeHtml(formatNumber(summary.peak_vram_allocated_gib, 2))}</strong></div>
                </div>
                <p class="muted summary-copy">Recent steps: ${escapeHtml((run.recent_steps || []).map((step) => step.label).slice(0, 4).join(", ") || "none")}</p>
              `
              : '<p class="muted summary-copy">No saved annotation outputs were found for this model on the selected scene.</p>'
          }
        </article>
      `;
    })
    .join("");

  if (data.comparison?.left_summary && data.comparison?.right_summary) {
    const left = data.comparison.left_summary;
    const right = data.comparison.right_summary;
    els.annotationComparison.innerHTML = renderMetricBars(
      [
        {
          label: "Parse Success",
          hint: "Higher is better",
          left: left.parse_success_rate,
          right: right.parse_success_rate,
          format: (value) => formatNumber(value, 2),
        },
        {
          label: "Other Rate",
          hint: "Lower is better",
          left: left.other_rate,
          right: right.other_rate,
          format: (value) => formatNumber(value, 2),
        },
        {
          label: "Seconds per Clip",
          hint: "Lower is faster",
          left: left.sec_per_clip,
          right: right.sec_per_clip,
          format: (value) => formatNumber(value, 2),
        },
        {
          label: "Peak VRAM (GiB)",
          hint: "Lower is lighter",
          left: left.peak_vram_allocated_gib,
          right: right.peak_vram_allocated_gib,
          format: (value) => formatNumber(value, 2),
        },
        {
          label: "Step Compression",
          hint: "Higher means more clips merged into steps",
          left: left.step_compression_ratio,
          right: right.step_compression_ratio,
          format: (value) => formatNumber(value, 2),
        },
      ],
      data.comparison.left_name || "Qwen",
      data.comparison.right_name || "Gemma 4",
    );
  } else {
    els.annotationComparison.innerHTML = '<div class="empty-state">This scene does not currently have both Qwen and Gemma outputs, so side-by-side comparison is unavailable.</div>';
  }

  const labelRows = Array.isArray(libraryEvaluation.label_rows) ? libraryEvaluation.label_rows : [];
  els.annotationLibraryReview.innerHTML = renderLibraryRows(labelRows.slice(0, 8));
  els.annotationLogs.innerHTML = state.annotationLogs.length
    ? state.annotationLogs
        .map((line) => {
          const text = String(line);
          const isError = /(error|failed|traceback|exception)/i.test(text);
          return `<div class="log-entry ${isError ? "error" : ""}">${escapeHtml(text)}</div>`;
        })
        .join("")
    : '<div class="empty-state">No annotation logs yet.</div>';
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
  renderAnnotationPage();
  renderRoute();
}

async function refreshAnnotationData() {
  const context = currentAnnotationContext();
  if (!context) {
    state.annotationStatus = "idle";
    state.annotationRunner = null;
    state.annotationStartedAt = null;
    state.annotationLogs = [];
    state.annotationData = null;
    state.annotationError = null;
    return;
  }

  const query = new URLSearchParams({
    item_id: context.item.id,
    scene_id: context.scene.id,
  });

  try {
    const [statusPayload, dataPayload] = await Promise.all([
      fetchJson(`/api/annotation/status?${query.toString()}`),
      fetchJson(`/api/annotation/data?${query.toString()}`),
    ]);
    state.annotationStatus = statusPayload.status || "idle";
    state.annotationRunner = statusPayload.runner || null;
    state.annotationStartedAt = statusPayload.started_at || null;
    state.annotationLogs = statusPayload.logs || [];
    state.annotationError = statusPayload.last_error || null;
    state.annotationData = dataPayload;
  } catch (error) {
    state.annotationError = error.message;
    state.annotationData = null;
  }
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

  await refreshAnnotationData();
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

async function runAnnotation(runner) {
  const context = currentAnnotationContext();
  if (!context) {
    appendLog("Choose a dataset and scene before running annotation.");
    return;
  }
  if (!annotationCanRunAutomated(context.item)) {
    appendLog(`Automated annotation is not enabled for dataset ${context.item.dataset}. This page can still inspect any saved annotations.`);
    return;
  }

  try {
    state.annotationLoading = true;
    renderAnnotationPage();
    await fetchJson("/api/annotation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_id: context.item.id,
        scene_id: context.scene.id,
        runner,
      }),
    });
    appendLog(`Started ${modelDisplayName(runner)} annotation for ${context.scene.label}.`);
    await refreshAll();
  } catch (error) {
    appendLog(`Annotation run failed: ${error.message}`);
  } finally {
    state.annotationLoading = false;
    renderAnnotationPage();
  }
}

async function stopAnnotation() {
  try {
    state.annotationLoading = true;
    renderAnnotationPage();
    await fetchJson("/api/annotation/stop", { method: "POST" });
    appendLog("Stopped annotation job.");
    await refreshAll();
  } catch (error) {
    appendLog(`Stopping annotation failed: ${error.message}`);
  } finally {
    state.annotationLoading = false;
    renderAnnotationPage();
  }
}

async function evaluateLibrary() {
  try {
    state.annotationLoading = true;
    renderAnnotationPage();
    await fetchJson("/api/annotation/evaluate-library", { method: "POST" });
    appendLog("Refreshed universal label library evaluation.");
    await refreshAll();
  } catch (error) {
    appendLog(`Library evaluation failed: ${error.message}`);
  } finally {
    state.annotationLoading = false;
    renderAnnotationPage();
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
  els.presetSearch.addEventListener("input", (event) => {
    state.presetSearch = event.target.value;
    renderItems();
  });
  els.sceneSelect.addEventListener("change", (event) => {
    state.selectedSceneId = event.target.value || null;
    refreshAnnotationData()
      .catch((error) => appendLog(`Annotation refresh failed: ${error.message}`))
      .finally(() => renderAll());
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
  els.annotationRunQwenBtn.addEventListener("click", () => {
    runAnnotation("qwen").catch((error) => appendLog(`Annotation failed: ${error.message}`));
  });
  els.annotationRunGemmaBtn.addEventListener("click", () => {
    runAnnotation("gemma4").catch((error) => appendLog(`Annotation failed: ${error.message}`));
  });
  els.annotationStopBtn.addEventListener("click", () => {
    stopAnnotation().catch((error) => appendLog(`Stopping annotation failed: ${error.message}`));
  });
  els.annotationEvaluateBtn.addEventListener("click", () => {
    evaluateLibrary().catch((error) => appendLog(`Library evaluation failed: ${error.message}`));
  });
  els.annotationRefreshBtn.addEventListener("click", () => {
    refreshAll().catch((error) => appendLog(`Annotation refresh failed: ${error.message}`));
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
    home: $("page-home"),
    library: $("page-library"),
    viewer: $("page-viewer"),
    annotation: $("page-annotation"),
  };

  els.statusText = $("status-text");
  els.statusPill = $("status-pill");
  els.activeItem = $("active-item");
  els.activeScene = $("active-scene");
  els.lastError = $("last-error");
  els.viewerPort = $("viewer-port");
  els.grpcUrl = $("grpc-url");
  els.recordingName = $("recording-name");
  els.recordingPath = $("recording-path");
  els.startedAt = $("started-at");
  els.totalSessions = $("total-sessions");
  els.totalScenes = $("total-scenes");
  els.activeSessionSummary = $("active-session-summary");
  els.validatedCount = $("validated-count");
  els.datasetsReadyCopy = $("datasets-ready-copy");
  els.scenesAvailableCopy = $("scenes-available-copy");
  els.activeSessionCopy = $("active-session-copy");
  els.validatedCopy = $("validated-copy");
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
  els.viewerSceneDetailList = $("viewer-scene-detail-list");
  els.viewerSceneCount = $("viewer-scene-count");
  els.viewerSceneApplyBtn = $("viewer-scene-apply-btn");
  els.saveRecordingToggle = $("save-recording-toggle");
  els.presetSearch = $("preset-search");
  els.logSearch = $("log-search");
  els.presetCount = $("preset-count");
  els.sceneSelector = $("scene-selector");
  els.sceneSelect = $("scene-select");
  els.sceneDescription = $("scene-description");
  els.sceneDetailList = $("scene-detail-list");
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
  els.annotationSceneName = $("annotation-scene-name");
  els.annotationSceneCopy = $("annotation-scene-copy");
  els.annotationRunsReady = $("annotation-runs-ready");
  els.annotationRunsCopy = $("annotation-runs-copy");
  els.annotationLibraryCount = $("annotation-library-count");
  els.annotationLibraryCopy = $("annotation-library-copy");
  els.annotationJobStatus = $("annotation-job-status");
  els.annotationJobCopy = $("annotation-job-copy");
  els.annotationSelectedDataset = $("annotation-selected-dataset");
  els.annotationSelectedScene = $("annotation-selected-scene");
  els.annotationSelectionSummary = $("annotation-selection-summary");
  els.annotationSupportCopy = $("annotation-support-copy");
  els.annotationRunQwenBtn = $("annotation-run-qwen-btn");
  els.annotationRunGemmaBtn = $("annotation-run-gemma-btn");
  els.annotationStopBtn = $("annotation-stop-btn");
  els.annotationEvaluateBtn = $("annotation-evaluate-btn");
  els.annotationRefreshBtn = $("annotation-refresh-btn");
  els.annotationRunsGrid = $("annotation-runs-grid");
  els.annotationComparison = $("annotation-comparison");
  els.annotationLibraryReview = $("annotation-library-review");
  els.annotationLogs = $("annotation-logs");

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
