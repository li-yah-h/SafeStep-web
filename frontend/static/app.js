

const elVideo = document.getElementById("video");
const elCanvas = document.getElementById("canvas");
const elOverlay = document.getElementById("overlay");
const elStatus = document.getElementById("status");
const elStartBtn = document.getElementById("startBtn");
const elStopBtn = document.getElementById("stopBtn");
const elDebugToggle = document.getElementById("debugToggle");

const overlayCtx = elOverlay.getContext("2d");

let ws = null;
let camera = null;
let audioEngine = null;
let showDebugOverlay = true; // mirrors SHOW_DEBUG_WINDOW in settings.py

// Same color mapping as draw_debug_overlay()'s colour_map in main.py
// (converted from OpenCV's BGR to CSS, same visual meaning: red = danger).
const PRIORITY_COLORS = {
  1: "rgb(255, 0, 0)",   // Red — Priority 1 (danger)
  2: "rgb(255, 200, 0)", // Yellow — Priority 2 (caution)
  3: "rgb(0, 200, 0)",   // Green — Priority 3 (environment)
};

function setStatus(text) {
  elStatus.textContent = text;
}

function buildWebSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

async function start() {
  try {
    audioEngine = new SafeStepAudioEngine();
  } catch (err) {
    setStatus(`Audio error: ${err.message}`);
    return;
  }

  ws = new WebSocket(buildWebSocketUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("Connected. Requesting camera access...");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleServerMessage(data);
  };

  ws.onerror = () => setStatus("WebSocket error — check server connection.");

  ws.onclose = () => {
    setStatus("Disconnected.");
    stop();
  };

  try {
    camera = new SafeStepCamera(elVideo, elCanvas, onFrameCaptured, {
      width: 640,
      height: 480,
      sendIntervalMs: 250, // matches SERVER_MAX_FPS=4 in settings.py
    });
    await camera.start();
    setStatus("Running — watching for obstacles.");
    elStartBtn.disabled = true;
    elStopBtn.disabled = false;
  } catch (err) {
    setStatus(`Camera error: ${err.message}`);
    ws.close();
  }
}

function stop() {
  if (camera) {
    camera.stop();
    camera = null;
  }
  if (audioEngine) {
    audioEngine.shutdown(); // mirrors shutdown_audio() — finishes current speech
    audioEngine = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
  overlayCtx.clearRect(0, 0, elOverlay.width, elOverlay.height);
  elStartBtn.disabled = false;
  elStopBtn.disabled = true;
  setStatus("Stopped.");
}

/**
 * Called by camera.js once per captured frame. Sends the JPEG blob to the
 * server as binary WebSocket data — equivalent to main.py handing a raw
 * frame to process_frame() directly, just over the wire instead of in-process.
 */
function onFrameCaptured(blob) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  blob.arrayBuffer().then((buf) => ws.send(buf));
}

/**
 * Handles {alerts, detections} from the server — the direct equivalent of
 * the second half of main.py's loop (audio.enqueue_alert + draw_debug_overlay).
 */
function handleServerMessage(data) {
  const { alerts = [], detections = [] } = data;

  // ── Audio (replaces: for alert in alerts: enqueue_alert(alert)) ──────
  for (const alert of alerts) {
    audioEngine.enqueueAlert(alert);
  }

  // ── Debug overlay (replaces: draw_debug_overlay + cv2.imshow) ────────
  if (showDebugOverlay) {
    drawDebugOverlay(detections, alerts);
  } else {
    overlayCtx.clearRect(0, 0, elOverlay.width, elOverlay.height);
  }
}

/**
 * Direct port of main.py's draw_debug_overlay(): colour-codes bounding
 * boxes by the highest-priority alert for that label, plus a status bar.
 */
function drawDebugOverlay(detections, alerts) {
  overlayCtx.clearRect(0, 0, elOverlay.width, elOverlay.height);

  // Build a priority lookup by label, same as priority_by_label in main.py
  const priorityByLabel = {};
  for (const alert of alerts) {
    const label = alert.object_id.split("_")[0];
    const existing = priorityByLabel[label] ?? 99;
    priorityByLabel[label] = Math.min(existing, alert.priority);
  }

  overlayCtx.lineWidth = 2;
  overlayCtx.font = "14px sans-serif";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const priority = priorityByLabel[det.label] ?? 3;
    const color = PRIORITY_COLORS[priority] ?? PRIORITY_COLORS[3];

    overlayCtx.strokeStyle = color;
    overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const labelText = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
    overlayCtx.fillStyle = color;
    overlayCtx.fillText(labelText, x1, Math.max(10, y1 - 6));
  }

  // Status bar, same info as main.py's bottom-of-frame overlay text
  overlayCtx.fillStyle = "rgb(200, 200, 200)";
  overlayCtx.font = "13px sans-serif";
  overlayCtx.fillText(
    `SafeStep  |  Objects: ${detections.length}  |  Alerts: ${alerts.length}`,
    8,
    elOverlay.height - 8
  );
}

elStartBtn.addEventListener("click", start);
elStopBtn.addEventListener("click", stop);
elDebugToggle.addEventListener("change", (e) => {
  showDebugOverlay = e.target.checked; // mirrors toggling SHOW_DEBUG_WINDOW
  if (!showDebugOverlay) overlayCtx.clearRect(0, 0, elOverlay.width, elOverlay.height);
});

window.addEventListener("beforeunload", stop);
