

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import asdict
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

from config import settings
from models.inference import InferenceEngine
from spatial.geometry import SpatialAnalyzer
from spatial.obstacle_heuristic import ObstacleHeuristic

# ---------------------------------------------------------------------------
# Logging — same structured style as the original modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s %(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("safestep.server")

app = FastAPI(title="SafeStep Web")

# CORS: allow the frontend (possibly hosted on a different origin/port during
# local dev) to open a WebSocket connection to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info("Loading shared YOLOv8 model weights at startup...")
_shared_model = YOLO(settings.MODEL_NAME)
logger.info("Shared model loaded.")


class SessionState:
    """
    Everything one connected browser needs, kept isolated from every other
    connected browser. Created on WebSocket connect, destroyed on disconnect.

    This is the direct replacement for SafeStep Final's two module-level
    singletons (_engine_instance in inference.py, _analyzer_instance in
    geometry.py) - those were correct for a single local process, but would
    silently mix tracking/approach-velocity state between users if reused
    across multiple simultaneous browser connections.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.engine = InferenceEngine(shared_model=_shared_model)
        self.analyzer = SpatialAnalyzer(settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        # New: catches large, close, unclassified obstacles (walls, doors,
        # blank surfaces) that YOLO structurally cannot label. Runs
        # alongside the analyzer, never instead of it. Isolated per
        # session for the same reason engine/analyzer are — frame history
        # from one user's camera must never bleed into another's.
        self.obstacle_heuristic = ObstacleHeuristic()
        self.last_processed_at: float = 0.0
        self.processing_lock = asyncio.Lock()
        self.last_activity: float = time.monotonic()
        # Tracks the (width, height) the analyzer was last built for, so we
        # can detect a mismatch against what the browser actually sends.
        self._analyzer_dims = (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        logger.info("Session %s: InferenceEngine + SpatialAnalyzer created.", session_id)

    def ensure_analyzer_matches(self, frame_width: int, frame_height: int) -> None:
        """
        BUGFIX (see chat history / README "Known issues"): server.py used to
        unconditionally assume every incoming frame was exactly
        settings.FRAME_WIDTH x settings.FRAME_HEIGHT (640x480), because that
        is what camera.js was *supposed* to send. In practice, phone camera
        capture can still hand back a frame whose actual decoded size
        differs from that assumption (camera.js now center-crops correctly,
        but this is a deliberate backstop in case a particular phone/browser
        still behaves unexpectedly).

        If the analyzer's assumed dimensions don't match the frame actually
        decoded this call, rebuild it for the real dimensions. Without this,
        SpatialAnalyzer's zone_width/distance-ratio math (geometry.py lines
        ~182, ~193) silently computes against the wrong canvas size, which
        is exactly what was causing alerts to report the wrong left/right
        zone and the wrong distance.
        """
        if (frame_width, frame_height) != self._analyzer_dims:
            logger.warning(
                "Session %s: frame size %dx%d != expected %dx%d — rebuilding "
                "SpatialAnalyzer for the real dimensions.",
                self.session_id, frame_width, frame_height, *self._analyzer_dims,
            )
            self.analyzer = SpatialAnalyzer(frame_width, frame_height)
            self._analyzer_dims = (frame_width, frame_height)


# Active sessions, keyed by a server-generated session id (one per WebSocket).
_sessions: Dict[str, SessionState] = {}


def _decode_frame(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes a JPEG byte string (sent from the browser's <canvas>.toBlob())
    into a BGR numpy array, matching the format cv2.VideoCapture used to
    hand to process_frame() in the original. Returns None on bad input
    instead of raising, mirroring how core/camera.py returned None on a
    dropped frame rather than crashing the pipeline.
    """
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.warning("Failed to decode incoming frame: %s", exc)
        return None


@app.get("/health")
async def health() -> dict:
    """Simple liveness check for Render/Railway health checks."""
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
   
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = SessionState(session_id)
    _sessions[session_id] = session

    logger.info("Session %s connected. Active sessions: %d", session_id, len(_sessions))

    min_interval = 1.0 / settings.SERVER_MAX_FPS

    try:
        while True:
            try:
                raw_bytes = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=settings.CONNECTION_IDLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Session %s idle for %ds — closing.",
                    session_id, settings.CONNECTION_IDLE_TIMEOUT_SECONDS,
                )
                break

            session.last_activity = time.monotonic()

            # ---- Throttle: drop frames arriving faster than SERVER_MAX_FPS ----
            now = time.monotonic()
            if now - session.last_processed_at < min_interval:
                # Too soon since the last processed frame - skip this one.
                # This is the server-side equivalent of MAX_QUEUED_FRAMES_PER_CONNECTION:
                # we always want the freshest frame, never a backlog of old ones.
                continue

            # Avoid overlapping inference calls on the same session if a
            # previous frame is still (rarely) being processed.
            if session.processing_lock.locked():
                continue

            async with session.processing_lock:
                session.last_processed_at = time.monotonic()

                frame = _decode_frame(raw_bytes)
                if frame is None:
                    continue

                # frame.shape is (height, width, channels) for a decoded
                # BGR image — confirm the analyzer matches reality before
                # running spatial math against it. See ensure_analyzer_matches()
                # docstring for why this exists.
                actual_height, actual_width = frame.shape[0], frame.shape[1]
                session.ensure_analyzer_matches(actual_width, actual_height)

                # ── 1. Inference (Jaliba's logic, session-isolated) ───────
                detections = await asyncio.to_thread(session.engine.run_inference, frame)

                # ── 2. Spatial (Lexmi's logic, byte-for-byte unchanged) ───
                alerts = session.analyzer.process_detections(detections)

                # ── 2b. Unclassified obstacle fallback (new) ───────────────
                # Only checks frames where YOLO found nothing covering the
                # center danger zone — see ObstacleHeuristic's docstring.
                # Runs in a thread for the same reason inference does: the
                # cv2 Laplacian call is synchronous CPU work and shouldn't
                # block the event loop, even though it's cheap relative to
                # YOLO inference.
                obstacle_alert = await asyncio.to_thread(
                    session.obstacle_heuristic.check, frame, detections
                )
                if obstacle_alert is not None:
                    alerts = [obstacle_alert] + alerts

                # ── 3. Send results back to the browser ───────────────────
                payload = {
                    "alerts": [asdict(a) for a in alerts],
                    "detections": [asdict(d) for d in detections],
                }
                await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("Session %s disconnected.", session_id)
    except Exception as exc:
        logger.error("Session %s crashed: %s", session_id, exc, exc_info=True)
    finally:
        _sessions.pop(session_id, None)
        logger.info("Session %s cleaned up. Active sessions: %d", session_id, len(_sessions))


# ---------------------------------------------------------------------------
# Serve the frontend as static files from the same server.
# Keeps deployment to one Render service instead of two.
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("../frontend/static/index.html")
