CAMERA_INDEX  = 0
FRAME_WIDTH   = 640   # Must match the resolution the browser captures at
FRAME_HEIGHT  = 480
TARGET_FPS    = 30
CAMERA_ROTATION = 0

# ------------------------------------------------------------------
# Inference Settings
# ------------------------------------------------------------------
MODEL_NAME           = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

PREFERRED_VOICE_INDEX = 0  # no longer used server-side; browser picks its own voice

# ------------------------------------------------------------------
# Spatial / Distance Settings
# ------------------------------------------------------------------
CLOSE_DISTANCE_RATIO = 0.85
MID_DISTANCE_RATIO   = 0.60

# ------------------------------------------------------------------
# Audio Cooldown Settings (seconds)
# Used twice now: once as the source of truth, and mirrored in
# frontend/static/audio.js so client-side cooldown logic matches exactly.
# ------------------------------------------------------------------
CRITICAL_COOLDOWN = 2.0
NAV_COOLDOWN      = 5.0
ENV_COOLDOWN      = 10.0

# ------------------------------------------------------------------
# UI / Debug
# ------------------------------------------------------------------
SHOW_DEBUG_WINDOW = True   # web equivalent: draw bounding boxes on the <canvas>
DEBUG_BOX_COLOR = (0, 255, 0)


# ==============================================================================
# WEB-SPECIFIC SETTINGS (new — required because of the browser/server split)
# ==============================================================================

# Max frames per second the server will actually run inference on, regardless
# of how fast the browser sends them. On a free-tier CPU box, YOLOv8n inference
# takes ~150-400ms per frame; processing every browser frame (usually 15-30fps)
# would queue the server solid under multiple simultaneous users. Frames that
# arrive faster than this are dropped, not queued, so alerts never lag behind
# real-world position by more than ~1 throttle interval.
SERVER_MAX_FPS = 4  # 1 frame every 250ms per connected user

# Max number of frames allowed to sit in a connection's processing queue.
# Kept at 1 deliberately: if the server is behind, we want the NEWEST frame,
# not a backlog of stale ones queued behind it.
MAX_QUEUED_FRAMES_PER_CONNECTION = 1

# Max JPEG dimension accepted from the browser (safety cap — also enforced
# client-side in camera.js, this is the server-side backstop).
MAX_FRAME_DIMENSION = 960

# Idle timeout: if a connected browser sends nothing for this long, the server
# assumes the tab was closed without a clean disconnect and frees the session.
CONNECTION_IDLE_TIMEOUT_SECONDS = 30
