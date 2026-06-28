# SafeStep 

A browser-based version of **SafeStep** (see the original repo, kept as-is
for backup/reference). SafeStep Final ran entirely on one local machine —
its own webcam in, its own speakers out. SafeStep moves the camera and
speakers to the user's browser, while the AI detection and spatial-reasoning
logic still runs on a server, reused almost unchanged from the original.

## Architecture

```
Browser                                    Server (FastAPI)
┌─────────────────────┐                    ┌──────────────────────────┐
│ camera.js            │  JPEG frames       │ server.py                │
│  getUserMedia()       │ ───────────────▶  │  per-connection session: │
│  captures @ ~4fps     │   (binary WS)      │   InferenceEngine        │
│                       │                    │   SpatialAnalyzer        │
│ app.js                │  {alerts,          │                          │
│  draws boxes on       │   detections}      │ models/inference.py      │
│  <canvas> overlay     │ ◀───────────────   │        │
│                       │   (JSON)           │                          │
│ audio.js              │                    │ spatial/geometry.py      │
│  speaks alerts via     │                    │          │
│  Web Speech API        │                    │                          │
└─────────────────────┘                    └──────────────────────────┘
```

Each browser tab that connects gets its **own** `InferenceEngine` +
`SpatialAnalyzer`, created on WebSocket connect and torn down on disconnect.
This matters because YOLOv8's tracker (ByteTrack) keeps per-stream state —
sharing one tracker across two different people's camera feeds would mix
their object tracking together. Model *weights* are loaded once at server
startup and reused read-only across every session, so adding more users
doesn't reload the ~6MB model repeatedly — only the lightweight per-session
tracker/confidence state is duplicated.

The server throttles inference to `SERVER_MAX_FPS` (default: 4 frames/sec
per connection — see `backend/config/settings.py`) and drops frames that
arrive faster than it can process them, rather than queuing a backlog. This
is deliberate: on free-tier CPU hosting, a queue of stale frames would mean
alerts lag behind where obstacles actually are, which is worse than
processing fewer, fresher frames.

## Project layout

```
safestep-web/
├── backend/
│   ├── server.py              # FastAPI app + WebSocket endpoint
│   ├── contracts.py           # unchanged from original
│   ├── config/settings.py     # original constants + a few web-specific additions
│   ├── spatial/geometry.py    # unchanged from original (Lexmi)
│   ├── models/inference.py    # same logic, session-isolated (Jaliba)
│   ├── tests/                 # ported test suites, + 2 new session-isolation tests
│   └── requirements.txt
└── frontend/
    └── static/
        ├── index.html
        ├── style.css
        ├── camera.js           # browser camera capture (new — replaces core/camera.py)
        ├── audio.js            # browser speech output (new — replaces audio/queue_manager.py)
        └── app.js              # wires camera + WebSocket + audio + overlay together
```

## Running locally

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in a browser, click **Start watching**,
and grant camera permission. The first request will download YOLOv8-Nano's
weights automatically (~6MB) — this requires normal internet access (it
fetches from Ultralytics' GitHub release assets).

> **Note on HTTPS for camera access**: most browsers only allow
> `getUserMedia()` (camera access) on `https://` or `http://localhost`.
> Local development on `localhost` is fine as-is. Once deployed, make sure
> your hosting URL serves over HTTPS (Render and Railway both do this
> automatically).

### Running the tests

```bash
cd backend
python3 tests/test_spatial.py     # Lexmi's logic — no model needed
python3 tests/test_inference.py   # Jaliba's logic — downloads model weights on first run
```

## Deploying to Render (free tier)

1. **Push this repo to GitHub** (a new repo — keep SafeStep Final's repo
   completely separate, as planned).
2. **Create a new Web Service on Render**, pointing at this repo.
3. **Root directory**: `backend`
4. **Build command**: `pip install -r requirements.txt`
5. **Start command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
6. **Instance type**: Free tier is enough to start, but read the
   "Free-tier reality check" section below before counting on it for
   multiple simultaneous users.
7. Render builds, deploys, and gives you a `https://your-app.onrender.com`
   URL — that's it, frontend and backend are served from the same service
   (see `server.py`'s static file mount), no separate frontend deploy needed.

### Free-tier reality check

A few things worth knowing going in, since you specifically chose
multi-user + free-tier CPU hosting:

- **Cold starts.** Render's free tier spins your service down after 15
  minutes of inactivity and takes 30-60s to wake back up on the next
  request. The first user after an idle period will see a delay before
  the WebSocket connects.
- **CPU inference is slow.** YOLOv8n on a shared free CPU is roughly
  150-400ms per frame. With `SERVER_MAX_FPS=4`, one user's experience
  should stay reasonable; 3-4 simultaneous users sharing the same CPU will
  likely need a lower `SERVER_MAX_FPS` (try 2) to keep each person's alerts
  from lagging behind. This is a one-line change in `config/settings.py`.
- **RAM.** Free tiers are typically 512MB. The shared YOLOv8n weights plus
  PyTorch's CPU runtime use a meaningful chunk of that on their own; each
  additional simultaneous session adds a smaller amount (just the
  tracker/confidence state, not the weights again). If you see
  out-of-memory restarts under real load, that's the signal to either
  upgrade the instance or cap concurrent sessions.
- **If this needs to be reliably real-time for several people at once**,
  a paid tier (more CPU, no cold start, more RAM) will matter a lot more
  here than it would for a typical web app, simply because YOLO inference
  is CPU-hungry in a way that, say, a CRUD API isn't.

## What to test before relying on this

- [ ] Single user, good lighting: alerts match what you'd expect from the
      original SafeStep Final for the same scene.
- [ ] Two browser tabs/devices connected at once: confirm each gets its own
      sensible alerts (not influenced by what the other tab sees).
- [ ] Slow/unstable network: confirm the app degrades gracefully (older
      frames dropped, not a frozen UI) rather than crashing.
- [ ] Mobile browser (the real target device for a navigation aid):
      confirm camera permission prompts work and the rear camera is used.
