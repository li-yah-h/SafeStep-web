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
│ camera.js           │  JPEG frames       │ server.py                │
│  getUserMedia()     │ ───────────────▶  │  per-connection session: │
│  captures @ ~4fps   │   (binary WS)      │   InferenceEngine        │
│                     │                    │   SpatialAnalyzer        │
│ app.js              │    {alerts,        │                          │
│  draws boxes on     │   detections}      │ models/inference.py      │
│  <canvas> overlay   │ ◀───────────────  │                          │
│                     │   (JSON)           │                          │
│ audio.js            │                    │ spatial/geometry.py      │
│  speaks alerts via  │                    │                          │
│  Web Speech API     │                    │                          │
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
│   ├── contracts.py     
│   ├── config/settings.py     
│   ├── spatial/geometry.py    
│   ├── models/inference.py  
│   ├── tests/                 # ported test suites, + 2 new session-isolation tests
│   └── requirements.txt
└── frontend/
    └── static/
        ├── index.html
        ├── style.css
        ├── camera.js       
        ├── audio.js           
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
python3 tests/test_spatial.py   
python3 tests/test_inference.py  
```
