```markdown
# SafeStep  
### Offline AI Navigation Assistant for the Visually Impaired

> **SafeStep** is a 100% offline, real-time AI assistant that detects nearby obstacles using a webcam and delivers priority-ranked spoken alerts to visually impaired users through any connected audio device.

---

## Overview

SafeStep continuously analyzes the user’s surroundings and provides concise, actionable voice guidance, helping users navigate safely without relying on an internet connection.

**Example Alerts**
- *“Stop! Car directly ahead, one step away.”*
- *“Person approaching from your left.”*
- *“Traffic signal detected on your right.”*

---

## Objectives

- Enable safe, independent navigation for visually impaired users  
- Operate entirely offline for reliability and privacy  
- Deliver real-time, prioritized audio feedback  
- Run efficiently on CPU-only, free-tier deployments

---

## System Architecture

```

[Webcam] ──► [YOLOv8-Nano Inference] ──► [Spatial Analysis] ──► [Priority Audio Queue] ──► [Earphones]  

````

---

## Processing Pipeline

| Stage | Module | Description |
|------|-------|-------------|
| Camera | `core/camera.py` | Captures live video, manages reconnection & rotation |
| Inference | `models/inference.py` | Object detection & tracking using YOLOv8 + ByteTrack |
| Spatial Analysis | `spatial/geometry.py` | Converts bounding boxes to clock positions & distance |
| Audio Output | `audio/queue_manager.py` | Priority-based TTS alerts (danger first) |

---

## Key Features

- **Fully Offline Operation** – No internet required  
- **Real-Time AI Detection** – Powered by YOLOv8-Nano  
- **Priority-Based Alerts** – Critical obstacles spoken first  
- **Low Latency Processing** – Drops stale frames to avoid lag  
- **Browser-Based Frontend** – No installation for users  
- **CPU-Friendly Design** – Suitable for free hosting tiers  
- **Highly Configurable** – FPS, confidence thresholds, cooldowns  

---

## Technology Stack

- **Backend**: Python, FastAPI, WebSockets  
- **AI Model**: YOLOv8-Nano, ByteTrack  
- **Frontend**: HTML, CSS, JavaScript  
- **Audio**: Web Speech API (Text-to-Speech)  
- **Deployment**: Render

---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/li-yah-h/SafeStep-web.git
cd SafeStep-web
````

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Launch Frontend

Open `frontend/static/index.html` in a browser and allow camera & audio permissions.

---

## Project Structure

```
safestep-web/
├── backend/
│   ├── server.py              # FastAPI app + WebSocket API
│   ├── contracts.py           # Data contracts & schemas
│   ├── config/
│   │   └── settings.py        # Global configuration
│   ├── spatial/
│   │   └── geometry.py        # Spatial reasoning logic
│   ├── models/
│   │   └── inference.py       # YOLOv8 inference pipeline
│   └── requirements.txt
│
└── frontend/
    └── static/
        ├── index.html         # Main UI
        ├── style.css          # Styling
        ├── camera.js          # Webcam handling
        ├── audio.js           # Speech synthesis
        └── app.js             # App orchestration
```

---

## Team Contributions

| Team Member        | Responsibility                                 |
| ------------------ | ---------------------------------------------- |
| Diya K             | Camera module (`core/camera.py`)               |
| Jaliba Nasrin O    | Spatial analysis (`spatial/geometry.py`)       |
| Lakshmi Priyanka M | Audio & alert queue (`audio/queue_manager.py`) |
| Liya Mary Paul     | Inference & tracking (`models/inference.py`)   |

---
