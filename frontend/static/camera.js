class SafeStepCamera {
  constructor(videoEl, canvasEl, onFrameBlob, opts = {}) {
    this.videoEl = videoEl;
    this.canvasEl = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.onFrameBlob = onFrameBlob;

    // Matches FRAME_WIDTH / FRAME_HEIGHT in config/settings.py so the
    // server's spatial math (clock zones, distance ratios) lines up with
    // what the browser actually sends.
    this.width = opts.width || 640;
    this.height = opts.height || 480;

    // Client-side throttle — same intent as SERVER_MAX_FPS, applied at the
    // source so we're not even encoding/sending frames the server would
    // just drop. 250ms = 4fps, matching the default server throttle.
    this.sendIntervalMs = opts.sendIntervalMs || 250;

    this.stream = null;
    this._captureTimer = null;
    this._running = false;
  }

  async start() {
    if (this._running) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: this.width },
        height: { ideal: this.height },
        facingMode: "environment", // prefer the rear/chest-facing camera on phones
      },
      audio: false,
    });

    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();

    this.canvasEl.width = this.width;
    this.canvasEl.height = this.height;

    this._running = true;
    this._captureTimer = setInterval(() => this._captureFrame(), this.sendIntervalMs);
  }

  stop() {
    this._running = false;

    if (this._captureTimer) {
      clearInterval(this._captureTimer);
      this._captureTimer = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  _captureFrame() {
    if (!this._running || this.videoEl.readyState < 2) return;

    this.ctx.drawImage(this.videoEl, 0, 0, this.width, this.height);

    this.canvasEl.toBlob(
      (blob) => {
        if (blob && this.onFrameBlob) this.onFrameBlob(blob);
      },
      "image/jpeg",
      0.7 // quality — balances bandwidth vs detail; tune if small objects are missed
    );
  }
}
