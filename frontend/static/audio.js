
const COOLDOWNS = {
  1: 2.0,  // CRITICAL_COOLDOWN
  2: 5.0,  // NAV_COOLDOWN
  3: 10.0, // ENV_COOLDOWN
};

class SafeStepAudioEngine {
  constructor() {
    if (!("speechSynthesis" in window)) {
      throw new Error(
        "Web Speech API not supported in this browser — SafeStep audio alerts require it."
      );
    }

    this.synth = window.speechSynthesis;

    // Same shape as queue_manager.py's PriorityQueue: each item is
    // {priority, seq, message}. Lower priority number = spoken sooner.
    this._queue = [];
    this._sequence = 0;

    // Per-object_id cooldown tracker: { object_id: lastSpokenTimestampMs }
    this._cooldownTracker = new Map();

    this._speaking = false;
    this._shuttingDown = false;

    this._preferredVoice = null;
    this._loadPreferredVoice();
  }

  /**
   * Checks cooldown and, if due, enqueues the alert for speech.
   * Direct port of queue_manager.py's process_alert().
   *
   * @param {{priority: number, message: string, object_id: string}} alert
   */
  enqueueAlert(alert) {
    if (this._shuttingDown) return;

    const now = performance.now();
    const requiredCooldownMs = (COOLDOWNS[alert.priority] ?? COOLDOWNS[3]) * 1000;
    const lastSpoken = this._cooldownTracker.get(alert.object_id) ?? -Infinity;
    const elapsed = now - lastSpoken;

    if (elapsed < requiredCooldownMs) {
      // Same as queue_manager.py: silently suppressed, logged at debug only.
      return;
    }

    // Mark as spoken now, pre-emptively — same reasoning as the original:
    // blocks duplicate alerts arriving in the same detection burst before
    // speech actually starts.
    this._cooldownTracker.set(alert.object_id, now);

    // Priority-1 flush: drop all pending P2/P3 items, exactly like
    // _flush_low_priority() in queue_manager.py.
    if (alert.priority === 1) {
      const before = this._queue.length;
      this._queue = this._queue.filter((item) => item.priority === 1);
      const dropped = before - this._queue.length;
      if (dropped > 0) {
        console.info(`[audio] P1 alert — flushed ${dropped} lower-priority item(s).`);
      }
    }

    this._queue.push({
      priority: alert.priority,
      seq: this._sequence++,
      message: alert.message,
    });

    this._sortQueue();
    this._maybeSpeakNext();
  }

  /**
   * Graceful shutdown — lets the current utterance finish, then stops.
   * Mirrors shutdown_audio()'s "last spoken alert always completes" promise.
   */
  shutdown() {
    this._shuttingDown = true;
    this._queue = [];
    // Intentionally NOT calling this.synth.cancel() here — cancelling mid-
    // utterance would cut off speech, which is exactly what the original's
    // graceful shutdown was designed to avoid. The current utterance (if
    // any) finishes naturally via its 'onend' handler.
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  _sortQueue() {
    // Stable sort by (priority, seq) — same tiebreak rule as the Python
    // PriorityQueue's (priority, sequence_number, message) tuples.
    this._queue.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
  }

  _maybeSpeakNext() {
    if (this._speaking || this._queue.length === 0 || this._shuttingDown) return;

    const item = this._queue.shift();
    this._speaking = true;

    const utterance = new SpeechSynthesisUtterance(item.message);
    utterance.rate = 1.15; // matches original's "slightly faster than default"
    utterance.volume = 1.0;
    if (this._preferredVoice) utterance.voice = this._preferredVoice;

    utterance.onend = () => {
      this._speaking = false;
      // Small buffer between utterances, mirroring queue_manager.py's
      // time.sleep(0.08) to avoid clipped boundaries between alerts.
      setTimeout(() => this._maybeSpeakNext(), 80);
    };

    utterance.onerror = (event) => {
      // Mirrors queue_manager.py's try/except around _speak(): one bad
      // utterance is logged and the engine keeps going, never crashes.
      console.error("[audio] Speech synthesis error (message dropped):", event);
      this._speaking = false;
      setTimeout(() => this._maybeSpeakNext(), 80);
    };

    this.synth.speak(utterance);
  }

  _loadPreferredVoice() {
    const pick = () => {
      const voices = this.synth.getVoices();
      if (voices.length > 0) {
        // PREFERRED_VOICE_INDEX in settings.py defaulted to 0 (system
        // default) — same idea here: take the first available voice.
        this._preferredVoice = voices[0];
      }
    };
    pick();
    // Some browsers populate the voice list asynchronously.
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = pick;
    }
  }
}
