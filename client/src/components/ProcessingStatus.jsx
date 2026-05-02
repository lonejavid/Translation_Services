import React, { useEffect, useRef, useState } from "react";
import { API_BASE } from "../App";

const STEP_LABELS = [
  "Downloading YouTube audio",
  "Detecting language and transcribing speech (mlx-whisper)",
  "Translating to [lang]",
  "Generating dubbed voice (XTTS / MMS-TTS)",
  "Syncing audio timeline",
];

// Max time we'll wait before showing a "taking longer than expected" warning
const WARN_AFTER_SEC = 300;   // 5 min
const TIMEOUT_SEC    = 1800;  // 30 min hard timeout

function ProcessingStatus({ jobId, targetLanguage, onResult, onError, onRetry }) {
  const [activeStep, setActiveStep]     = useState(1);
  const [segmentProgress, setSegmentProgress] = useState(null);
  const [error, setError]               = useState(null);
  const [elapsed, setElapsed]           = useState(0);
  const [reconnects, setReconnects]     = useState(0);

  // Reconnection state kept in a ref so the effect cleanup can access it
  const reconnectTimerRef = useRef(null);
  const reconnectDelayRef = useRef(2000);   // start at 2 s, double each time

  // Elapsed timer — resets when jobId changes
  useEffect(() => {
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [jobId]);

  useEffect(() => {
    setError(null);
    setElapsed(0);
    setReconnects(0);
    reconnectDelayRef.current = 2000;
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;

    let es = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      const url = `${API_BASE}/api/process-video/stream?job_id=${encodeURIComponent(jobId)}`;
      es = new EventSource(url);

      es.onmessage = (event) => {
        // Successful message — reset reconnect delay
        reconnectDelayRef.current = 2000;

        try {
          const data = JSON.parse(event.data);

          if (data.error) {
            setError(data.error);
            onError?.(data.error);
            es.close();
            return;
          }
          if (data.done) {
            es.close();
            return;
          }
          if (data.result) {
            onResult?.(data.result);
            es.close();
            return;
          }
          if (data.step != null) {
            setActiveStep(Math.min(Math.max(data.step, 1), 5));
            if (data.progress) setSegmentProgress(data.progress);
          }
        } catch (_) {}
      };

      es.onerror = () => {
        es.close();
        if (cancelled) return;

        // Exponential backoff: 2s → 4s → 8s → 16s → 32s → 60s cap
        const delay = Math.min(reconnectDelayRef.current, 60000);
        reconnectDelayRef.current = Math.min(delay * 2, 60000);

        setReconnects((n) => n + 1);
        reconnectTimerRef.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (es) es.close();
    };
  }, [jobId, onResult, onError]);

  // Hard timeout — surface an error so the user isn't stuck forever
  useEffect(() => {
    if (!jobId) return;
    const t = setTimeout(() => {
      setError(
        `Processing has taken over ${TIMEOUT_SEC / 60} minutes. ` +
        "The server may be overloaded or the video is too long. Please try again."
      );
      onError?.("Timeout");
    }, TIMEOUT_SEC * 1000);
    return () => clearTimeout(t);
  }, [jobId, onError]);

  if (error) {
    return (
      <div className="yt-proc-error">
        <div className="yt-proc-error-icon" aria-hidden>⚠</div>
        <h3>Something went wrong</h3>
        <p>{error}</p>
        <button
          type="button"
          onClick={() => { setError(null); onRetry?.(); }}
          className="yt-proc-retry"
        >
          Try Again
        </button>
      </div>
    );
  }

  const slow = elapsed >= WARN_AFTER_SEC;

  return (
    <div className="yt-proc">
      <div className="yt-proc-head">
        <span>Processing…</span>
        <span className="yt-proc-time" style={slow ? { color: "#f97316" } : undefined}>
          {elapsed}s elapsed
          {slow && " — taking longer than usual"}
        </span>
      </div>

      {reconnects > 0 && (
        <p style={{ fontSize: "0.78rem", color: "#94a3b8", marginBottom: "0.5rem" }}>
          Reconnected {reconnects}× — pipeline is still running
        </p>
      )}

      <ol className="yt-proc-list">
        {STEP_LABELS.map((label, i) => {
          const stepNum = i + 1;
          const labelText = label.replace("[lang]", targetLanguage || "your language");
          const done   = activeStep > stepNum;
          const active = activeStep === stepNum;
          const iconClass = [
            "yt-proc-icon",
            done ? "yt-proc-icon--done" : active ? "yt-proc-icon--active" : "",
          ].filter(Boolean).join(" ");
          const labelClass = [
            "yt-proc-label",
            active && "yt-proc-label--active",
            done   && "yt-proc-label--done",
            !active && !done && "yt-proc-label--pending",
          ].filter(Boolean).join(" ");

          return (
            <li key={stepNum} className="yt-proc-item">
              <div className={iconClass} aria-hidden>
                {done ? "✓" : active ? <span className="yt-proc-spinner" /> : stepNum}
              </div>
              <div className="yt-proc-body">
                <p className={labelClass}>{labelText}</p>
                {active && (stepNum === 3 || stepNum === 4) && segmentProgress && (
                  <p className="yt-proc-sub">Segments: {segmentProgress}</p>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      <div className="yt-proc-bar">
        <div className="yt-proc-bar-fill" style={{ width: `${(activeStep / 5) * 100}%` }} />
      </div>
    </div>
  );
}

export default ProcessingStatus;
