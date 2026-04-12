import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { API_BASE } from "../App";

function formatTc(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  const whole = Math.floor(r);
  const frac = Math.round((r - whole) * 10);
  return `${String(m).padStart(2, "0")}:${String(whole).padStart(2, "0")}:${String(frac).padStart(1, "0")}`;
}

// Safe literal string escape — prevents ReDoS from user-typed patterns
function escapeLiteral(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export default function TranscriptEditor() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const cacheKey = (searchParams.get("cache_key") || "").trim().toLowerCase();
  const youtubeUrl = searchParams.get("url") || "";
  const videoId = useMemo(() => {
    if (!youtubeUrl) return null;
    try {
      const u = new URL(youtubeUrl);
      if (u.hostname.endsWith("youtu.be")) return u.pathname.slice(1).split("?")[0] || null;
      if (u.hostname.includes("youtube.com")) {
        if (u.searchParams.has("v")) return u.searchParams.get("v");
        const shorts = u.pathname.match(/\/shorts\/([^/?]+)/);
        if (shorts) return shorts[1];
      }
    } catch (_) {}
    return null;
  }, [youtubeUrl]);

  const [segments, setSegments] = useState([]);
  const [savedSegments, setSavedSegments] = useState([]); // snapshot of last-saved state
  const [title, setTitle] = useState("Transcript editor");
  const [targetLang, setTargetLang] = useState("hi");
  const [loadError, setLoadError] = useState(null);
  const [saveState, setSaveState] = useState(null);
  const [redubBusy, setRedubBusy] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQ, setSearchQ] = useState("");
  const [searchRepl, setSearchRepl] = useState("");
  const [searchField, setSearchField] = useState("both");
  const isDirty = useRef(false);

  // Track whether user has unsaved edits
  const dirty = useMemo(() => {
    if (segments.length !== savedSegments.length) return segments.length > 0;
    return segments.some((s, i) => {
      const orig = savedSegments[i];
      return !orig || s.text !== orig.text || s.translated_text !== orig.translated_text;
    });
  }, [segments, savedSegments]);

  // Warn before navigating away with unsaved changes
  useEffect(() => {
    isDirty.current = dirty;
  }, [dirty]);

  useEffect(() => {
    const handler = (e) => {
      if (isDirty.current) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  const load = useCallback(async () => {
    if (!cacheKey || cacheKey.length !== 32) {
      setLoadError("Missing or invalid cache_key");
      return;
    }
    setLoadError(null);
    try {
      const res = await fetch(`${API_BASE}/api/subtitles/${cacheKey}.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!Array.isArray(data)) throw new Error("Invalid subtitles format");
      const segs = data.map((s, i) => ({
        ...s,
        _id: i,
        text: s.text ?? s.original ?? "",
        translated_text: s.translated_text ?? s.translated ?? "",
      }));
      setSegments(segs);
      setSavedSegments(segs.map((s) => ({ text: s.text, translated_text: s.translated_text })));
      const metaRes = await fetch(`${API_BASE}/api/subtitles/${cacheKey}_meta.json`).catch(() => null);
      if (metaRes && metaRes.ok) {
        const meta = await metaRes.json();
        if (meta.title) setTitle(meta.title);
        if (meta.target_language) setTargetLang(meta.target_language);
      }
    } catch (e) {
      setLoadError(String(e.message || e));
    }
  }, [cacheKey]);

  useEffect(() => {
    load();
  }, [load]);

  const updateSeg = (idx, field, value) => {
    setSegments((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: value };
      return next;
    });
  };

  const applySearchReplace = () => {
    if (!searchQ) return;
    // Always use literal (escaped) search — never user-supplied regex to prevent ReDoS
    const re = new RegExp(escapeLiteral(searchQ), "gi");
    setSegments((prev) =>
      prev.map((s) => {
        let t = s.text;
        let tr = s.translated_text;
        if (searchField === "source" || searchField === "both") t = t.replace(re, searchRepl);
        if (searchField === "translation" || searchField === "both") tr = tr.replace(re, searchRepl);
        return { ...s, text: t, translated_text: tr };
      })
    );
    setSearchOpen(false);
  };

  const publish = async () => {
    setSaveState({ type: "saving" });
    try {
      const body = {
        subtitles: segments.map((s) => ({
          start: s.start,
          end: s.end,
          text: s.text,
          translated_text: s.translated_text,
          words: s.words,
          stt_quality: s.stt_quality,
        })),
      };
      const res = await fetch(`${API_BASE}/api/subtitles/cache/${cacheKey}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      // Mark as saved — clear dirty flag
      setSavedSegments(segments.map((s) => ({ text: s.text, translated_text: s.translated_text })));
      setSaveState({ type: "ok", learned: data.learned });
    } catch (e) {
      setSaveState({ type: "err", msg: String(e.message || e) });
    }
  };

  const redub = async () => {
    if (dirty) {
      const ok = window.confirm("You have unsaved changes. Save them first before re-dubbing?\n\nClick Cancel to go back and save, or OK to re-dub with your current edits (they will be auto-saved).");
      if (!ok) return;
      // Auto-save first
      await publish();
    } else if (!window.confirm("Re-generate dubbed audio from edited text? This may take several minutes.")) {
      return;
    }
    setRedubBusy(true);
    setSaveState({ type: "redub" });
    try {
      const res = await fetch(`${API_BASE}/api/video/cache/${cacheKey}/redub`, {
        method: "POST",
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      const bust = `?v=${Date.now()}`;
      setSaveState({
        type: "ok",
        msg: "Re-dub complete. Return to the player and refresh the page to hear the new audio.",
        audio_url: data.audio_url ? data.audio_url + bust : null,
      });
    } catch (e) {
      setSaveState({ type: "err", msg: String(e.message || e) });
    } finally {
      setRedubBusy(false);
    }
  };

  const totalDur = segments.length
    ? Math.max(...segments.map((s) => Number(s.end) || 0))
    : 0;

  if (!cacheKey) {
    return (
      <div className="yt-editor-page">
        <p>Missing cache_key. Open this page from the player after a video finishes processing.</p>
        <button type="button" className="yt-btn-secondary" onClick={() => navigate("/")}>
          Home
        </button>
      </div>
    );
  }

  return (
    <div className="yt-editor-page">
      <header className="yt-editor-top">
        <nav className="yt-editor-tabs">
          <span className="yt-editor-tab">Overview</span>
          <span className="yt-editor-tab yt-editor-tab--active">Editor</span>
          <span className="yt-editor-tab">Downloads</span>
        </nav>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {dirty && (
            <span style={{ color: "#f59e0b", fontSize: "0.8rem", fontWeight: 600 }}>
              ● Unsaved changes
            </span>
          )}
          <button
            type="button"
            className="yt-editor-sr"
            onClick={() => setSearchOpen(true)}
          >
            Search &amp; Replace
          </button>
          {/* Export buttons */}
          <a
            href={`${API_BASE}/api/export/${cacheKey}/subtitles.srt`}
            className="yt-btn-secondary"
            style={{ textDecoration: "none", padding: "6px 12px", fontSize: "0.8rem" }}
            title="Download SRT subtitle file"
          >
            ↓ SRT
          </a>
          <a
            href={`${API_BASE}/api/export/${cacheKey}/subtitles.vtt`}
            className="yt-btn-secondary"
            style={{ textDecoration: "none", padding: "6px 12px", fontSize: "0.8rem" }}
            title="Download WebVTT subtitle file"
          >
            ↓ VTT
          </a>
          <a
            href={`${API_BASE}/api/export/${cacheKey}/video`}
            className="yt-btn-primary"
            style={{ textDecoration: "none", padding: "6px 12px", fontSize: "0.8rem" }}
            title="Download dubbed MP4 video (may take a moment to generate)"
          >
            ↓ MP4
          </a>
        </div>
      </header>

      <div className="yt-editor-workspace">
        <div className="yt-editor-video-col">
          <button type="button" className="yt-back-btn yt-editor-back" onClick={() => {
            if (dirty && !window.confirm("You have unsaved changes. Leave without saving?")) return;
            navigate(-1);
          }}>
            ← Back
          </button>
          <h1 className="yt-editor-title">{title}</h1>
          <p className="yt-editor-sub">
            Target: <strong>{targetLang}</strong> · Fixes are saved to{" "}
            <code className="yt-editor-code">learned_corrections.json</code> and applied on the{" "}
            <strong>next</strong> pipeline run. Use <strong>Re-dub audio</strong>{" "}
            to rebuild speech from edited lines without re-processing the whole video.
          </p>
          {videoId ? (
            <div className="yt-editor-video-wrap">
              <iframe
                title="source video"
                src={`https://www.youtube.com/embed/${videoId}`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          ) : (
            <div className="yt-editor-video-placeholder">No video URL — edit text only</div>
          )}
        </div>

        <div className="yt-editor-segments-col">
          {loadError && <div className="yt-error-box">{loadError}</div>}
          {segments.map((s, idx) => (
            <article key={s._id ?? idx} className="yt-seg-card">
              <div className="yt-seg-head">
                <span className="yt-seg-num">{idx + 1}</span>
                <span className="yt-seg-voice">Voice 1</span>
                <span className="yt-seg-time">
                  {formatTc(s.start)} – {formatTc(s.end)}
                </span>
                {s.cps_flag && (
                  <span title={`Readability issue: ${s.cps_flag}`} style={{ color: "#f59e0b", fontSize: "0.75rem", marginLeft: "auto" }}>
                    ⚠ {s.cps_flag}
                  </span>
                )}
              </div>
              <div className="yt-seg-dual">
                <div className="yt-seg-field">
                  <label>Translation ({targetLang})</label>
                  <textarea
                    value={s.translated_text}
                    onChange={(e) => updateSeg(idx, "translated_text", e.target.value)}
                    rows={3}
                    className="yt-seg-ta"
                    spellCheck="false"
                  />
                </div>
                <span className="yt-seg-arrow" aria-hidden="true">
                  →
                </span>
                <div className="yt-seg-field">
                  <label>Original (detected language)</label>
                  <textarea
                    value={s.text}
                    onChange={(e) => updateSeg(idx, "text", e.target.value)}
                    rows={3}
                    className="yt-seg-ta"
                    spellCheck="false"
                  />
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>

      <div className="yt-editor-timeline">
        <div className="yt-timeline-rail" aria-hidden="true">
          {totalDur > 0 &&
            segments.map((s, idx) => {
              const w = ((Number(s.end) - Number(s.start)) / totalDur) * 100;
              const left = (Number(s.start) / totalDur) * 100;
              return (
                <div
                  key={idx}
                  className="yt-timeline-block"
                  style={{ left: `${left}%`, width: `${Math.max(w, 0.4)}%` }}
                  title={`Segment ${idx + 1}`}
                />
              );
            })}
        </div>
        <div className="yt-timeline-scale">
          {totalDur > 0 &&
            Array.from({ length: Math.ceil(totalDur) + 1 }).map((_, i) => (
              <span key={i} className="yt-tick">
                {i}s
              </span>
            ))}
        </div>
      </div>

      <footer className="yt-editor-footer">
        <button type="button" className="yt-btn-secondary" onClick={load}>
          Refresh
        </button>
        <div className="yt-editor-footer-msg">
          {saveState?.type === "saving" && <span>Saving…</span>}
          {saveState?.type === "redub" && <span>Re-dubbing audio (please wait)…</span>}
          {saveState?.type === "ok" && (
            <span className="yt-ok">
              Saved
              {saveState.learned != null &&
                ` · learned +${saveState.learned.asr_phrases} ASR, +${saveState.learned.translation_memory} TM`}
              {saveState.msg && ` · ${saveState.msg}`}
              {saveState.audio_url && (
                <>
                  {" · "}
                  <a href={`${API_BASE}${saveState.audio_url}`} className="yt-editor-audio-link">
                    Open new dub MP3
                  </a>
                </>
              )}
            </span>
          )}
          {saveState?.type === "err" && <span className="yt-err">{saveState.msg}</span>}
        </div>
        <div className="yt-editor-actions">
          <button
            type="button"
            className="yt-btn-secondary"
            onClick={redub}
            disabled={redubBusy || segments.length === 0}
          >
            {redubBusy ? "Re-dubbing…" : "Re-dub audio"}
          </button>
          <button
            type="button"
            className={`yt-btn-primary yt-publish${dirty ? " yt-publish--dirty" : ""}`}
            onClick={publish}
            disabled={segments.length === 0}
          >
            {dirty ? "Save changes ●" : "Publish changes"}
          </button>
        </div>
      </footer>

      {searchOpen && (
        <div className="yt-modal-overlay" role="dialog" aria-modal="true">
          <div className="yt-modal">
            <h2>Search &amp; Replace</h2>
            <p style={{ fontSize: "0.8rem", color: "#666", margin: "0 0 8px" }}>
              Searches for exact text (not a regex).
            </p>
            <label className="yt-label">Find</label>
            <input
              className="yt-input"
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              placeholder="e.g. step in the right direction"
              maxLength={200}
            />
            <label className="yt-label">Replace with</label>
            <input
              className="yt-input"
              value={searchRepl}
              onChange={(e) => setSearchRepl(e.target.value)}
              maxLength={200}
            />
            <label className="yt-label">Apply to</label>
            <select
              className="yt-input"
              value={searchField}
              onChange={(e) => setSearchField(e.target.value)}
            >
              <option value="both">Original + translation</option>
              <option value="source">Original only</option>
              <option value="translation">Translation only</option>
            </select>
            <div className="yt-modal-actions">
              <button type="button" className="yt-btn-secondary" onClick={() => setSearchOpen(false)}>
                Cancel
              </button>
              <button type="button" className="yt-btn-primary" onClick={applySearchReplace}>
                Replace all
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
