/**
 * TranslationEditor — Video-synced correction editor.
 *
 * Layout  : video LEFT (50%)  |  two textareas RIGHT (50%)
 *
 * Left panel has two audio modes:
 *   Original   — YouTube plays normally with its own audio
 *   Translated — YouTube is muted; dubbed MP3 follows ``dub_sync`` (video time ≠ file time)
 *
 * "Play this segment" buttons let the user audition just one segment
 * in either mode so they can verify a correction immediately.
 *
 * Workflow:
 *   1. Play the video in Original or Translated mode.
 *   2. Both textareas update in real time to match the current segment.
 *   3. Pause, fix source or translation text, then Replay to verify.
 *   4. Save — persists JSON then regenerates dubbed MP3 on the server so Translated mode matches edits.
 */
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { API_BASE } from "../App";
import { videoTimeToDubAudioTime } from "../utils/dubAudioSync";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${String(m).padStart(2, "0")}:${String(Math.floor(r)).padStart(2, "0")}`;
}

function extractVideoId(url) {
  if (!url) return null;
  try {
    const u = new URL(url);
    if (u.hostname.endsWith("youtu.be")) return u.pathname.slice(1).split("?")[0] || null;
    if (u.hostname.includes("youtube.com")) {
      if (u.searchParams.has("v")) return u.searchParams.get("v");
      const shorts = u.pathname.match(/\/shorts\/([^/?]+)/);
      if (shorts) return shorts[1];
    }
  } catch (_) {}
  return null;
}

function segAtTime(segments, time) {
  for (let i = segments.length - 1; i >= 0; i--) {
    if (time >= (segments[i].start || 0) - 0.1) return i;
  }
  return 0;
}

/** Dub file duration clamp (same idea as main VideoPlayer). */
function clampDubAudioTime(wantSec, audioEl) {
  if (!audioEl) return wantSec;
  const dur = audioEl.duration;
  if (typeof dur === "number" && dur > 0 && Number.isFinite(dur)) {
    return Math.min(wantSec, Math.max(0, dur - 0.02));
  }
  return wantSec;
}

/** YouTube clock → position in dubbed MP3 (sequential TTS layout from server ``dub_sync``). */
function ytTimeToDubTime(ytSec, dubSync) {
  if (dubSync != null && dubSync.length > 0) {
    return videoTimeToDubAudioTime(ytSec, dubSync);
  }
  return ytSec;
}

const DUB_DRIFT_THRESHOLD = 0.35;

// ---------------------------------------------------------------------------
// YouTube IFrame API loader (singleton)
// ---------------------------------------------------------------------------

let _ytApiReady = false;
let _ytCallbacks = [];

function loadYTApi(cb) {
  if (_ytApiReady) { cb(); return; }
  _ytCallbacks.push(cb);
  if (document.getElementById("yt-iframe-api")) return;
  window.onYouTubeIframeAPIReady = () => {
    _ytApiReady = true;
    _ytCallbacks.forEach((fn) => fn());
    _ytCallbacks = [];
  };
  const tag = document.createElement("script");
  tag.id = "yt-iframe-api";
  tag.src = "https://www.youtube.com/iframe_api";
  document.head.appendChild(tag);
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function TranslationEditor() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const cacheKey   = (searchParams.get("cache_key") || "").trim().toLowerCase();
  const youtubeUrl  = searchParams.get("url") || "";
  const targetLang  = searchParams.get("lang") || "en";
  /** Relative `/api/audio/...` from Player link, or empty until we default from cache_key. */
  const audioPathQuery = (searchParams.get("audio") || "").trim();
  const videoId     = useMemo(() => extractVideoId(youtubeUrl), [youtubeUrl]);
  const [dubRelPath, setDubRelPath] = useState("");
  const [dubBust, setDubBust] = useState(0);
  const dubbedAudioUrl = useMemo(() => {
    if (!dubRelPath) return null;
    const sep = dubRelPath.includes("?") ? "&" : "?";
    return `${API_BASE}${dubRelPath}${sep}dubv=${dubBust}`;
  }, [dubRelPath, dubBust]);

  // ── Server data ──────────────────────────────────────────────────────────
  const [segments, setSegments] = useState([]);
  const [sourceLang, setSourceLang] = useState("en");
  const [title, setTitle] = useState("");
  const [loadError, setLoadError] = useState(null);
  /** From ``{cache_key}_meta.json`` — maps video timeline to sequential dub audio (required for correct sync). */
  const [dubSync, setDubSync] = useState([]);

  // ── Edit state ────────────────────────────────────────────────────────────
  const [sourceEdits, setSourceEdits] = useState({});
  const [translationEdits, setTranslationEdits] = useState({});

  // ── Playback state ────────────────────────────────────────────────────────
  const [activeIdx, setActiveIdx]   = useState(0);
  const [isPlaying, setIsPlaying]   = useState(false);
  const [videoEnded, setVideoEnded] = useState(false);
  // "original" → YouTube audio on  |  "translated" → YouTube muted, dubbed MP3 plays
  const [playMode, setPlayMode] = useState("original");

  // ── Save state ────────────────────────────────────────────────────────────
  const [saveStatus, setSaveStatus] = useState(null); // saving | redubbing | saved | warn | error
  const [saveDetail, setSaveDetail] = useState(null); // extra message (e.g. re-dub failed after text saved)
  const [totalSaved, setTotalSaved] = useState(0);
  const redubBusy = saveStatus === "redubbing";

  // ── Refs ──────────────────────────────────────────────────────────────────
  const playerRef          = useRef(null);
  const playerContainerRef = useRef(null);
  const audioRef           = useRef(null);   // <audio> for dubbed MP3
  const pollRef            = useRef(null);
  const segmentsRef        = useRef([]);     // always-current, no stale closure
  const playModeRef        = useRef("original");
  const dubSyncRef         = useRef([]);
  const autoStopRef        = useRef(null);   // stop playback at this time (segment preview)

  // ── Derived ───────────────────────────────────────────────────────────────
  const seg = segments[activeIdx] || null;

  const currentSource      = sourceEdits[activeIdx]      ?? seg?.text      ?? seg?.original ?? "";
  const currentTranslation = translationEdits[activeIdx] ?? seg?.translated_text ?? "";

  const isSourceEdited      = sourceEdits[activeIdx]      !== undefined &&
                              sourceEdits[activeIdx]      !== (seg?.text || seg?.original || "");
  const isTranslationEdited = translationEdits[activeIdx] !== undefined &&
                              translationEdits[activeIdx] !== (seg?.translated_text || "");

  const dirtyCount = useMemo(() => {
    return segments.reduce((acc, s, i) => {
      const srcDirty = sourceEdits[i]      !== undefined && sourceEdits[i]      !== (s.text || s.original || "");
      const tgtDirty = translationEdits[i] !== undefined && translationEdits[i] !== (s.translated_text || "");
      return acc + (srcDirty || tgtDirty ? 1 : 0);
    }, 0);
  }, [segments, sourceEdits, translationEdits]);

  // Keep refs in sync
  useEffect(() => { segmentsRef.current = segments; }, [segments]);
  useEffect(() => { playModeRef.current = playMode; }, [playMode]);
  useEffect(() => { dubSyncRef.current = dubSync; }, [dubSync]);

  // Dubbed file URL: query param from Player, or default cache path (same basename the pipeline uses).
  useEffect(() => {
    if (!cacheKey || cacheKey.length !== 32) {
      setDubRelPath("");
      return;
    }
    setDubRelPath(audioPathQuery || `/api/audio/${cacheKey}.mp3`);
    setDubBust(0);
  }, [cacheKey, audioPathQuery]);

  // ── Load segments ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (!cacheKey || cacheKey.length !== 32) {
      setLoadError("Missing or invalid cache_key — navigate here from the Player page.");
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [segsRes, metaRes] = await Promise.all([
          fetch(`${API_BASE}/api/subtitles/${cacheKey}.json`),
          fetch(`${API_BASE}/api/subtitles/${cacheKey}_meta.json`).catch(() => null),
        ]);
        if (!segsRes.ok) throw new Error(`Segments not found (${segsRes.status})`);
        const segsData = await segsRes.json();
        if (!Array.isArray(segsData)) throw new Error("Unexpected segment format");
        if (cancelled) return;
        setSegments(segsData);

        // Pre-populate already-saved corrections
        const corrRes = await fetch(`${API_BASE}/api/corrections/video/${cacheKey}`).catch(() => null);
        if (corrRes?.ok && !cancelled) {
          const corrData = await corrRes.json();
          const srcMap = {}, tgtMap = {};
          Object.entries(corrData.corrections || {}).forEach(([k, v]) => {
            const idx = parseInt(k, 10);
            if (v.corrected_translation) tgtMap[idx] = v.corrected_translation;
            if (v.source_text && v.original_source_text) srcMap[idx] = v.source_text;
          });
          if (Object.keys(tgtMap).length) setTranslationEdits(tgtMap);
          if (Object.keys(srcMap).length) setSourceEdits(srcMap);
        }

        if (metaRes?.ok && !cancelled) {
          const meta = await metaRes.json();
          setTitle(meta.title || "");
          setSourceLang(meta.source_language || "en");
          if (Array.isArray(meta.dub_sync)) setDubSync(meta.dub_sync);
        }
      } catch (err) {
        if (!cancelled) setLoadError(err.message);
      }
    })();
    return () => { cancelled = true; };
  }, [cacheKey]);

  // ── YouTube Player ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (!videoId) return;
    loadYTApi(() => {
      if (!playerContainerRef.current || playerRef.current) return;
      playerRef.current = new window.YT.Player(playerContainerRef.current, {
        videoId,
        width: "100%",
        height: "100%",
        playerVars: { rel: 0, modestbranding: 1 },
        events: {
          onReady: () => startPolling(),
          onStateChange: (e) => {
            const playing = e.data === 1;
            setIsPlaying(playing);
            if (e.data === 0) { // ended
              setVideoEnded(true);
              audioRef.current?.pause?.();
            }
            if (playing) setVideoEnded(false);
          },
        },
      });
    });
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (playerRef.current?.destroy) {
        playerRef.current.destroy();
        playerRef.current = null;
      }
    };
  }, [videoId]);

  // ── Polling: segment tracking + audio sync + auto-stop ───────────────────

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(() => {
      const player = playerRef.current;
      if (!player?.getCurrentTime) return;
      try {
        const t = player.getCurrentTime();
        if (typeof t !== "number" || isNaN(t)) return;

        // Auto-stop at segment end (used by "Play this segment")
        if (autoStopRef.current !== null && t >= autoStopRef.current) {
          player.pauseVideo?.();
          audioRef.current?.pause?.();
          autoStopRef.current = null;
        }

        // Keep dubbed audio in sync (dub file is sequential TTS — not 1:1 with YouTube seconds)
        const audio = audioRef.current;
        const ytPlaying = player.getPlayerState?.() === window.YT?.PlayerState?.PLAYING;
        if (audio && playModeRef.current === "translated" && ytPlaying) {
          const want = ytTimeToDubTime(t, dubSyncRef.current);
          const safe = clampDubAudioTime(want, audio);
          if (audio.paused || audio.ended) {
            audio.currentTime = safe;
            audio.play().catch(() => {});
          } else if (Math.abs(audio.currentTime - safe) > DUB_DRIFT_THRESHOLD) {
            audio.currentTime = safe;
          }
        }

        // Update active segment display
        setActiveIdx(segAtTime(segmentsRef.current, t));
      } catch (_) {}
    }, 100);
  }, []);

  useEffect(() => {
    if (segments.length) startPolling();
  }, [segments.length, startPolling]);

  // ── Audio mode helpers ────────────────────────────────────────────────────

  /** Switch full-video mode (original vs translated) */
  const switchMode = useCallback((mode) => {
    if (mode === "translated" && redubBusy) return;
    const player = playerRef.current;
    const audio  = audioRef.current;
    setPlayMode(mode);
    if (mode === "translated") {
      player?.mute?.();
      if (audio) {
        const yt = player?.getCurrentTime?.() || 0;
        const want = ytTimeToDubTime(yt, dubSyncRef.current);
        audio.currentTime = clampDubAudioTime(want, audio);
        if (isPlaying) audio.play().catch(() => {});
      }
    } else {
      player?.unMute?.();
      audio?.pause?.();
    }
  }, [isPlaying, redubBusy]);

  /** Play just one segment then auto-stop */
  const playSegment = useCallback((mode) => {
    if (!seg) return;
    if (mode === "translated" && redubBusy) return;
    const player = playerRef.current;
    const audio  = audioRef.current;
    const start  = seg.start || 0;
    const end    = (seg.end  || start + 5) + 0.15; // small buffer

    autoStopRef.current = end;
    player?.seekTo?.(start, true);

    if (mode === "original") {
      player?.unMute?.();
      audio?.pause?.();
      setPlayMode("original");
    } else {
      player?.mute?.();
      setPlayMode("translated");
      if (audio) {
        const want = ytTimeToDubTime(start, dubSyncRef.current);
        audio.currentTime = clampDubAudioTime(want, audio);
        audio.play().catch(() => {});
      }
    }
    player?.playVideo?.();
  }, [seg, redubBusy]);

  // ── Main play / pause ─────────────────────────────────────────────────────

  const handlePlayPause = useCallback(() => {
    if (playMode === "translated" && redubBusy) return;
    const player = playerRef.current;
    const audio  = audioRef.current;
    autoStopRef.current = null; // cancel segment auto-stop
    if (isPlaying) {
      player?.pauseVideo?.();
      audio?.pause?.();
    } else {
      player?.playVideo?.();
      if (playMode === "translated" && audio) {
        const yt = player?.getCurrentTime?.() || 0;
        const want = ytTimeToDubTime(yt, dubSyncRef.current);
        audio.currentTime = clampDubAudioTime(want, audio);
        audio.play().catch(() => {});
      }
    }
  }, [isPlaying, playMode, redubBusy]);

  const handlePrev = useCallback(() => {
    autoStopRef.current = null;
    const idx = Math.max(0, activeIdx - 1);
    const t = segmentsRef.current[idx]?.start || 0;
    playerRef.current?.seekTo?.(t, true);
    playerRef.current?.pauseVideo?.();
    audioRef.current?.pause?.();
    setActiveIdx(idx);
  }, [activeIdx]);

  const handleNext = useCallback(() => {
    autoStopRef.current = null;
    const idx = Math.min(segmentsRef.current.length - 1, activeIdx + 1);
    const t = segmentsRef.current[idx]?.start || 0;
    playerRef.current?.seekTo?.(t, true);
    playerRef.current?.pauseVideo?.();
    audioRef.current?.pause?.();
    setActiveIdx(idx);
  }, [activeIdx]);

  const handleReplay = useCallback(() => {
    if (!seg) return;
    if (playMode === "translated" && redubBusy) return;
    autoStopRef.current = null;
    playerRef.current?.seekTo?.(seg.start, true);
    playerRef.current?.playVideo?.();
    if (playMode === "translated" && audioRef.current) {
      const a = audioRef.current;
      const st = seg.start || 0;
      const want = ytTimeToDubTime(st, dubSyncRef.current);
      a.currentTime = clampDubAudioTime(want, a);
      a.play().catch(() => {});
    }
  }, [seg, playMode, redubBusy]);

  // ── Text editing ──────────────────────────────────────────────────────────

  const handleSourceChange = useCallback((value) => {
    setSourceEdits((prev) => ({ ...prev, [activeIdx]: value }));
  }, [activeIdx]);

  const handleTranslationChange = useCallback((value) => {
    setTranslationEdits((prev) => ({ ...prev, [activeIdx]: value }));
  }, [activeIdx]);

  const handleDiscardCurrent = useCallback(() => {
    setSourceEdits((prev)      => { const n = { ...prev }; delete n[activeIdx]; return n; });
    setTranslationEdits((prev) => { const n = { ...prev }; delete n[activeIdx]; return n; });
  }, [activeIdx]);

  // ── Save ──────────────────────────────────────────────────────────────────

  const handleSaveAll = useCallback(async () => {
    const toSave = segments
      .map((s, i) => {
        const origSrc = s.text || s.original || "";
        const origTgt = s.translated_text || "";
        const newSrc  = sourceEdits[i]      ?? origSrc;
        const newTgt  = translationEdits[i] ?? origTgt;
        if (newSrc === origSrc && newTgt === origTgt) return null;
        return {
          cache_key: cacheKey,
          segment_index: i,
          segment_id: `segment_${i}`,
          source_text: newSrc,
          ...(newSrc !== origSrc ? { original_source_text: origSrc } : {}),
          incorrect_translation: origTgt,
          corrected_translation: newTgt,
          source_lang: sourceLang,
          target_lang: targetLang,
          video_id: videoId || "",
        };
      })
      .filter(Boolean);

    if (!toSave.length) return;
    playerRef.current?.pauseVideo?.();
    audioRef.current?.pause?.();
    if (playModeRef.current === "translated") {
      playerRef.current?.unMute?.();
      setPlayMode("original");
    }
    setSaveDetail(null);
    setSaveStatus("saving");
    try {
      const res = await fetch(`${API_BASE}/api/corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrections: toSave,
          retranslate_source_changes: true,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reload = await fetch(`${API_BASE}/api/subtitles/${cacheKey}.json`);
      if (reload.ok) {
        const fresh = await reload.json();
        if (Array.isArray(fresh)) setSegments(fresh);
      } else {
        setSegments((prev) => {
          const next = [...prev];
          toSave.forEach((c) => {
            next[c.segment_index] = {
              ...next[c.segment_index],
              text: c.source_text,
              translated_text: c.corrected_translation,
              corrected: true,
            };
          });
          return next;
        });
      }
      setTotalSaved((n) => n + toSave.length);
      setSourceEdits({});
      setTranslationEdits({});
    } catch (_) {
      setSaveStatus("error");
      setTimeout(() => setSaveStatus(null), 5000);
      return;
    }

    setSaveStatus("redubbing");
    try {
      const rd = await fetch(`${API_BASE}/api/video/cache/${cacheKey}/redub`, {
        method: "POST",
      });
      const data = await rd.json().catch(() => ({}));
      if (!rd.ok) {
        const d = data?.detail;
        let msg = `HTTP ${rd.status}`;
        if (typeof d === "string") msg = d;
        else if (Array.isArray(d)) msg = d.map((x) => x?.msg || JSON.stringify(x)).join("; ");
        throw new Error(msg);
      }
      if (data.audio_url) setDubRelPath(data.audio_url);
      if (Array.isArray(data.dub_sync)) setDubSync(data.dub_sync);
      setDubBust((b) => b + 1);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus(null), 5000);
    } catch (e) {
      setSaveStatus("warn");
      setSaveDetail(
        `Subtitles are saved, but dubbed audio could not be rebuilt: ${e?.message || e}. ` +
          "Try again from the transcript editor (Re-dub audio), or check server logs."
      );
      setTimeout(() => {
        setSaveStatus(null);
        setSaveDetail(null);
      }, 12000);
    }
  }, [segments, sourceEdits, translationEdits, cacheKey, sourceLang, targetLang, videoId]);

  // ── Unsaved-changes guard ─────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e) => { if (dirtyCount > 0) { e.preventDefault(); e.returnValue = ""; } };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirtyCount]);

  // ── Language label ────────────────────────────────────────────────────────

  const langLabel = (code) =>
    ({ en: "English", hi: "Hindi", es: "Spanish", fr: "French", de: "German",
       ja: "Japanese", ko: "Korean", zh: "Chinese", ar: "Arabic", pt: "Portuguese",
       ru: "Russian", it: "Italian", nl: "Dutch", tr: "Turkish", pl: "Polish",
       sv: "Swedish" }[code] || (code || "").toUpperCase());

  // ── Render ────────────────────────────────────────────────────────────────

  if (loadError) {
    return (
      <div className="te-page">
        <div className="te-error-box">
          <p>{loadError}</p>
          <button className="yt-btn-secondary" onClick={() => navigate(-1)}>← Go back</button>
        </div>
      </div>
    );
  }

  const segCount = segments.length;
  const hasPrev  = activeIdx > 0;
  const hasNext  = activeIdx < segCount - 1;
  const segEnd   = seg?.end ?? (seg?.start ? seg.start + 5 : 0);

  return (
    <div className="te-page">

      {/* Hidden audio element for dubbed MP3 */}
      {dubbedAudioUrl && (
        <audio
          key={`dub-${dubBust}`}
          ref={audioRef}
          src={dubbedAudioUrl}
          preload="auto"
          style={{ display: "none" }}
        />
      )}

      {/* ── Header ── */}
      <header className="te-header">
        <button className="yt-back-btn" onClick={() => navigate(-1)}>← Back</button>
        <div className="te-header-center">
          <h1 className="te-title">{title || "Translation Editor"}</h1>
          <div className="te-lang-badges">
            <span className="yt-badge yt-badge--detected">{langLabel(sourceLang)}</span>
            <span className="te-lang-arrow">→</span>
            <span className="yt-badge yt-badge--target">{langLabel(targetLang)}</span>
          </div>
        </div>
        <div className="te-header-actions">
          {totalSaved > 0 && (
            <span className="te-correction-count">
              {totalSaved} correction{totalSaved !== 1 ? "s" : ""} saved
            </span>
          )}
          {dirtyCount > 0 && (
            <span className="te-dirty-hint">{dirtyCount} unsaved</span>
          )}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="te-body">

        {/* ════ LEFT — video + controls ════ */}
        <div className="te-video-col">

          {/* Video iframe */}
          <div className="te-video-frame">
            {videoId
              ? <div className="te-video-inner" ref={playerContainerRef} />
              : <div className="te-no-video"><p>No video URL provided.</p></div>
            }
          </div>

          {/* ── Audio mode switcher ── */}
          <div className="te-video-controls">

            <div className="te-mode-section">
              <span className="te-mode-label">Audio mode</span>
              <div className="te-mode-btns">
                <button
                  className={`te-mode-btn${playMode === "original" ? " te-mode-btn--active" : ""}`}
                  onClick={() => switchMode("original")}
                  title="Play with original YouTube audio"
                >
                  🎬 Original
                </button>
                <button
                  className={`te-mode-btn${playMode === "translated" ? " te-mode-btn--active" : ""}`}
                  onClick={() => switchMode("translated")}
                  disabled={!dubbedAudioUrl || redubBusy}
                  title={
                    redubBusy
                      ? "Regenerating dubbed audio after save…"
                      : dubbedAudioUrl
                        ? "Play with dubbed translated audio"
                        : "Dubbed audio path not set"
                  }
                >
                  🌐 Translated
                </button>
              </div>
              {!dubbedAudioUrl && (
                <p className="te-mode-hint">
                  Dubbed file path missing — save once to build audio from subtitles, or open from the Player page.
                </p>
              )}
            </div>

            {/* ── Segment preview ── */}
            {seg && segCount > 0 && (
              <div className="te-seg-preview">
                <span className="te-seg-preview-label">
                  Segment {activeIdx + 1}/{segCount}
                  <span className="te-seg-preview-time"> {fmtTime(seg.start)}–{fmtTime(segEnd)}</span>
                </span>
                <div className="te-seg-preview-btns">
                  <button
                    className="te-seg-play-btn"
                    onClick={() => playSegment("original")}
                    title="Play just this segment in original audio"
                  >
                    ▶ Original
                  </button>
                  <button
                    className="te-seg-play-btn te-seg-play-btn--translated"
                    onClick={() => playSegment("translated")}
                    disabled={!dubbedAudioUrl || redubBusy}
                    title={
                      redubBusy
                        ? "Wait for audio to finish rebuilding"
                        : dubbedAudioUrl
                          ? "Play just this segment in dubbed audio"
                          : "Dubbed audio not available"
                    }
                  >
                    ▶ Translated
                  </button>
                </div>
              </div>
            )}

          </div>{/* end te-video-controls */}
        </div>{/* end te-video-col */}

        {/* ════ RIGHT — editor ════ */}
        <div className="te-edit-col">

          {/* Segment counter */}
          <div className="te-seg-meta">
            <span className="te-seg-counter">
              Segment <strong>{segCount ? activeIdx + 1 : "—"}</strong> of <strong>{segCount || "—"}</strong>
            </span>
            {seg && (
              <span className="te-seg-time">{fmtTime(seg.start)} – {fmtTime(segEnd)}</span>
            )}
          </div>

          {/* Source textarea */}
          <div className="te-field">
            <div className="te-field-label">
              <span>Source — {langLabel(sourceLang)}</span>
              {isSourceEdited && <span className="te-badge-dirty">Edited</span>}
            </div>
            <textarea
              className={`te-textarea te-textarea--source${isSourceEdited ? " te-textarea--edited" : ""}`}
              value={currentSource}
              onChange={(e) => handleSourceChange(e.target.value)}
              placeholder="(no source text)"
              rows={3}
              aria-label="Source text"
              disabled={!seg}
            />
          </div>

          {/* Translation textarea */}
          <div className="te-field">
            <div className="te-field-label">
              <span>Translation — {langLabel(targetLang)}</span>
              {isTranslationEdited && <span className="te-badge-dirty">Edited</span>}
            </div>
            <textarea
              className={`te-textarea${isTranslationEdited ? " te-textarea--edited" : ""}`}
              value={currentTranslation}
              onChange={(e) => handleTranslationChange(e.target.value)}
              placeholder="(no translation)"
              rows={3}
              aria-label="Translation text"
              disabled={!seg}
            />
          </div>

          {/* Discard current */}
          {(isSourceEdited || isTranslationEdited) && (
            <button className="te-discard-btn" onClick={handleDiscardCurrent}>
              ✕ Discard edits for this segment
            </button>
          )}

          {/* Playback controls */}
          <div className="te-controls">
            <button className="te-ctrl-btn" onClick={handlePrev} disabled={!hasPrev} title="Previous segment">
              ◀ Prev
            </button>
            <button
              className="te-ctrl-btn te-ctrl-btn--play"
              onClick={handlePlayPause}
              disabled={!videoId}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button className="te-ctrl-btn" onClick={handleNext} disabled={!hasNext} title="Next segment">
              Next ▶
            </button>
          </div>

          <button
            className="te-replay-btn"
            onClick={handleReplay}
            disabled={!videoId || !seg || (playMode === "translated" && redubBusy)}
          >
            ↺ Replay this segment
          </button>

          {/* Save panel */}
          <div className="te-save-panel">
            {videoEnded && dirtyCount === 0 && totalSaved === 0 && (
              <p className="te-save-hint">Video finished — no corrections made.</p>
            )}
            {(dirtyCount > 0 || videoEnded) && (
              <button
                className={`te-save-btn${dirtyCount === 0 ? " te-save-btn--dim" : ""}`}
                onClick={handleSaveAll}
                disabled={dirtyCount === 0 || saveStatus === "saving" || saveStatus === "redubbing"}
              >
                {saveStatus === "saving"
                  ? "Saving…"
                  : saveStatus === "redubbing"
                    ? "Updating audio…"
                    : saveStatus === "saved"
                      ? "✓ Saved!"
                      : saveStatus === "error"
                        ? "Error — retry"
                        : dirtyCount > 0
                          ? `Save ${dirtyCount} correction${dirtyCount !== 1 ? "s" : ""}`
                          : "All saved"}
              </button>
            )}
            {saveStatus === "redubbing" && (
              <p className="te-save-ok">
                Regenerating dubbed MP3 from your saved lines — this can take a minute. Translated mode is paused until it finishes.
              </p>
            )}
            {saveStatus === "saved" && (
              <p className="te-save-ok">
                Subtitles and dubbed audio are updated. Switch to <strong>Translated</strong> and press Play to hear your corrections.
              </p>
            )}
            {saveStatus === "warn" && saveDetail && (
              <p className="te-save-warn" role="alert">{saveDetail}</p>
            )}
          </div>

          {/* Instructions */}
          <div className="te-instructions">
            <p>How to use</p>
            <ol>
              <li>Pick <strong>Original</strong> or <strong>Translated</strong> audio mode on the left.</li>
              <li>Press <strong>Play</strong> — textareas update in real time.</li>
              <li>Spot a mistake → <strong>Pause</strong> → fix the text.</li>
              <li>Use <strong>▶ Original / ▶ Translated</strong> (left panel) to audition just the current segment in either mode.</li>
              <li>Press <strong>↺ Replay</strong> to hear the whole segment again.</li>
              <li>
                <strong>Save</strong> — writes subtitles, then rebuilds the dubbed track so <strong>Translated</strong> playback matches your edits (wait for &quot;Updating audio&quot; to finish).
              </li>
            </ol>
          </div>

        </div>{/* end te-edit-col */}
      </div>{/* end te-body */}

      {/* ── Sticky footer ── */}
      {dirtyCount > 0 && (
        <div className="te-footer-bar">
          <span>{dirtyCount} unsaved correction{dirtyCount !== 1 ? "s" : ""}</span>
          <div className="te-footer-actions">
            <button
              className="yt-btn-secondary"
              onClick={() => { setSourceEdits({}); setTranslationEdits({}); }}
            >
              Discard all
            </button>
            <button
              className="te-save-btn"
              onClick={handleSaveAll}
              disabled={saveStatus === "saving" || saveStatus === "redubbing"}
            >
              {saveStatus === "redubbing"
                ? "Updating audio…"
                : saveStatus === "saving"
                  ? "Saving…"
                  : "Save All Corrections"}
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
