import React, { useEffect, useState, useMemo } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import VideoPlayer from "../components/VideoPlayer";
import ProcessingStatus from "../components/ProcessingStatus";
import LanguageSelector from "../components/LanguageSelector";
import { API_BASE } from "../App";

const RECENT_KEY = "youtube-translator-recent-v2";

function extractVideoId(url) {
  if (!url) return null;
  try {
    const u = new URL(url);
    if (u.hostname.endsWith("youtu.be")) {
      return u.pathname.slice(1).split("?")[0] || null;
    }
    if (u.hostname.includes("youtube.com")) {
      if (u.searchParams.has("v")) return u.searchParams.get("v");
      const embed = u.pathname.match(/\/embed\/([^/?]+)/);
      if (embed) return embed[1];
      // YouTube Shorts: /shorts/VIDEO_ID
      const shorts = u.pathname.match(/\/shorts\/([^/?]+)/);
      if (shorts) return shorts[1];
      // Legacy: /v/VIDEO_ID
      const vpath = u.pathname.match(/\/v\/([^/?]+)/);
      if (vpath) return vpath[1];
    }
  } catch (_) {}
  return null;
}

function Player() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const youtubeUrl = searchParams.get("url") || "";
  const targetLang = searchParams.get("lang") || "en";

  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [processError, setProcessError] = useState(null);
  const [languages, setLanguages] = useState([]);
  const [pendingLang, setPendingLang] = useState(targetLang);
  const [retryToken, setRetryToken] = useState(0);

  const videoId = useMemo(() => extractVideoId(youtubeUrl), [youtubeUrl]);

  useEffect(() => {
    setPendingLang(targetLang);
  }, [targetLang]);

  useEffect(() => {
    fetch(`${API_BASE}/api/languages`)
      .then((r) => r.json())
      .then(setLanguages)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!youtubeUrl) {
      navigate("/", { replace: true });
      return;
    }

    let cancelled = false;
    setProcessError(null);
    setResult(null);
    setJobId(null);

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/process-video`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            youtube_url: youtubeUrl,
            target_language: targetLang,
            include_transcripts: true, // server always writes .txt; URLs returned in result
          }),
        });
        const data = await res.json();
        if (cancelled) return;
        if (data.cached) {
          setResult({ ...data, youtube_url: youtubeUrl });
          updateRecentTitle(data.title, targetLang);
          return;
        }
        if (data.job_id) {
          setJobId(data.job_id);
        }
      } catch (e) {
        if (!cancelled) {
          console.error(e);
          setProcessError("Failed to reach server. Is the backend running on port 8000?");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [youtubeUrl, targetLang, retryToken, navigate]);

  const updateRecentTitle = (title, lang) => {
    if (!title) return;
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      const list = raw ? JSON.parse(raw) : [];
      const next = list.map((e) =>
        e.url === youtubeUrl ? { ...e, title, language: lang } : e
      );
      localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    } catch (_) {}
  };

  const handleResult = (data) => {
    setResult({ ...data, youtube_url: youtubeUrl, _ts: Date.now() });
    setJobId(null);
    setProcessError(null);
    updateRecentTitle(data.title, targetLang);
  };

  const handleStatusError = (err) => {
    if (err) setProcessError(err);
    else setProcessError(null);
  };

  const handleRetry = () => {
    setProcessError(null);
    setRetryToken((t) => t + 1);
  };

  const handleApplyLanguage = () => {
    const q = new URLSearchParams({ url: youtubeUrl, lang: pendingLang });
    navigate(`/player?${q.toString()}`, { replace: true });
  };

  // Cache-bust the audio URL so the browser never serves a stale MP3 after
  // a re-dub.  Only added for fresh pipeline results (not already-busted URLs).
  const fullAudioUrl = result?.audio_url
    ? `${API_BASE}${result.audio_url}${result.audio_url.includes("?") ? "" : `?v=${result._ts || ""}`}`
    : null;
  const originalTxtUrl = result?.original_txt_url
    ? `${API_BASE}${result.original_txt_url}`
    : result?.cache_key
      ? `${API_BASE}/api/transcript/${result.cache_key}/original.txt`
      : null;
  const translatedTxtUrl = result?.translated_txt_url
    ? `${API_BASE}${result.translated_txt_url}`
    : result?.cache_key
      ? `${API_BASE}/api/transcript/${result.cache_key}/translated.txt`
      : null;

  const langLabel = (code) => languages.find((l) => l.code === code)?.name || code;

  if (!youtubeUrl) return null;

  return (
    <div className="yt-player-page">
      <div className="yt-player-inner">
        <button type="button" onClick={() => navigate("/")} className="yt-back-btn">
          ← Back to Home
        </button>

        <div className="yt-player-layout">
          <div className="yt-player-main">
            {!videoId && youtubeUrl && (
              <p className="yt-video-id-missing" role="alert">
                Could not read a YouTube video ID from this link. Use a normal watch URL (
                <code>youtube.com/watch?v=…</code>), Shorts (<code>youtube.com/shorts/…</code>), or{" "}
                <code>youtu.be/…</code>
              </p>
            )}
            <VideoPlayer
              videoId={videoId}
              dubbedAudioUrl={fullAudioUrl}
              subtitles={result?.subtitles}
              dubSync={result?.dub_sync}
            />
          </div>

          <aside className="yt-player-sidebar">
            <h1 className="yt-video-title">
              {result?.title || (jobId ? "Processing…" : "Loading…")}
            </h1>

            <div className="yt-badge-row">
              {result?.source_language && (
                <span className="yt-badge yt-badge--detected">
                  Detected: {langLabel(result.source_language)} ({result.source_language})
                </span>
              )}
              <span className="yt-badge yt-badge--target">
                Target: {langLabel(result?.target_language || targetLang)}
              </span>
              {(result?.speaker_routing_gender || result?.speaker_gender) && (
                <span className="yt-badge yt-badge--gender" title="From voice reference (pitch-based)">
                  Speaker voice:{" "}
                  <strong>
                    {result.speaker_routing_gender || result.speaker_gender}
                  </strong>
                  {result.speaker_gender_confidence != null &&
                    result.speaker_gender_confidence !== "" && (
                      <>
                        {" "}
                        ({Math.round(Number(result.speaker_gender_confidence) * 100)}% conf.)
                      </>
                    )}
                </span>
              )}
            </div>

            <div className="yt-row-lang">
              <LanguageSelector
                languages={languages}
                value={pendingLang}
                onChange={setPendingLang}
                disabled={!!jobId}
              />
              <button
                type="button"
                onClick={handleApplyLanguage}
                disabled={!!jobId || pendingLang === targetLang}
                className="yt-btn-secondary"
              >
                Apply language
              </button>
            </div>

            <p className="yt-sidebar-hint">
              Source text is saved after speech detection; translation is saved before TTS. The dubbed MP3
              is generated from the translated lines (same as <strong>translated.txt</strong>). YouTube plays
              muted; only the dub is audible.
            </p>

            {jobId && (
              <ProcessingStatus
                jobId={jobId}
                targetLanguage={targetLang}
                onResult={handleResult}
                onError={handleStatusError}
                onRetry={handleRetry}
              />
            )}

            {processError && !jobId && (
              <div className="yt-error-box">
                <p>{processError}</p>
                <button type="button" onClick={handleRetry} className="yt-btn-danger">
                  Try Again
                </button>
              </div>
            )}

            {result && !jobId && (
              <div className="yt-download-stack">
                {result.cache_key && (
                  <>
                    <Link
                      className="yt-btn-editor"
                      to={`/translation-editor?${new URLSearchParams({
                        cache_key: result.cache_key,
                        url: youtubeUrl,
                        lang: result.target_language || targetLang,
                        ...(result.audio_url ? { audio: result.audio_url } : {}),
                      }).toString()}`}
                    >
                      ✏️ Translation Editor
                    </Link>
                    <Link
                      className="yt-btn-editor yt-btn-editor--secondary"
                      to={`/editor?${new URLSearchParams({
                        cache_key: result.cache_key,
                        url: youtubeUrl,
                        lang: result.target_language || targetLang,
                      }).toString()}`}
                    >
                      Open transcript editor
                    </Link>
                  </>
                )}
                <a href={fullAudioUrl} download className="yt-btn-download">
                  Download dubbed audio
                </a>
                {originalTxtUrl && translatedTxtUrl && (
                  <div className="yt-transcript-dl">
                    <span className="yt-transcript-dl-label">Transcript files (cache)</span>
                    <a
                      href={originalTxtUrl}
                      download
                      className="yt-btn-download yt-btn-download--secondary"
                    >
                      Original (detected language)
                    </a>
                    <a
                      href={translatedTxtUrl}
                      download
                      className="yt-btn-download yt-btn-download--secondary"
                    >
                      Translated ({langLabel(result.target_language || targetLang)})
                    </a>
                  </div>
                )}
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}

export default Player;
