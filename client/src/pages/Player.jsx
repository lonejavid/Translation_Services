import React, { useEffect, useState, useMemo } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import VideoPlayer from "../components/VideoPlayer";
import ProcessingStatus from "../components/ProcessingStatus";
import LanguageSelector from "../components/LanguageSelector";
import { API_BASE } from "../App";

const RECENT_KEY = "youtube-translator-recent-v2";

// Module-level cache — survives React unmount/remount on navigation so the
// Player never loses its result when the user goes to TranslationEditor and back.
const _playerCache = { key: null, result: null };

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
      const shorts = u.pathname.match(/\/shorts\/([^/?]+)/);
      if (shorts) return shorts[1];
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

  // Initialize from module cache so back-navigation restores the result instantly
  const [result, setResult] = useState(() => {
    const key = `${(searchParams.get("url") || "")}||${(searchParams.get("lang") || "en")}`;
    return _playerCache.key === key ? _playerCache.result : null;
  });
  const [jobId, setJobId] = useState(null);
  const [processError, setProcessError] = useState(null);
  const [languages, setLanguages] = useState([]);
  const [pendingLang, setPendingLang] = useState(targetLang);
  const [retryToken, setRetryToken] = useState(0);
  const [dlVideoStatus, setDlVideoStatus] = useState("idle"); // idle|preparing|done|error
  const [dlVideoError, setDlVideoError] = useState(null);
  const dlPollRef = React.useRef(null);

  const resultCacheKey = `${youtubeUrl}||${targetLang}`;
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

    // Only clear state for a genuinely new video/language or an explicit retry.
    // Back-navigation to the same URL+lang must NOT reset the result.
    const isNewVideo = _playerCache.key !== resultCacheKey || retryToken > 0;

    let cancelled = false;
    setProcessError(null);
    if (isNewVideo) {
      _playerCache.key = null;
      _playerCache.result = null;
      setResult(null);
      setJobId(null);
    }

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/process-video`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            youtube_url: youtubeUrl,
            target_language: targetLang,
            include_transcripts: true,
          }),
        });
        const data = await res.json();
        if (cancelled) return;
        if (data.cached) {
          const r = { ...data, youtube_url: youtubeUrl };
          _playerCache.key = resultCacheKey;
          _playerCache.result = r;
          setResult(r);
          updateRecentTitle(data.title, targetLang);
          return;
        }
        if (data.job_id) {
          setJobId(data.job_id);
        }
      } catch (e) {
        if (!cancelled) {
          console.error(e);
          if (!_playerCache.result) {
            setProcessError("Failed to reach server. Is the backend running on port 8000?");
          }
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [youtubeUrl, targetLang, retryToken, navigate, resultCacheKey]);

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
    const r = { ...data, youtube_url: youtubeUrl, _ts: Date.now() };
    _playerCache.key = resultCacheKey;
    _playerCache.result = r;
    setResult(r);
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

  // Stop polling when component unmounts
  React.useEffect(() => {
    return () => { if (dlPollRef.current) clearInterval(dlPollRef.current); };
  }, []);

  // Fetch the file as a blob and trigger a real download with the correct
  // filename + .mp4 extension. Using <a download> with a cross-origin URL is
  // silently ignored by Electron/Chromium — the blob URL approach is same-origin
  // so the download attribute is honoured and the file arrives as a proper .mp4.
  const triggerVideoDownload = async (cacheKey, title) => {
    setDlVideoStatus("downloading");
    try {
      const response = await fetch(`${API_BASE}/api/export/${cacheKey}/video-file`);
      if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const safeTitle = (title || "dubbed-video").replace(/[\\/:*?"<>|]/g, "").trim().slice(0, 80);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = blobUrl;
      a.download = `${safeTitle} (dubbed).mp4`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(blobUrl); document.body.removeChild(a); }, 2000);
      setDlVideoStatus("done");
    } catch (e) {
      setDlVideoStatus("error");
      setDlVideoError(e.message || "Download failed.");
    }
  };

  const handleDownloadVideo = async () => {
    if (!result?.cache_key) return;
    const ck = result.cache_key;
    const title = result.title || "dubbed-video";
    setDlVideoStatus("preparing");
    setDlVideoError(null);
    try {
      const res = await fetch(`${API_BASE}/api/export/${ck}/video`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      if (data.status === "done") {
        triggerVideoDownload(ck, title);
        return;
      }
      // Poll until the background job finishes
      dlPollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`${API_BASE}/api/export/${ck}/video-status`);
          const sd = await sr.json();
          if (sd.status === "done") {
            clearInterval(dlPollRef.current);
            dlPollRef.current = null;
            triggerVideoDownload(ck, title);
          } else if (sd.status === "error") {
            clearInterval(dlPollRef.current);
            dlPollRef.current = null;
            setDlVideoStatus("error");
            setDlVideoError(sd.error || "Export failed.");
          }
        } catch (_) {}
      }, 3000);
    } catch (e) {
      setDlVideoStatus("error");
      setDlVideoError(e.message || "Could not start export.");
    }
  };

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

        {/* Top navigation bar */}
        <nav className="yt-player-topbar">
          <button type="button" onClick={() => navigate("/")} className="yt-back-btn">
            <svg className="yt-back-icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
              <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
            </svg>
            Back to Home
          </button>
          <div className="yt-topbar-status">
            {jobId && <span className="yt-topbar-pill yt-topbar-pill--processing">Processing…</span>}
            {result && !jobId && <span className="yt-topbar-pill yt-topbar-pill--ready">Ready</span>}
          </div>
        </nav>

        <div className="yt-player-layout">

          {/* Video column */}
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

          {/* Sidebar column */}
          <aside className="yt-player-sidebar">

            {/* Video info card */}
            <div className="yt-sidebar-card yt-sidebar-card--info">
              <h1 className="yt-video-title">
                {result?.title || (jobId ? "Processing…" : "Loading…")}
              </h1>
              <div className="yt-badge-row">
                {result?.source_language && (
                  <span className="yt-badge yt-badge--detected">
                    <span className="yt-badge-dot yt-badge-dot--blue" />
                    Detected: {langLabel(result.source_language)}
                  </span>
                )}
                <span className="yt-badge yt-badge--target">
                  <span className="yt-badge-dot yt-badge-dot--green" />
                  Target: {langLabel(result?.target_language || targetLang)}
                </span>
                {(result?.speaker_routing_gender || result?.speaker_gender) && (
                  <span className="yt-badge yt-badge--gender" title="From voice reference (pitch-based)">
                    <span className="yt-badge-dot yt-badge-dot--purple" />
                    {result.speaker_routing_gender || result.speaker_gender}
                    {result.speaker_gender_confidence != null &&
                      result.speaker_gender_confidence !== "" && (
                        <> · {Math.round(Number(result.speaker_gender_confidence) * 100)}%</>
                      )}
                  </span>
                )}
              </div>
            </div>

            {/* Language Settings card */}
            <div className="yt-sidebar-card">
              <div className="yt-card-header">
                <div className="yt-card-icon-wrap">
                  <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                    <path fillRule="evenodd" d="M7 2a1 1 0 011 1v1h3a1 1 0 110 2H9.578a18.87 18.87 0 01-1.724 4.78c.29.354.596.696.914 1.026a1 1 0 11-1.44 1.389c-.188-.196-.373-.396-.554-.6a19.098 19.098 0 01-3.107 3.567 1 1 0 01-1.334-1.49 17.087 17.087 0 003.13-3.733 18.992 18.992 0 01-1.487-3.754 1 1 0 111.934-.516c.358 1.205.79 2.307 1.266 3.270A17.88 17.88 0 008.453 7H3a1 1 0 110-2h3V3a1 1 0 011-1zm6 6a1 1 0 01.894.553l2.991 5.982a.869.869 0 01.02.037l.99 1.98a1 1 0 11-1.79.895L15.383 16h-4.764l-.724 1.447a1 1 0 11-1.788-.894l.99-1.98.019-.038 2.99-5.982A1 1 0 0113 8zm-1.382 6h2.764L13 11.236 11.618 14z" clipRule="evenodd" />
                  </svg>
                </div>
                <h3 className="yt-card-title">Language</h3>
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
                  className="yt-btn-apply"
                >
                  Apply
                </button>
              </div>
              <p className="yt-sidebar-hint">
                Source text is saved after speech detection; translation is saved before TTS. The dubbed
                MP3 is generated from the translated lines. YouTube plays muted; only the dub is audible.
              </p>
            </div>

            {/* Processing status */}
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

            {/* Tools card */}
            {result && !jobId && result.cache_key && (
              <div className="yt-sidebar-card">
                <div className="yt-card-header">
                  <div className="yt-card-icon-wrap">
                    <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                      <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                    </svg>
                  </div>
                  <h3 className="yt-card-title">Edit</h3>
                </div>
                <div className="yt-tool-stack">
                  <Link
                    className="yt-btn-tool yt-btn-tool--primary"
                    to={`/translation-editor?${new URLSearchParams({
                      cache_key: result.cache_key,
                      url: youtubeUrl,
                      lang: result.target_language || targetLang,
                      ...(result.audio_url ? { audio: result.audio_url } : {}),
                    }).toString()}`}
                  >
                    <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                      <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                    </svg>
                    Translation Editor
                  </Link>
                  <Link
                    className="yt-btn-tool yt-btn-tool--secondary"
                    to={`/editor?${new URLSearchParams({
                      cache_key: result.cache_key,
                      url: youtubeUrl,
                      lang: result.target_language || targetLang,
                    }).toString()}`}
                  >
                    <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                    Transcript Editor
                  </Link>
                </div>
              </div>
            )}

            {/* Export card */}
            {result && !jobId && (
              <div className="yt-sidebar-card">
                <div className="yt-card-header">
                  <div className="yt-card-icon-wrap">
                    <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                      <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="yt-card-title">Export</h3>
                </div>
                <div className="yt-tool-stack">
                  {/* Download full translated video */}
                  <button
                    type="button"
                    className={`yt-btn-tool yt-btn-tool--video-dl${(dlVideoStatus === "preparing" || dlVideoStatus === "downloading") ? " yt-btn-tool--busy" : ""}`}
                    onClick={handleDownloadVideo}
                    disabled={dlVideoStatus === "preparing" || dlVideoStatus === "downloading"}
                    title="Download the original video with dubbed audio merged in"
                  >
                    <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                      <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                    </svg>
                    {dlVideoStatus === "preparing"
                      ? "Preparing video…"
                      : dlVideoStatus === "downloading"
                        ? "Saving file…"
                        : dlVideoStatus === "done"
                          ? "✓ Downloaded — click to re-download"
                          : "Download Translated Video"}
                  </button>
                  {dlVideoStatus === "error" && dlVideoError && (
                    <p className="yt-dl-video-error">{dlVideoError}</p>
                  )}
                  {dlVideoStatus === "preparing" && (
                    <p className="yt-dl-video-hint">
                      Downloading the original video and merging the dubbed audio — this can take a few minutes depending on video length.
                    </p>
                  )}

                  <div className="yt-export-divider"><span>Audio only</span></div>
                  <a href={fullAudioUrl} download className="yt-btn-tool yt-btn-tool--download">
                    <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                      <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v8.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 111.414-1.414L9 12.586V4a1 1 0 01.383-.924zM3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                    </svg>
                    Download Dubbed Audio (MP3)
                  </a>
                  <div className="yt-export-divider"><span>AI Validation</span></div>
                  <a
                    href={`${API_BASE}/api/llm-validation-log`}
                    download="llm_validation_log.txt"
                    className="yt-btn-tool yt-btn-tool--secondary"
                    title="Download a log of every translation the LLM validated or improved"
                  >
                    <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                    </svg>
                    Download LLM validation log
                  </a>

                  {originalTxtUrl && translatedTxtUrl && (
                    <>
                      <div className="yt-export-divider">
                        <span>Transcripts</span>
                      </div>
                      <a
                        href={originalTxtUrl}
                        download
                        className="yt-btn-tool yt-btn-tool--secondary"
                      >
                        <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                        </svg>
                        Original transcript
                      </a>
                      <a
                        href={translatedTxtUrl}
                        download
                        className="yt-btn-tool yt-btn-tool--secondary"
                      >
                        <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15">
                          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                        </svg>
                        Translated ({langLabel(result.target_language || targetLang)})
                      </a>
                    </>
                  )}
                </div>
              </div>
            )}

          </aside>
        </div>
      </div>
    </div>
  );
}

export default Player;
