import React, { useEffect, useState, useMemo } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import VideoPlayer from "../components/VideoPlayer";
import ProcessingStatus from "../components/ProcessingStatus";
import LanguageSelector from "../components/LanguageSelector";
import { API_BASE } from "../App";

const RECENT_KEY = "youtube-translator-recent-v2";
const _playerCache = { key: null, result: null };

function extractVideoId(url) {
  if (!url) return null;
  try {
    const u = new URL(url);
    if (u.hostname.endsWith("youtu.be")) return u.pathname.slice(1).split("?")[0] || null;
    if (u.hostname.includes("youtube.com")) {
      if (u.searchParams.has("v")) return u.searchParams.get("v");
      const embed = u.pathname.match(/\/embed\/([^/?]+)/);
      if (embed) return embed[1];
      const shorts = u.pathname.match(/\/shorts\/([^/?]+)/);
      if (shorts) return shorts[1];
    }
  } catch (_) {}
  return null;
}

function Player() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const youtubeUrl = searchParams.get("url") || "";
  const targetLang = searchParams.get("lang") || "en";

  const [result, setResult] = useState(() => {
    const key = `${youtubeUrl}||${targetLang}`;
    return _playerCache.key === key ? _playerCache.result : null;
  });
  const [jobId, setJobId] = useState(null);
  const [processError, setProcessError] = useState(null);
  const [languages, setLanguages] = useState([]);
  const [pendingLang, setPendingLang] = useState(targetLang);
  const [retryToken, setRetryToken] = useState(0);

  const resultCacheKey = `${youtubeUrl}||${targetLang}`;
  const videoId = useMemo(() => extractVideoId(youtubeUrl), [youtubeUrl]);

  // Determine the final video source
  // If the backend result contains a video_url, we use that local .mp4
  const mergedVideoUrl = result?.video_url ? `${API_BASE}${result.video_url}` : null;

  useEffect(() => {
    fetch(`${API_BASE}/api/languages`).then(r => r.json()).then(setLanguages).catch(() => {});
  }, []);

  useEffect(() => {
    if (!youtubeUrl) { navigate("/", { replace: true }); return; }

    const isNewVideo = _playerCache.key !== resultCacheKey || retryToken > 0;
    let cancelled = false;

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
          handleResult(data);
          return;
        }
        if (data.job_id) setJobId(data.job_id);
      } catch (e) {
        if (!cancelled) setProcessError("Backend unreachable. Check if FastAPI is running.");
      }
    })();

    return () => { cancelled = true; };
  }, [youtubeUrl, targetLang, retryToken]);

  const handleResult = (data) => {
    const r = { ...data, youtube_url: youtubeUrl, _ts: Date.now() };
    _playerCache.key = resultCacheKey;
    _playerCache.result = r;
    setResult(r);
    setJobId(null);
  };

  const handleApplyLanguage = () => {
    navigate(`/player?url=${encodeURIComponent(youtubeUrl)}&lang=${pendingLang}`, { replace: true });
  };

  return (
    <div className="yt-player-page">
      <nav className="yt-player-topbar">
        <button onClick={() => navigate("/")} className="yt-back-btn">Back to Home</button>
        <div className="yt-topbar-status">
          {jobId && <span className="yt-topbar-pill yt-topbar-pill--processing">Processing…</span>}
          {result && <span className="yt-topbar-pill yt-topbar-pill--ready">Ready</span>}
        </div>
      </nav>

      <div className="yt-player-layout">
        <main className="yt-player-main">
          {mergedVideoUrl ? (
            /* PRIORITY: Play the local translated .mp4 from your server */
            <video 
              src={mergedVideoUrl} 
              controls 
              autoPlay 
              style={{ width: "100%", borderRadius: "12px", background: "#000" }} 
            />
          ) : (
            /* FALLBACK: Show YouTube preview while processing */
            <VideoPlayer videoId={videoId} />
          )}
        </main>

        <aside className="yt-player-sidebar">
          <div className="yt-sidebar-card">
            <h1 className="yt-video-title">{result?.title || "Video Translation"}</h1>
            <LanguageSelector languages={languages} value={pendingLang} onChange={setPendingLang} disabled={!!jobId} />
            <button onClick={handleApplyLanguage} disabled={!!jobId || pendingLang === targetLang} className="yt-btn-apply">
              Apply
            </button>
          </div>

          {jobId && (
            <ProcessingStatus 
              jobId={jobId} 
              targetLanguage={targetLang} 
              onResult={handleResult} 
              onError={(err) => setProcessError(err)} 
            />
          )}

          {processError && <div className="yt-error-box">{processError}</div>}
        </aside>
      </div>
    </div>
  );
}

export default Player;