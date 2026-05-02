import React, { useEffect, useRef, useState, useCallback } from "react";
import SubtitleDisplay from "./SubtitleDisplay";
import { videoTimeToDubAudioTime } from "../utils/dubAudioSync";

const SYNC_INTERVAL_MS = 500;
const DRIFT_THRESHOLD = 0.35;

function VideoPlayer({ videoId, dubbedAudioUrl, subtitles, dubSync }) {
  const containerRef = useRef(null);
  const playerRef = useRef(null);
  const audioRef = useRef(null);
  const dubSyncRef = useRef(dubSync);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioReady, setAudioReady] = useState(false);
  const [audioError, setAudioError] = useState(null);
  const [volume, setVolume] = useState(1);
  const [embedBlocked, setEmbedBlocked] = useState(false);

  useEffect(() => { dubSyncRef.current = dubSync; }, [dubSync]);
  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  const mapVideoToAudio = useCallback((ytTime) => {
    const sync = dubSyncRef.current;
    if (sync && sync.length > 0) return videoTimeToDubAudioTime(ytTime, sync);
    return ytTime;
  }, []);

  const tryPlayDub = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !dubbedAudioUrl) return;
    audio.play().catch(() => {});
  }, [dubbedAudioUrl]);

  const initPlayer = useCallback(() => {
    if (!containerRef.current || !videoId || !window.YT?.Player) return;
    if (playerRef.current) {
      try { playerRef.current.destroy(); } catch (_) {}
      playerRef.current = null;
    }
    const player = new window.YT.Player(containerRef.current, {
      videoId,
      width: "100%",
      height: "100%",
      playerVars: {
        autoplay: 0,
        mute: 1,
        controls: 1,
        rel: 0,
        origin: window.location.origin,
      },
      events: {
        onError: (e) => {
          // Error 101 / 150: embedding disabled by uploader
          if (e.data === 101 || e.data === 150) {
            setEmbedBlocked(true);
          }
        },
        onStateChange: (e) => {
          const audio = audioRef.current;
          if (!audio || !dubbedAudioUrl) return;
          const p = e.target;
          try {
            const ytTime = p.getCurrentTime?.();
            if (typeof ytTime === "number" && !isNaN(ytTime)) {
              const want = mapVideoToAudio(ytTime);
              const dur = audio.duration;
              const safe = typeof dur === "number" && dur > 0 && isFinite(dur)
                ? Math.min(want, Math.max(0, dur - 0.02)) : want;
              if (e.data === window.YT.PlayerState.BUFFERING ||
                 (e.data === window.YT.PlayerState.PLAYING &&
                  Math.abs(audio.currentTime - safe) > DRIFT_THRESHOLD)) {
                audio.currentTime = safe;
              }
            }
          } catch (_) {}
          if (e.data === window.YT.PlayerState.PLAYING) {
            tryPlayDub();
          } else if (e.data === window.YT.PlayerState.PAUSED ||
                     e.data === window.YT.PlayerState.ENDED) {
            audio.pause();
          }
        },
      },
    });
    playerRef.current = player;
  }, [videoId, dubbedAudioUrl, tryPlayDub, mapVideoToAudio]);

  useEffect(() => {
    if (!videoId) return;
    setEmbedBlocked(false);
    if (window.YT?.Player) { initPlayer(); return; }
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
    window.onYouTubeIframeAPIReady = initPlayer;
    return () => {
      if (window.onYouTubeIframeAPIReady === initPlayer)
        window.onYouTubeIframeAPIReady = null;
    };
  }, [videoId, initPlayer]);

  useEffect(() => {
    setAudioReady(false);
    setAudioError(null);
    const audio = audioRef.current;
    if (!audio || !dubbedAudioUrl) return;
    const markReady = () => setAudioReady(true);
    const onErr = () => { setAudioError("Could not load dubbed audio."); setAudioReady(true); };
    audio.pause();
    audio.src = dubbedAudioUrl;
    audio.addEventListener("canplay", markReady);
    audio.addEventListener("loadedmetadata", markReady);
    audio.addEventListener("error", onErr);
    audio.load();
    const t = setTimeout(markReady, 8000);
    return () => {
      clearTimeout(t);
      audio.removeEventListener("canplay", markReady);
      audio.removeEventListener("loadedmetadata", markReady);
      audio.removeEventListener("error", onErr);
    };
  }, [dubbedAudioUrl]);

  useEffect(() => {
    if (!playerRef.current || !dubbedAudioUrl) return;
    const id = setInterval(() => {
      try {
        const p = playerRef.current;
        const audio = audioRef.current;
        if (!p?.getCurrentTime || !audio) return;
        const ytTime = p.getCurrentTime();
        if (typeof ytTime !== "number" || isNaN(ytTime)) return;
        const want = mapVideoToAudio(ytTime);
        const dur = audio.duration;
        const safe = typeof dur === "number" && dur > 0 && isFinite(dur)
          ? Math.min(want, Math.max(0, dur - 0.02)) : want;
        if (Math.abs(audio.currentTime - safe) > DRIFT_THRESHOLD) {
          audio.currentTime = safe;
        }
      } catch (_) {}
    }, SYNC_INTERVAL_MS);
    return () => clearInterval(id);
  }, [dubbedAudioUrl, audioReady, mapVideoToAudio]);

  useEffect(() => {
    if (!videoId) return;
    const id = setInterval(() => {
      try {
        const t = playerRef.current?.getCurrentTime?.();
        if (typeof t === "number" && !isNaN(t)) setCurrentTime(t);
      } catch (_) {}
    }, 200);
    return () => clearInterval(id);
  }, [videoId, dubbedAudioUrl]);

  useEffect(() => {
    if (!audioReady || !dubbedAudioUrl) return;
    try {
      if (playerRef.current?.getPlayerState?.() === window.YT?.PlayerState?.PLAYING) {
        tryPlayDub();
      }
    } catch (_) {}
  }, [audioReady, dubbedAudioUrl, tryPlayDub]);

  return (
    <div className="yt-vp-wrap">
      <div className="yt-vp-aspect">
        {embedBlocked ? (
          <div className="yt-vp-embed-blocked">
            <p>This video cannot be embedded.</p>
            <a
              href={`https://www.youtube.com/watch?v=${videoId}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Watch on YouTube ↗
            </a>
            <p style={{ fontSize: "0.8em", opacity: 0.7, marginTop: 8 }}>
              The dubbed audio still plays — use the controls below.
            </p>
          </div>
        ) : (
          <div ref={containerRef} className="yt-vp-frame" />
        )}
        {subtitles && <SubtitleDisplay subtitles={subtitles} currentTime={currentTime} />}
        {audioError && (
          <div className="yt-vp-toast" style={{ borderColor: "rgba(248,113,113,0.5)", color: "#fecaca" }}>
            {audioError}
          </div>
        )}
        {!audioError && !audioReady && dubbedAudioUrl && (
          <div className="yt-vp-toast">Loading dubbed audio…</div>
        )}
      </div>
      {dubbedAudioUrl && <audio ref={audioRef} preload="auto" className="yt-vp-audio-hidden" />}
      {dubbedAudioUrl && (
        <div className="yt-vp-volume">
          <label>
            <span>Dub volume</span>
            <input type="range" min="0" max="1" step="0.05" value={volume}
              onChange={(e) => { setVolume(Number(e.target.value)); tryPlayDub(); }}
              className="yt-vp-range"
            />
          </label>
        </div>
      )}
    </div>
  );
}

export default VideoPlayer;
