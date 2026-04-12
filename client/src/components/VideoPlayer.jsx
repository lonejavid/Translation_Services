import React, { useEffect, useRef, useState, useCallback } from "react";
import SubtitleDisplay from "./SubtitleDisplay";
import { videoTimeToDubAudioTime } from "../utils/dubAudioSync";

const SYNC_INTERVAL_MS = 500;
const DRIFT_THRESHOLD = 0.35;

/**
 * YouTube iframe is created with mute: 1 — original audio off; dubbed MP3 carries all speech.
 * Dubbed MP3 must be same-origin as the page in dev (see package.json "proxy")
 * so the <audio> element can load and play without CORS blocking playback.
 */
function VideoPlayer({ videoId, dubbedAudioUrl, subtitles, dubSync }) {
  const containerRef = useRef(null);
  const playerRef = useRef(null);
  const audioRef = useRef(null);
  const dubSyncRef = useRef(dubSync);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioReady, setAudioReady] = useState(false);
  const [audioError, setAudioError] = useState(null);
  const [volume, setVolume] = useState(1);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  const mapVideoToAudio = useCallback(
    (ytTime) => {
      if (dubSync && dubSync.length > 0) {
        return videoTimeToDubAudioTime(ytTime, dubSync);
      }
      return ytTime;
    },
    [dubSync]
  );

  const tryPlayDub = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !dubbedAudioUrl) return;
    const p = audio.play();
    if (p && typeof p.catch === "function") {
      p.catch((err) => {
        console.warn("[dub] audio.play() blocked or failed:", err?.message || err);
      });
    }
  }, [dubbedAudioUrl]);

  const initPlayer = useCallback(() => {
    if (!containerRef.current || !videoId || !window.YT?.Player) return;
    if (playerRef.current) {
      playerRef.current.destroy();
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
      },
      events: {
        onStateChange: (e) => {
          const audio = audioRef.current;
          if (!audio || !dubbedAudioUrl) return;
          const p = e.target;
          try {
            if (typeof p.getCurrentTime === "function") {
              const ytTime = p.getCurrentTime();
              if (typeof ytTime === "number" && !Number.isNaN(ytTime)) {
                const want = mapVideoToAudio(ytTime);
                const dur = audio.duration;
                const safe =
                  typeof dur === "number" && dur > 0 && Number.isFinite(dur)
                    ? Math.min(want, Math.max(0, dur - 0.02))
                    : want;
                if (e.data === window.YT.PlayerState.BUFFERING) {
                  audio.currentTime = safe;
                } else if (
                  e.data === window.YT.PlayerState.PLAYING &&
                  Math.abs(audio.currentTime - safe) > DRIFT_THRESHOLD
                ) {
                  audio.currentTime = safe;
                }
              }
            }
          } catch (_) {}
          if (e.data === window.YT.PlayerState.PLAYING) {
            tryPlayDub();
          } else if (
            e.data === window.YT.PlayerState.PAUSED ||
            e.data === window.YT.PlayerState.ENDED
          ) {
            audio.pause();
          }
        },
      },
    });
    playerRef.current = player;
  }, [videoId, dubbedAudioUrl, tryPlayDub]);

  useEffect(() => {
    if (!videoId) return;
    if (window.YT && window.YT.Player) {
      initPlayer();
      return;
    }
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    const firstScript = document.getElementsByTagName("script")[0];
    firstScript.parentNode.insertBefore(tag, firstScript);
    window.onYouTubeIframeAPIReady = initPlayer;
    return () => {
      if (window.onYouTubeIframeAPIReady === initPlayer) {
        window.onYouTubeIframeAPIReady = null;
      }
    };
  }, [videoId, initPlayer]);

  useEffect(() => {
    setAudioReady(false);
    setAudioError(null);
    const audio = audioRef.current;
    if (!audio || !dubbedAudioUrl) return;

    const markReady = () => setAudioReady(true);
    const onErr = () => {
      setAudioError("Could not load dubbed audio (check server / CORS).");
      setAudioReady(true);
    };

    audio.pause();
    audio.src = dubbedAudioUrl;
    // No crossOrigin — CORS mode + wrong headers often blocks <audio> decode/play across ports.

    audio.addEventListener("loadedmetadata", markReady);
    audio.addEventListener("loadeddata", markReady);
    audio.addEventListener("canplay", markReady);
    audio.addEventListener("error", onErr);

    audio.load();

    const t = window.setTimeout(markReady, 8000);

    return () => {
      window.clearTimeout(t);
      audio.removeEventListener("loadedmetadata", markReady);
      audio.removeEventListener("loadeddata", markReady);
      audio.removeEventListener("canplay", markReady);
      audio.removeEventListener("error", onErr);
    };
  }, [dubbedAudioUrl]);

  useEffect(() => {
    if (!playerRef.current || !dubbedAudioUrl) return;
    const id = setInterval(() => {
      const p = playerRef.current;
      const audio = audioRef.current;
      if (!p?.getCurrentTime || !audio) return;
      const ytTime = p.getCurrentTime();
      if (typeof ytTime !== "number" || Number.isNaN(ytTime)) return;
      const want = mapVideoToAudio(ytTime);
      const dur = audio.duration;
      const safe =
        typeof dur === "number" && dur > 0 && Number.isFinite(dur)
          ? Math.min(want, Math.max(0, dur - 0.02))
          : want;
      if (Math.abs(audio.currentTime - safe) > DRIFT_THRESHOLD) {
        audio.currentTime = safe;
      }
    }, SYNC_INTERVAL_MS);
    return () => clearInterval(id);
  }, [dubbedAudioUrl, audioReady, mapVideoToAudio]);

  // Subtitles follow YouTube time (cues are timed to the video), not raw MP3 position.
  useEffect(() => {
    if (!videoId) return;
    const id = setInterval(() => {
      try {
        const p = playerRef.current;
        const t = p?.getCurrentTime?.();
        if (typeof t === "number" && !Number.isNaN(t)) {
          setCurrentTime(t);
        }
      } catch (_) {}
    }, 200);
    return () => clearInterval(id);
  }, [videoId, dubbedAudioUrl]);

  // If YouTube is already playing when dub finishes buffering, start dub now.
  useEffect(() => {
    if (!audioReady || !dubbedAudioUrl) return;
    const p = playerRef.current;
    try {
      if (p?.getPlayerState?.() === window.YT?.PlayerState?.PLAYING) {
        tryPlayDub();
      }
    } catch (_) {}
  }, [audioReady, dubbedAudioUrl, tryPlayDub]);

  return (
    <div className="yt-vp-wrap">
      <div className="yt-vp-aspect">
        <div ref={containerRef} className="yt-vp-frame" />
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
      {dubbedAudioUrl && (
        <audio ref={audioRef} preload="auto" className="yt-vp-audio-hidden" />
      )}
      {dubbedAudioUrl && (
        <div className="yt-vp-volume">
          <label>
            <span>Dub volume</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={volume}
              onChange={(e) => {
                const v = Number(e.target.value);
                setVolume(v);
                tryPlayDub();
              }}
              className="yt-vp-range"
            />
          </label>
        </div>
      )}
    </div>
  );
}

export default VideoPlayer;
