"""Download YouTube audio as WAV using yt-dlp (100% local, no API keys)."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import yt_dlp


def _cookie_opts() -> dict:
    opts = {}
    cookies_file = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    if cookies_file and os.path.isfile(cookies_file):
        opts["cookiefile"] = cookies_file
    browser = os.environ.get("YT_DLP_BROWSER", "").strip().lower()
    if browser in ("chrome", "chromium", "firefox", "safari", "edge", "opera", "brave", "vivaldi"):
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def _js_remote_opts() -> dict:
    opts: dict = {}
    js = os.environ.get("YT_DLP_JS_RUNTIMES", "").strip().lower()
    if js:
        parts = js.split(":", 1)
        runtime, path = parts[0], parts[1] if len(parts) > 1 else None
        if runtime in ("deno", "node", "bun", "quickjs"):
            opts["js_runtimes"] = {runtime: {} if not path else {"path": path}}
    remote = os.environ.get("YT_DLP_REMOTE_COMPONENTS", "").strip().lower()
    if remote in ("ejs:github", "ejs:npm"):
        opts["remote_components"] = [remote]
    return opts


def download_audio(youtube_url: str, output_path_base: str) -> Tuple[str, str, float]:
    """
    Download best audio and convert to WAV via ffmpeg postprocessor.
    Tries player clients in order: tv_embedded, android_vr, ios, default (yt-dlp default clients).

    Returns:
        (wav_path, video_title, duration_seconds)
    Raises:
        RuntimeError: if all strategies fail (hints to use YT_DLP_BROWSER=chrome).
    """
    base = output_path_base.rstrip(".wav").rstrip(".mp3")
    outtmpl = base + ".%(ext)s"
    wav_path = base + ".wav"

    cookie_opts = _cookie_opts()
    extra = _js_remote_opts()
    base_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
        "quiet": True,
        **cookie_opts,
        **extra,
    }

    # Spec order; "default" = no forced player_client (last attempt)
    client_attempts: list[Optional[list[str]]] = [
        ["tv_embedded"],
        ["android_vr"],
        ["ios"],
        None,  # default / extractor chooses
    ]

    last_err: Optional[Exception] = None

    for player_clients in client_attempts:
        ydl_opts = {**base_opts}
        if player_clients is not None:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
            if not info:
                continue
            if not os.path.isfile(wav_path):
                continue
            title = (info.get("title") or "YouTube Video").strip()
            duration = float(info.get("duration") or 0.0)
            return wav_path, title, duration
        except Exception as e:
            last_err = e
            continue

    msg = (
        "Could not download YouTube audio. Try: export YT_DLP_BROWSER=chrome "
        "(same machine as browser, logged into YouTube), install Deno + pip install -U 'yt-dlp[default]', "
        "or see yt-dlp EJS wiki."
    )
    if last_err:
        msg = f"{last_err}. {msg}"
    raise RuntimeError(msg)
