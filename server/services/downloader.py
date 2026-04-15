"""Download YouTube audio as WAV using yt-dlp (100% local, no API keys)."""
from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
from typing import Optional, Tuple

import yt_dlp


def _log(msg: str) -> None:
    """Print that silently swallows BrokenPipeError (stderr pipe gone)."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass


# Ignore SIGPIPE: Python converts EPIPE to BrokenPipeError (caught by our
# exception handlers).  SIG_DFL would terminate the process on any pipe write
# to a closed reader — exactly the wrong behaviour in a long-running server.
try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except (AttributeError, OSError):
    pass  # Windows has no SIGPIPE


class _QuietLogger:
    """Redirect all yt-dlp output to /dev/null.

    yt-dlp writes progress / warnings to sys.stdout/stderr even when
    quiet=True in some code paths.  Passing a custom logger object
    completely bypasses those writes and prevents BrokenPipeError from
    propagating out of the downloader.
    """
    def debug(self, msg: str) -> None:   # noqa: D102
        pass
    def info(self, msg: str) -> None:    # noqa: D102
        pass
    def warning(self, msg: str) -> None: # noqa: D102
        pass
    def error(self, msg: str) -> None:   # noqa: D102
        _log(f"[yt-dlp] {msg}")


def _find_ffmpeg() -> str | None:
    """Return the absolute path to ffmpeg, checking common locations explicitly."""
    env_path = os.environ.get("FFMPEG_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _cookie_opts() -> dict:
    """Return yt-dlp cookie options based on env or macOS Safari auto-detection."""
    cookies_file = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    if cookies_file and os.path.isfile(cookies_file):
        return {"cookiefile": cookies_file}

    browser = os.environ.get("YT_DLP_BROWSER", "").strip().lower()
    if browser in ("chrome", "chromium", "firefox", "safari", "edge", "opera", "brave", "vivaldi"):
        return {"cookiesfrombrowser": (browser,)}

    # macOS: auto-use Safari cookies (most users are logged into YouTube there)
    if platform.system() == "Darwin":
        return {"cookiesfrombrowser": ("safari",)}

    return {}


def _convert_to_wav(src: str, wav_path: str, ffmpeg: str) -> bool:
    """Convert any audio file to 16-bit mono WAV via a direct ffmpeg subprocess (no pipe)."""
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", src,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                wav_path,
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            _log(f"[downloader] ffmpeg stderr: {result.stderr.decode(errors='replace')[-500:]}")
        return result.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 1024
    except Exception as exc:
        _log(f"[downloader] ffmpeg conversion exception: {exc}")
        return False


def _download_raw(
    youtube_url: str,
    base: str,
    ydl_opts: dict,
) -> tuple[str | None, str, float]:
    """
    Download without any postprocessor (no ffmpeg pipe).
    Returns (downloaded_file_path, title, duration) or (None, "", 0).
    """
    # Remove postprocessors so yt-dlp never pipes to ffmpeg
    opts = {k: v for k, v in ydl_opts.items() if k != "postprocessors"}
    opts["outtmpl"] = base + ".%(ext)s"

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
        if not info:
            return None, "", 0.0

        title = (info.get("title") or "YouTube Video").strip()
        duration = float(info.get("duration") or 0.0)

        # Find the file yt-dlp wrote
        for ext in ("m4a", "mp3", "webm", "opus", "ogg", "aac", "wav", "mp4"):
            candidate = f"{base}.{ext}"
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 1024:
                return candidate, title, duration

        # Fallback: use info dict filename
        for key in ("filename", "_filename", "requested_downloads"):
            val = info.get(key)
            if isinstance(val, list):
                val = val[0].get("filename") if val else None
            if val and os.path.isfile(str(val)):
                return str(val), title, duration

        return None, title, duration
    except Exception as exc:
        _log(f"[downloader] raw download error: {exc}")
        return None, "", 0.0


def download_audio(youtube_url: str, output_path_base: str) -> Tuple[str, str, float]:
    """
    Download YouTube audio and convert to WAV.

    Always downloads the raw audio file first (no ffmpeg pipe), then converts
    with a direct ffmpeg subprocess call. This eliminates [Errno 32] Broken pipe.

    Returns:
        (wav_path, video_title, duration_seconds)
    Raises:
        RuntimeError if all strategies fail.
    """
    base = output_path_base.rstrip(".wav").rstrip(".mp3")
    wav_path = base + ".wav"

    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg\nThen relaunch the app."
        )
    _log(f"[downloader] ffmpeg: {ffmpeg_bin}")

    cookie_opts = _cookie_opts()
    _log(f"[downloader] cookie strategy: {list(cookie_opts.keys())}")

    base_opts: dict = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best",
        # quiet=True + no_warnings=True: suppresses most yt-dlp output.
        "quiet": True,
        "no_warnings": True,
        # logger: custom object replaces ALL yt-dlp writes to sys.stdout/stderr.
        # Without this, some yt-dlp code paths write directly even with quiet=True,
        # risking BrokenPipeError on the stdout pipe to the Electron parent.
        "logger": _QuietLogger(),
        # fixup=never: prevents automatic ffmpeg fixup postprocessors (e.g.
        # FixupM4a) that yt-dlp adds internally — we convert manually below.
        "fixup": "never",
        # ffmpeg_location: directory containing ffmpeg binary.
        "ffmpeg_location": os.path.dirname(ffmpeg_bin),
    }

    # Player clients to try, paired with whether to use cookies
    attempts: list[tuple[Optional[list[str]], bool]] = [
        (["web"],          True),
        (["web_embedded"], True),
        (["tv_embedded"],  True),
        (["ios"],          True),
        (["android"],      True),
        (None,             True),   # yt-dlp default + cookies
        (["tv_embedded"],  False),
        (["android_vr"],   False),
        (["ios"],          False),
        (["mweb"],         False),
        (None,             False),  # yt-dlp default, no cookies
    ]

    last_err: Optional[Exception] = None

    for player_clients, use_cookies in attempts:
        # Clean up leftovers from previous attempt
        for ext in ("wav", "mp3", "m4a", "webm", "opus", "ogg", "aac", "mp4"):
            p = f"{base}.{ext}"
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        ydl_opts = {**base_opts}
        if use_cookies and cookie_opts:
            ydl_opts.update(cookie_opts)
        if player_clients is not None:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": player_clients}}

        _log(f"[downloader] trying client={player_clients} cookies={use_cookies}")

        try:
            downloaded, title, duration = _download_raw(youtube_url, base, ydl_opts)
        except Exception as exc:
            last_err = exc
            _log(f"[downloader] attempt exception: {exc}")
            continue

        if not downloaded:
            _log(f"[downloader] no file produced for client={player_clients}")
            continue

        _log(f"[downloader] downloaded: {downloaded} ({os.path.getsize(downloaded)} bytes)")

        # Already a WAV — just return it
        if downloaded == wav_path and os.path.getsize(wav_path) > 1024:
            return wav_path, title, duration

        # Convert to WAV with direct ffmpeg call (no pipe)
        if _convert_to_wav(downloaded, wav_path, ffmpeg_bin):
            try:
                if downloaded != wav_path:
                    os.remove(downloaded)
            except OSError:
                pass
            _log(f"[downloader] success: {wav_path}")
            return wav_path, title, duration
        else:
            _log(f"[downloader] ffmpeg conversion failed for {downloaded}")

    raise RuntimeError(
        "Could not download YouTube audio.\n"
        "• Make sure you are logged into YouTube in Safari and relaunch the app.\n"
        "• Or add  YT_DLP_BROWSER=chrome  to your .env if you use Chrome.\n"
        "• Or run: pip install -U yt-dlp"
        + (f"\n\nLast error: {last_err}" if last_err else "")
    )
