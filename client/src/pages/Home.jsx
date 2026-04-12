import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import LanguageSelector from "../components/LanguageSelector";
import { API_BASE } from "../App";

const RECENT_KEY = "youtube-translator-recent-v2";
const MAX_RECENT = 5;

function Home() {
  const [url, setUrl] = useState("");
  const [language, setLanguage] = useState("en");
  const [includeTranscripts, setIncludeTranscripts] = useState(false);
  const [languages, setLanguages] = useState([]);
  const [recent, setRecent] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`${API_BASE}/api/languages`)
      .then((r) => r.json())
      .then(setLanguages)
      .catch(() => {});
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setRecent(parsed);
        else localStorage.removeItem(RECENT_KEY); // stale / corrupted
      }
    } catch (_) {
      localStorage.removeItem(RECENT_KEY);
    }
  }, []);

  const saveRecent = (entry) => {
    const next = [entry, ...(recent || []).filter((e) => e.url !== entry.url)].slice(0, MAX_RECENT);
    setRecent(next);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (submitting) return;
    const trimmed = url.trim();
    if (!trimmed) return;
    setSubmitting(true);
    saveRecent({ url: trimmed, language, title: null });
    const q = new URLSearchParams({ url: trimmed, lang: language });
    if (includeTranscripts) q.set("tx", "1");
    navigate(`/player?${q.toString()}`);
  };

  const langName = (code) => languages.find((l) => l.code === code)?.name || code;

  return (
    <div className="yt-home">
      <div className="yt-home-inner">
        <header className="yt-hero">
          <h1 className="yt-hero-title">YouTube Video Translator</h1>
          <p className="yt-hero-sub">
            Watch any YouTube video in your language —{" "}
            <span className="yt-accent-emerald">100% free</span>,{" "}
            <span className="yt-accent-blue">100% local</span>, zero API keys
          </p>
        </header>

        <form onSubmit={handleSubmit} className="yt-card">
          <label className="yt-label" htmlFor="yt-url-input">
            YouTube URL
          </label>
          <input
            id="yt-url-input"
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste YouTube URL here…"
            className="yt-input"
            required
          />

          <label className="yt-label yt-label-spaced" htmlFor="yt-lang-select">
            Target language
          </label>
          <LanguageSelector
            id="yt-lang-select"
            languages={languages}
            value={language}
            onChange={setLanguage}
          />
          <p className="yt-hint">
            Source language is <span className="yt-hint-highlight">auto-detected by AI</span> — no
            selection needed.
          </p>

          <label className="yt-check-row">
            <input
              type="checkbox"
              checked={includeTranscripts}
              onChange={(e) => setIncludeTranscripts(e.target.checked)}
            />
            <span>
              Offer <strong>.txt</strong> downloads (original language + translated) after
              processing
            </span>
          </label>

          <button type="submit" className="yt-btn-primary" disabled={submitting}>
            {submitting ? "Opening…" : "Translate and Watch"}
          </button>
        </form>

        <div className="yt-feature-grid">
          {[
            ["No API Keys", "Everything runs without subscriptions or secrets."],
            ["Runs Locally", "Whisper, translation, and TTS on your machine."],
            ["Auto Language Detection", "Whisper detects the spoken language for you."],
            ["Instant Cache", "Same video + language returns immediately."],
          ].map(([title, desc]) => (
            <div key={title} className="yt-feature-card">
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>

        <div className="yt-warning">
          First run takes <strong>5–10 min</strong> while AI models download (~3GB one-time only).
        </div>

        {recent.length > 0 && (
          <section className="yt-recent">
            <h2 className="yt-recent-title">Recently watched</h2>
            <ul className="yt-recent-list">
              {recent.map((entry, i) => (
                <li key={i}>
                  <button
                    type="button"
                    onClick={() => {
                      setUrl(entry.url);
                      setLanguage(entry.language || "en");
                    }}
                    className="yt-recent-btn"
                  >
                    <span>{entry.title || entry.url}</span>
                    {entry.language && (
                      <span className="yt-badge-lang">{langName(entry.language)}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}

export default Home;
