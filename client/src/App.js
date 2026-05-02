import React from "react";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Player from "./pages/Player";
import TranscriptEditor from "./pages/TranscriptEditor";
import TranslationEditor from "./pages/TranslationEditor";
import ErrorBoundary from "./components/ErrorBoundary";

/**
 * API_BASE defines the connection to your FastAPI backend.
 * In Electron or local development, this defaults to localhost:8000.
 */
export const API_BASE =
  process.env.REACT_APP_API_URL !== undefined && process.env.REACT_APP_API_URL !== ""
    ? process.env.REACT_APP_API_URL
    : "http://127.0.0.1:8000";

function App() {
  return (
    <ErrorBoundary>
      {/* 
          HashRouter is used because Electron serves files via the file:// protocol, 
           which doesn't support the HTML5 History API used by BrowserRouter.
      */}
      <HashRouter>
        <Routes>
          {/* Main landing page to input YouTube URLs */}
          <Route path="/" element={<Home />} />
          
          {/* 
              The Player page now handles both the YouTube preview 
              and the local .mp4 playback once translation is complete.
          */}
          <Route path="/player" element={<Player />} />
          
          {/* Editors for fine-tuning AI-generated content */}
          <Route path="/editor" element={<TranscriptEditor />} />
          <Route path="/translation-editor" element={<TranslationEditor />} />
          
          {/* Redirects for legacy routes or typos */}
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  );
}

export default App;