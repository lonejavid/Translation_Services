import React from "react";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Player from "./pages/Player";
import TranscriptEditor from "./pages/TranscriptEditor";
import TranslationEditor from "./pages/TranslationEditor";
import ErrorBoundary from "./components/ErrorBoundary";

// In Electron the app is served via file:// so BrowserRouter breaks.
// HashRouter (#/) works in both Electron (file://) and browser (dev server).
// API always points to the local FastAPI backend on port 8000.
export const API_BASE =
  process.env.REACT_APP_API_URL !== undefined && process.env.REACT_APP_API_URL !== ""
    ? process.env.REACT_APP_API_URL
    : "http://127.0.0.1:8000";

function App() {
  return (
    <ErrorBoundary>
      <HashRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/player" element={<Player />} />
          <Route path="/editor" element={<TranscriptEditor />} />
          <Route path="/translation-editor" element={<TranslationEditor />} />
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  );
}

export default App;
