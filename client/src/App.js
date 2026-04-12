import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Player from "./pages/Player";
import TranscriptEditor from "./pages/TranscriptEditor";
import TranslationEditor from "./pages/TranslationEditor";
import ErrorBoundary from "./components/ErrorBoundary";

// Default matches server/run.sh (port 8000). If you use another port (e.g. 8001 when
// Docker uses 8000), create client/.env.development.local with:
//   REACT_APP_API_URL=http://127.0.0.1:8001
// and restart `npm start`. Server must list your dev origin in CORS (see CORS_ORIGINS in server .env).
export const API_BASE =
  process.env.REACT_APP_API_URL !== undefined && process.env.REACT_APP_API_URL !== ""
    ? process.env.REACT_APP_API_URL
    : "http://127.0.0.1:8000";

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/player" element={<Player />} />
          <Route path="/editor" element={<TranscriptEditor />} />
          <Route path="/translation-editor" element={<TranslationEditor />} />
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
