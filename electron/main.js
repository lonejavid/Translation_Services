/**
 * Electron main process.
 * - Spawns the FastAPI backend (server/run.sh) as a child process.
 * - Serves the React build via a local HTTP server (http://localhost:PORT) so
 *   YouTube IFrame embeds work — YouTube rejects file:// / null origins (Error 153).
 * - Shows a loading screen until the backend is ready.
 * - Cleans up the backend process when the app quits.
 */

const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn, execSync } = require("child_process");
const http = require("http");
const fs = require("fs");
const net = require("net");

// ── Constants ────────────────────────────────────────────────────────────────
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const SERVER_DIR = path.join(__dirname, "..", "server");
const REQUIREMENTS = path.join(SERVER_DIR, "requirements.txt");
const RUN_SH = path.join(SERVER_DIR, "run.sh");
const CLIENT_BUILD_DIR = path.join(__dirname, "..", "client", "build");
const CLIENT_BUILD = path.join(CLIENT_BUILD_DIR, "index.html");

// Venv lives in userData so it survives app updates and is writable in /Applications.
// app.getPath("userData") = ~/Library/Application Support/Video Translator
// We can't use app here yet (before app.whenReady), so we build the path manually.
const USER_DATA = path.join(
  process.env.HOME || process.env.USERPROFILE || "~",
  "Library", "Application Support", "Video Translator"
);
const VENV_DIR = path.join(USER_DATA, "venv");
const VENV_PYTHON = path.join(VENV_DIR, "bin", "python3");
const CACHE_DIR = path.join(USER_DATA, "cache");

let mainWindow = null;
let loadingWindow = null;
let backendProcess = null;
let ollamaProcess = null;
let backendReady = false;
let staticServer = null;
let staticPort = 0;
let intentionalShutdown = false;

// ── Ollama (local LLM for translation validation) ────────────────────────────
const OLLAMA_PORT = 11434;
const OLLAMA_MODELS_DIR = path.join(USER_DATA, "ollama-models");

/** Find the bundled Ollama binary — works both in packaged app and dev mode. */
function getBundledOllamaBin() {
  const candidates = [
    process.resourcesPath && path.join(process.resourcesPath, "ollama", "ollama"),
    path.join(__dirname, "resources", "ollama", "ollama"),
    path.join(__dirname, "..", "electron", "resources", "ollama", "ollama"),
  ].filter(Boolean);
  return candidates.find((p) => fs.existsSync(p)) || null;
}

/** Check if Ollama HTTP API is already responding on localhost. */
function isOllamaRunning() {
  return new Promise((resolve) => {
    http
      .get(`http://127.0.0.1:${OLLAMA_PORT}`, (res) => { resolve(true); })
      .on("error", () => { resolve(false); });
  });
}

/**
 * Pull llama3 in background — fire-and-forget.
 * Only runs if the model manifest isn't already on disk.
 * Shows a log message; does NOT block app startup.
 */
function pullModelBackground(ollamaBin) {
  // Ollama stores model manifests here
  const manifestDir = path.join(
    OLLAMA_MODELS_DIR, "manifests", "registry.ollama.ai", "library", "llama3"
  );
  if (fs.existsSync(manifestDir)) {
    console.log("[ollama] llama3 model already present — skipping pull.");
    return;
  }

  console.log("[ollama] Pulling llama3 in background (~4.7 GB, one-time download)…");
  const ollamaDir = path.dirname(ollamaBin);
  const pull = spawn(ollamaBin, ["pull", "llama3"], {
    env: {
      ...process.env,
      PATH: FULL_PATH,
      OLLAMA_HOST: `127.0.0.1:${OLLAMA_PORT}`,
      OLLAMA_MODELS: OLLAMA_MODELS_DIR,
      DYLD_LIBRARY_PATH: `${ollamaDir}:${process.env.DYLD_LIBRARY_PATH || ""}`,
      LD_LIBRARY_PATH: `${ollamaDir}:${process.env.LD_LIBRARY_PATH || ""}`,
    },
    stdio: "ignore",
  });
  pull.on("close", (code) => {
    if (code === 0) console.log("[ollama] llama3 pull complete.");
    else console.warn(`[ollama] llama3 pull exited with code ${code}.`);
  });
  pull.on("error", (err) => {
    console.warn("[ollama] llama3 pull error:", err.message);
  });
}

/**
 * Start Ollama server if not already running, then pull llama3 in background.
 * Non-blocking — returns immediately. App works fine even if this fails.
 */
async function ensureOllama() {
  const ollamaBin = getBundledOllamaBin();
  if (!ollamaBin) {
    console.log("[ollama] No bundled binary found — LLM validator will be skipped.");
    return;
  }

  const alreadyRunning = await isOllamaRunning();
  if (alreadyRunning) {
    console.log("[ollama] Already running on port", OLLAMA_PORT);
    pullModelBackground(ollamaBin);
    return;
  }

  sendStatus("Starting local AI assistant (background)…");
  fs.mkdirSync(OLLAMA_MODELS_DIR, { recursive: true });

  const ollamaDir = path.dirname(ollamaBin);
  ollamaProcess = spawn(ollamaBin, ["serve"], {
    env: {
      ...process.env,
      PATH: FULL_PATH,
      OLLAMA_HOST: `127.0.0.1:${OLLAMA_PORT}`,
      OLLAMA_MODELS: OLLAMA_MODELS_DIR,
      // Needed so Ollama finds its bundled .dylib / .so files next to the binary
      DYLD_LIBRARY_PATH: `${ollamaDir}:${process.env.DYLD_LIBRARY_PATH || ""}`,
      LD_LIBRARY_PATH: `${ollamaDir}:${process.env.LD_LIBRARY_PATH || ""}`,
    },
    stdio: "ignore",
  });

  ollamaProcess.on("error", (err) => {
    console.warn("[ollama] Failed to start:", err.message);
    ollamaProcess = null;
  });

  ollamaProcess.on("exit", (code) => {
    if (!intentionalShutdown) console.log("[ollama] Server exited with code", code);
    ollamaProcess = null;
  });

  // Give Ollama a moment to bind to the port, then start the model pull
  setTimeout(() => pullModelBackground(ollamaBin), 3000);
}

// ── Static file server for React build ──────────────────────────────────────
// YouTube IFrame API refuses to load on file:// / null origins (Error 153).
// Serving the build from http://localhost:PORT gives a valid HTTP origin.
const MIME_TYPES = {
  ".html": "text/html",
  ".js":   "application/javascript",
  ".css":  "text/css",
  ".json": "application/json",
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".svg":  "image/svg+xml",
  ".ico":  "image/x-icon",
  ".woff": "font/woff",
  ".woff2":"font/woff2",
  ".ttf":  "font/ttf",
};

function getFreePorts(count) {
  return Promise.all(
    Array.from({ length: count }, () =>
      new Promise((resolve, reject) => {
        const s = net.createServer();
        s.listen(0, "127.0.0.1", () => {
          const port = s.address().port;
          s.close(() => resolve(port));
        });
        s.on("error", reject);
      })
    )
  );
}

function startStaticServer() {
  return new Promise((resolve, reject) => {
    staticServer = http.createServer((req, res) => {
      // Strip query string and hash; default to index.html (HashRouter handles routing)
      let urlPath = req.url.split("?")[0].split("#")[0];
      if (urlPath === "/" || urlPath === "") urlPath = "/index.html";

      const filePath = path.join(CLIENT_BUILD_DIR, urlPath);
      const ext = path.extname(filePath).toLowerCase();
      const mime = MIME_TYPES[ext] || "application/octet-stream";

      fs.readFile(filePath, (err, data) => {
        if (err) {
          // SPA fallback: any unknown route → index.html
          fs.readFile(CLIENT_BUILD, (err2, data2) => {
            if (err2) { res.writeHead(404); res.end("Not found"); return; }
            res.writeHead(200, { "Content-Type": "text/html" });
            res.end(data2);
          });
          return;
        }
        res.writeHead(200, { "Content-Type": mime });
        res.end(data);
      });
    });

    staticServer.listen(0, "127.0.0.1", () => {
      staticPort = staticServer.address().port;
      console.log(`[static] Serving React build on http://127.0.0.1:${staticPort}`);
      resolve(staticPort);
    });
    staticServer.on("error", reject);
  });
}

// ── Loading window ───────────────────────────────────────────────────────────
function createLoadingWindow() {
  loadingWindow = new BrowserWindow({
    width: 480,
    height: 320,
    resizable: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
    },
  });
  loadingWindow.loadFile(path.join(__dirname, "loading.html"));
}

// ── Main window ──────────────────────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      // Required so fetch() from http://127.0.0.1:staticPort → http://127.0.0.1:8000 works.
      webSecurity: false,
    },
  });

  // Load from the local static HTTP server so YouTube embeds have a valid origin.
  // (file:// origin is null → YouTube IFrame API rejects it with Error 153)
  mainWindow.loadURL(`http://127.0.0.1:${staticPort}/index.html`);

  mainWindow.once("ready-to-show", () => {
    if (loadingWindow && !loadingWindow.isDestroyed()) {
      loadingWindow.close();
    }
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ── Check if backend is up ───────────────────────────────────────────────────
// Increased to 5 minutes: first cold Python import of torch/TTS/transformers
// can take 3-5 min on a fresh machine before uvicorn finishes loading.
const BACKEND_TIMEOUT_S = 300;

// Allows the backend exit handler to abort the wait early (fail fast) instead
// of making the user sit through the full 5-minute countdown on a crash.
let _backendWaitReject = null;

function pollBackend(resolve, reject, attempts = 0) {
  http
    .get(`${BACKEND_URL}/health`, (res) => {
      if (res.statusCode === 200) {
        _backendWaitReject = null;
        resolve();
      } else {
        retryPoll(resolve, reject, attempts);
      }
    })
    .on("error", () => {
      retryPoll(resolve, reject, attempts);
    });
}

function retryPoll(resolve, reject, attempts) {
  if (attempts >= BACKEND_TIMEOUT_S) {
    _backendWaitReject = null;
    reject(new Error(
      `Backend did not start within ${BACKEND_TIMEOUT_S / 60} minutes.\n\n` +
      `Check the log for details:\n${path.join(USER_DATA, "backend.log")}`
    ));
    return;
  }
  // Show progress hints so the user knows we're still working
  if (attempts === 8)  sendStatus("Loading AI libraries — first launch can take a few minutes…");
  if (attempts === 60) sendStatus("Still loading (large AI models take time on first run)…");
  if (attempts === 150) sendStatus("Almost there — finishing model loading…");
  setTimeout(() => pollBackend(resolve, reject, attempts + 1), 1000);
}

function waitForBackend() {
  return new Promise((resolve, reject) => {
    _backendWaitReject = reject;
    pollBackend(resolve, reject);
  });
}

// ── Setup: create venv + install deps ───────────────────────────────────────
function sendStatus(msg) {
  if (loadingWindow && !loadingWindow.isDestroyed()) {
    loadingWindow.webContents.send("status", msg);
  }
  console.log("[setup]", msg);
}

/** Run a shell command as a promise so the event loop stays free. */
function runAsync(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"], ...opts });
    let stderr = "";
    proc.stderr.on("data", (d) => { stderr += d.toString(); });
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} failed (code ${code}):\n${stderr.slice(-400)}`));
    });
    proc.on("error", reject);
  });
}

// Shared augmented PATH: macOS .app bundles get a stripped PATH (/usr/bin:/bin only).
// Prepend Homebrew locations so python3, ffmpeg, pip, etc. are always found.
const HOMEBREW_PATH = [
  "/opt/homebrew/bin",
  "/opt/homebrew/sbin",
  "/usr/local/bin",
  "/usr/local/sbin",
].join(":");
const FULL_PATH = `${HOMEBREW_PATH}:${process.env.PATH || "/usr/bin:/bin:/usr/sbin:/sbin"}`;

async function ensureSetup() {
  // 0. Ensure writable userData directories exist
  fs.mkdirSync(USER_DATA, { recursive: true });
  fs.mkdirSync(CACHE_DIR, { recursive: true });

  // Remove any leftover stub directories from previous install attempts
  // (old pyproject.toml / setup.cfg files there caused build failures)
  const oldStubDir = path.join(USER_DATA, "stub_pkuseg");
  if (fs.existsSync(oldStubDir)) fs.rmSync(oldStubDir, { recursive: true, force: true });

  const spawnEnv = { ...process.env, PATH: FULL_PATH };

  // 1. Find Python 3.10+
  sendStatus("Checking Python…");
  let python = "";
  for (const candidate of ["python3.11", "python3.10", "python3", "python"]) {
    try {
      const v = execSync(`${candidate} --version 2>&1`, { env: spawnEnv }).toString().trim();
      const m = v.match(/Python 3\.(\d+)/);
      if (m && parseInt(m[1]) >= 9) {
        python = candidate;
        break;
      }
    } catch (_) {}
  }
  if (!python) {
    throw new Error(
      "Python 3.9+ not found.\nInstall it from https://www.python.org/downloads/macos/ and relaunch the app."
    );
  }

  // 2. Create venv if missing
  if (!fs.existsSync(VENV_PYTHON)) {
    sendStatus("Creating Python environment (first launch, ~30 seconds)…");
    await runAsync(python, ["-m", "venv", VENV_DIR], { cwd: SERVER_DIR, env: spawnEnv });
  }

  // 3. Install / upgrade deps
  sendStatus("Installing dependencies (first launch may take several minutes)…");
  await runAsync(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });

  // Install numpy first — some packages need it present before they can build.
  await runAsync(VENV_PYTHON, ["-m", "pip", "install", "numpy>=1.22.0,<2.0.0", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });

  // ── Install chatterbox-tts without pkuseg ───────────────────────────────────
  // chatterbox-tts==0.1.3 has Requires-Dist: pkuseg==0.0.25.
  // pkuseg has no pre-built wheel for Python 3.9 + macOS ARM64 and fails to compile.
  // chatterbox only uses pkuseg for Chinese segmentation inside a try/except, so
  // installing it with --no-deps is safe — Chinese text is not used in our pipeline.
  //
  // We also write a pkuseg stub directly into site-packages at version 0.0.25 so
  // that pip's resolver sees pkuseg as already installed if any other package asks
  // for it, and never tries to fetch/build the real package from PyPI.
  {
    const libDir = path.join(VENV_DIR, "lib");
    const pyDirName = fs.readdirSync(libDir).find(d => d.startsWith("python3.")) || "python3.9";
    const sitePackages = path.join(libDir, pyDirName, "site-packages");

    const pkgDir  = path.join(sitePackages, "pkuseg");
    const distDir = path.join(sitePackages, "pkuseg-0.0.25.dist-info");
    // Remove any old stub dist-info (different version) to avoid conflicts
    for (const existing of fs.readdirSync(sitePackages)) {
      if (existing.startsWith("pkuseg-") && existing.endsWith(".dist-info") && existing !== "pkuseg-0.0.25.dist-info") {
        fs.rmSync(path.join(sitePackages, existing), { recursive: true, force: true });
      }
    }
    fs.mkdirSync(pkgDir,  { recursive: true });
    fs.mkdirSync(distDir, { recursive: true });

    fs.writeFileSync(path.join(pkgDir,  "__init__.py"), "# stub — Chinese tokenizer not used\n");
    fs.writeFileSync(path.join(distDir, "INSTALLER"), "pip\n");
    fs.writeFileSync(path.join(distDir, "METADATA"),
      "Metadata-Version: 2.1\nName: pkuseg\nVersion: 0.0.25\n");
    fs.writeFileSync(path.join(distDir, "WHEEL"),
      "Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n");
    fs.writeFileSync(path.join(distDir, "RECORD"), [
      "pkuseg/__init__.py,,",
      "pkuseg-0.0.25.dist-info/INSTALLER,,",
      "pkuseg-0.0.25.dist-info/METADATA,,",
      "pkuseg-0.0.25.dist-info/WHEEL,,",
      "pkuseg-0.0.25.dist-info/RECORD,,",
    ].join("\n") + "\n");
  }

  // Install chatterbox-tts without deps so pip never sees the pkuseg==0.0.25 requirement.
  // chatterbox-tts<=0.1.3 is removed from requirements.txt for the same reason.
  await runAsync(VENV_PYTHON, ["-m", "pip", "install", "chatterbox-tts<=0.1.3",
    "--no-deps", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });

  // Install TTS (Coqui) without deps: on Python 3.9 it pins numpy==1.22.0 exactly,
  // which conflicts with demucs/noisereduce/librosa needing newer numpy.
  // TTS>=0.22.0 is removed from requirements.txt for the same reason.
  sendStatus("Installing Coqui TTS…");
  await runAsync(VENV_PYTHON, ["-m", "pip", "install", "TTS>=0.22.0",
    "--no-deps", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });

  await runAsync(VENV_PYTHON, ["-m", "pip", "install", "-r", REQUIREMENTS, "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });

  // Always upgrade yt-dlp to latest — YouTube regularly changes its bot-detection
  // and old yt-dlp versions stop working within weeks. Fast (cached by pip).
  sendStatus("Updating yt-dlp…");
  try {
    await runAsync(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "yt-dlp", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });
  } catch (_) { /* non-fatal — existing version will be used */ }

  // Install mlx-whisper only on Python 3.10+ (requires mlx>=0.11 which needs Python >=3.10).
  // On Python 3.9 the transcriber falls back to faster-whisper automatically.
  try {
    const pyVer = execSync(`"${VENV_PYTHON}" -c "import sys; print(sys.version_info.minor)"`, { env: spawnEnv }).toString().trim();
    if (parseInt(pyVer) >= 10) {
      sendStatus("Installing MLX Whisper (Apple Silicon accelerated)…");
      await runAsync(VENV_PYTHON, ["-m", "pip", "install", "mlx-whisper>=0.4.0", "--quiet"], { cwd: SERVER_DIR, env: spawnEnv });
    }
  } catch (_) { /* optional — skip silently */ }

  sendStatus("Setup complete.");
}

// ── Free port 8000 if something is already LISTENING on it ──────────────────
/** Return PIDs listening on BACKEND_PORT, excluding our own PID. */
function getListeningPids() {
  return new Promise((resolve) => {
    const lsof = spawn("lsof", ["-ti", String(BACKEND_PORT), "-sTCP:LISTEN"], {
      stdio: ["ignore", "pipe", "ignore"],
      env: { ...process.env, PATH: FULL_PATH },
    });
    let out = "";
    lsof.stdout.on("data", (d) => { out += d.toString(); });
    lsof.on("close", () => {
      const myPid = process.pid;
      const pids = out.trim().split(/\s+/).filter((p) => {
        const n = Number(p);
        return p && !isNaN(n) && n !== myPid;
      });
      resolve(pids.map(Number));
    });
    lsof.on("error", () => resolve([]));
  });
}

/** Kill any lingering uvicorn processes from a previous session by name. */
function killStrayUvicorn() {
  return new Promise((resolve) => {
    // pkill -9 -f matches the full command line — catches uvicorn even when
    // lsof misses it (e.g. process just started and not yet in LISTEN state).
    const pk = spawn("pkill", ["-9", "-f", `uvicorn.*${BACKEND_PORT}`], {
      stdio: "ignore",
      env: { ...process.env, PATH: FULL_PATH },
    });
    pk.on("close", () => resolve());
    pk.on("error", () => resolve()); // pkill not found / no match → ignore
  });
}

/** Poll until port is free, escalating SIGTERM → SIGKILL aggressively. */
function freeBackendPort() {
  return new Promise(async (resolve) => {
    // Belt-and-suspenders: kill by name first, then by port.
    await killStrayUvicorn();

    const pids = await getListeningPids();
    if (pids.length === 0) { resolve(); return; }

    console.log(`[setup] Port ${BACKEND_PORT} held by PID(s) ${pids.join(",")} — sending SIGTERM…`);
    for (const pid of pids) {
      try { process.kill(pid, "SIGTERM"); } catch (_) {}
    }

    // Poll every 200 ms; escalate to SIGKILL after 1.5 s; give up at 5 s.
    // (Shorter than before — uvicorn is fast to die once it receives SIGKILL.)
    const start = Date.now();
    const poll = setInterval(async () => {
      const remaining = await getListeningPids();
      if (remaining.length === 0) {
        clearInterval(poll);
        console.log(`[setup] Port ${BACKEND_PORT} is now free.`);
        resolve();
        return;
      }
      const elapsed = Date.now() - start;
      if (elapsed > 1500 && elapsed < 1800) {
        console.log(`[setup] Port still held after 1.5 s — sending SIGKILL…`);
        for (const pid of remaining) {
          try { process.kill(pid, "SIGKILL"); } catch (_) {}
        }
        // Also re-run pkill in case new processes appeared.
        await killStrayUvicorn();
      }
      if (elapsed > 5000) {
        clearInterval(poll);
        console.warn(`[setup] Port ${BACKEND_PORT} still busy after 5 s — proceeding anyway.`);
        resolve();
      }
    }, 200);
  });
}

// ── Start FastAPI backend ────────────────────────────────────────────────────
let _backendRetried = false; // only auto-retry once per app session

function startBackend() {
  sendStatus("Starting backend server…");
  _backendRetried = false;

  // run.sh activates the venv and starts uvicorn.
  // Pass VENV_DIR and CACHE_DIR so run.sh uses the writable userData location.
  // FULL_PATH includes /opt/homebrew/bin so ffmpeg is found even in a .app bundle.
  backendProcess = spawn("bash", [RUN_SH], {
    cwd: SERVER_DIR,
    env: {
      ...process.env,
      PATH: FULL_PATH,
      PYTHONUNBUFFERED: "1",
      ELECTRON_APP: "1",
      ELECTRON_RUN_AS_NODE: undefined, // never inherit this into backend
      VENV_DIR: VENV_DIR,
      CACHE_DIR: CACHE_DIR,
      // Allow the React static server (random port) to call the backend.
      CORS_ORIGINS: [
        `http://127.0.0.1:${staticPort}`,
        `http://localhost:${staticPort}`,
        // Keep dev ports so `npm start` still works without this env var.
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
      ].join(","),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  // Accumulate last 3 KB of stderr so the crash dialog shows the real error.
  const LOG_PATH = path.join(USER_DATA, "backend.log");
  let stderrTail = "";
  const logStream = fs.createWriteStream(LOG_PATH, { flags: "w" });

  backendProcess.stdout.on("data", (d) => {
    const line = d.toString().trim();
    logStream.write("[stdout] " + line + "\n");
    if (line) console.log("[backend]", line);
    if (line.includes("Application startup complete")) {
      sendStatus("Ready!");
    }
  });

  backendProcess.stderr.on("data", (d) => {
    const chunk = d.toString();
    logStream.write(chunk);
    stderrTail += chunk;
    // Keep only last 3 KB to avoid unbounded memory
    if (stderrTail.length > 3000) stderrTail = stderrTail.slice(-3000);
    const line = chunk.trim();
    if (line) console.error("[backend-err]", line);
  });

  const backendStartTime = Date.now();

  backendProcess.on("exit", (code, signal) => {
    logStream.end();
    console.log("[backend] exited with code", code, "signal", signal);
    if (intentionalShutdown || signal === "SIGTERM") return;

    const errSnippet = stderrTail.trim().slice(-800) || "(no stderr captured)";
    console.error("[backend] crash log:\n" + errSnippet);

    // Auto-retry once if backend crashed quickly due to port conflict.
    // "address already in use" means a stray uvicorn from a previous session
    // survived; free the port and restart rather than showing an error dialog.
    const uptime = Date.now() - backendStartTime;
    const isPortConflict = errSnippet.includes("address already in use") ||
                           errSnippet.includes("Errno 48") ||
                           errSnippet.includes("Errno 98");
    if (code === 1 && uptime < 8000 && isPortConflict && !_backendRetried) {
      _backendRetried = true;
      console.log("[backend] Port conflict detected — freeing port and retrying…");
      sendStatus("Port conflict — retrying backend…");
      freeBackendPort().then(() => {
        // Brief pause so the OS fully releases the socket.
        setTimeout(() => startBackend(), 800);
      });
      return;
    }

    // Show the actual error so we know what to fix
    const target = mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;
    const msg = `Backend exited (code ${code}).\n\nLog: ${LOG_PATH}\n\n--- Last output ---\n${errSnippet}`;
    if (target) {
      dialog.showErrorBox("Backend stopped", msg);
    } else if (loadingWindow && !loadingWindow.isDestroyed()) {
      // Crashed during startup — show on loading screen
      loadingWindow.webContents.send("error", msg);
    }
  });
}

// ── App lifecycle ────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  createLoadingWindow();

  try {
    await startStaticServer();
    await ensureSetup();
    await freeBackendPort();
    startBackend();
    ensureOllama(); // fire-and-forget — app works without it

    sendStatus("Waiting for server to be ready…");
    await waitForBackend();

    backendReady = true;
    createMainWindow();
  } catch (err) {
    console.error("Startup failed:", err);
    if (loadingWindow && !loadingWindow.isDestroyed()) {
      loadingWindow.webContents.send("error", err.message);
    } else {
      dialog.showErrorBox("Startup failed", err.message);
      app.quit();
    }
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (mainWindow === null && backendReady) createMainWindow();
});

app.on("before-quit", () => {
  intentionalShutdown = true;
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
  }
  if (ollamaProcess) {
    ollamaProcess.kill("SIGTERM");
  }
  if (staticServer) {
    staticServer.close();
  }
});

// ── IPC ──────────────────────────────────────────────────────────────────────
ipcMain.handle("quit-error", () => {
  app.quit();
});
