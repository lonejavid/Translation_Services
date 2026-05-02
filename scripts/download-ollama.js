/**
 * Downloads and extracts the Ollama macOS release into electron/resources/ollama/.
 * Run automatically before `npm run dist` via the updated dist script.
 * Idempotent — skipped if the binary is already present and valid.
 */

const { execFileSync, execSync } = require("child_process");
const https = require("https");
const fs = require("fs");
const path = require("path");

const OUT_DIR = path.join(__dirname, "..", "electron", "resources", "ollama");
const OLLAMA_BIN = path.join(OUT_DIR, "ollama");
const TGZ_URL =
  "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz";

// Already downloaded and valid (>10 MB binary)
if (fs.existsSync(OLLAMA_BIN) && fs.statSync(OLLAMA_BIN).size > 10_000_000) {
  console.log("[download-ollama] Ollama already present — skipping download.");
  process.exit(0);
}

fs.mkdirSync(OUT_DIR, { recursive: true });

const TMP_TGZ = path.join(OUT_DIR, "_ollama-darwin.tgz");

console.log("[download-ollama] Downloading ollama-darwin.tgz …");
try {
  execFileSync("curl", ["-L", "--progress-bar", "-o", TMP_TGZ, TGZ_URL], {
    stdio: "inherit",
  });
} catch (err) {
  console.error("[download-ollama] Download failed:", err.message);
  process.exit(0); // non-fatal — app works without Ollama
}

console.log("[download-ollama] Extracting …");
try {
  execFileSync("tar", ["-xzf", TMP_TGZ, "-C", OUT_DIR], { stdio: "inherit" });
  fs.unlinkSync(TMP_TGZ);
} catch (err) {
  console.error("[download-ollama] Extraction failed:", err.message);
  try { fs.unlinkSync(TMP_TGZ); } catch (_) {}
  process.exit(0); // non-fatal
}

// electron-builder cannot copy symlinks — replace each symlink with a real copy
// of the file it points to (typical macOS .dylib versioned aliases).
function dereferenceSymlinks(dir) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
  catch (_) { return; }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isSymbolicLink()) {
      const target = fs.readlinkSync(full);
      const targetFull = path.isAbsolute(target) ? target : path.join(dir, target);
      if (fs.existsSync(targetFull) && fs.statSync(targetFull).isFile()) {
        fs.unlinkSync(full);
        fs.copyFileSync(targetFull, full);
        console.log(`[download-ollama] Deref symlink: ${entry.name} → ${target}`);
      }
    } else if (entry.isDirectory()) {
      dereferenceSymlinks(full);
    }
  }
}

console.log("[download-ollama] Resolving symlinks for electron-builder compatibility…");
dereferenceSymlinks(OUT_DIR);

if (fs.existsSync(OLLAMA_BIN)) {
  fs.chmodSync(OLLAMA_BIN, 0o755);
  const sizeMB = Math.round(fs.statSync(OLLAMA_BIN).size / 1024 / 1024);
  console.log(`[download-ollama] Done — ollama binary is ${sizeMB} MB`);
} else {
  console.error("[download-ollama] ollama binary not found after extraction.");
}
