/**
 * Electron preload — exposes a minimal safe bridge to the renderer.
 * contextIsolation: true, so we use contextBridge.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  onStatus: (cb) => ipcRenderer.on("status", (_e, msg) => cb(msg)),
  onError:  (cb) => ipcRenderer.on("error",  (_e, msg) => cb(msg)),
  quitApp:  ()   => ipcRenderer.invoke("quit-error"),
});
