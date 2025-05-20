const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  openFileDialog: () => ipcRenderer.invoke("open-file-dialog"),
  getFileUrl: (filePath) => {
    try {
      return pathToFileURL(filePath).href;
    } catch (error) {
      console.error('Error converting path to file URL:', error);
      return null;
    }
  },
  addWaveformColumnIfNotExists: () => ipcRenderer.invoke('add-waveform-column'),
  runPythonScript: (filePath) =>
    ipcRenderer.invoke("run-python-script", filePath),
  on: (channel, callback) => {
    // Whitelist channels to avoid security issues
    const validChannels = ["python-output", "python-error"];
    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, (event, ...args) => callback(event, ...args));
    }
    
  },
  off: (channel, callback) => {
    // Whitelist channels to avoid security issues
    const validChannels = ["python-output", "python-error"];
    if (validChannels.includes(channel)) {
      ipcRenderer.removeListener(channel, callback);
    }
  },
  extractArtworks: (filePath) =>
    ipcRenderer.invoke("extract-artworks", filePath),
});

contextBridge.exposeInMainWorld("electron", {
  setZoomFactor: (factor) => ipcRenderer.send("set-zoom-factor", factor),
});



// All the Node.js APIs are available in the preload process.
// It has the same sandbox as a Chrome extension.
window.addEventListener("DOMContentLoaded", () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector);
    if (element) element.innerText = text;
  };

  for (const dependency of ["chrome", "node", "electron"]) {
    replaceText(`${dependency}-version`, process.versions[dependency]);
  }
});
