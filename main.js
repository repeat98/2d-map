/************************************************************************************/
/**                           Your Original Script + Spectral Columns              **/
/************************************************************************************/

const { app, BrowserWindow, ipcMain, dialog, session } = require("electron");
const { spawn } = require("child_process");
const express = require("express");
const sqlite3 = require("sqlite3").verbose();
const path = require("path");
const cors = require("cors");
const fs = require("fs");
const port = 3000;

const expressApp = express();

// Initialize the Express app with middleware
expressApp.use(express.json({ limit: "1024mb" })); // Increase payload limit
expressApp.use(cors());

// Get the path to the user's data directory
const appPath = app.getAppPath();
const parentPath = path.dirname(appPath);

// Paths for development
const dbPath = path.resolve(appPath, "db/tracks.db");
const scriptPath = path.join(appPath, "scripts", "analyze.py");
const pythonPath = path.join(appPath, "scripts", "venv", "bin", "python");
const extractArtworksPath = path.join(
  appPath,
  "scripts",
  "extract_artworks.py"
);

// Paths for packaging (commented out for development)
// const dbPath = path.resolve(parentPath, "app.asar.unpacked/db/tracks.db");
// const scriptPath = path.join(
//   parentPath,
//   "app.asar.unpacked",
//   "scripts",
//   "analyze.py"
// );
// const pythonPath = path.join(
//   parentPath,
//   "app.asar.unpacked",
//   "scripts",
//   "venv",
//   "bin",
//   "python"
// );
// const extractArtworksPath = path.join(
//   parentPath,
//   "app.asar.unpacked",
//   "scripts",
//   "extract_artworks.py"
// );

const analyzeBinaryPath = path.join(appPath, "scripts", "analyze"); // Ensure correct path

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error("Database connection error:", err.message);
  } else {
    console.log("Connected to the SQLite database.");
    // Initialize the database schema
    initDatabase();
  }
});

// Define the waveforms directory
const waveformsDir = path.join(__dirname, "waveforms");

// Ensure the waveforms directory exists
if (!fs.existsSync(waveformsDir)) {
  fs.mkdirSync(waveformsDir, { recursive: true });
  console.log(`Created waveforms directory at ${waveformsDir}`);
} else {
  console.log(`Waveforms directory already exists at ${waveformsDir}`);
}

let mainWindow;

// Create the main application window
const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1564,
    height: 860,
    titleBarStyle: "hidden",
    trafficLightPosition: { x: 10, y: 10 },
    webPreferences: {
      preload: path.join(__dirname, "preload.js"), // Use the preload script
      contextIsolation: true, // Ensure context isolation is enabled
      nodeIntegration: false, // Ensure nodeIntegration is disabled
    },
  });

  mainWindow.loadFile(path.join(__dirname, "dist", "index.html"));
};

app.whenReady().then(() => {
  // Configure session headers with a more secure CSP
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self';" +
          "script-src 'self' 'unsafe-inline' 'unsafe-eval';" + // Required for React and PixiJS
          "style-src 'self' 'unsafe-inline';" +
          "img-src 'self' data: http://localhost:3000 blob: file:;" + // Allow data URLs for images
          "media-src 'self' http://localhost:3000 blob: file: data:;" + // Allow data URLs for media
          "worker-src 'self' blob:;" + // For web workers
          "child-src 'self' blob:;" + // For web workers
          "connect-src 'self' http://localhost:3000 data: blob:;" // Allow data and blob URLs for connections
        ]
      }
    });
  });

  createWindow();
  initRoutes();

  // IPC Handlers
  ipcMain.handle("open-file-dialog", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile", "openDirectory", "multiSelections"], // Allow selecting files, directories, and multiple files
      filters: [
        {
          name: "Audio Files",
          extensions: [
            "wav",
            "mp3",
            "flac",
            "m4a",
            "aiff",
            "aif",
            "ogg",
            "opus",
            "wma",
            "aac",
          ],
        },
      ],
    });
    return result.filePaths; // Return an array of file paths
  });

  ipcMain.handle("get-file-path", async (event, fileName) => {
    // Optionally, perform operations or validations here
    return path.resolve(fileName); // Resolve the path if needed
  });

  // Handle IPC communication for extracting artworks
  ipcMain.handle("extract-artworks", async (event, filePath) => {
    return new Promise((resolve, reject) => {
      const extractArtworks = spawn(pythonPath, [
        extractArtworksPath,
        filePath,
      ]);

      let output = "";
      let error = "";

      extractArtworks.stdout.on("data", (data) => {
        output += data.toString();
      });

      extractArtworks.stderr.on("data", (data) => {
        error += data.toString();
      });

      extractArtworks.on("close", (code) => {
        if (code === 0) {
          resolve(output); // Or parse as needed
        } else {
          reject(`Python script exited with code ${code}: ${error}`);
        }
      });
    });
  });

  // Handle IPC communication for running the analyze script
  ipcMain.handle("run-python-script", async (event, filePath) => {
    return new Promise((resolve, reject) => {
      // Spawn the binary process with the filePath as an argument
      const analyzeProcess = spawn(analyzeBinaryPath, [
        filePath,
        "--threads",
        "4",
        "--plot", // Include the --plot flag if needed
      ]);

      let output = "";
      let errorOutput = "";
      const outputBuffer = [];
      const intervalMs = 1000; // Adjust the interval as needed

      // Set up an interval to send updates to the renderer
      const intervalId = setInterval(() => {
        if (outputBuffer.length > 0) {
          console.log("Sending output to renderer:", outputBuffer.join("")); // Debugging
          event.sender.send("python-output", outputBuffer.join(""));
          outputBuffer.length = 0; // Clear buffer after sending
        }
      }, intervalMs);

      // Handle stdout stream
      analyzeProcess.stdout.on("data", (data) => {
        output += data.toString();
        outputBuffer.push(data.toString()); // Add data to the buffer
      });

      // Handle stderr stream
      analyzeProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
      });

      // Handle process close event
      analyzeProcess.on("close", (code) => {
        clearInterval(intervalId); // Clear the interval when process closes
        if (code === 0) {
          console.log("Final output sent to renderer:", output); // Debugging
          event.sender.send("python-output", output);
          resolve(output.trim());
        } else {
          console.error("Final error sent to renderer:", errorOutput); // Debugging
          event.sender.send("python-error", errorOutput);
          reject(
            new Error(`Process exited with code ${code}: ${errorOutput.trim()}`)
          );
        }
      });

      // Handle process error event
      analyzeProcess.on("error", (err) => {
        clearInterval(intervalId); // Clear the interval on error
        console.error("Error in analyze process:", err.message); // Debugging
        event.sender.send(
          "python-error",
          `Failed to start analyze process: ${err.message}`
        );
        reject(new Error(`Failed to start analyze process: ${err.message}`));
      });
    });
  });
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// Initialize the database schema
function initDatabase() {
  // Check if new columns exist in 'classified_tracks' table
  const checkColumnsQuery = "PRAGMA table_info(classified_tracks);";
  db.all(checkColumnsQuery, [], (err, columns) => {
    if (err) {
      console.error("Error retrieving table info:", err.message);
      return;
    }
    const columnNames = columns.map((column) => column.name);
    const newColumns = [
      { name: "idx", type: "INTEGER" },
      // Add 'x' and 'y' columns here
      { name: "x", type: "REAL" },
      { name: "y", type: "REAL" },

      // ***** ADDED SPECTRAL FEATURE COLUMNS BELOW *****
      { name: "spectral_centroid", type: "REAL" },
      { name: "spectral_bandwidth", type: "REAL" },
      { name: "spectral_rolloff", type: "REAL" },
      { name: "spectral_contrast", type: "REAL" },
      { name: "spectral_flatness", type: "REAL" },
      // ***** END SPECTRAL FEATURE COLUMNS *****
    ];

    const addColumnPromises = newColumns.map((column) => {
      return new Promise((resolve, reject) => {
        if (!columnNames.includes(column.name)) {
          const addColumnQuery = `ALTER TABLE classified_tracks ADD COLUMN ${column.name} ${column.type};`;
          db.run(addColumnQuery, [], (err) => {
            if (err) {
              console.error(`Error adding ${column.name} column:`, err.message);
              reject(err);
            } else {
              console.log(
                `Added ${column.name} column to classified_tracks table.`
              );
              resolve();
            }
          });
        } else {
          console.log(
            `${column.name} column already exists in classified_tracks table.`
          );
          resolve();
        }
      });
    });

    Promise.all(addColumnPromises)
      .then(() => {
        // Recompute 'idx' values for existing tracks
        recomputeIdx();
      })
      .catch((err) => {
        console.error("Error adding new columns:", err.message);
      });
  });
}

// Function to recompute idx based on tags
function recomputeIdx() {
  const selectTracksQuery = `
    SELECT id FROM classified_tracks
    ORDER BY tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10;
  `;
  db.all(selectTracksQuery, [], (err, rows) => {
    if (err) {
      console.error("Error selecting tracks for idx recompute:", err.message);
      return;
    }
    const updateIdxStmt = db.prepare(
      "UPDATE classified_tracks SET idx = ? WHERE id = ?"
    );
    rows.forEach((row, index) => {
      updateIdxStmt.run(index + 1, row.id, (err) => {
        if (err) {
          console.error(
            `Error updating idx for track id ${row.id}:`,
            err.message
          );
        }
      });
    });
    updateIdxStmt.finalize();
    console.log("Recomputed idx for classified_tracks.");
  });
}

// Initialize Express routes
function initRoutes() {
  // Route to get all classified tracks ordered by idx
  expressApp.get("/tracks", (req, res) => {
    const checkTableQuery = `
      SELECT name FROM sqlite_master WHERE type='table' AND name='classified_tracks';
    `;

    db.get(checkTableQuery, (err, row) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to check if table exists" });
        return;
      }

      if (!row) {
        // Table does not exist
        res.status(404).json({ error: "Table classified_tracks does not exist" });
        return;
      }

      // Table exists, proceed with retrieving tracks ordered by idx
      const query = `
        SELECT id, idx, path, artist, title, album, year, bpm, time, key, date,
               features, instrument_features,
               tag1, tag1_prob,
               tag2, tag2_prob,
               tag3, tag3_prob,
               tag4, tag4_prob,
               tag5, tag5_prob,
               tag6, tag6_prob,
               tag7, tag7_prob,
               tag8, tag8_prob,
               tag9, tag9_prob,
               tag10, tag10_prob,
               artwork_path, artwork_thumbnail_path,
               x, y,
               spectral_centroid, spectral_bandwidth, spectral_rolloff,
               spectral_contrast, spectral_flatness, rms,
               happiness, party, aggressive, danceability,
               relaxed, sad, engagement, approachability
        FROM classified_tracks
        ORDER BY idx ASC
      `;

      db.all(query, [], (err, rows) => {
        if (err) {
          console.error("SQL Error:", err.message);
          res.status(500).json({ error: "Failed to retrieve tracks" });
          return;
        }
        // Parse the 'features' and 'instrument_features' JSON strings into objects for each track
        const tracksWithParsedFeatures = rows.map(track => {
          try {
            return {
              ...track,
              features: track.features ? JSON.parse(track.features) : null,
              instrument_features: track.instrument_features ? JSON.parse(track.instrument_features) : null
            };
          } catch (e) {
            console.error(`Error parsing features for track ID ${track.id}:`, e);
            return {
              ...track,
              features: null,
              instrument_features: null
            };
          }
        });
        res.json(tracksWithParsedFeatures);
      });
    });
  });


  expressApp.post("/recommend", (req, res) => {
    const queryVector = req.body.vector; // Expect a JSON array representing your query embedding.
    if (!queryVector || !Array.isArray(queryVector)) {
      return res.status(400).json({ error: "Invalid query vector" });
    }

    // Path to your Python recommendation script
    const scriptPath = path.join(
      __dirname,
      "scripts",
      "query_recommendations.py"
    );

    // Pass the query vector as a JSON string argument
    const pythonProcess = spawn("python3", [
      scriptPath,
      JSON.stringify({ vector: queryVector }),
    ]);

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output);
          res.json(result);
        } catch (err) {
          res.status(500).json({ error: "Failed to parse recommendations" });
        }
      } else {
        res.status(500).json({ error: errorOutput || "Python process error" });
      }
    });
  });

  // Route to get a track by ID
  expressApp.get("/tracks/:id", (req, res) => {
    const trackId = req.params.id;

    const query = `
      SELECT id, idx, path, artist, title, album, year, BPM, TIME, DANCE, KEY, DATE,
             tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10,
             artwork_path, artwork_thumbnail_path,
             x, y,

             -- ***** ADD SPECTRAL FEATURES HERE *****
             spectral_centroid,
             spectral_bandwidth,
             spectral_rolloff,
             spectral_contrast,
             spectral_flatness
             -- ***** END SPECTRAL FEATURES *****
      FROM classified_tracks
      WHERE id = ?
    `;

    db.get(query, [trackId], (err, row) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to retrieve track" });
        return;
      }

      if (!row) {
        res.status(404).json({ error: "Track not found" });
        return;
      }

      res.json(row);
    });
  });

  // Route to add a new classified track
  expressApp.post("/tracks", (req, res) => {
    const {
      path,
      artist,
      title,
      album,
      year,
      BPM,
      TIME,
      DANCE,
      KEY,
      DATE,
      tag1,
      tag2,
      tag3,
      tag4,
      tag5,
      tag6,
      tag7,
      tag8,
      tag9,
      tag10,
      artwork_path,
      artwork_thumbnail_path,
      x, // Optional: If you plan to set x and y from Electron
      y,
    } = req.body;

    const sql = `
      INSERT INTO classified_tracks (
        path, artist, title, album, year, BPM, TIME, DANCE, KEY, DATE,
        tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10,
        artwork_path, artwork_thumbnail_path,
        x, y
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    db.run(
      sql,
      [
        path,
        artist,
        title,
        album,
        year,
        BPM,
        TIME,
        DANCE,
        KEY,
        DATE,
        tag1,
        tag2,
        tag3,
        tag4,
        tag5,
        tag6,
        tag7,
        tag8,
        tag9,
        tag10,
        artwork_path,
        artwork_thumbnail_path,
        x, // Pass x if available
        y, // Pass y if available
      ],
      function (err) {
        if (err) {
          console.error("SQL Error:", err.message);
          res.status(500).json({ error: "Failed to insert track" });
          return;
        }
        // Recompute idx after inserting a new track
        recomputeIdx();
        res.status(201).json({ id: this.lastID });
      }
    );
  });

  // Route to delete a track by ID
  expressApp.delete("/tracks/:id", (req, res) => {
    const { id } = req.params;
    db.run("DELETE FROM classified_tracks WHERE id = ?", [id], function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to delete track" });
        return;
      }
      if (this.changes === 0) {
        res.status(404).json({ error: "Track not found" });
        return;
      }
      // Recompute idx after deleting a track
      recomputeIdx();
      res.json({ message: "Track deleted successfully" });
    });
  });

  // Route to update tags for a specific track
  expressApp.put("/tracks/:id/tags", (req, res) => {
    const trackId = req.params.id;
    const { tags } = req.body; // Expecting an array of tags

    if (!tags || !Array.isArray(tags)) {
      return res.status(400).json({ error: "Tags must be an array" });
    }

    if (tags.length > 10) {
      return res
        .status(400)
        .json({ error: "A maximum of 10 tags are allowed" });
    }

    // Prepare tag1 to tag10 fields
    const tagFields = Array(10).fill(null);
    tags.forEach((tag, index) => {
      tagFields[index] = tag;
    });

    // Construct the SQL query dynamically based on the number of tags
    let sql = `UPDATE classified_tracks SET `;
    const params = [];
    tagFields.forEach((tag, index) => {
      sql += `tag${index + 1} = ?, `;
      params.push(tag);
    });
    // Remove the trailing comma and space
    sql = sql.slice(0, -2);
    sql += ` WHERE id = ?`;
    params.push(trackId);

    db.run(sql, params, function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        return res.status(500).json({ error: "Failed to update tags" });
      }

      if (this.changes === 0) {
        return res.status(404).json({ error: "Track not found" });
      }

      // Recompute idx after updating tags
      recomputeIdx();
      res.status(200).json({ message: "Tags updated successfully" });
    });
  });

  // Route to save column sizes
  expressApp.post("/column-size", (req, res) => {
    const { index, width } = req.body;
    const query =
      "INSERT OR REPLACE INTO column_sizes (column_index, col_width) VALUES (?, ?)";

    db.run(query, [index, width], function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to save column size" });
        return;
      }
      res.status(200).json({ message: "Column size saved successfully" });
    });
  });

  expressApp.get("/column-size", (req, res) => {
    const createColumnSizesTable = `
      CREATE TABLE IF NOT EXISTS column_sizes (
        column_index INTEGER PRIMARY KEY,
        col_width TEXT
      );
    `;

    db.run(createColumnSizesTable, (err) => {
      if (err) {
        console.error("Failed to create column_sizes table:", err.message);
        res.status(500).json({ error: "Failed to initialize database table" });
        return;
      }

      console.log("Table created or already exists.");

      // Query to get column sizes with quotes around reserved keywords
      const query =
        'SELECT "column_index" AS "index", "col_width" AS "width" FROM column_sizes ORDER BY "column_index"';
      db.all(query, [], (err, rows) => {
        if (err) {
          console.error(
            "SQL Error while retrieving column sizes:",
            err.message
          );
          res.status(500).json({ error: "Failed to retrieve column sizes" });
          return;
        }

        console.log("Column sizes retrieved successfully:", rows);
        res.json(rows);
      });
    });
  });

  // Route to search tracks based on a query
  expressApp.get("/search", (req, res) => {
    const { query } = req.query; // Retrieve the search query from the request

    if (!query) {
      return res.status(400).json({ error: "Search query is required" });
    }

    const searchQuery = `
      SELECT id, idx, path, artist, title, album, year, BPM, TIME, DANCE, KEY, DATE,
             tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10,
             artwork_path, artwork_thumbnail_path,
             x, y,

             -- ***** ADD SPECTRAL FEATURES HERE *****
             spectral_centroid,
             spectral_bandwidth,
             spectral_rolloff,
             spectral_contrast,
             spectral_flatness
             -- ***** END SPECTRAL FEATURES *****
      FROM classified_tracks
      WHERE artist LIKE ? OR title LIKE ? OR album LIKE ? OR year LIKE ? OR BPM LIKE ? OR TIME LIKE ? 
            OR DANCE LIKE ? OR KEY LIKE ? OR DATE LIKE ?
      ORDER BY idx
    `;

    const params = Array(9).fill(`%${query}%`);

    db.all(searchQuery, params, (err, rows) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to search tracks" });
        return;
      }

      res.json(rows);
    });
  });

  // Route to save waveform data for a track based on its id
  expressApp.put("/tracks/waveform", (req, res) => {
    const { id, waveform } = req.body;

    if (!id || !waveform) {
      return res
        .status(400)
        .json({ error: "Track id and waveform data are required" });
    }

    const waveformPath = path.join(waveformsDir, `${id}.json`);

    fs.writeFile(waveformPath, JSON.stringify(waveform), "utf8", (err) => {
      if (err) {
        console.error("Error saving waveform to file:", err.message);
        return res.status(500).json({ error: "Failed to save waveform" });
      }

      res.status(200).json({ message: "Waveform saved successfully" });
    });
  });

  // Route to get waveform data for a track by ID
  expressApp.get("/tracks/waveform/:id", (req, res) => {
    const trackId = req.params.id;
    const waveformPath = path.join(waveformsDir, `${trackId}.json`);

    fs.readFile(waveformPath, "utf8", (err, data) => {
      if (err) {
        if (err.code === "ENOENT") {
          // File does not exist
          res.status(404).json({ error: "Waveform not found" });
        } else {
          console.error("Error reading waveform file:", err.message);
          res.status(500).json({ error: "Failed to retrieve waveform data" });
        }
        return;
      }

      try {
        const waveformData = JSON.parse(data);
        res.json({ waveform: waveformData });
      } catch (parseError) {
        console.error("Error parsing waveform data:", parseError.message);
        res.status(500).json({ error: "Error parsing waveform data" });
      }
    });
  });

  // Create playlists table if it does not exist
  const createPlaylistsTable = `
    CREATE TABLE IF NOT EXISTS playlists (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      tracks TEXT
    );
  `;

  db.run(createPlaylistsTable, (err) => {
    if (err) {
      console.error("Failed to create playlists table:", err.message);
    } else {
      console.log("Playlists table created or already exists.");
    }
  });

  // Route to get all playlists
  expressApp.get("/playlists", (req, res) => {
    const query = "SELECT id, name FROM playlists ORDER BY id";

    db.all(query, [], (err, rows) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to retrieve playlists" });
        return;
      }

      res.json(rows);
    });
  });

  // Route to get a playlist by ID
  expressApp.get("/playlists/:id", (req, res) => {
    const { id } = req.params;
    const sql = `SELECT * FROM playlists WHERE id = ?`;
    db.get(sql, [id], (err, row) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to retrieve playlist" });
        return;
      }
      if (!row) {
        res.status(404).json({ error: "Playlist not found" });
        return;
      }
      res.json(row);
    });
  });

  // Route to create a new playlist
  expressApp.post("/playlists", (req, res) => {
    const { name, tracks } = req.body;
    const sql = `INSERT INTO playlists (name, tracks) VALUES (?, ?)`;
    db.run(sql, [name, tracks], function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to create playlist" });
        return;
      }
      res.status(201).json({ id: this.lastID });
    });
  });

  // Route to update a playlist
  expressApp.put("/playlists/:id", (req, res) => {
    const playlistId = req.params.id;
    const { tracks } = req.body;

    if (!tracks) {
      return res.status(400).json({ error: "Tracks data is required" });
    }

    // Ensure tracks is a valid JSON string
    let parsedTracks;
    try {
      parsedTracks = JSON.parse(tracks);
      if (!Array.isArray(parsedTracks)) {
        throw new Error("Tracks must be an array");
      }
    } catch (e) {
      return res.status(400).json({ error: "Invalid JSON format for tracks" });
    }

    const sql = `UPDATE playlists SET tracks = ? WHERE id = ?`;

    db.run(sql, [JSON.stringify(parsedTracks), playlistId], function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        return res.status(500).json({ error: "Failed to update playlist" });
      }

      res.status(200).json({ message: "Playlist updated successfully" });
    });
  });

  // Route to delete a playlist by ID
  expressApp.delete("/playlists/:id", (req, res) => {
    const playlistId = req.params.id;

    const sql = "DELETE FROM playlists WHERE id = ?";

    db.run(sql, [playlistId], function (err) {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to delete playlist" });
        return;
      }

      if (this.changes === 0) {
        res.status(404).json({ error: "Playlist not found" });
        return;
      }

      res.json({ message: "Playlist deleted successfully" });
    });
  });

  // Route to filter tracks by selected tags
  expressApp.get("/tracks/filter", (req, res) => {
    const { tags } = req.query; // Retrieve tags from the query parameters

    if (!tags || !Array.isArray(tags) || tags.length === 0) {
      return res.status(400).json({ error: "Tags are required" });
    }

    // Construct the SQL query dynamically based on the number of tags
    let sql =
      "SELECT id, idx, path, artist, title, album, year, BPM, TIME, DANCE, KEY, DATE, " +
      "tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, " +
      "x, y, " +
      // ***** ADD SPECTRAL FEATURES HERE *****
      "spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast, spectral_flatness " +
      // ***** END SPECTRAL FEATURES *****
      "FROM classified_tracks WHERE ";

    // Prepare placeholders and parameters for the query
    const placeholders = [];
    const params = [];

    // Generate conditions for each tag
    tags.forEach((tag, index) => {
      placeholders.push(
        `(tag1 LIKE ? OR tag2 LIKE ? OR tag3 LIKE ? OR tag4 LIKE ? OR tag5 LIKE ? ` +
          `OR tag6 LIKE ? OR tag7 LIKE ? OR tag8 LIKE ? OR tag9 LIKE ? OR tag10 LIKE ?)`
      );
      const likeTag = `%${tag}%`;
      params.push(
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag,
        likeTag
      );
    });

    // Join conditions with OR to handle multiple tags
    sql += placeholders.join(" OR ");

    // Execute the query
    db.all(sql, params, (err, rows) => {
      if (err) {
        console.error("SQL Error:", err.message);
        res.status(500).json({ error: "Failed to filter tracks" });
        return;
      }

      res.json(rows);
    });
  });

  // Route to get the artwork image for a track
  expressApp.get("/artwork/:trackId", (req, res) => {
    const trackId = req.params.trackId;

    // Query to get the artwork path from the database
    const sql = `SELECT artwork_path FROM classified_tracks WHERE id = ?`;

    db.get(sql, [trackId], (err, row) => {
      if (err) {
        console.error("Database Error:", err.message);
        return res.status(500).json({ error: "Failed to fetch artwork" });
      }

      const defaultArtworkPath = path.resolve("assets/default-artwork.png");

      // Check if a valid artwork path exists in the database
      if (row && row.artwork_path) {
        const artworkPath = path.resolve(row.artwork_path);

        // Check if the file exists before attempting to send it
        fs.access(artworkPath, fs.constants.F_OK, (err) => {
          if (err) {
            console.warn(
              `Artwork file not found at ${artworkPath}. Sending default artwork.`
            );
            return res.sendFile(defaultArtworkPath); // Send default artwork if the file is missing
          }

          // If file exists, send the requested artwork file
          res.sendFile(artworkPath, (err) => {
            if (err) {
              console.error("Error sending artwork file:", err.message);
              res.status(500).json({ error: "Failed to send artwork" });
            }
          });
        });
      } else {
        // If no artwork path is found in the database, send the default artwork
        console.log(
          `No artwork path found for track ${trackId}. Sending default artwork.`
        );
        res.sendFile(defaultArtworkPath);
      }
    });
  });

  // Start the server
  expressApp.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });

  // Close the database connection on exit
  process.on("SIGINT", () => {
    db.close((err) => {
      if (err) {
        console.error("Error closing the database connection", err);
      }
      console.log("Database connection closed.");
      process.exit(0);
    });
  });
}
