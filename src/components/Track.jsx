// src/components/Track/Track.jsx

import React, { useContext } from "react";
import "./Track.scss";
import PropTypes from "prop-types";
import defaultArtwork from "../../assets/default-artwork.png";
import Waveform from "./Waveform";
import { PlaybackContext } from "../context/PlaybackContext";
import { getTagColors } from "../utils/tagUtils"; // Import getTagColors

const Track = ({ track, column }) => {
  const { setCurrentTrack } = useContext(PlaybackContext);

  const handlePlay = () => {
    setCurrentTrack(track);
  };

  // Helper function to strip the file extension from the filename
  const stripSuffix = (filename) => {
    if (typeof filename !== "string") {
      console.error("Invalid filename:", filename);
      return "";
    }

    // Extract the base filename without the path
    const baseName = filename.substring(filename.lastIndexOf("/") + 1);

    if (!baseName) {
      console.warn(
        "Base name extraction resulted in an empty string for filename:",
        filename
      );
      return "";
    }

    // Split by '.' and remove the last segment (extension)
    const parts = baseName.split(".");
    if (parts.length === 1) {
      // No extension found
      return baseName;
    }

    // Remove the last part (extension) and join the rest
    const nameWithoutSuffix = parts.slice(0, -1).join(".") || baseName;
    return nameWithoutSuffix;
  };

  // Convert traditional key notation to Camelot notation
  const convertToCamelot = (key) => {
    const camelotWheel = {
      "C Major": "8B",
      "C# Major": "3B",
      "Db Major": "3B",
      "D Major": "10B",
      "D# Major": "5B",
      "Eb Major": "5B",
      "E Major": "12B",
      "F Major": "7B",
      "F# Major": "2B",
      "Gb Major": "2B",
      "G Major": "9B",
      "G# Major": "4B",
      "Ab Major": "4B",
      "A Major": "11B",
      "A# Major": "6B",
      "Bb Major": "6B",
      "B Major": "1B",

      "C Minor": "5A",
      "C# Minor": "12A",
      "Db Minor": "12A",
      "D Minor": "7A",
      "D# Minor": "2A",
      "Eb Minor": "2A",
      "E Minor": "9A",
      "F Minor": "4A",
      "F# Minor": "11A",
      "Gb Minor": "11A",
      "G Minor": "6A",
      "G# Minor": "1A",
      "Ab Minor": "1A",
      "A Minor": "8A",
      "A# Minor": "3A",
      "Bb Minor": "3A",
      "B Minor": "10A",
    };
    return camelotWheel[key] || "Unknown Key"; // Return 'Unknown Key' if not found
  };

  switch (column) {
    case "Cover":
      const artworkPath = track.artwork_thumbnail_path || defaultArtwork;
      return (
        <img
          src={artworkPath}
          alt={`${track.title} Cover`}
          width="50"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = defaultArtwork; // Fallback to default image on error
          }}
          className="track-cover"
        />
      );
    case "Title":
      if (
        typeof track.title === "string" &&
        track.title.trim().toLowerCase() === "unknown title"
      ) {
        if (typeof track.path === "string" && track.path.trim() !== "") {
          return <span className="track-title">{stripSuffix(track.path)}</span>;
        } else {
          console.warn(
            `Track ID: ${track.id} has title "Unknown Title" but missing or invalid path: "${track.path}"`
          );
          return <span className="track-title">Unknown Track</span>;
        }
      }
      return <span className="track-title">{track.title}</span>;
    case "Artist":
      return <span className="track-artist">{track.artist}</span>;
    case "Album":
      return <span className="track-album">{track.album || "Unknown Album"}</span>;
    case "BPM":
      return (
        <span className="track-bpm">
          {parseFloat(Math.floor(track.BPM)) || 0}
        </span>
      );
    case "Year":
      return <span className="track-year">{track.year || "Unknown Year"}</span>;
    case "TIME":
      return <span className="track-time">{track.TIME || "0:00"}</span>;
    case "Key":
      return <span className="track-key">{convertToCamelot(track.KEY)}</span>;
    case "Date":
      return <span className="track-date">{track.DATE.split(" ")[0]}</span>;
    case "Tags":
      // Collect all tag fields into an array
      const tags = [
        track.tag1,
        track.tag2,
        track.tag3,
        track.tag4,
        track.tag5,
        track.tag6,
        track.tag7,
        track.tag8,
        track.tag9,
        track.tag10,
      ].filter((tag) => tag); // Filter out undefined or empty tags

      return (
        <div className="tags-container">
          {tags.length > 0 ? (
            tags.map((tag, index) => {
              // Strip main genre prefix for display
              const displayTag = tag.includes('---') ? tag.split('---')[1].trim() : tag;

              const { backgroundColor, textColor } = getTagColors(tag); // Use full tag for color consistency
              return (
                <div
                  key={index}
                  className="tag-track"
                  style={{ backgroundColor, color: textColor }}
                >
                  {displayTag}
                </div>
              );
            })
          ) : (
            <span className="no-tags">No tags</span> // Provide feedback if there are no tags
          )}
        </div>
      );
    case "Wave":
      return (
        <div className="wave-column">
          <Waveform
            trackId={track.id.toString()}
            audioPath={track.path}
            isInteractive={true} // Make it interactive to handle play events
            onPlay={handlePlay}
          />
        </div>
      );
    default:
      return null;
  }
};

// Updated PropTypes
Track.propTypes = {
  track: PropTypes.shape({
    cover: PropTypes.string,
    title: PropTypes.string.isRequired,
    path: PropTypes.string,
    artist: PropTypes.string,
    album: PropTypes.string,
    year: PropTypes.string,
    TIME: PropTypes.string, // Assuming it's a formatted string like "3:45"
    BPM: PropTypes.number,
    KEY: PropTypes.string,
    DATE: PropTypes.string,
    tag1: PropTypes.string,
    tag2: PropTypes.string,
    tag3: PropTypes.string,
    tag4: PropTypes.string,
    tag5: PropTypes.string,
    tag6: PropTypes.string,
    tag7: PropTypes.string,
    tag8: PropTypes.string,
    tag9: PropTypes.string,
    tag10: PropTypes.string,
    wave: PropTypes.string,
    artwork_thumbnail_path: PropTypes.string,
  }).isRequired,
  column: PropTypes.string.isRequired,
};

export default Track;