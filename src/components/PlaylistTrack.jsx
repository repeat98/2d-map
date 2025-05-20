// PlaylistTrack.jsx
import React, { useContext } from "react";
import PropTypes from "prop-types";
import "./PlaylistTrack.scss";
import Waveform from "./Waveform"; // Ensure Waveform component is correctly imported
import { PlaybackContext } from "../context/PlaybackContext";

const PlaylistTrack = ({
  track,
  index,
  isRecommended,
  onAccept,
  onDeny,
  onRemove,
}) => {
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
      return baseName;
    }

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
    return camelotWheel[key] || "Unknown Key";
  };

  // Helper function to get the first two tags
  const getTrackTags = (track) => {
    return [
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
    ]
      .filter((tag) => tag)
      .slice(0, 2);
  };

  return (
    <div className={`playlist-track ${isRecommended ? "recommended" : ""}`}>
      <div className="track-info">
        {/* Track Details */}
        <div
          className="track-details"
          onClick={handlePlay}
          style={{ cursor: "pointer" }}
        >
          <span className="track-title">
            {typeof track.title === "string" &&
            track.title.trim().toLowerCase() === "unknown title"
              ? stripSuffix(track.path)
              : track.title}
          </span>
          <span className="track-artist">
            {track.artist} {track.BPM ? `| ${track.BPM} BPM` : ""}
          </span>
        </div>

        {/* Waveform Column */}
        <div className="track-waveform">
          <Waveform
            trackId={track.id.toString()}
            audioPath={track.path}
            isInteractive={true}
            onPlay={handlePlay}
            height={8}
            width={100}
          />
        </div>
      </div>

      {/* Tags */}
      <div className="track-tags">
        {getTrackTags(track).length > 0 ? (
          getTrackTags(track).map((tag, index) => (
            <span key={index} className="track-tag">
              {tag.includes("---") ? tag.split("---")[1].trim() : tag}
            </span>
          ))
        ) : (
          <span className="no-tags">No tags</span>
        )}
      </div>

      {/* Actions */}
      <div className="track-actions">
        {isRecommended ? (
          <>
            <button
              className="button-accept"
              onClick={onAccept}
              aria-label="Accept Recommendation"
            >
              ✓
            </button>
            <button
              className="button-deny"
              onClick={onDeny}
              aria-label="Deny Recommendation"
            >
              ✖
            </button>
          </>
        ) : (
          <button
            className="button-remove"
            onClick={onRemove}
            aria-label="Remove Track"
          >
            ✖
          </button>
        )}
      </div>
    </div>
  );
};

PlaylistTrack.propTypes = {
  track: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    artist: PropTypes.string,
    BPM: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
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
  }).isRequired,
  index: PropTypes.number,
  isRecommended: PropTypes.bool,
  onAccept: PropTypes.func,
  onDeny: PropTypes.func,
  onRemove: PropTypes.func,
};

PlaylistTrack.defaultProps = {
  index: null,
  isRecommended: false,
  onAccept: () => {},
  onDeny: () => {},
  onRemove: () => {},
};

export default React.memo(PlaylistTrack);