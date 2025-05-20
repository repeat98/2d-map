// Playlist.jsx
import React, { useState, useEffect, useCallback, useMemo } from "react";
import { useDrop } from "react-dnd";
import PlaylistTrack from "./PlaylistTrack";
import "./Playlist.scss";
import PropTypes from "prop-types";

// --- Utility: Debounce function ---
function debounce(func, delay) {
  let timeoutId;
  return function (...args) {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func(...args);
    }, delay);
  };
}

const BPM_DEVIATION_PERCENT = 5; // (retained for potential future use)
const RECOMMENDATION_LIMIT = 5; // Number of last tracks to use for averaging (if available)

/**
 * Checks whether a given track has all required data (BPM, coordinates,
 * and spectral features). Returns false if any field is missing or non-numeric.
 */
function hasAllData(track) {
  const fields = [
    "BPM",
    "x",
    "y",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "spectral_flatness",
  ];

  for (const field of fields) {
    const value = parseFloat(track[field]);
    if (isNaN(value)) {
      return false;
    }
  }
  return true;
}

const Playlist = ({
  playlistTracks,
  setPlaylistTracks,
  allTracks,
  selectedTags,
  bpmRange,
}) => {
  // --- Drag & Drop Setup ---
  const [{ isOver }, drop] = useDrop({
    accept: ["TRACK", "PLAYLIST_TRACK"],
    drop: (item) => {
      let tracksToAdd = [];
      if (item.tracks && Array.isArray(item.tracks)) {
        tracksToAdd = item.tracks;
      } else if (item.track) {
        tracksToAdd = [item.track];
      }
      if (tracksToAdd.length > 0) {
        setPlaylistTracks((prevTracks) => {
          // Filter out duplicates.
          const newTracks = tracksToAdd.filter(
            (t) => !prevTracks.some((pt) => pt.id === t.id)
          );
          return [...prevTracks, ...newTracks];
        });
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  // --- Recommendation Filtering States ---
  const [deniedTrackIds, setDeniedTrackIds] = useState([]);
  const [recommendedTitles, setRecommendedTitles] = useState(new Set());
  const [matchBpm, setMatchBpm] = useState(true);
  // NEW: Spectral match toggle state
  const [spectralMatch, setSpectralMatch] = useState(true);

  // --- Slider & Recommendation Display States ---
  const [sliderValue, setSliderValue] = useState(0);
  const [activeRecIndex, setActiveRecIndex] = useState(0);
  const [recommendations, setRecommendations] = useState([]);
  // This controls how many recommendations are displayed concurrently.
  const [displayCount, setDisplayCount] = useState(3);

  // When recommendations update, reset slider values.
  useEffect(() => {
    if (recommendations.length > 0) {
      setActiveRecIndex(0);
      setSliderValue(0);
    }
  }, [recommendations]);

  const debouncedSetActiveRecIndex = useCallback(
    debounce((value) => {
      setActiveRecIndex(value);
    }, 300),
    []
  );

  // --- Helper: Convert "mm:ss" to seconds ---
  const convertTimeToSeconds = (timeStr) => {
    if (typeof timeStr !== "string") return 0;
    const parts = timeStr.split(":");
    if (parts.length !== 2) return 0;
    const minutes = parseInt(parts[0], 10);
    const seconds = parseInt(parts[1], 10);
    if (isNaN(minutes) || isNaN(seconds)) return 0;
    return minutes * 60 + seconds;
  };

  // --- Sorted list of all tracks by idx ---
  const sortedTracks = useMemo(() => {
    return allTracks.slice().sort((a, b) => a.idx - b.idx);
  }, [allTracks]);

  // --- Filtering Helpers ---
  const isWithinBpmRange = useCallback(
    (trackBpm) => {
      if (!matchBpm) return true;
      const bpm = parseFloat(trackBpm);
      return bpmRange.length === 2 && !isNaN(bpm)
        ? bpm >= bpmRange[0] && bpm <= bpmRange[1]
        : false;
    },
    [bpmRange, matchBpm]
  );

  const doesTrackMatchSelectedTags = useCallback(
    (track) => {
      if (selectedTags.length === 0) return true;
      return selectedTags.some((selectedTag) => {
        for (let pos = 1; pos <= 10; pos++) {
          const trackTag = track[`tag${pos}`];
          if (trackTag === selectedTag) {
            return true;
          } else if (
            !selectedTag.includes("---") &&
            trackTag &&
            trackTag.startsWith(`${selectedTag}---`)
          ) {
            return true;
          }
        }
        return false;
      });
    },
    [selectedTags]
  );

  const getTrackTags = (track) => {
    const tags = [];
    for (let i = 1; i <= 10; i++) {
      const tag = track[`tag${i}`];
      if (tag) tags.push(tag);
    }
    return tags;
  };

  /**
   * Checks if a track is a valid candidate for recommendation.
   */
  const isValidCandidate = useCallback(
    (track) => {
      if (!hasAllData(track)) return false;
      if (playlistTracks.some((pt) => pt.id === track.id)) return false;
      if (deniedTrackIds.includes(track.id)) return false;
      if (!isWithinBpmRange(track.BPM)) return false;
      if (!doesTrackMatchSelectedTags(track)) return false;
      const trackTitle = track.title ? track.title.toLowerCase() : "";
      if (trackTitle && recommendedTitles.has(trackTitle)) return false;
      if (
        playlistTracks.some(
          (pt) => pt.title && pt.title.toLowerCase() === trackTitle
        )
      )
        return false;
      return true;
    },
    [
      playlistTracks,
      deniedTrackIds,
      recommendedTitles,
      isWithinBpmRange,
      doesTrackMatchSelectedTags,
    ]
  );

  // --- Simple Array Shuffling ---
  const shuffleArray = (array) => {
    const shuffled = array.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  };

  // --- Averaging Functions (using last few playlist tracks) ---
  const calculateAverageBpm = useCallback(() => {
    const lastTracks = playlistTracks.slice(-RECOMMENDATION_LIMIT).filter(hasAllData);
    if (lastTracks.length === 0) return null;
    const totalBpm = lastTracks.reduce(
      (sum, track) => sum + parseFloat(track.BPM),
      0
    );
    return totalBpm / lastTracks.length;
  }, [playlistTracks]);

  const calculateAverageIdx = useCallback(() => {
    const lastTracks = playlistTracks.slice(-RECOMMENDATION_LIMIT).filter(hasAllData);
    if (lastTracks.length === 0) return null;
    const totalIdx = lastTracks.reduce((sum, track) => sum + track.idx, 0);
    return totalIdx / lastTracks.length;
  }, [playlistTracks]);

  const calculateAverageCoordinates = useCallback(() => {
    const lastTracks = playlistTracks.slice(-RECOMMENDATION_LIMIT).filter(hasAllData);
    if (lastTracks.length === 0) return null;
    let xSum = 0,
      ySum = 0;
    lastTracks.forEach((track) => {
      xSum += parseFloat(track.x);
      ySum += parseFloat(track.y);
    });
    return { x: xSum / lastTracks.length, y: ySum / lastTracks.length };
  }, [playlistTracks]);

  const computeAverageSpectralFeatures = useCallback(() => {
    const lastTracks = playlistTracks.slice(-RECOMMENDATION_LIMIT).filter(hasAllData);
    if (lastTracks.length === 0) return null;
    let centroidSum = 0,
      bandwidthSum = 0,
      rolloffSum = 0,
      contrastSum = 0,
      flatnessSum = 0;
    lastTracks.forEach((track) => {
      centroidSum += parseFloat(track.spectral_centroid);
      bandwidthSum += parseFloat(track.spectral_bandwidth);
      rolloffSum += parseFloat(track.spectral_rolloff);
      contrastSum += parseFloat(track.spectral_contrast);
      flatnessSum += parseFloat(track.spectral_flatness);
    });
    return {
      centroid: centroidSum / lastTracks.length,
      bandwidth: bandwidthSum / lastTracks.length,
      rolloff: rolloffSum / lastTracks.length,
      contrast: contrastSum / lastTracks.length,
      flatness: flatnessSum / lastTracks.length,
    };
  }, [playlistTracks]);

  const collectTagsFromLastTracks = useCallback(() => {
    const lastTracks = playlistTracks.slice(-RECOMMENDATION_LIMIT).filter(hasAllData);
    const tagCounts = {};
    lastTracks.forEach((track) => {
      const tags = getTrackTags(track);
      tags.forEach((tag) => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });
    return tagCounts;
  }, [playlistTracks]);

  // --- Similarity Helpers ---
  const computeTagSimilarity = (candidateTags, lastTrackTagCounts) => {
    let similarity = 0;
    candidateTags.forEach((tag) => {
      if (lastTrackTagCounts[tag]) similarity += lastTrackTagCounts[tag];
    });
    return similarity;
  };

  const computeSelectedTagSimilarity = (candidateTags, selectedTags) => {
    if (selectedTags.length === 0) return 0;
    let similarity = 0;
    candidateTags.forEach((tag) => {
      if (selectedTags.includes(tag)) {
        similarity += 1;
      } else {
        const tagMain = tag.split("---")[0];
        selectedTags.forEach((selTag) => {
          const selTagMain = selTag.split("---")[0];
          if (tagMain === selTagMain) similarity += 0.5;
        });
      }
    });
    return similarity;
  };

  /**
   * MAIN RECOMMENDATION LOGIC:
   * Scores every valid candidate track from the entire database.
   * If the playlist is empty, averages are computed from all valid tracks.
   * Otherwise, averages come from the last few playlist tracks.
   * Returns a sorted array (highest score first) of recommended tracks.
   */
  const findBestMatches = useCallback(() => {
    let avgBpm, avgIdx, avgCoords, avgSpectral, lastTrackTagCounts;
    if (playlistTracks.length === 0) {
      // Use all valid tracks from the database.
      const validTracks = sortedTracks.filter(hasAllData);
      if (validTracks.length === 0) return [];
      avgBpm =
        validTracks.reduce((sum, track) => sum + parseFloat(track.BPM), 0) /
        validTracks.length;
      avgIdx =
        validTracks.reduce((sum, track) => sum + track.idx, 0) / validTracks.length;
      avgCoords = {
        x:
          validTracks.reduce((sum, track) => sum + parseFloat(track.x), 0) /
          validTracks.length,
        y:
          validTracks.reduce((sum, track) => sum + parseFloat(track.y), 0) /
          validTracks.length,
      };
      avgSpectral = {
        centroid:
          validTracks.reduce((sum, track) => sum + parseFloat(track.spectral_centroid), 0) /
          validTracks.length,
        bandwidth:
          validTracks.reduce((sum, track) => sum + parseFloat(track.spectral_bandwidth), 0) /
          validTracks.length,
        rolloff:
          validTracks.reduce((sum, track) => sum + parseFloat(track.spectral_rolloff), 0) /
          validTracks.length,
        contrast:
          validTracks.reduce((sum, track) => sum + parseFloat(track.spectral_contrast), 0) /
          validTracks.length,
        flatness:
          validTracks.reduce((sum, track) => sum + parseFloat(track.spectral_flatness), 0) /
          validTracks.length,
      };
      lastTrackTagCounts = {};
    } else {
      avgBpm = calculateAverageBpm();
      avgIdx = calculateAverageIdx();
      avgCoords = calculateAverageCoordinates();
      avgSpectral = computeAverageSpectralFeatures();
      lastTrackTagCounts = collectTagsFromLastTracks();
    }
    if (avgBpm === null || avgIdx === null || !avgCoords || !avgSpectral) {
      return [];
    }
    // Filter valid candidate tracks.
    let candidateTracks = sortedTracks.filter(isValidCandidate);
    if (candidateTracks.length === 0) return [];
    candidateTracks = shuffleArray(candidateTracks);
    candidateTracks = candidateTracks.map((track) => {
      const idxVal = parseFloat(track.idx);
      const bpmVal = parseFloat(track.BPM);
      const xVal = parseFloat(track.x);
      const yVal = parseFloat(track.y);
      const centroid = parseFloat(track.spectral_centroid);
      const bandwidth = parseFloat(track.spectral_bandwidth);
      const rolloff = parseFloat(track.spectral_rolloff);
      const contrast = parseFloat(track.spectral_contrast);
      const flatness = parseFloat(track.spectral_flatness);
      const idxDiff = Math.abs(idxVal - avgIdx);
      const bpmDiff = Math.abs(bpmVal - avgBpm);
      const spatialDistance = Math.sqrt(
        Math.pow(xVal - avgCoords.x, 2) + Math.pow(yVal - avgCoords.y, 2)
      );
      const centroidDiff = Math.abs(centroid - avgSpectral.centroid);
      const bandwidthDiff = Math.abs(bandwidth - avgSpectral.bandwidth);
      const rolloffDiff = Math.abs(rolloff - avgSpectral.rolloff);
      const contrastDiff = Math.abs(contrast - avgSpectral.contrast);
      const flatnessDiff = Math.abs(flatness - avgSpectral.flatness);
      const candidateTags = getTrackTags(track);
      const tagSimilarity = computeTagSimilarity(candidateTags, lastTrackTagCounts);
      const selTagSimilarity = computeSelectedTagSimilarity(candidateTags, selectedTags);
      return {
        track,
        idxDiff,
        bpmDiff,
        spatialDistance,
        centroidDiff,
        bandwidthDiff,
        rolloffDiff,
        contrastDiff,
        flatnessDiff,
        tagSimilarity,
        selTagSimilarity,
      };
    });
    // Compute maximum differences for normalization.
    const maxIdxDiff = Math.max(...candidateTracks.map(item => item.idxDiff));
    const maxBpmDiff = Math.max(...candidateTracks.map(item => item.bpmDiff));
    const maxSpatialDistance = Math.max(...candidateTracks.map(item => item.spatialDistance));
    const maxCentroidDiff = Math.max(...candidateTracks.map(item => item.centroidDiff));
    const maxBandwidthDiff = Math.max(...candidateTracks.map(item => item.bandwidthDiff));
    const maxRolloffDiff = Math.max(...candidateTracks.map(item => item.rolloffDiff));
    const maxContrastDiff = Math.max(...candidateTracks.map(item => item.contrastDiff));
    const maxFlatnessDiff = Math.max(...candidateTracks.map(item => item.flatnessDiff));
    const maxTagSimilarity = Math.max(...candidateTracks.map(item => item.tagSimilarity));
    const maxSelectedTagSimilarity = Math.max(
      ...candidateTracks.map(item => item.selTagSimilarity)
    );
    // Normalization functions.
    const normDist = (val, maxVal) => (maxVal > 0 ? (maxVal - val) / maxVal : 1);
    const normSim = (val, maxVal) => (maxVal > 0 ? val / maxVal : 0);
    // Weights for each metric.
    // Note: When spectralMatch is disabled, theta is set to 0 so that the spectralScore is ignored.
    const alpha = 1,
      beta = 1,
      gamma = 1,
      delta = 1,
      zeta = selectedTags.length > 0 ? 1 : 0,
      theta = spectralMatch ? 1 : 0;
    candidateTracks = candidateTracks.map((item) => {
      const {
        track,
        idxDiff,
        bpmDiff,
        spatialDistance,
        centroidDiff,
        bandwidthDiff,
        rolloffDiff,
        contrastDiff,
        flatnessDiff,
        tagSimilarity,
        selTagSimilarity,
      } = item;
      const idxScore = normDist(idxDiff, maxIdxDiff);
      const bpmScore = normDist(bpmDiff, maxBpmDiff);
      const spatialScore = normDist(spatialDistance, maxSpatialDistance);
      const centroidScore = normDist(centroidDiff, maxCentroidDiff);
      const bandwidthScore = normDist(bandwidthDiff, maxBandwidthDiff);
      const rolloffScore = normDist(rolloffDiff, maxRolloffDiff);
      const contrastScore = normDist(contrastDiff, maxContrastDiff);
      const flatnessScore = normDist(flatnessDiff, maxFlatnessDiff);
      const spectralScore =
        (centroidScore + bandwidthScore + rolloffScore + contrastScore + flatnessScore) / 5;
      const tagSimScore = normSim(tagSimilarity, maxTagSimilarity);
      const selTagSimScore = normSim(selTagSimilarity, maxSelectedTagSimilarity);
      const totalScore =
        alpha * idxScore +
        beta * bpmScore +
        gamma * spatialScore +
        delta * tagSimScore +
        zeta * selTagSimScore +
        theta * spectralScore;
      return { track, totalScore, scores: { idxScore, bpmScore, spatialScore, centroidScore, bandwidthScore, rolloffScore, contrastScore, flatnessScore, tagSimScore, selTagSimScore, spectralScore } };
    });
    // Sort candidates by total score descending.
    candidateTracks.sort((a, b) => b.totalScore - a.totalScore);
    return candidateTracks.map((item) => item.track);
  }, [
    sortedTracks,
    playlistTracks,
    deniedTrackIds,
    recommendedTitles,
    selectedTags,
    isWithinBpmRange,
    doesTrackMatchSelectedTags,
    calculateAverageBpm,
    calculateAverageIdx,
    calculateAverageCoordinates,
    computeAverageSpectralFeatures,
    collectTagsFromLastTracks,
    isValidCandidate,
    spectralMatch, // include spectralMatch dependency
  ]);

  // Update recommendations whenever dependencies change.
  useEffect(() => {
    const newRecommendations = findBestMatches();
    setRecommendations(newRecommendations);
  }, [findBestMatches]);

  // --- Accept or Deny a Recommendation ---
  const acceptRecommendation = useCallback(
    (track) => {
      if (!track) return;
      setPlaylistTracks((prevTracks) => [...prevTracks, track]);
      // Optionally, clear denies after acceptance.
      setDeniedTrackIds([]);
    },
    [setPlaylistTracks]
  );

  const denyRecommendation = useCallback((trackId, trackTitle) => {
    if (!trackId) return;
    setDeniedTrackIds((prevDenied) => [...prevDenied, trackId]);
    if (trackTitle) {
      setRecommendedTitles((prevTitles) => {
        const newSet = new Set(prevTitles);
        newSet.add(trackTitle.toLowerCase());
        return newSet;
      });
    }
  }, []);

  // --- Export Playlist as .m3u ---
  const handleExport = () => {
    if (playlistTracks.length === 0) {
      alert("Playlist is empty. Add tracks to export.");
      return;
    }
    let m3uContent = "#EXTM3U\n";
    playlistTracks.forEach((track) => {
      const durationSeconds = convertTimeToSeconds(track.TIME);
      const artist = track.artist || "Unknown Artist";
      const title = track.title || "Unknown Title";
      const filePath = track.path || "";
      if (filePath.trim() === "") {
        console.warn(`Track ID: ${track.id} has an empty path. Skipping.`);
        return;
      }
      m3uContent += `#EXTINF:${durationSeconds},${artist} - ${title}\n`;
      m3uContent += `${filePath}\n`;
    });
    const blob = new Blob([m3uContent], { type: "audio/x-mpegurl" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.download = `playlist-${timestamp}.m3u`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // --- Remove a Track from the Playlist ---
  const handleRemoveTrack = useCallback(
    (trackId) => {
      setPlaylistTracks((prevTracks) =>
        prevTracks.filter((t) => t.id !== trackId)
      );
    },
    [setPlaylistTracks]
  );

  // Determine the subset of recommendations to display concurrently.
  const displayedRecommendations = recommendations.slice(
    activeRecIndex,
    activeRecIndex + displayCount
  );
  const sliderMax = Math.max(recommendations.length - displayCount, 0);

  return (
    <div
      ref={drop}
      className={`playlist-container ${isOver ? "highlight" : ""}`}
    >
      <div className="playlist-header">
        <span className="playlist-title">Tags</span>
        <button
          className="button-export"
          onClick={handleExport}
          disabled={playlistTracks.length === 0}
        >
          Export Playlist
        </button>
      </div>

      {/* Recommendation Controls */}
      <div className="recommendation-controls">
        <label>
          <input
            type="checkbox"
            checked={matchBpm}
            onChange={(e) => setMatchBpm(e.target.checked)}
          />
          Match BPM
        </label>
        <label>
          <input
            type="checkbox"
            checked={spectralMatch}
            onChange={(e) => setSpectralMatch(e.target.checked)}
          />
          Spectral Match
        </label>
        <label>
          Display Count:
          <input
            type="number"
            min="1"
            value={displayCount}
            onChange={(e) =>
              setDisplayCount(parseInt(e.target.value, 10) || 1)
            }
          />
        </label>
      </div>

      <div className="playlist-scrollable">
        {/* Slider placed above the recommendations */}
        {recommendations.length > displayCount && (
          <div className="recommendation-slider" style={{ marginBottom: "10px" }}>
            <input
              type="range"
              min="0"
              max={sliderMax}
              value={sliderValue}
              onChange={(e) => {
                const newValue = parseInt(e.target.value, 10);
                setSliderValue(newValue);
                debouncedSetActiveRecIndex(newValue);
              }}
            />
            <div>
              Showing recommendations {activeRecIndex + 1} to{" "}
              {Math.min(activeRecIndex + displayCount, recommendations.length)}
            </div>
          </div>
        )}

        {/* Recommendation Display */}
        <div className="recommendation-container">
          {displayedRecommendations.length > 0 ? (
            displayedRecommendations.map((track) => (
              <PlaylistTrack
                key={`recommendation-${track.id}`}
                track={track}
                isRecommended={true}
                recommendationType="Recommended"
                onAccept={() => acceptRecommendation(track)}
                onDeny={() => denyRecommendation(track.id, track.title)}
              />
            ))
          ) : (
            <div>No recommendations available</div>
          )}
        </div>

        {/* Actual Playlist Tracks */}
        <div className="playlist">
          {playlistTracks.map((track, index) => (
            <PlaylistTrack
              key={track.id}
              track={track}
              index={index}
              isRecommended={false}
              onRemove={() => handleRemoveTrack(track.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

Playlist.propTypes = {
  playlistTracks: PropTypes.array.isRequired,
  setPlaylistTracks: PropTypes.func.isRequired,
  allTracks: PropTypes.array.isRequired,
  selectedTags: PropTypes.array.isRequired,
  bpmRange: PropTypes.array.isRequired,
};

export default React.memo(Playlist);