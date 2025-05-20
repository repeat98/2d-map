// src/components/Sidebar.jsx
import React, { useState, useEffect, useRef } from 'react';
import './Sidebar.scss';
import { getTagColors } from '../utils/tagUtils';

const Sidebar = ({ onTagSelectionChange, onBpmRangeChange }) => {
  const [availableTags, setAvailableTags] = useState({});
  const [selectedTags, setSelectedTags] = useState([]);
  const [minBpm, setMinBpm] = useState(60);
  const [maxBpm, setMaxBpm] = useState(200);
  const [expandedGenres, setExpandedGenres] = useState({});
  const [genreFrequency, setGenreFrequency] = useState({}); // New state for genre frequency
  const [mainGenreTrackCount, setMainGenreTrackCount] = useState({}); // New state for unique track counts
  const [searchTerm, setSearchTerm] = useState(''); // New state for search

  // Refs for debouncing
  const bpmTimeoutRef = useRef(null);

  useEffect(() => {
    // Fetch tracks and extract tags
    const fetchTracks = async () => {
      try {
        const response = await fetch('http://localhost:3000/tracks');
        if (!response.ok) throw new Error('Failed to fetch tracks');
        const data = await response.json();

        // Extract tags from tracks
        const tagData = {};
        const genreFrequencyTemp = {};
        const mainGenreTrackSet = {}; // Temporary storage for unique track IDs

        data.forEach((track) => {
          const trackTags = [
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
          ];
          const trackId = track.id;

          trackTags.forEach((tag) => {
            if (tag) {
              const parts = tag.split('---');
              const mainGenre = parts[0].trim();
              const subGenre = parts[1] ? parts[1].trim() : '';

              // Initialize structures if not present
              if (!tagData[mainGenre]) {
                tagData[mainGenre] = {};
              }
              if (!tagData[mainGenre][subGenre]) {
                tagData[mainGenre][subGenre] = 0;
              }

              // Increment subgenre frequency
              tagData[mainGenre][subGenre] += 1;

              // Increment main genre frequency
              if (!genreFrequencyTemp[mainGenre]) {
                genreFrequencyTemp[mainGenre] = 0;
              }
              genreFrequencyTemp[mainGenre] += 1;

              // Track unique track IDs per main genre
              if (!mainGenreTrackSet[mainGenre]) {
                mainGenreTrackSet[mainGenre] = new Set();
              }
              mainGenreTrackSet[mainGenre].add(trackId);
            }
          });
        });

        setAvailableTags(tagData);
        setGenreFrequency(genreFrequencyTemp);

        // Convert Sets to unique track counts
        const mainGenreTrackCountTemp = {};
        Object.keys(mainGenreTrackSet).forEach((genre) => {
          mainGenreTrackCountTemp[genre] = mainGenreTrackSet[genre].size;
        });
        setMainGenreTrackCount(mainGenreTrackCountTemp);

        // Initialize expanded genres (collapsed by default)
        const initialExpandedGenres = {};
        Object.keys(tagData).forEach((genre) => {
          initialExpandedGenres[genre] = false;
        });
        setExpandedGenres(initialExpandedGenres);
      } catch (err) {
        console.error(err);
      }
    };

    fetchTracks();
  }, []);

  // Handle tag selection change
  const handleTagChange = (fullTag) => {
    setSelectedTags((prevSelectedTags) => {
      const isSelected = prevSelectedTags.includes(fullTag);
      const newSelectedTags = isSelected
        ? prevSelectedTags.filter((t) => t !== fullTag)
        : [...prevSelectedTags, fullTag];
      onTagSelectionChange(newSelectedTags);
      return newSelectedTags;
    });
  };

  // Handle main genre selection change
  const handleMainGenreChange = (mainGenre) => {
    setSelectedTags((prevSelectedTags) => {
      const isSelected = prevSelectedTags.includes(mainGenre);
      const newSelectedTags = isSelected
        ? prevSelectedTags.filter((t) => t !== mainGenre)
        : [...prevSelectedTags, mainGenre];
      onTagSelectionChange(newSelectedTags);
      return newSelectedTags;
    });
  };

  // Handle removing a selected tag
  const handleRemoveSelectedTag = (tagToRemove) => {
    setSelectedTags((prevSelectedTags) => {
      const newSelectedTags = prevSelectedTags.filter((t) => t !== tagToRemove);
      onTagSelectionChange(newSelectedTags);
      return newSelectedTags;
    });
  };

  // Debounce BPM range change
  const handleBpmRangeChangeLocal = (newMinBpm, newMaxBpm) => {
    setMinBpm(newMinBpm);
    setMaxBpm(newMaxBpm);

    if (bpmTimeoutRef.current) {
      clearTimeout(bpmTimeoutRef.current);
    }

    bpmTimeoutRef.current = setTimeout(() => {
      onBpmRangeChange([newMinBpm, newMaxBpm]);
    }, 300); // Debounce delay of 300ms
  };

  // Toggle genre expansion
  const toggleGenre = (genre) => {
    setExpandedGenres((prev) => ({
      ...prev,
      [genre]: !prev[genre],
    }));
  };

  // Ensure minBpm is not greater than maxBpm
  useEffect(() => {
    if (minBpm > maxBpm) {
      setMinBpm(maxBpm);
    }
  }, [minBpm, maxBpm]);

  // Compute the left and width percentages for the slider range
  const minPosition = ((minBpm - 60) / 140) * 100;
  const maxPosition = ((maxBpm - 60) / 140) * 100;

  // Filter availableTags based on searchTerm
  const filteredAvailableTags = Object.keys(availableTags)
    .filter((mainGenre) => {
      // Check if mainGenre matches the search term
      const mainGenreMatch = mainGenre.toLowerCase().includes(searchTerm.toLowerCase());
      // Check if any subGenre matches the search term
      const subGenreMatch = Object.keys(availableTags[mainGenre]).some((subGenre) =>
        subGenre.toLowerCase().includes(searchTerm.toLowerCase())
      );
      return mainGenreMatch || subGenreMatch;
    })
    .reduce((acc, mainGenre) => {
      const mainGenreMatch = mainGenre.toLowerCase().includes(searchTerm.toLowerCase());
      if (mainGenreMatch) {
        // If mainGenre matches, include all its subGenres
        acc[mainGenre] = availableTags[mainGenre];
      } else {
        // Otherwise, include only the matching subGenres
        const matchingSubGenres = Object.keys(availableTags[mainGenre]).filter((subGenre) =>
          subGenre.toLowerCase().includes(searchTerm.toLowerCase())
        );
        if (matchingSubGenres.length > 0) {
          acc[mainGenre] = {};
          matchingSubGenres.forEach((subGenre) => {
            acc[mainGenre][subGenre] = availableTags[mainGenre][subGenre];
          });
        }
      }
      return acc;
    }, {});

  // Sort main genres by genreFrequency (high to low)
  const sortedMainGenres = Object.keys(filteredAvailableTags).sort((a, b) => {
    const freqA = genreFrequency[a] || 0;
    const freqB = genreFrequency[b] || 0;
    return freqB - freqA;
  });

  return (
    <div className="side-bar">
      <div className="filter-container">
        <div className="filter-label">Filter options</div>
        <div className="divider"></div>
        <div className="filter-bpm">
          <div className="filter-name">BPM</div>
          <div className="slider-container">
            {/* Custom dual-range slider */}
            <div className="dual-range-slider">
              <div className="slider-track"></div>
              <div
                className="slider-range"
                style={{
                  left: `${minPosition}%`,
                  width: `${maxPosition - minPosition}%`,
                }}
              ></div>
              <input
                type="range"
                min="60"
                max="200"
                value={minBpm}
                onChange={(e) => {
                  const value = Math.min(Number(e.target.value), maxBpm - 1);
                  handleBpmRangeChangeLocal(value, maxBpm);
                }}
                id="minRange"
              />
              <input
                type="range"
                min="60"
                max="200"
                value={maxBpm}
                onChange={(e) => {
                  const value = Math.max(Number(e.target.value), minBpm + 1);
                  handleBpmRangeChangeLocal(minBpm, value);
                }}
                id="maxRange"
              />
            </div>
            <div id="bpmRangeDisplay">
              <span id="bpmRangeMinValue">{minBpm}</span> -{' '}
              <span id="bpmRangeMaxValue">{maxBpm}</span>
            </div>
          </div>
        </div>
        <div className="filter-label">Selected Tags</div>
        <div className="selected-tags-container">
          {selectedTags.map((tag) => {
            // Strip main genre prefix for display
            const displayTag = tag.includes('---') ? tag.split('---')[1].trim() : tag;

            const { backgroundColor, textColor } = getTagColors(tag); // Use full tag for color consistency
            return (
              <div
                key={tag}
                className="tag selected-tag"
                style={{ backgroundColor, color: textColor }}
              >
                {displayTag}
                <span
                  className="remove-tag"
                  onClick={() => handleRemoveSelectedTag(tag)}
                >
                  ×
                </span>
              </div>
            );
          })}
        </div>
        <div className="filter-label">Tags</div>
        <div className="filter-tag-container">
          {/* Search Input */}
          <div className="tag-search-container">
            <input
              type="text"
              placeholder="Search tags"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="tag-search-input"
            />
          </div>

          {/* Render filtered and sorted tags */}
          {sortedMainGenres.map((mainGenre) => {
            const subGenres = filteredAvailableTags[mainGenre];
            const uniqueTrackCount = mainGenreTrackCount[mainGenre] || 0;

            return (
              <div key={mainGenre} className="main-genre-section">
                <div className="main-genre-header">
                  <span
                    className="genre-toggle"
                    onClick={() => toggleGenre(mainGenre)}
                  >
                    {expandedGenres[mainGenre] ? '▼' : '▶'}
                  </span>
                  <label className="main-genre-label">
                    <input
                      type="checkbox"
                      checked={selectedTags.includes(mainGenre)}
                      onChange={() => handleMainGenreChange(mainGenre)}
                    />
                    <div className="main-genre-name">
                      {mainGenre} ({uniqueTrackCount})
                    </div>
                  </label>
                </div>
                {expandedGenres[mainGenre] && (
                  <div className="subgenre-container">
                    {Object.entries(subGenres)
                      .sort((a, b) => b[1] - a[1])
                      .map(([subGenre, frequency]) => {
                        const fullTag = subGenre
                          ? `${mainGenre}---${subGenre}`
                          : mainGenre;
                        const { backgroundColor, textColor } = getTagColors(
                          fullTag
                        );
                        return (
                          <label key={fullTag} className="tag-checkbox">
                            <input
                              type="checkbox"
                              checked={selectedTags.includes(fullTag)}
                              onChange={() => handleTagChange(fullTag)}
                            />
                            <div
                              className="tag"
                              style={{ backgroundColor, color: textColor }}
                            >
                              {subGenre || mainGenre} ({frequency})
                            </div>
                          </label>
                        );
                      })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;