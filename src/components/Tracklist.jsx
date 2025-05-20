// Tracklist.jsx
import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import './Tracklist.scss';
import Track from './Track';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import PropTypes from 'prop-types';
import ColumnContextMenu from './ColumnContextMenu';
import { useDrag } from 'react-dnd';

const MIN_COLUMN_WIDTH = 50; // Minimum width for a column in pixels
const ROW_HEIGHT = 56; // Height of each row

const DRAG_TYPE = 'TRACK';

const Tracklist = ({
  searchTerm,
  onTrackSelect,
  selectedTags,
  bpmRange,
  setAllTracks,
}) => {
  const [tracks, setTracks] = useState([]);
  const [filteredTracks, setFilteredTracks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Define columns with visibility settings
  const columns = [
    { name: 'Cover', visible: true },
    { name: 'Title', visible: true },
    { name: 'Artist', visible: true },
    { name: 'Album', visible: true },
    { name: 'BPM', visible: true },
    { name: 'Year', visible: true },
    { name: 'TIME', visible: true },
    { name: 'Key', visible: true },
    { name: 'Date', visible: true },
    { name: 'Tags', visible: true },
    { name: 'Wave', visible: true },
  ];

  // State for visible columns
  const [visibleColumns, setVisibleColumns] = useState(
    () => JSON.parse(localStorage.getItem('visibleColumns')) || columns
  );

  // State for column widths from localStorage or default
  const [columnWidths, setColumnWidths] = useState(() => {
    const saved = localStorage.getItem('columnWidths');
    return saved ? JSON.parse(saved) : {};
  });

  // State for sorting: { key: 'Title', direction: 'ascending' }
  const [sortConfig, setSortConfig] = useState({ key: null, direction: null });

  // Ref to store the current container width
  const containerWidthRef = useRef(0);

  // Refs for resizing logic
  const resizingColumn = useRef(null);
  const startX = useRef(0);
  const startWidth = useRef(0);

  // Ref to always have the latest columnWidths inside callbacks
  const columnWidthsRef = useRef(columnWidths);
  useEffect(() => {
    columnWidthsRef.current = columnWidths;
  }, [columnWidths]);

  // Fetch tracks data
  useEffect(() => {
    const fetchTracks = async () => {
      try {
        const response = await fetch('http://localhost:3000/tracks');
        if (!response.ok) throw new Error('Failed to fetch tracks');
        const data = await response.json();

        setTracks(data);
        setFilteredTracks(data);
        setAllTracks(data); // Set allTracks in App.jsx
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTracks();
  }, [setAllTracks]); // Added setAllTracks as dependency

  // Persist column widths to localStorage
  useEffect(() => {
    if (Object.keys(columnWidths).length > 0) {
      localStorage.setItem('columnWidths', JSON.stringify(columnWidths));
    }
  }, [columnWidths]);

  // Helper function to get tags from a track
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
    ].filter((tag) => tag);
  };

  // Filter tracks based on searchTerm, selectedTags, and bpmRange
  useEffect(() => {
    let filtered = tracks;

    // Filter based on searchTerm
    if (searchTerm !== '') {
      const lowerCaseTerm = searchTerm.toLowerCase();
      filtered = filtered.filter((track) => {
        const trackTags = getTrackTags(track);
        return (
          (track.title && track.title.toLowerCase().includes(lowerCaseTerm)) ||
          (track.artist && track.artist.toLowerCase().includes(lowerCaseTerm)) ||
          (track.album && track.album.toLowerCase().includes(lowerCaseTerm)) ||
          (track.year && track.year.toString().includes(lowerCaseTerm)) ||
          (track.TIME && track.TIME.toLowerCase().includes(lowerCaseTerm)) ||
          trackTags.some((tag) => tag.toLowerCase().includes(lowerCaseTerm))
        );
      });
    }

    // Filter based on selectedTags
    if (selectedTags.length > 0) {
      filtered = filtered.filter((track) => {
        const trackTags = getTrackTags(track);

        return selectedTags.some((selectedTag) => {
          if (selectedTag.includes('---')) {
            // It's a full tag (main genre + subgenre)
            return trackTags.includes(selectedTag);
          } else {
            // It's a main genre, check if any track tag starts with this main genre
            return trackTags.some((tag) => tag.startsWith(`${selectedTag}---`));
          }
        });
      });
    }

    // Filter based on bpmRange
    if (bpmRange && bpmRange.length === 2) {
      const [minBpm, maxBpm] = bpmRange;
      filtered = filtered.filter((track) => {
        const bpm = parseFloat(track.BPM);
        if (isNaN(bpm)) {
          return false; // Exclude tracks without BPM
        }
        return bpm >= minBpm && bpm <= maxBpm;
      });
    }

    setFilteredTracks(filtered);
  }, [searchTerm, selectedTags, bpmRange, tracks]);

  // Initialize column widths based on container width if not set
  const initializeColumnWidths = useCallback(
    (containerWidth) => {
      if (Object.keys(columnWidths).length === 0) {
        const totalMinWidth = columns.reduce((acc, col) => acc + MIN_COLUMN_WIDTH, 0);
        const availableWidth = containerWidth > totalMinWidth ? containerWidth : totalMinWidth;
        const initialWidth = Math.floor(availableWidth / columns.length);
        const initialWidths = {};
        columns.forEach((col) => {
          initialWidths[col.name] = initialWidth;
        });
        setColumnWidths(initialWidths);
      }
    },
    [columns, columnWidths]
  );

  // Function to update visible columns
  const updateVisibleColumns = useCallback((columnName, isVisible) => {
    setVisibleColumns((prev) => {
      const updated = prev.map((col) =>
        col.name === columnName ? { ...col, visible: isVisible } : col
      );
      localStorage.setItem('visibleColumns', JSON.stringify(updated));
      return updated;
    });
  }, []);

  // Function to show the context menu
  const [contextMenu, setContextMenu] = useState(null);

  const handleContextMenu = useCallback((e, column) => {
    e.preventDefault();
    setContextMenu({
      x: e.pageX,
      y: e.pageY,
      column,
    });
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Handle mouse movement during resizing
  const handleMouseMove = useCallback(
    (e) => {
      if (resizingColumn.current) {
        const deltaX = e.clientX - startX.current;
        let newWidth = startWidth.current + deltaX;

        // Ensure newWidth is within bounds
        newWidth = Math.max(MIN_COLUMN_WIDTH, newWidth);

        // Update the column width
        setColumnWidths((prevWidths) => ({
          ...prevWidths,
          [resizingColumn.current]: newWidth,
        }));
      }
    },
    []
  );

  // Handle mouse up to stop resizing
  const handleMouseUp = useCallback(() => {
    if (resizingColumn.current) {
      resizingColumn.current = null;
      document.body.style.cursor = 'default';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
  }, [handleMouseMove]);

  // Handle mouse down to initiate resizing
  const handleMouseDown = (e, column) => {
    e.preventDefault();
    resizingColumn.current = column;
    startX.current = e.clientX;
    startWidth.current = columnWidths[column] || MIN_COLUMN_WIDTH;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
  };

  // Handle sorting when a header is clicked
  const handleSort = (column) => {
    setSortConfig((prevConfig) => {
      if (prevConfig.key === column) {
        // Cycle through ascending -> descending -> no sort
        if (prevConfig.direction === 'ascending') {
          return { key: column, direction: 'descending' };
        } else if (prevConfig.direction === 'descending') {
          return { key: null, direction: null };
        } else {
          return { key: column, direction: 'ascending' };
        }
      } else {
        // New sort
        return { key: column, direction: 'ascending' };
      }
    });
  };

  // Updated useMemo for sortedTracks with matchScore considering tag positions
  const sortedTracks = useMemo(() => {
    let tracksToSort = filteredTracks.slice();

    if (selectedTags.length > 0) {
      // Compute matchScore for each track based on selectedTags and tag positions
      tracksToSort = tracksToSort.map((track) => {
        const trackTags = getTrackTags(track);
        let matchScore = 0;

        selectedTags.forEach((selectedTag) => {
          if (selectedTag.includes('---')) {
            // Full tag match (e.g., "Genre---SubGenre")
            const tagIndex = trackTags.findIndex((tag) => tag === selectedTag);
            if (tagIndex !== -1) {
              // Higher score for lower index (tag1:10, tag2:9, ..., tag10:1)
              matchScore += 10 - tagIndex;
            }
          } else {
            // Main genre match (e.g., "Genre")
            // Check if any track tag starts with this main genre
            trackTags.forEach((tag) => {
              if (tag.startsWith(`${selectedTag}---`)) {
                const tagIndex = trackTags.findIndex((t) => t === tag);
                if (tagIndex !== -1) {
                  // Assign a base score and adjust based on position
                  matchScore += 5 - tagIndex; // tag1:5, tag2:4, ..., tag10:-4
                }
              }
            });
          }
        });

        return { ...track, matchScore };
      });

      // Sort by matchScore descending, then by id ascending
      tracksToSort.sort((a, b) => {
        if (b.matchScore !== a.matchScore) {
          return b.matchScore - a.matchScore;
        }
        return a.id - b.id; // Assuming id is a number
      });
    } else {
      // If no tags are selected, sort by id ascending
      tracksToSort.sort((a, b) => {
        if (typeof a.id === 'number' && typeof b.id === 'number') {
          return a.id - b.id;
        } else {
          // If id is not a number, sort lexicographically
          return a.id.toString().localeCompare(b.id.toString());
        }
      });
    }

    // Apply additional sorting if needed
    if (sortConfig.key && sortConfig.direction) {
      const keyMapping = {
        Cover: 'cover',
        Title: 'title',
        Artist: 'artist',
        Album: 'album',
        BPM: 'BPM',
        Year: 'year',
        TIME: 'TIME',
        Key: 'KEY',
        Date: 'DATE',
        Tags: 'Tags',
        Wave: 'wave',
      };

      const key = keyMapping[sortConfig.key];

      tracksToSort.sort((a, b) => {
        // === Updated Sorting Logic for Tags Column ===
        if (sortConfig.key === 'Tags') {
          // Sort based on tag1 to tag10 in sequence
          for (let i = 1; i <= 10; i++) {
            const aTag = a[`tag${i}`] ? a[`tag${i}`].toLowerCase() : '';
            const bTag = b[`tag${i}`] ? b[`tag${i}`].toLowerCase() : '';
            if (aTag < bTag) return sortConfig.direction === 'ascending' ? -1 : 1;
            if (aTag > bTag) return sortConfig.direction === 'ascending' ? 1 : -1;
          }
          return 0;
        }
        // === End of Updated Sorting Logic ===

        let aValue = a[key];
        let bValue = b[key];

        // Handle undefined or null values
        if (aValue === undefined || aValue === null) aValue = '';
        if (bValue === undefined || bValue === null) bValue = '';

        // Special handling for numerical columns like BPM, Year
        if (['BPM', 'Year'].includes(sortConfig.key)) {
          aValue = parseFloat(aValue) || 0;
          bValue = parseFloat(bValue) || 0;
          return sortConfig.direction === 'ascending' ? aValue - bValue : bValue - aValue;
        }

        // Special handling for TIME (formatted as "mm:ss")
        if (sortConfig.key === 'TIME') {
          const timeToSeconds = (timeStr) => {
            const [mins, secs] = timeStr.split(':').map(Number);
            return (isNaN(mins) ? 0 : mins) * 60 + (isNaN(secs) ? 0 : secs);
          };
          aValue = timeToSeconds(aValue);
          bValue = timeToSeconds(bValue);
          return sortConfig.direction === 'ascending' ? aValue - bValue : bValue - aValue;
        }

        // Handle strings for other columns
        aValue = aValue.toString().toLowerCase();
        bValue = bValue.toString().toLowerCase();

        if (aValue < bValue) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }

    return tracksToSort;
  }, [filteredTracks, sortConfig, selectedTags]);

  // Create a ref for the list
  const listRef = useRef();

  // Cleanup event listeners on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'default';
    };
  }, [handleMouseMove, handleMouseUp]);

  // === Added Selection Logic ===
  const [selectedTrackIds, setSelectedTrackIds] = useState([]);

  // Handle Cmd + A to select all tracks
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.metaKey && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        setSelectedTrackIds(sortedTracks.map((track) => track.id));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [sortedTracks]);

  // Memoized selected tracks based on selectedTrackIds
  const selectedTracks = useMemo(() => {
    return sortedTracks.filter((track) => selectedTrackIds.includes(track.id));
  }, [sortedTracks, selectedTrackIds]);

  return (
    <div className="tracklist-container" id="tracklistContainer" onClick={closeContextMenu}>
      {loading && <div className="loading">Loading tracks...</div>}
      {error && <div className="error">Error: {error}</div>}
      {!loading && !error && (
        <AutoSizer>
          {({ height, width }) => {
            // Initialize column widths if not already set
            initializeColumnWidths(width);
            containerWidthRef.current = width;

            return (
              <>
                {/* Header */}
                <div className="header" style={{ width: `${width}px`, height: `28px` }}>
                  {visibleColumns.map(
                    (column) =>
                      column.visible && (
                        <div
                          key={column.name}
                          className="header-cell"
                          style={{ width: `${columnWidths[column.name] || MIN_COLUMN_WIDTH}px` }}
                          onClick={() => handleSort(column.name)}
                          onContextMenu={(e) => handleContextMenu(e, column.name)}
                          role="button"
                          tabIndex={0}
                          aria-sort={
                            sortConfig.key === column.name
                              ? sortConfig.direction === 'ascending'
                                ? 'ascending'
                                : 'descending'
                              : 'none'
                          }
                        >
                          <span>{column.name}</span>
                          {/* Sort Indicator */}
                          {sortConfig.key === column.name && sortConfig.direction && (
                            <span
                              className={`sort-indicator ${
                                sortConfig.direction === 'ascending' ? 'sort-asc' : 'sort-desc'
                              }`}
                            ></span>
                          )}
                          <div
                            className="resizer"
                            onMouseDown={(e) => handleMouseDown(e, column.name)}
                            role="separator"
                            aria-orientation="vertical"
                            tabIndex={0}
                            aria-label={`Resize ${column.name} column`}
                          />
                        </div>
                      )
                  )}
                </div>

                {/* List Wrapper */}
                <div
                  className="list-wrapper"
                  style={{
                    position: 'relative',
                    width: `${width}px`,
                    height: `${height - ROW_HEIGHT}px`,
                  }}
                >
                  {/* Table Body */}
                  <div
                    className="table-wrapper"
                    style={{ width: `${width}px`, height: `${height - ROW_HEIGHT}px` }}
                  >
                    {sortedTracks.length > 0 ? (
                      <List
                        ref={listRef}
                        height={height - ROW_HEIGHT}
                        itemCount={sortedTracks.length}
                        itemSize={ROW_HEIGHT}
                        width={width}
                        itemData={{
                          tracks: sortedTracks,
                          columns: visibleColumns.filter((col) => col.visible),
                          columnWidths,
                          onTrackSelect,
                          selectedTrackIds,
                          setSelectedTrackIds,
                        }}
                      >
                        {Row}
                      </List>
                    ) : (
                      <div className="tracklist-empty-state">
                        No tracks match your search criteria.
                      </div>
                    )}
                  </div>
                </div>

                {/* Render context menu if visible */}
                {contextMenu && (
                  <ColumnContextMenu
                    x={contextMenu.x}
                    y={contextMenu.y}
                    column={contextMenu.column}
                    visibleColumns={visibleColumns}
                    onToggleColumn={updateVisibleColumns}
                  />
                )}
              </>
            );
          }}
        </AutoSizer>
      )}
    </div>
  );
};

// Memoized Row component to prevent unnecessary re-renders
const Row = React.memo(({ index, style, data }) => {
  const {
    tracks,
    columns,
    columnWidths,
    onTrackSelect,
    selectedTrackIds,
    setSelectedTrackIds,
  } = data;
  const track = tracks[index];

  const handleClick = (e) => {
    // If Cmd key is pressed, toggle selection
    if (e.metaKey) {
      if (selectedTrackIds.includes(track.id)) {
        setSelectedTrackIds(selectedTrackIds.filter((id) => id !== track.id));
      } else {
        setSelectedTrackIds([...selectedTrackIds, track.id]);
      }
    } else {
      onTrackSelect(track);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      onTrackSelect(track);
    }
  };

  // Determine if the current track is selected
  const isSelected = selectedTrackIds.includes(track.id);

  // Use useDrag to make the row draggable
  const [{ isDragging }, drag] = useDrag({
    type: DRAG_TYPE,
    item: isSelected
      ? { tracks: data.tracks.filter((t) => data.selectedTrackIds.includes(t.id)) }
      : { track },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  return (
    <div
      ref={drag}
      style={{
        ...style,
        opacity: isDragging ? 0.5 : 1,
        backgroundColor: isSelected ? '#e0e0e0' : 'transparent',
      }}
      className={`track-container ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyPress={handleKeyPress}
      aria-label={`Select track ${track.title} by ${track.artist}`}
      key={track.id}
    >
      {columns.map((column) => (
        <div
          key={column.name}
          className="cell"
          style={{ width: `${columnWidths[column.name] || MIN_COLUMN_WIDTH}px` }}
        >
          <Track track={track} column={column.name} />
        </div>
      ))}
    </div>
  );
});

Row.propTypes = {
  index: PropTypes.number.isRequired,
  style: PropTypes.object.isRequired,
  data: PropTypes.shape({
    tracks: PropTypes.array.isRequired,
    columns: PropTypes.array.isRequired,
    columnWidths: PropTypes.object.isRequired,
    onTrackSelect: PropTypes.func.isRequired,
    selectedTrackIds: PropTypes.array.isRequired,
    setSelectedTrackIds: PropTypes.func.isRequired,
  }).isRequired,
};

Tracklist.propTypes = {
  searchTerm: PropTypes.string.isRequired,
  onTrackSelect: PropTypes.func.isRequired,
  selectedTags: PropTypes.array.isRequired,
  bpmRange: PropTypes.array.isRequired,
  setAllTracks: PropTypes.func.isRequired, // Added PropType for setAllTracks
};

export default Tracklist;