// src/context/PlaybackContext.jsx

import React, { createContext, useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';

export const PlaybackContext = createContext();

export const PlaybackProvider = ({ children }) => {
  const [currentTrack, setCurrentTrack] = useState(null);
  const currentWaveSurfer = useRef(null);

  /**
   * Sets the currently playing WaveSurfer instance.
   * Pauses any previously playing instance to ensure only one plays at a time.
   * @param {WaveSurfer} wavesurfer - The WaveSurfer instance to set as currently playing.
   */
  const setPlayingWaveSurfer = (wavesurfer) => {
    if (currentWaveSurfer.current && currentWaveSurfer.current !== wavesurfer) {
      currentWaveSurfer.current.pause();
    }
    currentWaveSurfer.current = wavesurfer;
  };

  /**
   * Handles global keydown events.
   * Specifically listens for the spacebar to toggle play/pause.
   * Ensures that the event doesn't interfere when typing in inputs or textareas.
   * @param {KeyboardEvent} e - The keyboard event.
   */
  const handleKeyDown = (e) => {
    if (e.code === 'Space') {
      const activeElement = document.activeElement;
      const isInputFocused =
        activeElement.tagName === 'INPUT' ||
        activeElement.tagName === 'TEXTAREA' ||
        activeElement.isContentEditable;

      if (!isInputFocused) {
        e.preventDefault(); // Prevent default spacebar actions like scrolling
        if (currentWaveSurfer.current) {
          if (currentWaveSurfer.current.isPlaying()) {
            currentWaveSurfer.current.pause();
          } else {
            currentWaveSurfer.current.play();
          }
        }
      }
    }
  };

  useEffect(() => {
    // Add the keydown event listener when the component mounts
    window.addEventListener('keydown', handleKeyDown);

    // Clean up the event listener when the component unmounts
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <PlaybackContext.Provider
      value={{
        currentTrack,
        setCurrentTrack,
        setPlayingWaveSurfer,
      }}
    >
      {children}
    </PlaybackContext.Provider>
  );
};

PlaybackProvider.propTypes = {
  children: PropTypes.node.isRequired,
};