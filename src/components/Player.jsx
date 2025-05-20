// src/components/Player/Player.jsx

import React, { useEffect, useRef, useContext } from 'react';
import WaveSurfer from 'wavesurfer.js';
import PropTypes from 'prop-types';
import './Player.scss';
import { PlaybackContext } from '../context/PlaybackContext';

const Player = () => {
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const { currentTrack, setPlayingWaveSurfer } = useContext(PlaybackContext);

  // Function to fetch waveform data from the server
  const fetchWaveformData = async (trackId) => {
    try {
      const response = await fetch(`http://localhost:3000/tracks/waveform/${trackId}`);
      const data = await response.json();
      if (data.waveform) {
        return data.waveform;
      } else {
        console.warn(`No waveform data found for track ID: ${trackId}`);
        return null;
      }
    } catch (error) {
      console.error('Error fetching waveform data:', error);
      return null;
    }
  };

  useEffect(() => {
    // Initialize WaveSurfer once when the component mounts
    if (!wavesurferRef.current) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#CDCDCD',
        progressColor: '#7f56d9',
        height: 60,
        barWidth: 2,
        responsive: true,
        normalize: true,
        autoCenter: true,
        backend: 'MediaElement',
        interact: true, // Allow interaction
        cursorColor: '#333',
        cursorWidth: 1,
        partialRender: true,
      });

      // Handle play event to manage global playback state
      wavesurferRef.current.on('play', () => {
        setPlayingWaveSurfer(wavesurferRef.current);
      });

      // Optional: Handle pause, finish, and other events as needed
      wavesurferRef.current.on('pause', () => {
        // Handle pause if necessary
      });

      wavesurferRef.current.on('finish', () => {
        // Handle track finish if necessary
      });
    }

    // Cleanup on unmount
    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, [setPlayingWaveSurfer]);

  useEffect(() => {
    const loadTrack = async () => {
      if (currentTrack && wavesurferRef.current) {
        const { id, path } = currentTrack;

        // Fetch waveform data from the server
        const waveformData = await fetchWaveformData(id);

        if (waveformData) {
          // Load the audio with waveform peaks
          wavesurferRef.current.load(path, waveformData);
        } else {
          // Load the audio without waveform peaks
          wavesurferRef.current.load(path);
        }
      }
    };

    loadTrack();
  }, [currentTrack]);

  return (
    <div className="player-container">
      <div ref={waveformRef} id="waveform-big"></div>
      {/* Add playback controls as needed */}
    </div>
  );
};

Player.propTypes = {
  // No props required as it uses context
};

export default Player;