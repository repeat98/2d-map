// src/components/Track/Waveform.jsx

import React, { useEffect, useRef, useContext } from 'react';
import WaveSurfer from 'wavesurfer.js';
import PropTypes from 'prop-types';
import './Waveform.scss';
import { PlaybackContext } from '../context/PlaybackContext';

const Waveform = ({ trackId, audioPath, isInteractive = false, onPlay }) => {
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const { setPlayingWaveSurfer } = useContext(PlaybackContext);

  // Function to save waveform data to the server
  const saveWaveform = (waveformData) => {
    fetch('http://localhost:3000/tracks/waveform', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ id: trackId, waveform: waveformData }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          console.error('Error saving waveform:', data.error);
        } else {
          console.log('Waveform saved successfully:', data.message);
        }
      })
      .catch((error) => {
        console.error('Fetch Error:', error);
      });
  };

  // Function to load waveform data from the server
  const loadWaveform = async () => {
    try {
      const response = await fetch(`http://localhost:3000/tracks/waveform/${trackId}`);
      const data = await response.json();
      if (data.waveform) {
        return data.waveform;
      } else {
        return null;
      }
    } catch (error) {
      console.error('Error loading waveform data:', error);
      return null;
    }
  };

  useEffect(() => {
    const initializeWaveSurfer = async () => {
      // Initialize WaveSurfer
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#ddd',
        progressColor: '#7f56d9',
        cursorColor: '#333',
        responsive: true,
        normalize: true,
        autoCenter: true,
        height: isInteractive ? 32 : 8,
        barWidth: isInteractive ? 1 : 1,
        barRadius: 1,
        cursorWidth: 1,
        backend: 'MediaElement',
        interact: isInteractive,
        partialRender: true,
      });

      // Load existing waveform data if available
      const existingWaveform = await loadWaveform();
      if (existingWaveform) {
        wavesurferRef.current.load(audioPath, existingWaveform);
      } else {
        wavesurferRef.current.load(audioPath);
      }

      // Once ready, if no waveform data was loaded, save the generated waveform
      wavesurferRef.current.on('ready', () => {
        if (!existingWaveform) {
          const waveformData = wavesurferRef.current.exportPeaks(1024);
          saveWaveform(waveformData);
        }
      });

      // **Modify Interaction Handling Here**
      if (isInteractive) {
        wavesurferRef.current.on("interaction", () => {
          // **Prevent toggling play/pause on interaction**
          // **Option 1: Do Nothing**
          // This allows interactions like seeking without affecting playback.

          // **Option 2: Only Play if Not Already Playing**
          if (!wavesurferRef.current.isPlaying()) {
            wavesurferRef.current.play();
          }

          // **Option 3: Remove the Interaction Handler Completely**
          // If you don't need any action on interaction, you can skip adding this handler.
        });
      }

      // Handle play event to ensure only one WaveSurfer plays at a time
      wavesurferRef.current.on('play', () => {
        setPlayingWaveSurfer(wavesurferRef.current);
        if (onPlay) onPlay();
      });

      // Cleanup on unmount
      return () => {
        if (wavesurferRef.current) {
          wavesurferRef.current.destroy();
        }
      };
    };

    initializeWaveSurfer();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioPath, isInteractive, trackId]);

  return <div ref={waveformRef} className="waveform-container"></div>;
};

Waveform.propTypes = {
  trackId: PropTypes.string.isRequired,
  audioPath: PropTypes.string.isRequired,
  isInteractive: PropTypes.bool,
  onPlay: PropTypes.func,
};

export default Waveform;