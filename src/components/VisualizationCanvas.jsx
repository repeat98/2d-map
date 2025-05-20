import React, { useEffect, useState, useRef, useCallback } from 'react';
import 'pixi.js/unsafe-eval'; // For PixiJS v7+, often needed for certain features/performance.
import * as PIXI from 'pixi.js';
import WaveSurfer from 'wavesurfer.js'; // Ensure Wavesurfer.js is installed
import './VisualizationCanvas.scss';
import defaultArtwork from "../../assets/default-artwork.png";
import Waveform from './Waveform';
import ReactDOM from 'react-dom/client';
import { PlaybackContext } from '../context/PlaybackContext';
import { PCA } from 'ml-pca';

/*
* IMPORTANT NOTE FOR ELECTRON USERS (Content Security Policy - CSP):
* (CSP Note remains the same)
*/

// Core feature configurations (moods, spectral, bpm)
// Instrument and genre features will be added dynamically.
const coreFeaturesConfig = [
  { value: 'bpm', label: 'BPM', axisTitleStyle: { fill: 0xe74c3c, fontWeight: 'bold' }, isNumeric: true },
  { value: 'danceability', label: 'Danceability', axisTitleStyle: { fill: 0x3498db }, isNumeric: true },
  { value: 'happiness', label: 'Happiness', axisTitleStyle: { fill: 0xf1c40f }, isNumeric: true },
  { value: 'party', label: 'Party Vibe', isNumeric: true, axisTitleStyle: { fill: 0x9b59b6} },
  { value: 'aggressive', label: 'Aggressiveness', axisTitleStyle: { fill: 0xc0392b }, isNumeric: true },
  { value: 'relaxed', label: 'Relaxed Vibe', axisTitleStyle: { fill: 0x2ecc71 }, isNumeric: true },
  { value: 'sad', label: 'Sadness', isNumeric: true, axisTitleStyle: { fill: 0x7f8c8d } },
  { value: 'engagement', label: 'Engagement', isNumeric: true, axisTitleStyle: { fill: 0x1abc9c } },
  { value: 'approachability', label: 'Approachability', isNumeric: true, axisTitleStyle: { fill: 0x34495e } },
  { value: 'rms', label: 'RMS (Loudness)', isNumeric: true, axisTitleStyle: { fill: 0x16a085 } },
  { value: 'spectral_centroid', label: 'Spectral Centroid (Brightness)', isNumeric: true, axisTitleStyle: { fill: 0x27ae60 } },
  { value: 'spectral_bandwidth', label: 'Spectral Bandwidth', isNumeric: true, axisTitleStyle: { fill: 0x2980b9 } },
  { value: 'spectral_rolloff', label: 'Spectral Rolloff', isNumeric: true, axisTitleStyle: { fill: 0x8e44ad } },
  { value: 'spectral_contrast', label: 'Spectral Contrast (Peakiness)', isNumeric: true, axisTitleStyle: { fill: 0xf39c12 } },
  { value: 'spectral_flatness', label: 'Spectral Flatness (Noisiness)', isNumeric: true, axisTitleStyle: { fill: 0xd35400 } },
];

// Add probability threshold constant
const STYLE_PROBABILITY_THRESHOLD = 0.3; // Only count styles that appear with at least 30% probability

// Style constants
const PADDING = 70; const AXIS_COLOR = 0xAAAAAA; const TEXT_COLOR = 0xE0E0E0;
const DOT_RADIUS = 5; const DOT_RADIUS_HOVER = 7; const DEFAULT_DOT_COLOR = 0x00A9FF;
const HAPPINESS_COLOR = 0xFFD700; const AGGRESSIVE_COLOR = 0xFF4136; const RELAXED_COLOR = 0x2ECC40;
const TOOLTIP_BG_COLOR = 0x333333; const TOOLTIP_TEXT_COLOR = 0xFFFFFF;
const TOOLTIP_PADDING = 10; const COVER_ART_SIZE = 80;
const MIN_ZOOM = 1; const MAX_ZOOM = 5; const ZOOM_SENSITIVITY = 0.0005;
const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;

const VisualizationCanvas = () => {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  const [isSimilarityMode, setIsSimilarityMode] = useState(false);
  const [styleThreshold, setStyleThreshold] = useState(0.3);
  
  const pixiCanvasContainerRef = useRef(null);
  const pixiAppRef = useRef(null);
  const chartAreaRef = useRef(null);
  
  const tooltipContainerRef = useRef(null);
  const coverArtSpriteRef = useRef(null);
  const trackTitleTextRef = useRef(null);
  const trackFeaturesTextRef = useRef(null);
  const currentTooltipTrackRef = useRef(null);

  const wavesurferContainerRef = useRef(null);
  const wavesurferRef = useRef(null);
  const activeAudioUrlRef = useRef(null);

  const waveformContainerRef = useRef(null);

  const [selectableFeatures, setSelectableFeatures] = useState([...coreFeaturesConfig]);
  const [xAxisFeature, setXAxisFeature] = useState(coreFeaturesConfig[0]?.value || '');
  const [yAxisFeature, setYAxisFeature] = useState(coreFeaturesConfig[1]?.value || '');
  
  const [axisMinMax, setAxisMinMax] = useState({ x: null, y: null });
  const [isPixiAppReady, setIsPixiAppReady] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0});
  const onWheelZoomRef = useRef(null);

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [currentHoverTrack, setCurrentHoverTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const tooltipTimeoutRef = useRef(null);

  const playButtonRef = useRef(null);
  const playIconRef = useRef(null);

  const [tsneData, setTsneData] = useState(null); // For PCA/t-SNE results
  const [isTsneCalculating, setIsTsneCalculating] = useState(false);

  useEffect(() => {
    const fetchTracksAndPrepareFeatures = async () => {
      console.log("🚀 Fetching tracks and preparing features...");
      setIsLoadingTracks(true); setError(null);

      try {
        const response = await fetch("http://localhost:3000/tracks");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const rawTracks = await response.json();

        const allDiscoveredFeatureKeys = new Set();
        const featureFrequencies = {}; // To store frequencies of dynamic features

        const processedTracks = rawTracks.map(track => {
          const currentParsedFeatures = {};

          // Process core features with threshold
          coreFeaturesConfig.forEach(coreFeatureConf => {
            const featureKey = coreFeatureConf.value;
            if (track[featureKey] !== undefined && track[featureKey] !== null) {
              const val = parseFloat(track[featureKey]);
              if (!isNaN(val) && val >= styleThreshold) {
                currentParsedFeatures[featureKey] = val;
                allDiscoveredFeatureKeys.add(featureKey);
                featureFrequencies[featureKey] = (featureFrequencies[featureKey] || 0) + 1;
              }
            }
          });

          // Process genre features
          if (track.features) {
            try {
              const genreObject = typeof track.features === 'string' ? JSON.parse(track.features) : track.features;
              Object.entries(genreObject).forEach(([genreKey, value]) => {
                const val = parseFloat(value);
                if (!isNaN(val) && val >= styleThreshold) {
                  currentParsedFeatures[genreKey] = val;
                  allDiscoveredFeatureKeys.add(genreKey);
                  featureFrequencies[genreKey] = (featureFrequencies[genreKey] || 0) + 1;
                }
              });
            } catch (e) { console.error("Error parsing genre features for track:", track.id, e); }
          }

          // Process instrument features
          if (track.instrument_features) {
            try {
              const instrumentObject = typeof track.instrument_features === 'string'
                ? JSON.parse(track.instrument_features)
                : track.instrument_features;
              Object.entries(instrumentObject).forEach(([instrumentKey, probability]) => {
                const val = parseFloat(probability);
                if (!isNaN(val) && val >= styleThreshold) {
                  currentParsedFeatures[instrumentKey] = val;
                  allDiscoveredFeatureKeys.add(instrumentKey);
                  featureFrequencies[instrumentKey] = (featureFrequencies[instrumentKey] || 0) + 1;
                }
              });
            } catch (e) { console.error("Error parsing instrument features for track:", track.id, e); }
          }
          
          if (rawTracks.indexOf(track) === 0) {
            console.log("Sample parsedFeatures for first track:", currentParsedFeatures);
          }

          return { ...track, parsedFeatures: currentParsedFeatures };
        });
        
        console.log("Collected Feature Frequencies:", JSON.parse(JSON.stringify(featureFrequencies)));

        const coreFeatureValues = new Set(coreFeaturesConfig.map(f => f.value));
        const dynamicFeatureConfigs = [];

        Array.from(allDiscoveredFeatureKeys).forEach(featureKey => {
          if (coreFeatureValues.has(featureKey)) {
            return;
          }

          let label = featureKey;
          if (featureKey.includes("---")) { 
            label = featureKey.substring(featureKey.indexOf("---") + 3);
          }
          
          label = label
            .replace(/_/g, ' ') 
            .split(/[\s-]+/) 
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

          const frequency = featureFrequencies[featureKey] || 0;
          if (frequency > 0) { // Only add features that have a frequency
            dynamicFeatureConfigs.push({
              value: featureKey,
              label: `${label} (${frequency})`, // Add frequency to label
              isNumeric: true, 
              axisTitleStyle: { fill: 0x95a5a6 }, 
              frequency: frequency
            });
          }
        });

        // Log before sorting
        console.log("Dynamic Feature Configs (before sort - label, freq):", 
            JSON.parse(JSON.stringify(dynamicFeatureConfigs.map(f => ({label: f.label, freq: f.frequency, val: f.value})))));

        dynamicFeatureConfigs.sort((a, b) => {
          if (b.frequency !== a.frequency) {
            return b.frequency - a.frequency; // Sort by frequency descending
          }
          return a.label.localeCompare(b.label); // Then by label ascending
        });
        
        // Log after sorting
        console.log("Dynamic Feature Configs (after sort - label, freq):", 
            JSON.parse(JSON.stringify(dynamicFeatureConfigs.map(f => ({label: f.label, freq: f.frequency, val: f.value})))));
        
        const finalSelectableFeatures = [...coreFeaturesConfig, ...dynamicFeatureConfigs];

        // Log final list sample
        console.log("Final Selectable Features (first 20 entries with freq):", 
            finalSelectableFeatures.slice(0, 20).map(f => ({
                label: f.label, 
                value: f.value, 
                // For core features, frequency might be undefined, show N/A
                frequency: f.frequency !== undefined ? f.frequency : 'N/A (core)' 
            }))
        );


        setSelectableFeatures(finalSelectableFeatures);

        if (finalSelectableFeatures.length > 0 && !finalSelectableFeatures.find(f => f.value === xAxisFeature)) {
            setXAxisFeature(finalSelectableFeatures[0].value);
        }
        if (finalSelectableFeatures.length > 1 && !finalSelectableFeatures.find(f => f.value === yAxisFeature)) {
            setYAxisFeature(finalSelectableFeatures[1].value);
        } else if (finalSelectableFeatures.length === 1 && xAxisFeature !== finalSelectableFeatures[0].value) { 
            setYAxisFeature(finalSelectableFeatures[0].value);
        } else if (finalSelectableFeatures.length > 0 && yAxisFeature === xAxisFeature && finalSelectableFeatures.length > 1) {
            const secondFeature = finalSelectableFeatures.find(f => f.value !== xAxisFeature);
            if (secondFeature) setYAxisFeature(secondFeature.value);
            else setYAxisFeature(finalSelectableFeatures[0].value); 
        }


        setTracks(processedTracks);
      } catch (error) {
        console.error("Error fetching or processing tracks:", error);
        setError(`Failed to load tracks: ${error.message}`);
      } finally {
        setIsLoadingTracks(false);
      }
    };
    fetchTracksAndPrepareFeatures();
  }, [styleThreshold]);

  useEffect(() => { 
    if (isLoadingTracks || !tracks || tracks.length === 0 || !xAxisFeature || !yAxisFeature || selectableFeatures.length === 0) return;
    
    const calculateMinMax = (featureKey, tracksToCalc) => {
      let min = Infinity;
      let max = -Infinity;
      let hasValidValues = false;

      tracksToCalc.forEach(track => {
        const value = track.parsedFeatures?.[featureKey];
        if (typeof value === 'number' && !isNaN(value)) {
          min = Math.min(min, value);
          max = Math.max(max, value);
          hasValidValues = true;
        }
      });

      if (!hasValidValues) {
        console.warn(`No valid numeric values found for feature: ${featureKey}. Defaulting range to 0-1.`);
        return { min: 0, max: 1, range: 1, hasData: false }; 
      }
      const range = max - min;
      return { min, max, range: range === 0 ? 1 : range, hasData: true }; 
    };

    const xRange = calculateMinMax(xAxisFeature, tracks);
    const yRange = calculateMinMax(yAxisFeature, tracks);

    console.log(`🎯 Axis ranges calculated:`, {
      xAxis: { feature: xAxisFeature, ...xRange },
      yAxis: { feature: yAxisFeature, ...yRange }
    });
    setAxisMinMax({ x: xRange, y: yRange });
  }, [tracks, xAxisFeature, yAxisFeature, isLoadingTracks, selectableFeatures]);

  const updateAxesTextScale = useCallback((chartArea) => { 
    if (!chartArea || !chartArea.scale) return;
    const currentChartScale = chartArea.scale.x; const inverseScale = 1 / currentChartScale;
    for (const child of chartArea.children) { if (child.isAxisTextElement) { child.scale.set(inverseScale); } }
  }, []);

  useEffect(() => { 
    if (!pixiCanvasContainerRef.current || pixiAppRef.current) return;
    let app = new PIXI.Application();
    const initPrimaryApp = async (retryCount = 0) => {
      try {
        const { clientWidth: cw, clientHeight: ch } = pixiCanvasContainerRef.current;
        if (cw <= 0 || ch <= 0) { 
          if (retryCount < 5) { 
            setTimeout(() => initPrimaryApp(retryCount + 1), 250); 
            return; 
          } 
          throw new Error("Container zero dimensions"); 
        }

        await app.init({ 
          width: cw, 
          height: ch, 
          backgroundColor: 0x101010, 
          antialias: true, 
          resolution: window.devicePixelRatio || 1, 
          autoDensity: true 
        });

        pixiCanvasContainerRef.current.appendChild(app.canvas);
        pixiAppRef.current = app;
        setCanvasSize({width: app.screen.width, height: app.screen.height});
        
        chartAreaRef.current = new PIXI.Container();
        app.stage.addChild(chartAreaRef.current);

        tooltipContainerRef.current = new PIXI.Container();
        tooltipContainerRef.current.visible = false;
        tooltipContainerRef.current.eventMode = 'static';
        tooltipContainerRef.current.cursor = 'default';
        app.stage.addChild(tooltipContainerRef.current);

        const tooltipBg = new PIXI.Graphics()
          .roundRect(0, 0, 300, 200, 8)
          .fill({ color: 0x333333 });
        tooltipContainerRef.current.addChild(tooltipBg);

        coverArtSpriteRef.current = new PIXI.Sprite(PIXI.Texture.EMPTY);
        coverArtSpriteRef.current.position.set(10, 10);
        coverArtSpriteRef.current.width = 80;
        coverArtSpriteRef.current.height = 80;
        tooltipContainerRef.current.addChild(coverArtSpriteRef.current);

        trackTitleTextRef.current = new PIXI.Text({
          text: '',
          style: {
            fontFamily: 'Arial',
            fontSize: 16,
            fontWeight: 'bold',
            fill: 0xFFFFFF,
            wordWrap: true,
            wordWrapWidth: 200
          }
        });
        trackTitleTextRef.current.position.set(100, 10);
        tooltipContainerRef.current.addChild(trackTitleTextRef.current);

        trackFeaturesTextRef.current = new PIXI.Text({
          text: '',
          style: {
            fontFamily: 'Arial',
            fontSize: 14,
            fill: 0xAAAAAA,
            wordWrap: true,
            wordWrapWidth: 200,
            lineHeight: 18
          }
        });
        trackFeaturesTextRef.current.position.set(100, 40);
        tooltipContainerRef.current.addChild(trackFeaturesTextRef.current);

        playButtonRef.current = new PIXI.Graphics()
          .circle(0, 0, 12)
          .fill({ color: 0x6A82FB });
        playButtonRef.current.position.set(280, 30);
        playButtonRef.current.eventMode = 'static';
        playButtonRef.current.cursor = 'pointer';
        tooltipContainerRef.current.addChild(playButtonRef.current);

        playIconRef.current = new PIXI.Graphics();
        playIconRef.current.fill({ color: 0xFFFFFF })
          .moveTo(-4, -6)
          .lineTo(-4, 6)
          .lineTo(6, 0);
        playButtonRef.current.addChild(playIconRef.current);

        waveformContainerRef.current = new PIXI.Container();
        waveformContainerRef.current.position.set(10, 100); 
        tooltipContainerRef.current.addChild(waveformContainerRef.current); 

        playButtonRef.current.on('pointerover', () => {
          playButtonRef.current.clear()
            .circle(0, 0, 12)
            .fill({ color: 0x8BA3FF });
          playButtonRef.current.addChild(playIconRef.current);
        });

        playButtonRef.current.on('pointerout', () => {
          playButtonRef.current.clear()
            .circle(0, 0, 12)
            .fill({ color: 0x6A82FB });
          playButtonRef.current.addChild(playIconRef.current);
        });

        playButtonRef.current.on('pointerdown', async (event) => {
          event.stopPropagation(); 
          const trackToPlay = currentTooltipTrackRef.current; 
          if (trackToPlay && wavesurferRef.current) {
            if (wavesurferRef.current.isPlaying() && activeAudioUrlRef.current === trackToPlay.path) {
              wavesurferRef.current.pause();
              setIsPlaying(false);
            } else {
              if (activeAudioUrlRef.current !== trackToPlay.path) {
                console.log(`🌊 Loading new track for Wavesurfer: ${trackToPlay.path}`);
                await wavesurferRef.current.load(trackToPlay.path); 
                activeAudioUrlRef.current = trackToPlay.path;
              } else {
                wavesurferRef.current.play(); 
              }
              setIsPlaying(true); 
            }
          }
        });
        
        wavesurferRef.current?.on('ready', () => {
          const currentSrc = wavesurferRef.current?.getMediaElement()?.src;
          console.log('🌊 Wavesurfer ready for URL:', currentSrc);
          const tooltipTrack = currentTooltipTrackRef.current;
          if (tooltipTrack && tooltipTrack.path === activeAudioUrlRef.current && tooltipContainerRef.current?.visible) {
            console.log("🌊 Autoplaying on ready (tooltip is active for this track).");
            wavesurferRef.current.play().catch(e => console.error("🌊 Error auto-playing on ready:", e));
            setIsPlaying(true);
          }
        });


        onWheelZoomRef.current = (event) => {
          event.preventDefault();
          if (!chartAreaRef.current) return;

          const rect = pixiAppRef.current.canvas.getBoundingClientRect();
          const mouseX = event.clientX - rect.left;
          const mouseY = event.clientY - rect.top;
          const chartPoint = chartAreaRef.current.toLocal(new PIXI.Point(mouseX, mouseY));
          const zoomFactor = 1 - (event.deltaY * ZOOM_SENSITIVITY);
          const prevScale = chartAreaRef.current.scale.x;
          let newScale = prevScale * zoomFactor;
          newScale = Math.max(MIN_ZOOM, Math.min(newScale, MAX_ZOOM));
          if (prevScale === newScale) return;

          const scaleFactor = newScale / prevScale;
          const newX = chartPoint.x - (chartPoint.x - chartAreaRef.current.x) * scaleFactor;
          const newY = chartPoint.y - (chartPoint.y - chartAreaRef.current.y) * scaleFactor;

          chartAreaRef.current.scale.set(newScale);
          chartAreaRef.current.position.set(newX, newY);
          updateAxesTextScale(chartAreaRef.current);
        };

        pixiAppRef.current.canvas.addEventListener('wheel', onWheelZoomRef.current, { passive: false });

        app.stage.eventMode = 'static';
        app.stage.cursor = 'grab';
        let localIsDragging = false; 
        let localDragStart = { x: 0, y: 0 };
        let chartStartPos = { x: 0, y: 0 };


        app.stage.on('pointerdown', (event) => {
          if (event.target === app.stage || event.target === chartAreaRef.current) { 
            localIsDragging = true;
            localDragStart = { x: event.global.x, y: event.global.y };
            chartStartPos = { x: chartAreaRef.current.x, y: chartAreaRef.current.y };
            app.stage.cursor = 'grabbing';
          }
        });

        app.stage.on('pointermove', (event) => {
          if (localIsDragging) {
            const dx = event.global.x - localDragStart.x;
            const dy = event.global.y - localDragStart.y;
            chartAreaRef.current.x = chartStartPos.x + dx;
            chartAreaRef.current.y = chartStartPos.y + dy;
          }
        });

        const onPointerUp = () => {
          if (localIsDragging) {
            localIsDragging = false;
            app.stage.cursor = 'grab';
          }
        };
        app.stage.on('pointerup', onPointerUp);
        app.stage.on('pointerupoutside', onPointerUp);
        
        tooltipContainerRef.current.on('pointerover', () => {
          if (tooltipTimeoutRef.current) {
            clearTimeout(tooltipTimeoutRef.current);
            tooltipTimeoutRef.current = null;
          }
        });

        tooltipContainerRef.current.on('pointerout', () => {
          if (!tooltipTimeoutRef.current) {
            tooltipTimeoutRef.current = setTimeout(() => {
              setCurrentHoverTrack(null); 
              currentTooltipTrackRef.current = null;
              if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
              if (wavesurferRef.current && wavesurferRef.current.isPlaying()) {
                wavesurferRef.current.pause(); setIsPlaying(false);
              }
            }, 300);
          }
        });

        setIsPixiAppReady(true);
        console.log("✅ Pixi App, Tooltip Listeners, Wavesurfer, Zoom initialized.");
      } catch (initError) {
        console.error("💥 AppCreate: Failed to init Pixi App:", initError);
        setError(e => e || `Pixi Init Error: ${initError.message}`);
        if (app.destroy) app.destroy(true, {children:true, texture:true, basePath:true});
        app = null;
        pixiAppRef.current = null;
      }
    };
    initPrimaryApp();
    return () => {
      const currentApp = pixiAppRef.current;
      if (currentApp && currentApp.canvas && onWheelZoomRef.current) {
        currentApp.canvas.removeEventListener('wheel', onWheelZoomRef.current);
      }
      if (currentApp && currentApp.destroy) {
        currentApp.destroy(true, { children: true, texture: true, basePath: true });
      }
      pixiAppRef.current = null;
      if (wavesurferRef.current) {
        wavesurferRef.current.stop();
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
        console.log("🌊 Wavesurfer instance destroyed.");
      }
      chartAreaRef.current = null;
      tooltipContainerRef.current = null;
      currentTooltipTrackRef.current = null;
      setIsPixiAppReady(false);
    };
  }, [updateAxesTextScale]); 

  useEffect(() => {
    if (!currentHoverTrack || !tooltipContainerRef.current || !pixiAppRef.current) {
      if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
      const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
      existingReactContainers?.forEach(container => container.remove());
      return;
    }
    currentTooltipTrackRef.current = currentHoverTrack; 

    const updateTooltipVisuals = async () => {
      try {
        trackTitleTextRef.current.text = currentHoverTrack.title || 'Unknown Title';
        const xFeat = selectableFeatures.find(f => f.value === xAxisFeature);
        const yFeat = selectableFeatures.find(f => f.value === yAxisFeature);
        const xFeatureLabel = xFeat?.label || xAxisFeature;
        const yFeatureLabel = yFeat?.label || yAxisFeature;
        
        trackFeaturesTextRef.current.text = 
          `${xFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[xAxisFeature])}\n` +
          `${yFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[yAxisFeature])}`;

        const artworkPath = currentHoverTrack.artwork_thumbnail_path || defaultArtwork;
        const img = new Image();
        img.onload = () => { coverArtSpriteRef.current.texture = PIXI.Texture.from(img); };
        img.onerror = () => { coverArtSpriteRef.current.texture = PIXI.Texture.from(defaultArtwork); };
        img.src = artworkPath;
        
        tooltipContainerRef.current.visible = true; 

        const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
        existingReactContainers?.forEach(container => container.remove());

        const waveformHostElement = document.createElement('div');
        waveformHostElement.className = 'waveform-react-container';
        waveformHostElement.style.width = '150px'; 
        waveformHostElement.style.height = '40px';
        waveformHostElement.style.position = 'absolute'; 
        waveformHostElement.style.pointerEvents = 'auto'; 

        const tooltipGlobalPos = tooltipContainerRef.current.getGlobalPosition(new PIXI.Point());
        const canvasRect = pixiCanvasContainerRef.current.getBoundingClientRect();

        waveformHostElement.style.left = `${tooltipGlobalPos.x - canvasRect.left + waveformContainerRef.current.x}px`;
        waveformHostElement.style.top = `${tooltipGlobalPos.y - canvasRect.top + waveformContainerRef.current.y}px`;
        
        pixiCanvasContainerRef.current.appendChild(waveformHostElement);
        
        const root = ReactDOM.createRoot(waveformHostElement);
        const playbackContextValue = {
          setPlayingWaveSurfer: (ws) => {  }, 
          currentTrack: currentHoverTrack,
          setCurrentTrack: () => {},
        };
        
        root.render(
          <PlaybackContext.Provider value={playbackContextValue}>
            <Waveform
              key={currentHoverTrack.id} 
              trackId={currentHoverTrack.id.toString()}
              audioPath={currentHoverTrack.path} 
              isInteractive={true}
              wavesurferInstanceRef={wavesurferRef} 
              onPlay={() => {
                setIsPlaying(true);
                activeAudioUrlRef.current = currentHoverTrack.path;
              }}
              onPause={() => setIsPlaying(false)}
              onReadyToPlay={(ws) => { 
                if (activeAudioUrlRef.current === currentHoverTrack.path && tooltipContainerRef.current?.visible) {
                   ws.play();
                   setIsPlaying(true);
                }
              }}
            />
          </PlaybackContext.Provider>
        );
        
        return () => { 
          root.unmount();
          if (waveformHostElement.parentElement) {
            waveformHostElement.remove();
          }
        };

      } catch (error) {
        console.error("💥 Error updating tooltip:", error);
        if (coverArtSpriteRef.current) coverArtSpriteRef.current.texture = PIXI.Texture.from(defaultArtwork);
      }
      return () => {}; 
    };

    let cleanupPromise = updateTooltipVisuals();
     return () => { 
        if (cleanupPromise && typeof cleanupPromise.then === 'function') {
            cleanupPromise.then(actualCleanup => {
                if(typeof actualCleanup === 'function') actualCleanup();
            }).catch(e => console.warn("Error in async cleanup:", e));
        } else if (typeof cleanupPromise === 'function') {
            cleanupPromise(); 
        }
    };


  }, [currentHoverTrack, xAxisFeature, yAxisFeature, selectableFeatures, formatTickValue]);


  const formatTickValue = useCallback((value, isGenreAxis) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number' || isNaN(value)) return String(value); 
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
    if (Math.abs(value) >= 10000) return value.toExponential(1);
    const numStr = value.toFixed(2); 
    return parseFloat(numStr).toString(); 
  }, []);

  const drawAxes = useCallback((chartArea, currentXAxisFeatureKey, currentYAxisFeatureKey, xRange, yRange, currentCanvasSize) => { 
    if (!chartArea || !xRange || !yRange || !currentCanvasSize.width || !currentCanvasSize.height || selectableFeatures.length === 0) return;
    const graphics = new PIXI.Graphics();
    const { width: canvasWidth, height: canvasHeight } = currentCanvasSize;
    const drawableWidth = canvasWidth - 2 * PADDING; const drawableHeight = canvasHeight - 2 * PADDING;
    if (drawableWidth <=0 || drawableHeight <=0) return;
    
    const xFeatureInfo = selectableFeatures.find(f => f.value === currentXAxisFeatureKey);
    const yFeatureInfo = selectableFeatures.find(f => f.value === currentYAxisFeatureKey);

    const defaultAxisTextStyle = { fontFamily: 'Arial, sans-serif', fontSize: 12, fill: TEXT_COLOR, align: 'center' };
    const defaultTitleTextStyle = { fontFamily: 'Arial, sans-serif', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center'};
    
    const xTitleStyle = {...defaultTitleTextStyle, ...(xFeatureInfo?.axisTitleStyle || {})};
    const yTitleStyle = {...defaultTitleTextStyle, ...(yFeatureInfo?.axisTitleStyle || {})};

    graphics.moveTo(PADDING, canvasHeight - PADDING).lineTo(canvasWidth - PADDING, canvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
    graphics.moveTo(PADDING, PADDING).lineTo(PADDING, canvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
    
    const xTitleText = xFeatureInfo?.label || currentXAxisFeatureKey;
    const yTitleText = yFeatureInfo?.label || currentYAxisFeatureKey;

    const xTitle = new PIXI.Text({text: xTitleText, style:xTitleStyle});
    xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5, 0); xTitle.position.set(PADDING + drawableWidth / 2, canvasHeight - PADDING + 25);
    chartArea.addChild(xTitle);
    
    const yTitle = new PIXI.Text({text: yTitleText, style:yTitleStyle});
    yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5, 1); yTitle.rotation = -Math.PI / 2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight / 2);
    chartArea.addChild(yTitle);
    
    const numTicks = 5; 
    for (let i = 0; i <= numTicks; i++) {
        const xVal = xRange.min + (xRange.range / numTicks) * i;
        const xTickPos = PADDING + (i / numTicks) * drawableWidth;
        graphics.moveTo(xTickPos, canvasHeight - PADDING).lineTo(xTickPos, canvasHeight - PADDING + 5).stroke({width:1, color:AXIS_COLOR});
        const xLabel = new PIXI.Text({text:formatTickValue(xVal), style:defaultAxisTextStyle});
        xLabel.isAxisTextElement = true; xLabel.anchor.set(0.5, 0); xLabel.position.set(xTickPos, canvasHeight - PADDING + 8);
        chartArea.addChild(xLabel);
        
        const yVal = yRange.min + (yRange.range / numTicks) * i;
        const yTickPos = canvasHeight - PADDING - (i / numTicks) * drawableHeight;
        graphics.moveTo(PADDING, yTickPos).lineTo(PADDING - 5, yTickPos).stroke({width:1, color:AXIS_COLOR});
        const yLabel = new PIXI.Text({text:formatTickValue(yVal), style:defaultAxisTextStyle});
        yLabel.isAxisTextElement = true; yLabel.anchor.set(1, 0.5); yLabel.position.set(PADDING - 8, yTickPos);
        chartArea.addChild(yLabel);
    }
    chartArea.addChild(graphics);
  }, [formatTickValue, selectableFeatures]);

  const preparePcaData = useCallback((tracksToProcess, allSelectableFeaturesForPca) => {
    if (!tracksToProcess || tracksToProcess.length === 0) {
        console.warn("PCA: No tracks provided.");
        return null;
    }
    if (!allSelectableFeaturesForPca || allSelectableFeaturesForPca.length === 0) {
        console.warn("PCA: No selectable features defined.");
        return null;
    }

    const featureOrder = allSelectableFeaturesForPca
        .filter(f => f.isNumeric) 
        .map(f => f.value)
        .sort(); 

    if (featureOrder.length === 0) {
        console.warn("PCA: No numeric features identified from selectableFeatures for PCA.");
        return null;
    }
    console.log("PCA: Using features in order:", featureOrder.join(", "));

    const featureVectors = [];
    const trackIdsForPca = [];

    tracksToProcess.forEach(track => {
      const vector = [];
      let hasAnyValidData = false;
      featureOrder.forEach(featureKey => {
        const value = track.parsedFeatures?.[featureKey];
        if (typeof value === 'number' && !isNaN(value)) {
          vector.push(value);
          hasAnyValidData = true;
        } else {
          vector.push(0); 
        }
      });

      if (hasAnyValidData || featureOrder.length === 0) { 
          featureVectors.push(vector);
          trackIdsForPca.push(track.id);
      }
    });

    if (featureVectors.length === 0) {
        console.warn("PCA: No tracks with valid data for PCA after filtering.");
        return null;
    }
    if (featureVectors.length < 2) {
        console.warn("PCA: Need at least 2 data points for PCA. Got:", featureVectors.length);
        return null; 
    }
     if (featureVectors[0].length < 2 ) {
        console.warn(`PCA: Need at least 2 features (dimensions) for 2-component PCA. Got: ${featureVectors[0].length}`);
        if (featureVectors[0].length === 1) { 
             const oneDdata = {
                features: featureVectors.map(v => [v[0], 0.0]), 
                trackIds: trackIdsForPca,
                featureNames: [featureOrder[0], " artificial_second_component"]
            };
            console.log("PCA: Handling 1D data by padding with a zero component.");
            return oneDdata;
        }
        return null;
    }

    const numFeatures = featureOrder.length;
    const normalizedFeatureVectors = JSON.parse(JSON.stringify(featureVectors)); 

    for (let j = 0; j < numFeatures; j++) { 
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let i = 0; i < normalizedFeatureVectors.length; i++) { 
        minVal = Math.min(minVal, normalizedFeatureVectors[i][j]);
        maxVal = Math.max(maxVal, normalizedFeatureVectors[i][j]);
      }
      const range = maxVal - minVal;
      if (range === 0) { 
        for (let i = 0; i < normalizedFeatureVectors.length; i++) {
          normalizedFeatureVectors[i][j] = 0.5; 
        }
      } else {
        for (let i = 0; i < normalizedFeatureVectors.length; i++) {
          normalizedFeatureVectors[i][j] = (normalizedFeatureVectors[i][j] - minVal) / range;
        }
      }
    }
    
    console.log("PCA: Data prepared with", normalizedFeatureVectors.length, "tracks and", numFeatures, "normalized features per track.");
    return { features: normalizedFeatureVectors, trackIds: trackIdsForPca, featureNames: featureOrder };
  }, []);

  const calculatePca = useCallback((data) => {
    if (!data || !data.features || data.features.length === 0) {
      console.warn("PCA: Invalid or empty data provided to calculatePca.");
      setIsTsneCalculating(false);
      setTsneData(null);
      return;
    }
    const nActualFeatures = data.features[0]?.length || 0;
    if (nActualFeatures < 1) { 
        console.warn(`PCA: Not enough features (${nActualFeatures}) to perform PCA for 2 components.`);
        setIsTsneCalculating(false);
        setTsneData(null);
        return;
    }
    const nComponentsToUse = Math.min(2, nActualFeatures); 

    console.log(`🔄 Starting PCA calculation with ${data.features.length} tracks, ${nActualFeatures} features, for ${nComponentsToUse} components.`);
    
    try {
      const pca = new PCA(data.features);
      const result = pca.predict(data.features, { nComponents: nComponentsToUse });
      let resultArray = result.data || result; 

      if (nComponentsToUse === 1 && resultArray.every(p => typeof p[0] === 'number')) {
          resultArray = resultArray.map(point => [point[0], 0.0]);
          console.log("PCA: Result was 1D, padded to 2D for visualization.");
      }


      if (!resultArray || resultArray.length === 0 || !resultArray[0] || resultArray[0].length < nComponentsToUse ) {
          console.error("PCA: Prediction did not return expected 2D data. Result:", resultArray);
          setIsTsneCalculating(false);
          setTsneData(null);
          return;
      }
      
      const xValues = resultArray.map(point => point[0]);
      const yValues = resultArray.map(point => point[1]);
      
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);
      
      const xRangePCA = (xMax - xMin) === 0 ? 1 : (xMax - xMin); 
      const yRangePCA = (yMax - yMin) === 0 ? 1 : (yMax - yMin);

      const normalizedResult = resultArray.map((point, index) => ({
        x: (point[0] - xMin) / xRangePCA,
        y: (point[1] - yMin) / yRangePCA,
        trackId: data.trackIds[index]
      }));

      console.log("✅ PCA results normalized for", normalizedResult.length, "tracks.");
      setTsneData(normalizedResult);
    } catch (error) {
      console.error("❌ Error calculating PCA:", error);
      console.error("Error details:", error.stack);
      setTsneData(null);
    } finally {
      setIsTsneCalculating(false);
    }
  }, []); 

  useEffect(() => { 
    if (!tracks || tracks.length === 0) {
      setTsneData(null); return;
    }
    if (!isSimilarityMode) {
      setTsneData(null); return;
    }

    console.log("✅ Requesting PCA calculation for all tracks in similarity mode.");
    setIsTsneCalculating(true);
    
    setTimeout(() => {
      const dataForPca = preparePcaData(tracks, selectableFeatures);
      if (dataForPca && dataForPca.features.length > 0) {
        console.log("PCA: Data prepared, proceeding to calculatePca.", dataForPca.features.length, "tracks,", dataForPca.features[0].length, "features.");
        calculatePca(dataForPca);
      } else {
        console.warn("PCA: Data preparation failed or yielded no usable data. Clearing t-SNE data.");
        setIsTsneCalculating(false);
        setTsneData(null);
      }
    }, 0);
  }, [tracks, isSimilarityMode, preparePcaData, selectableFeatures, calculatePca]);


  const calculateVisualizationBounds = useCallback((points, padding = 0.1) => {
    if (!points || points.length === 0) return { xMin:0, xMax:1, yMin:0, yMax:1, xRange:1, yRange:1 };

    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    
    let xMin = Math.min(...xValues); let xMax = Math.max(...xValues);
    let yMin = Math.min(...yValues); let yMax = Math.max(...yValues);
    
    let xRange = xMax - xMin; let yRange = yMax - yMin;

    if (xRange === 0) { xRange = 1; xMin -= 0.5; xMax += 0.5; } 
    if (yRange === 0) { yRange = 1; yMin -= 0.5; yMax += 0.5; } 

    const xPadding = xRange * padding;
    const yPadding = yRange * padding;
    
    return {
      xMin: xMin - xPadding, xMax: xMax + xPadding,
      yMin: yMin - yPadding, yMax: yMax + yPadding,
      xRange: xRange + (2 * xPadding), yRange: yRange + (2 * yPadding)
    };
  }, []);

  useEffect(() => { 
    if (!isPixiAppReady || !pixiAppRef.current || !chartAreaRef.current || isLoadingTracks || error || !tracks || !canvasSize.width || !canvasSize.height) return;
    
    const app = pixiAppRef.current;
    const chartArea = chartAreaRef.current;
    chartArea.removeChildren();

    if (tracks.length === 0 && !isLoadingTracks) {
      const msgText = new PIXI.Text({text: "No tracks to display.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
      msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
      msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea); return;
    }

    if (isSimilarityMode && isTsneCalculating) {
      const loadingText = new PIXI.Text({text: "Calculating similarity visualization...", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
      loadingText.anchor.set(0.5); loadingText.position.set(app.screen.width / 2, app.screen.height / 2);
      loadingText.isAxisTextElement = true; chartArea.addChild(loadingText); updateAxesTextScale(chartArea); return;
    }

    const { width: currentCanvasWidth, height: currentCanvasHeight } = app.screen;
    const drawableWidth = currentCanvasWidth - 2 * PADDING;
    const drawableHeight = currentCanvasHeight - 2 * PADDING;
    if (drawableWidth <= 0 || drawableHeight <= 0) return;

    const commonDotLogic = (track, screenX, screenY) => {
        let fillColor = DEFAULT_DOT_COLOR;
        const happinessVal = track.parsedFeatures?.happiness;
        const aggressiveVal = track.parsedFeatures?.aggressive;
        const relaxedVal = track.parsedFeatures?.relaxed;

        if (typeof happinessVal === 'number' && happinessVal > 0.7) fillColor = HAPPINESS_COLOR;
        else if (typeof aggressiveVal === 'number' && aggressiveVal > 0.6) fillColor = AGGRESSIVE_COLOR;
        else if (typeof relaxedVal === 'number' && relaxedVal > 0.7) fillColor = RELAXED_COLOR;

        const dotContainer = new PIXI.Container();
        dotContainer.position.set(screenX, screenY);
        dotContainer.eventMode = 'static'; dotContainer.cursor = 'pointer';
        const dataDot = new PIXI.Graphics().circle(0, 0, DOT_RADIUS).fill({ color: fillColor });
        dotContainer.addChild(dataDot);
        const hitArea = new PIXI.Graphics().circle(0, 0, DOT_RADIUS * 1.5).fill({ color: 0xFFFFFF, alpha: 0.001 }); 
        dotContainer.addChild(hitArea);

        dotContainer.on('pointerover', (event) => {
          event.stopPropagation(); dataDot.scale.set(DOT_RADIUS_HOVER / DOT_RADIUS);
          setCurrentHoverTrack(track);
          if (tooltipTimeoutRef.current) { clearTimeout(tooltipTimeoutRef.current); tooltipTimeoutRef.current = null; }
          
          const mousePosition = event.global; const tooltipWidth = 300; const tooltipHeight = 200; 
          let x = mousePosition.x + 20; let y = mousePosition.y - tooltipHeight / 2;
          if (x + tooltipWidth > app.screen.width) x = mousePosition.x - tooltipWidth - 20;
          if (y + tooltipHeight > app.screen.height) y = app.screen.height - tooltipHeight - 10;
          if (y < 0) y = 10;
          
          if (tooltipContainerRef.current) { tooltipContainerRef.current.position.set(x, y); tooltipContainerRef.current.visible = true; }
        });
        dotContainer.on('pointerout', (event) => {
          event.stopPropagation(); dataDot.scale.set(1.0);
          tooltipTimeoutRef.current = setTimeout(() => {
            setCurrentHoverTrack(null); currentTooltipTrackRef.current = null;
            if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
          }, 300);
        });
        chartArea.addChild(dotContainer);
    };


    if (isSimilarityMode && tsneData && tsneData.length > 0) {
      const pcaBounds = calculateVisualizationBounds(tsneData);
      if (!pcaBounds) return;

      const graphics = new PIXI.Graphics(); 
      graphics.moveTo(PADDING, currentCanvasHeight - PADDING).lineTo(currentCanvasWidth - PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
      graphics.moveTo(PADDING, PADDING).lineTo(PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
      const xTitle = new PIXI.Text({text: "Principal Component 1", style: { fontFamily: 'Arial', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center' }});
      xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5,0); xTitle.position.set(PADDING + drawableWidth/2, currentCanvasHeight - PADDING + 25); chartArea.addChild(xTitle);
      const yTitle = new PIXI.Text({text: "Principal Component 2", style: { fontFamily: 'Arial', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center' }});
      yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5,1); yTitle.rotation = -Math.PI/2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight/2); chartArea.addChild(yTitle);
      chartArea.addChild(graphics);

      tsneData.forEach(({ x: pcaX, y: pcaY, trackId }) => {
        const track = tracks.find(t => t.id === trackId);
        if (!track) return;
        const screenX = PADDING + ((pcaX - pcaBounds.xMin) / pcaBounds.xRange) * drawableWidth;
        const screenY = PADDING + (1 - ((pcaY - pcaBounds.yMin) / pcaBounds.yRange)) * drawableHeight; 
        commonDotLogic(track, screenX, screenY);
      });

    } else if (!isSimilarityMode) {
      if (!axisMinMax.x || !axisMinMax.y || !axisMinMax.x.hasData || !axisMinMax.y.hasData) {
        const msgText = new PIXI.Text({ text: "Select features or wait for axis range calculation.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
        msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
        msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea); return;
      }
      const { x: xRange, y: yRange } = axisMinMax;
      drawAxes(chartArea, xAxisFeature, yAxisFeature, xRange, yRange, {width: currentCanvasWidth, height: currentCanvasHeight});

      tracks.forEach((track) => {
        const rawXVal = track.parsedFeatures?.[xAxisFeature];
        const rawYVal = track.parsedFeatures?.[yAxisFeature];
        if (typeof rawXVal !== 'number' || isNaN(rawXVal) || typeof rawYVal !== 'number' || isNaN(rawYVal)) {
          return; 
        }
        const screenX = PADDING + ((rawXVal - xRange.min) / xRange.range) * drawableWidth;
        const screenY = PADDING + (1 - ((rawYVal - yRange.min) / yRange.range)) * drawableHeight; 
        commonDotLogic(track, screenX, screenY);
      });
    }
    updateAxesTextScale(chartArea);
  }, [isPixiAppReady, tracks, axisMinMax, xAxisFeature, yAxisFeature, isLoadingTracks, error, drawAxes, formatTickValue, canvasSize, updateAxesTextScale, selectableFeatures, setCurrentHoverTrack, setIsPlaying, tsneData, isTsneCalculating, isSimilarityMode, calculateVisualizationBounds]);

  useEffect(() => { 
    return () => { if (tooltipTimeoutRef.current) clearTimeout(tooltipTimeoutRef.current); };
  }, []);

  useEffect(() => { 
    const handleResize = () => {
      if (pixiAppRef.current && pixiCanvasContainerRef.current) {
        const { clientWidth, clientHeight } = pixiCanvasContainerRef.current;
        pixiAppRef.current.renderer.resize(clientWidth, clientHeight);
        setCanvasSize({ width: clientWidth, height: clientHeight });
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize(); 
    return () => window.removeEventListener('resize', handleResize);
  }, [isPixiAppReady]); 

  useEffect(() => {
    if (wavesurferContainerRef.current && !wavesurferRef.current) { 
      const wsInstance = WaveSurfer.create({
        container: wavesurferContainerRef.current, 
        waveColor: '#6A82FB', progressColor: '#3B4D9A', height: 40, barWidth: 1,
        barGap: 1, cursorWidth: 0, interact: false, backend: 'MediaElement',
        normalize: true, autoCenter: true, partialRender: true, responsive: false,
      });
      wavesurferRef.current = wsInstance;
      console.log("🌊 Global Wavesurfer instance created for tooltip potential control.");
      wsInstance.on('error', (err) => console.error('🌊 Global WS Error:', err, "Attempted URL:", activeAudioUrlRef.current));
    }
  }, []);


  return ( 
    <div className="visualization-outer-container">
      <div className="controls-panel" style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 1000, backgroundColor: "rgba(30,30,30,0.8)", padding: "10px", borderRadius: "5px" }}>
        <div className="mode-toggle" style={{ marginBottom: '10px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#E0E0E0', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={isSimilarityMode}
              onChange={(e) => {
                setIsSimilarityMode(e.target.checked);
                if(chartAreaRef.current) chartAreaRef.current.scale.set(MIN_ZOOM); 
              }}
              style={{ width: '16px', height: '16px' }}
            />
            Similarity Mode
          </label>
        </div>
        <div className="style-threshold" style={{ marginBottom: '10px' }}>
          <label style={{ color: '#E0E0E0', display: 'block', marginBottom: '5px' }}>
            Feature Threshold: {(styleThreshold * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={styleThreshold * 100}
            onChange={(e) => setStyleThreshold(parseInt(e.target.value) / 100)}
            style={{ width: '100%' }}
          />
        </div>
        {!isSimilarityMode && (
          <div className="axis-selectors" style={{display: "flex", gap: "10px"}}>
            <div className="axis-selector">
              <label htmlFor="xAxisSelect" style={{color: "#ccc", marginRight:"5px"}}>X-Axis:</label>
              <select id="xAxisSelect" value={xAxisFeature} onChange={(e) => setXAxisFeature(e.target.value)} style={{padding:"3px"}}>
                {selectableFeatures.map((feature) => (
                  <option key={`x-${feature.value}`} value={feature.value}>
                    {feature.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="axis-selector">
              <label htmlFor="yAxisSelect" style={{color: "#ccc", marginRight:"5px"}}>Y-Axis:</label>
              <select id="yAxisSelect" value={yAxisFeature} onChange={(e) => setYAxisFeature(e.target.value)} style={{padding:"3px"}}>
                {selectableFeatures.map((feature) => (
                  <option key={`y-${feature.value}`} value={feature.value}>
                    {feature.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>
      <div className="canvas-wrapper">
        <div ref={pixiCanvasContainerRef} className="pixi-canvas-target" />
        <div ref={wavesurferContainerRef} className="wavesurfer-container-hidden" style={{ display: 'none' }}></div>
        {(isLoadingTracks || (isSimilarityMode && isTsneCalculating)) && 
            <div className="loading-overlay">
                {isLoadingTracks ? "Loading tracks..." : "Calculating similarity..."}
            </div>
        }
        {error && <div className="error-overlay">{error}</div>}
      </div>
    </div>
  );
};

export default VisualizationCanvas;