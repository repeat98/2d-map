import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import 'pixi.js/unsafe-eval';
import * as PIXI from 'pixi.js';
import WaveSurfer from 'wavesurfer.js';
import './VisualizationCanvas.scss';
import defaultArtwork from "../../assets/default-artwork.png";
import Waveform from './Waveform';
import ReactDOM from 'react-dom/client';
import { PlaybackContext } from '../context/PlaybackContext';
import { PCA } from 'ml-pca';
import { kmeans } from 'ml-kmeans';

/*
* IMPORTANT NOTE FOR ELECTRON USERS (Content Security Policy - CSP):
* (CSP Note remains the same)
*/

// --- FEATURE CATEGORIES ---
const FEATURE_CATEGORIES = {
  MOOD: 'Mood',
  SPECTRAL: 'Spectral',
  TECHNICAL: 'Technical',
  STYLE: 'Style',
  INSTRUMENT: 'Instrument'
};

const coreFeaturesConfig = [
  { value: 'bpm', label: 'BPM', axisTitleStyle: { fill: 0xe74c3c, fontWeight: 'bold' }, isNumeric: true, category: FEATURE_CATEGORIES.TECHNICAL },
  { value: 'danceability', label: 'Danceability', axisTitleStyle: { fill: 0x3498db }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'happiness', label: 'Happiness', axisTitleStyle: { fill: 0xf1c40f }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'party', label: 'Party Vibe', isNumeric: true, axisTitleStyle: { fill: 0x9b59b6}, category: FEATURE_CATEGORIES.MOOD },
  { value: 'aggressive', label: 'Aggressiveness', axisTitleStyle: { fill: 0xc0392b }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'relaxed', label: 'Relaxed Vibe', axisTitleStyle: { fill: 0x2ecc71 }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'sad', label: 'Sadness', isNumeric: true, axisTitleStyle: { fill: 0x7f8c8d }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'engagement', label: 'Engagement', isNumeric: true, axisTitleStyle: { fill: 0x1abc9c }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'approachability', label: 'Approachability', isNumeric: true, axisTitleStyle: { fill: 0x34495e }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'rms', label: 'RMS (Loudness)', isNumeric: true, axisTitleStyle: { fill: 0x16a085 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'spectral_centroid', label: 'Spectral Centroid (Brightness)', isNumeric: true, axisTitleStyle: { fill: 0x27ae60 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'spectral_bandwidth', label: 'Spectral Bandwidth', isNumeric: true, axisTitleStyle: { fill: 0x2980b9 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'spectral_rolloff', label: 'Spectral Rolloff', isNumeric: true, axisTitleStyle: { fill: 0x8e44ad }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'spectral_contrast', label: 'Spectral Contrast (Peakiness)', isNumeric: true, axisTitleStyle: { fill: 0xf39c12 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'spectral_flatness', label: 'Spectral Flatness (Noisiness)', isNumeric: true, axisTitleStyle: { fill: 0xd35400 }, category: FEATURE_CATEGORIES.SPECTRAL },
];

const DYNAMIC_FEATURE_MIN_PROBABILITY = 0.3; // Renamed for clarity
const PADDING = 70; const AXIS_COLOR = 0xAAAAAA; const TEXT_COLOR = 0xE0E0E0;
const DOT_RADIUS = 5; const DOT_RADIUS_HOVER = 7; const DEFAULT_DOT_COLOR = 0x00A9FF;
const HAPPINESS_COLOR = 0xFFD700; const AGGRESSIVE_COLOR = 0xFF4136; const RELAXED_COLOR = 0x2ECC40;
const TOOLTIP_BG_COLOR = 0x333333; const TOOLTIP_TEXT_COLOR = 0xFFFFFF;
const TOOLTIP_PADDING = 10; const COVER_ART_SIZE = 80;
const MIN_ZOOM = 1; const MAX_ZOOM = 5; const ZOOM_SENSITIVITY = 0.0005;
const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;

// Helper for generating distinct colors for clusters
const generateClusterColors = (numColors) => {
  const colors = [];
  if (numColors <=0) return [0xCCCCCC]; // Default color if k is 0 or invalid
  for (let i = 0; i < numColors; i++) {
    const hue = (i * (360 / numColors)) % 360;
    const hex = hslToHex(hue, 70, 60);
    colors.push(parseInt(hex.replace('#', ''), 16));
  }
  return colors;
};

function hslToHex(h, s, l) {
  l /= 100;
  const a = s * Math.min(l, 1 - l) / 100;
  const f = n => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color).toString(16).padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}


const VisualizationCanvas = () => {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  const [isSimilarityMode, setIsSimilarityMode] = useState(false);
  const [styleThreshold, setStyleThreshold] = useState(DYNAMIC_FEATURE_MIN_PROBABILITY); // Initialize with constant

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

  // --- CLUSTERING STATE ---
  const [clusterSettings, setClusterSettings] = useState({
    [FEATURE_CATEGORIES.MOOD]: true,
    [FEATURE_CATEGORIES.SPECTRAL]: true,
    [FEATURE_CATEGORIES.TECHNICAL]: true,
    [FEATURE_CATEGORIES.STYLE]: true,
    [FEATURE_CATEGORIES.INSTRUMENT]: true,
    enabled: true,
  });
  const [numClustersControl, setNumClustersControl] = useState(7); // User-configurable k
  const clusterColors = useMemo(() => generateClusterColors(numClustersControl > 0 ? numClustersControl : 1), [numClustersControl]);

  const [clusters, setClusters] = useState([]);
  const [isClusteringCalculating, setIsClusteringCalculating] = useState(false);
  // --- END CLUSTERING STATE ---

  const [tsneData, setTsneData] = useState(null);
  const [isPcaCalculating, setIsPcaCalculating] = useState(false);


  useEffect(() => {
    const fetchTracksAndPrepareFeatures = async () => {
      console.log("🚀 Fetching tracks and preparing features...");
      setIsLoadingTracks(true); setError(null);

      try {
        const response = await fetch("http://localhost:3000/tracks");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const rawTracks = await response.json();

        const allDiscoveredFeatureKeys = new Set();
        const featureFrequencies = {};
        const dynamicFeatureSourceMap = new Map(); // featureKey -> original category

        const processedTracks = rawTracks.map(track => {
          const currentParsedFeatures = {};
          coreFeaturesConfig.forEach(coreFeatureConf => {
            const featureKey = coreFeatureConf.value;
            if (track[featureKey] !== undefined && track[featureKey] !== null) {
              const val = parseFloat(track[featureKey]);
              if (!isNaN(val)) {
                currentParsedFeatures[featureKey] = val;
                allDiscoveredFeatureKeys.add(featureKey); // Still add here for dynamicFeatureConfigs loop
                // dynamicFeatureSourceMap.set(featureKey, coreFeatureConf.category); // Core features already have category
                featureFrequencies[featureKey] = (featureFrequencies[featureKey] || 0) + 1;
              }
            }
          });

          const processDynamicFeatures = (featureObject, category) => {
             if (featureObject) {
                try {
                    const parsedObj = typeof featureObject === 'string' ? JSON.parse(featureObject) : featureObject;
                    Object.entries(parsedObj).forEach(([key, value]) => {
                        const val = parseFloat(value);
                        if (!isNaN(val) && val >= styleThreshold) { // Use component state styleThreshold
                            currentParsedFeatures[key] = val;
                            allDiscoveredFeatureKeys.add(key);
                            dynamicFeatureSourceMap.set(key, category); // Store source category for dynamic keys
                            featureFrequencies[key] = (featureFrequencies[key] || 0) + 1;
                        }
                    });
                } catch (e) { console.error(`Error parsing ${category} features for track:`, track.id, e); }
            }
          };

          processDynamicFeatures(track.features, FEATURE_CATEGORIES.STYLE);
          processDynamicFeatures(track.instrument_features, FEATURE_CATEGORIES.INSTRUMENT); // Corrected category


          if (rawTracks.indexOf(track) === 0) {
            console.log("Sample parsedFeatures for first track:", currentParsedFeatures);
          }
          return { ...track, parsedFeatures: currentParsedFeatures, id: track.id.toString() };
        });

        const coreFeatureValues = new Set(coreFeaturesConfig.map(f => f.value));
        const dynamicFeatureConfigs = [];

        Array.from(allDiscoveredFeatureKeys).forEach(featureKey => {
          if (coreFeatureValues.has(featureKey)) return; // Skip core features already defined

          let label = featureKey;
          if (featureKey.includes("---")) {
            label = featureKey.substring(featureKey.indexOf("---") + 3);
          }
          label = label.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          const frequency = featureFrequencies[featureKey] || 0;
          const determinedCategory = dynamicFeatureSourceMap.get(featureKey) || FEATURE_CATEGORIES.STYLE; // Fallback

          if (frequency > 0) {
            dynamicFeatureConfigs.push({
              value: featureKey,
              label: `${label} (${frequency})`,
              isNumeric: true,
              axisTitleStyle: { fill: 0x95a5a6 },
              frequency: frequency,
              category: determinedCategory // Use the determined category
            });
          }
        });

        dynamicFeatureConfigs.sort((a, b) => {
          if (b.frequency !== a.frequency) return b.frequency - a.frequency;
          return a.label.localeCompare(b.label);
        });

        const finalSelectableFeatures = [...coreFeaturesConfig, ...dynamicFeatureConfigs];
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

  // Axis Min/Max Calculation (Unchanged)
  useEffect(() => {
    if (isLoadingTracks || !tracks || tracks.length === 0 || !xAxisFeature || !yAxisFeature || selectableFeatures.length === 0) return;
    const calculateMinMax = (featureKey, tracksToCalc) => {
      let min = Infinity; let max = -Infinity; let hasValidValues = false;
      tracksToCalc.forEach(track => {
        const value = track.parsedFeatures?.[featureKey];
        if (typeof value === 'number' && !isNaN(value)) {
          min = Math.min(min, value); max = Math.max(max, value); hasValidValues = true;
        }
      });
      if (!hasValidValues) return { min: 0, max: 1, range: 1, hasData: false };
      const range = max - min;
      return { min, max, range: range === 0 ? 1 : range, hasData: true };
    };
    const xRange = calculateMinMax(xAxisFeature, tracks);
    const yRange = calculateMinMax(yAxisFeature, tracks);
    setAxisMinMax({ x: xRange, y: yRange });
  }, [tracks, xAxisFeature, yAxisFeature, isLoadingTracks, selectableFeatures]);

  // Pixi App Initialization (Largely unchanged)
  useEffect(() => {
    // ... (Full Pixi init from the original code should be here) ...
    // This includes: app creation, tooltip setup, play button, wavesurfer global instance, zoom/pan listeners
    // For brevity, I'm omitting the full copy-paste of the lengthy Pixi setup from the problem description.
    // Ensure the existing PIXI setup logic is present.
    // --- START OF COPIED PIXI INIT (FROM ORIGINAL, MINUS WAVESURFER PART MOVED) ---
    if (!pixiCanvasContainerRef.current || pixiAppRef.current) return;
    let app = new PIXI.Application();
    const initPrimaryApp = async (retryCount = 0) => {
      try {
        const { clientWidth: cw, clientHeight: ch } = pixiCanvasContainerRef.current;
        if (cw <= 0 || ch <= 0) {
          if (retryCount < 5) { setTimeout(() => initPrimaryApp(retryCount + 1), 250); return; }
          throw new Error("Container zero dimensions");
        }
        await app.init({ width: cw, height: ch, backgroundColor: 0x101010, antialias: true, resolution: window.devicePixelRatio || 1, autoDensity: true });
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
        const tooltipBg = new PIXI.Graphics().roundRect(0, 0, 300, 200, 8).fill({ color: 0x333333 }); // Adjusted size slightly for potential longer labels
        tooltipContainerRef.current.addChild(tooltipBg);
        coverArtSpriteRef.current = new PIXI.Sprite(PIXI.Texture.EMPTY);
        coverArtSpriteRef.current.position.set(10, 10);
        coverArtSpriteRef.current.width = 80;
        coverArtSpriteRef.current.height = 80;
        tooltipContainerRef.current.addChild(coverArtSpriteRef.current);
        trackTitleTextRef.current = new PIXI.Text({ text: '', style: { fontFamily: 'Arial', fontSize: 16, fontWeight: 'bold', fill: 0xFFFFFF, wordWrap: true, wordWrapWidth: 200 }});
        trackTitleTextRef.current.position.set(100, 10);
        tooltipContainerRef.current.addChild(trackTitleTextRef.current);
        trackFeaturesTextRef.current = new PIXI.Text({ text: '', style: { fontFamily: 'Arial', fontSize: 14, fill: 0xAAAAAA, wordWrap: true, wordWrapWidth: 200, lineHeight: 18 }});
        trackFeaturesTextRef.current.position.set(100, 40); // Adjusted y for potentially more feature text
        tooltipContainerRef.current.addChild(trackFeaturesTextRef.current);
        playButtonRef.current = new PIXI.Graphics().circle(0, 0, 12).fill({ color: 0x6A82FB });
        playButtonRef.current.position.set(280, 30); // Position within new tooltip size
        playButtonRef.current.eventMode = 'static';
        playButtonRef.current.cursor = 'pointer';
        tooltipContainerRef.current.addChild(playButtonRef.current);
        playIconRef.current = new PIXI.Graphics();
        playIconRef.current.fill({ color: 0xFFFFFF }).moveTo(-4, -6).lineTo(-4, 6).lineTo(6, 0);
        playButtonRef.current.addChild(playIconRef.current);
        waveformContainerRef.current = new PIXI.Container();
        waveformContainerRef.current.position.set(10, 100);
        tooltipContainerRef.current.addChild(waveformContainerRef.current);
        playButtonRef.current.on('pointerover', () => { playButtonRef.current.clear().circle(0, 0, 12).fill({ color: 0x8BA3FF }); playButtonRef.current.addChild(playIconRef.current); });
        playButtonRef.current.on('pointerout', () => { playButtonRef.current.clear().circle(0, 0, 12).fill({ color: 0x6A82FB }); playButtonRef.current.addChild(playIconRef.current); });
        playButtonRef.current.on('pointerdown', async (event) => {
          event.stopPropagation();
          const trackToPlay = currentTooltipTrackRef.current;
          if (trackToPlay && wavesurferRef.current) {
            if (wavesurferRef.current.isPlaying() && activeAudioUrlRef.current === trackToPlay.path) {
              wavesurferRef.current.pause(); setIsPlaying(false);
            } else {
              if (activeAudioUrlRef.current !== trackToPlay.path) {
                console.log(`🌊 Loading new track for Wavesurfer: ${trackToPlay.path}`);
                activeAudioUrlRef.current = trackToPlay.path; // Set before load to prevent race conditions with ready handler
                await wavesurferRef.current.load(trackToPlay.path);
                // Play will be handled by 'ready' event if conditions match
              } else {
                 wavesurferRef.current.play().catch(e => console.error("Play error", e));
                 setIsPlaying(true);
              }
            }
          }
        });

        // Global Wavesurfer Instance Setup
        if (wavesurferContainerRef.current && !wavesurferRef.current) {
            const wsInstance = WaveSurfer.create({
                container: wavesurferContainerRef.current,
                waveColor: '#6A82FB', progressColor: '#3B4D9A', height: 40, barWidth: 1,
                barGap: 1, cursorWidth: 0, interact: false,
                backend: 'MediaElement', normalize: true, autoCenter: true, partialRender: true, responsive: false,
            });
            wavesurferRef.current = wsInstance;
            console.log("🌊 Global Wavesurfer instance created.");
            wsInstance.on('error', (err) => console.error('🌊 Global WS Error:', err, "Attempted URL:", activeAudioUrlRef.current));
            wsInstance.on('ready', () => {
              const currentSrc = wsInstance.getMediaElement()?.src;
              console.log('🌊 Wavesurfer ready for URL:', currentSrc);
              const tooltipTrack = currentTooltipTrackRef.current;
              // Autoplay only if this track is the active one for the tooltip and tooltip is visible
              if (tooltipTrack && tooltipTrack.path === activeAudioUrlRef.current && tooltipContainerRef.current?.visible) {
                console.log("🌊 Autoplaying on ready (tooltip is active for this track).");
                wsInstance.play().catch(e => console.error("🌊 Error auto-playing on ready:", e));
                setIsPlaying(true);
              }
            });
             wsInstance.on('play', () => setIsPlaying(true));
             wsInstance.on('pause', () => setIsPlaying(false));
             wsInstance.on('finish', () => setIsPlaying(false));
        }

        onWheelZoomRef.current = (event) => {
          event.preventDefault(); if (!chartAreaRef.current) return;
          const rect = pixiAppRef.current.canvas.getBoundingClientRect();
          const mouseX = event.clientX - rect.left; const mouseY = event.clientY - rect.top;
          const chartPoint = chartAreaRef.current.toLocal(new PIXI.Point(mouseX, mouseY));
          const zoomFactor = 1 - (event.deltaY * ZOOM_SENSITIVITY);
          const prevScale = chartAreaRef.current.scale.x; let newScale = prevScale * zoomFactor;
          newScale = Math.max(MIN_ZOOM, Math.min(newScale, MAX_ZOOM)); if (prevScale === newScale) return;
          const scaleFactor = newScale / prevScale;
          const newX = chartPoint.x - (chartPoint.x - chartAreaRef.current.x) * scaleFactor;
          const newY = chartPoint.y - (chartPoint.y - chartAreaRef.current.y) * scaleFactor;
          chartAreaRef.current.scale.set(newScale); chartAreaRef.current.position.set(newX, newY);
          updateAxesTextScale(chartAreaRef.current);
        };
        pixiAppRef.current.canvas.addEventListener('wheel', onWheelZoomRef.current, { passive: false });
        app.stage.eventMode = 'static'; app.stage.cursor = 'grab';
        let localIsDragging = false; let localDragStart = { x: 0, y: 0 }; let chartStartPos = { x: 0, y: 0 };
        app.stage.on('pointerdown', (event) => {
          if (event.target === app.stage || event.target === chartAreaRef.current) {
            localIsDragging = true; localDragStart = { x: event.global.x, y: event.global.y };
            chartStartPos = { x: chartAreaRef.current.x, y: chartAreaRef.current.y }; app.stage.cursor = 'grabbing';
          }
        });
        app.stage.on('pointermove', (event) => {
          if (localIsDragging) {
            const dx = event.global.x - localDragStart.x; const dy = event.global.y - localDragStart.y;
            chartAreaRef.current.x = chartStartPos.x + dx; chartAreaRef.current.y = chartStartPos.y + dy;
          }
        });
        const onPointerUp = () => { if (localIsDragging) { localIsDragging = false; app.stage.cursor = 'grab'; }};
        app.stage.on('pointerup', onPointerUp); app.stage.on('pointerupoutside', onPointerUp);
        tooltipContainerRef.current.on('pointerover', () => { if (tooltipTimeoutRef.current) { clearTimeout(tooltipTimeoutRef.current); tooltipTimeoutRef.current = null; }});
        tooltipContainerRef.current.on('pointerout', () => {
          if (!tooltipTimeoutRef.current) {
            tooltipTimeoutRef.current = setTimeout(() => {
              setCurrentHoverTrack(null); currentTooltipTrackRef.current = null;
              if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
              if (wavesurferRef.current && wavesurferRef.current.isPlaying()) {
                // Only pause if the audio is not for the main player (if you had one)
                // For now, tooltip pause is fine.
                wavesurferRef.current.pause();
                setIsPlaying(false);
              }
            }, 300);
          }
        });
        setIsPixiAppReady(true); console.log("✅ Pixi App, Tooltip, Wavesurfer, Zoom initialized.");
      } catch (initError) {
        console.error("💥 AppCreate: Failed to init Pixi App:", initError);
        setError(e => e || `Pixi Init Error: ${initError.message}`);
        if (app.destroy) app.destroy(true, {children:true, texture:true, basePath:true});
        app = null; pixiAppRef.current = null;
      }
    };
    initPrimaryApp();
    return () => {
      const currentApp = pixiAppRef.current;
      if (currentApp && currentApp.canvas && onWheelZoomRef.current) { currentApp.canvas.removeEventListener('wheel', onWheelZoomRef.current); }
      if (currentApp && currentApp.destroy) { currentApp.destroy(true, { children: true, texture: true, basePath: true });}
      pixiAppRef.current = null;
      if (wavesurferRef.current) { wavesurferRef.current.stop(); wavesurferRef.current.destroy(); wavesurferRef.current = null; console.log("🌊 Wavesurfer instance destroyed."); }
      chartAreaRef.current = null; tooltipContainerRef.current = null; currentTooltipTrackRef.current = null; setIsPixiAppReady(false);
    };
    // --- END OF COPIED PIXI INIT ---
  }, [updateAxesTextScale]); // updateAxesTextScale is defined below.

  const formatTickValue = useCallback((value, isGenreAxis) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number' || isNaN(value)) return String(value);
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
    if (Math.abs(value) >= 10000) return value.toExponential(1);
    const numStr = value.toFixed(2);
    return parseFloat(numStr).toString();
  }, []);

  // Tooltip Update Logic (Add cluster label)
  useEffect(() => {
    if (!currentHoverTrack || !tooltipContainerRef.current || !pixiAppRef.current) {
      if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
      const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
      existingReactContainers?.forEach(container => {
        const root = container._reactRoot; // Retrieve stored root
        if (root) root.unmount();
        container.remove();
      });
      return;
    }
    currentTooltipTrackRef.current = currentHoverTrack;

    const updateTooltipVisuals = async () => {
      try {
        const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
        existingReactContainers?.forEach(container => {
          const root = container._reactRoot;
          if (root) root.unmount();
          container.remove();
        });

        trackTitleTextRef.current.text = currentHoverTrack.title || 'Unknown Title';
        let featuresText = '';
        if (isSimilarityMode && currentHoverTrack.clusterId !== undefined && clusters.length > 0) {
            const cluster = clusters.find(c => c.id === currentHoverTrack.clusterId);
            featuresText = `Cluster: ${cluster ? cluster.label : 'N/A'}\n`; // Use improved label
        }

        const xFeat = selectableFeatures.find(f => f.value === xAxisFeature);
        const yFeat = selectableFeatures.find(f => f.value === yAxisFeature);
        const xFeatureLabel = xFeat?.label.split('(')[0].trim() || xAxisFeature; // Cleaner label
        const yFeatureLabel = yFeat?.label.split('(')[0].trim() || yAxisFeature; // Cleaner label

        featuresText +=
          `${xFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[xAxisFeature])}\n` +
          `${yFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[yAxisFeature])}`;
        trackFeaturesTextRef.current.text = featuresText;

        const artworkPath = currentHoverTrack.artwork_thumbnail_path || defaultArtwork;
        // PIXI's texture cache handles image loading efficiently. Direct assignment is fine.
        coverArtSpriteRef.current.texture = await PIXI.Assets.load(artworkPath).catch(() => PIXI.Texture.from(defaultArtwork));

        tooltipContainerRef.current.visible = true;

        const waveformHostElement = document.createElement('div');
        waveformHostElement.className = 'waveform-react-container';
        waveformHostElement.style.width = '150px'; // Match Waveform component if it has fixed size
        waveformHostElement.style.height = '40px';
        waveformHostElement.style.position = 'absolute'; // Relative to pixiCanvasContainerRef
        waveformHostElement.style.pointerEvents = 'auto'; // Allow interaction with React component

        const tooltipGlobalPos = tooltipContainerRef.current.getGlobalPosition(new PIXI.Point());
        const canvasRect = pixiCanvasContainerRef.current.getBoundingClientRect();

        waveformHostElement.style.left = `${tooltipGlobalPos.x - canvasRect.left + waveformContainerRef.current.x}px`;
        waveformHostElement.style.top = `${tooltipGlobalPos.y - canvasRect.top + waveformContainerRef.current.y}px`;

        pixiCanvasContainerRef.current.appendChild(waveformHostElement);

        const root = ReactDOM.createRoot(waveformHostElement);
        waveformHostElement._reactRoot = root; // Store root for unmounting

        const playbackContextValue = {
          setPlayingWaveSurfer: (ws) => { /* Allow Waveform to control a global player if needed */ },
          currentTrack: currentHoverTrack, // Pass current track for context
          setCurrentTrack: () => { /* If Waveform needs to set global track */ },
        };

        root.render(
          <PlaybackContext.Provider value={playbackContextValue}>
            <Waveform
              key={`${currentHoverTrack.id}-tooltip`} // Ensure re-render if track changes
              trackId={currentHoverTrack.id.toString()}
              audioPath={currentHoverTrack.path}
              isInteractive={true}
              wavesurferInstanceRef={wavesurferRef} // Pass the global ref
              onPlay={() => {
                setIsPlaying(true);
                activeAudioUrlRef.current = currentHoverTrack.path;
              }}
              onPause={() => setIsPlaying(false)}
              onReadyToPlay={(wsInstance) => {
                // Autoplay logic moved to global wavesurfer 'ready' handler to ensure it's the correct track
                if (activeAudioUrlRef.current === currentHoverTrack.path &&
                    tooltipContainerRef.current?.visible &&
                    currentTooltipTrackRef.current?.id === currentHoverTrack.id) {
                  wsInstance.play().catch(e => console.warn("Waveform play failed on ready:", e));
                  setIsPlaying(true);
                }
              }}
            />
          </PlaybackContext.Provider>
        );
         return () => { // Cleanup for this specific waveform render
          if (root) root.unmount();
          if (waveformHostElement.parentElement) waveformHostElement.remove();
        };
      } catch (error) {
        console.error("💥 Error updating tooltip:", error);
        if (coverArtSpriteRef.current) {
          coverArtSpriteRef.current.texture = PIXI.Texture.from(defaultArtwork);
        }
      }
    };

    let cleanupPromise = updateTooltipVisuals();
    return () => { // Main useEffect cleanup
      if (cleanupPromise && typeof cleanupPromise.then === 'function') {
        cleanupPromise.then(actualCleanup => {
          if (typeof actualCleanup === 'function') actualCleanup();
        }).catch(e => console.warn("Error in async tooltip cleanup:", e));
      } else if (typeof cleanupPromise === 'function') { // If not async
        cleanupPromise();
      }
    };
  }, [currentHoverTrack, xAxisFeature, yAxisFeature, selectableFeatures, formatTickValue, isSimilarityMode, clusters, isPlaying]); // Added isPlaying to re-evaluate play icon

  const updateAxesTextScale = useCallback((chartArea) => {
    if (!chartArea || !chartArea.scale) return;
    const currentChartScale = chartArea.scale.x; const inverseScale = 1 / currentChartScale;
    for (const child of chartArea.children) { if (child.isAxisTextElement) { child.scale.set(inverseScale); } }
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
    const xTitleText = xFeatureInfo?.label.split('(')[0].trim() || currentXAxisFeatureKey;
    const yTitleText = yFeatureInfo?.label.split('(')[0].trim() || currentYAxisFeatureKey;
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


  // --- DATA PREPARATION FOR CLUSTERING & PCA ---
  const prepareFeatureData = useCallback((tracksToProcess, allSelectableFeatures, currentClusterSettings) => {
    if (!tracksToProcess?.length || !allSelectableFeatures?.length) return null;

    const activeFeatureConfigs = allSelectableFeatures.filter(featureConf => 
      featureConf.isNumeric && currentClusterSettings[featureConf.category]
    );

    if (!activeFeatureConfigs.length) {
      console.warn("DataPrep: No numeric features selected based on active categories.");
      return null;
    }

    const featureVectors = [];
    const trackIdsForProcessing = [];
    const validTracksForProcessing = [];

    // Pre-allocate arrays for better performance
    const vectorLength = activeFeatureConfigs.length;
    const tempVector = new Array(vectorLength);

    tracksToProcess.forEach(track => {
      let hasAnyValidData = false;
      for (let i = 0; i < vectorLength; i++) {
        const value = track.parsedFeatures?.[activeFeatureConfigs[i].value];
        tempVector[i] = typeof value === 'number' && !isNaN(value) ? value : 0;
        if (tempVector[i] !== 0) hasAnyValidData = true;
      }

      if (hasAnyValidData) {
        featureVectors.push([...tempVector]); // Create new array to avoid reference issues
        trackIdsForProcessing.push(track.id);
        validTracksForProcessing.push(track);
      }
    });

    if (!featureVectors.length) return null;

    // Normalize features in-place for better performance
    const numFeatures = vectorLength;
    const normalizationParams = new Array(numFeatures);

    for (let j = 0; j < numFeatures; j++) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      
      // Find min/max in one pass
      for (let i = 0; i < featureVectors.length; i++) {
        const val = featureVectors[i][j];
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }

      const range = maxVal - minVal;
      normalizationParams[j] = { min: minVal, range: range === 0 ? 1 : range };

      // Normalize in-place
      if (range === 0) {
        for (let i = 0; i < featureVectors.length; i++) {
          featureVectors[i][j] = 0.5;
        }
      } else {
        for (let i = 0; i < featureVectors.length; i++) {
          featureVectors[i][j] = (featureVectors[i][j] - minVal) / range;
        }
      }
    }

    return {
      features: featureVectors,
      trackIds: trackIdsForProcessing,
      originalTracks: validTracksForProcessing,
      featureConfigsUsed: activeFeatureConfigs,
      normalizationParams
    };
  }, []);


  // --- K-MEANS CLUSTERING ---
  const runKMeansClustering = useCallback((data, kValue) => {
    if (!data || !data.features || data.features.length === 0) return null;
    let actualK = kValue;
    if (data.features.length < actualK) {
        console.warn(`KMeans: Not enough data points (${data.features.length}) for ${actualK} clusters. Reducing k.`);
        actualK = Math.max(1, data.features.length);
    }
    if (actualK === 0 && data.features.length > 0) actualK = 1; // Ensure k is at least 1 if there's data
    if (actualK === 0) return null;


    console.log(`🔄 Starting K-Means with ${data.features.length} tracks, ${actualK} clusters.`);
    try {
      const ans = kmeans(data.features, actualK, {seed: 42});
      const trackIdToClusterId = {};
      ans.clusters.forEach((clusterIndex, trackIndex) => {
        trackIdToClusterId[data.trackIds[trackIndex]] = clusterIndex;
      });
      console.log("✅ K-Means completed.");
      return { assignments: ans.clusters, centroids: ans.centroids, trackIdToClusterId, actualKUsed: actualK };
    } catch (error) {
      console.error("❌ Error calculating K-Means:", error);
      return null;
    }
  }, []);

  // --- DESCRIPTIVE CLUSTER LABELING (IMPROVED) ---
  const generateClusterLabels = useCallback((clusteringResult, dataForClustering, allSelectableFeatures, currentStyleThreshold) => {
    if (!clusteringResult || !dataForClustering || !clusteringResult.assignments) return [];
    console.log("🏷️ Generating improved cluster labels...");

    const newClusters = [];
    const { originalTracks, trackIds, featureConfigsUsed } = dataForClustering;
    const { assignments, actualKUsed } = clusteringResult;

    // Get enabled categories
    const enabledCategories = Object.entries(clusterSettings)
      .filter(([key, value]) => key !== 'enabled' && value === true)
      .map(([key]) => key);

    for (let i = 0; i < actualKUsed; i++) {
        const trackIndicesInCluster = [];
        assignments.forEach((clusterId, index) => {
            if (clusterId === i) trackIndicesInCluster.push(index);
        });

        if (trackIndicesInCluster.length === 0) continue;

        const tracksInCluster = trackIndicesInCluster.map(idx => originalTracks[idx]);
        let labelParts = [];

        // Only include BPM if Technical category is enabled
        if (enabledCategories.includes(FEATURE_CATEGORIES.TECHNICAL)) {
            const bpmValues = tracksInCluster.map(t => t.parsedFeatures?.bpm).filter(v => typeof v === 'number');
            if (bpmValues.length > 0) {
                const avgBpm = bpmValues.reduce((sum, v) => sum + v, 0) / bpmValues.length;
                if (avgBpm < 95) labelParts.push(`Slow (${Math.round(avgBpm)} BPM)`);
                else if (avgBpm < 130) labelParts.push(`Mid-Tempo (${Math.round(avgBpm)} BPM)`);
                else labelParts.push(`Fast (${Math.round(avgBpm)} BPM)`);
            }
        }

        // Characteristic Mood/Technical/Spectral Features
        const moodTechSpectralDescriptors = [];
        const relevantFeatureConfs = featureConfigsUsed
            .map(fc => allSelectableFeatures.find(sf => sf.value === fc.value) || fc)
            .filter(featureConf => featureConf &&
                enabledCategories.includes(featureConf.category) && // Only include enabled categories
                (featureConf.category === FEATURE_CATEGORIES.MOOD ||
                 featureConf.category === FEATURE_CATEGORIES.TECHNICAL ||
                 featureConf.category === FEATURE_CATEGORIES.SPECTRAL) &&
                 featureConf.value !== 'bpm' // Already handled
            );

        relevantFeatureConfs.forEach(featureConf => {
            const values = tracksInCluster.map(t => t.parsedFeatures?.[featureConf.value]).filter(v => typeof v === 'number');
            if (values.length === 0) return;
            const avgValue = values.reduce((sum, v) => sum + v, 0) / values.length;

            if (avgValue > 0.60) {
                moodTechSpectralDescriptors.push({
                    name: featureConf.label.split('(')[0].trim(),
                    score: avgValue,
                    category: featureConf.category
                });
            }
        });

        // Sort by category priority and then by score
        const categoryPriority = {
            [FEATURE_CATEGORIES.MOOD]: 1,
            [FEATURE_CATEGORIES.TECHNICAL]: 2,
            [FEATURE_CATEGORIES.SPECTRAL]: 3
        };
        
        moodTechSpectralDescriptors.sort((a, b) => {
            if (categoryPriority[a.category] !== categoryPriority[b.category]) {
                return categoryPriority[a.category] - categoryPriority[b.category];
            }
            return b.score - a.score;
        });

        // Add top features from each enabled category
        const addedCategories = new Set();
        moodTechSpectralDescriptors.forEach(desc => {
            if (!addedCategories.has(desc.category) && labelParts.length < 3) {
                labelParts.push(desc.name);
                addedCategories.add(desc.category);
            }
        });

        // Style/Instrument Features
        if (enabledCategories.includes(FEATURE_CATEGORIES.STYLE) || enabledCategories.includes(FEATURE_CATEGORIES.INSTRUMENT)) {
            const styleInstrumentCounts = {};
            const styleInstrumentFeatureDefs = featureConfigsUsed
                .map(fc => allSelectableFeatures.find(sf => sf.value === fc.value) || fc)
                .filter(featureConf => featureConf &&
                    enabledCategories.includes(featureConf.category) && // Only include enabled categories
                    (featureConf.category === FEATURE_CATEGORIES.STYLE ||
                     featureConf.category === FEATURE_CATEGORIES.INSTRUMENT)
                );

            styleInstrumentFeatureDefs.forEach(featureConf => {
                let trackCountWithFeature = 0;
                tracksInCluster.forEach(track => {
                    if (track.parsedFeatures?.[featureConf.value] >= currentStyleThreshold) {
                        trackCountWithFeature++;
                    }
                });
                if (trackCountWithFeature / tracksInCluster.length > 0.3) {
                    styleInstrumentCounts[featureConf.label.split('(')[0].trim()] = {
                        count: trackCountWithFeature,
                        category: featureConf.category
                    };
                }
            });

            // Sort by category priority and then by count
            const styleInstrumentPriority = {
                [FEATURE_CATEGORIES.STYLE]: 1,
                [FEATURE_CATEGORIES.INSTRUMENT]: 2
            };

            const sortedStylesInstruments = Object.entries(styleInstrumentCounts)
                .sort((a, b) => {
                    if (styleInstrumentPriority[a[1].category] !== styleInstrumentPriority[b[1].category]) {
                        return styleInstrumentPriority[a[1].category] - styleInstrumentPriority[b[1].category];
                    }
                    return b[1].count - a[1].count;
                });

            // Add top features from each enabled category
            const addedStyleInstrumentCategories = new Set();
            sortedStylesInstruments.forEach(([label, info]) => {
                if (!addedStyleInstrumentCategories.has(info.category) && labelParts.length < 4) {
                    labelParts.push(label);
                    addedStyleInstrumentCategories.add(info.category);
                }
            });
        }

        let finalLabel = labelParts.slice(0, 3).join(' | ');
        if (!finalLabel) {
            // If no features were selected, show enabled categories
            const enabledCategoryLabels = enabledCategories
                .map(cat => cat.toLowerCase())
                .join('/');
            finalLabel = `Cluster ${i + 1} (${enabledCategoryLabels})`;
        }

        newClusters.push({
            id: i,
            label: finalLabel,
            trackIds: trackIndicesInCluster.map(idx => trackIds[idx]),
            color: clusterColors[i % clusterColors.length]
        });
    }
    console.log("🏷️ Generated labels:", newClusters.map(c=> ({id: c.id, label: c.label, tracks: c.trackIds.length }) ));
    return newClusters;
  }, [selectableFeatures, styleThreshold, clusterColors, clusterSettings]); // Added clusterSettings dependency


  // --- PCA CALCULATION ---
  const calculatePca = useCallback((data, forClusteringVis = false) => {
    if (!data || !data.features || data.features.length === 0) {
      if (forClusteringVis) setIsPcaCalculating(false);
      setTsneData(null);
      return;
    }

    const nActualFeatures = data.features[0]?.length || 0;
    if (nActualFeatures < 1) {
      if (forClusteringVis) setIsPcaCalculating(false);
      setTsneData(null);
      return;
    }

    // Use requestAnimationFrame for smoother UI
    requestAnimationFrame(() => {
      try {
        const pca = new PCA(data.features);
        const result = pca.predict(data.features, { nComponents: 2 });
        let resultArray = result.data || result;

        if (!resultArray?.length || !resultArray[0]?.length) {
          if (forClusteringVis) setIsPcaCalculating(false);
          setTsneData(null);
          return;
        }

        // Calculate bounds in one pass
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        resultArray.forEach(point => {
          xMin = Math.min(xMin, point[0]);
          xMax = Math.max(xMax, point[0]);
          yMin = Math.min(yMin, point[1]);
          yMax = Math.max(yMax, point[1]);
        });

        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;

        // Create normalized result in one pass
        const normalizedResult = resultArray.map((point, index) => ({
          x: (point[0] - xMin) / xRange,
          y: (point[1] - yMin) / yRange,
          trackId: data.trackIds[index]
        }));

        setTsneData(normalizedResult);
      } catch (error) {
        console.error("❌ Error calculating PCA:", error);
        setTsneData(null);
      } finally {
        if (forClusteringVis) setIsPcaCalculating(false);
      }
    });
  }, []);


  // --- EFFECT FOR CLUSTERING & SUBSEQUENT PCA ---
  useEffect(() => {
    if (!isSimilarityMode || !clusterSettings.enabled || !tracks.length || !selectableFeatures.length || numClustersControl <= 0) {
      setClusters([]);
      setTsneData(null);
      return;
    }

    const activeCategories = Object.entries(clusterSettings)
      .filter(([key, value]) => key !== 'enabled' && value === true)
      .map(([key]) => key);

    if (!activeCategories.length) {
      setClusters([]);
      setTsneData(null);
      return;
    }

    // Use requestAnimationFrame for smoother UI
    let animationFrameId;
    const timeoutId = setTimeout(() => {
      setIsClusteringCalculating(true);
      setIsPcaCalculating(true);

      // Use requestAnimationFrame to batch state updates
      animationFrameId = requestAnimationFrame(() => {
        const dataForClustering = prepareFeatureData(tracks, selectableFeatures, clusterSettings);

        if (dataForClustering?.features.length) {
          // Run clustering in a separate animation frame
          requestAnimationFrame(() => {
            const clusteringResult = runKMeansClustering(dataForClustering, numClustersControl);

            if (clusteringResult) {
              // Batch state updates
              requestAnimationFrame(() => {
                const trackIdToClusterId = clusteringResult.trackIdToClusterId;
                setTracks(prevTracks => prevTracks.map(t => ({...t, clusterId: trackIdToClusterId[t.id]})));
                const labeledClusters = generateClusterLabels(clusteringResult, dataForClustering, selectableFeatures, styleThreshold);
                setClusters(labeledClusters);
                calculatePca(dataForClustering, true);
              });
            } else {
              setClusters([]);
              setTsneData(null);
              setIsPcaCalculating(false);
            }
            setIsClusteringCalculating(false);
          });
        } else {
          setClusters([]);
          setTsneData(null);
          setIsPcaCalculating(false);
          setIsClusteringCalculating(false);
        }
      });
    }, 500); // Reduced debounce time since we're using requestAnimationFrame

    return () => {
      clearTimeout(timeoutId);
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [
    isSimilarityMode,
    clusterSettings,
    tracks.length,
    selectableFeatures.length,
    numClustersControl,
    prepareFeatureData,
    runKMeansClustering,
    generateClusterLabels,
    calculatePca,
    styleThreshold
  ]);


  const calculateVisualizationBounds = useCallback((points, padding = 0.1) => {
    if (!points || points.length === 0) return { xMin:0, xMax:1, yMin:0, yMax:1, xRange:1, yRange:1 };
    const xValues = points.map(p => p.x); const yValues = points.map(p => p.y);
    let xMin = Math.min(...xValues); let xMax = Math.max(...xValues);
    let yMin = Math.min(...yValues); let yMax = Math.max(...yValues);
    let xRange = xMax - xMin; let yRange = yMax - yMin;
    if (xRange === 0) { xRange = 1; xMin -= 0.5; xMax += 0.5; }
    if (yRange === 0) { yRange = 1; yMin -= 0.5; yMax += 0.5; }
    const xPadding = xRange * padding; const yPadding = yRange * padding;
    return {
      xMin: xMin - xPadding, xMax: xMax + xPadding,
      yMin: yMin - yPadding, yMax: yMax + yPadding,
      xRange: xRange + (2 * xPadding), yRange: yRange + (2 * yPadding)
    };
  }, []);

  // Main PIXI Rendering useEffect (Modified for Clustering)
  useEffect(() => {
    if (!isPixiAppReady || !pixiAppRef.current || !chartAreaRef.current || isLoadingTracks || error || !tracks || !canvasSize.width || !canvasSize.height) return;

    const app = pixiAppRef.current;
    const chartArea = chartAreaRef.current;
    chartArea.removeChildren(); // Clear previous drawings

    if (tracks.length === 0 && !isLoadingTracks) {
      const msgText = new PIXI.Text({text:"No tracks to display.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
      msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
      msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea); return;
    }

    let currentLoadingMessage = "";
    if (isSimilarityMode) {
        if (isClusteringCalculating) currentLoadingMessage = "Calculating clusters...";
        else if (isPcaCalculating) currentLoadingMessage = "Calculating similarity projection...";
    }

    if (currentLoadingMessage) {
      const loadingText = new PIXI.Text({text:currentLoadingMessage, style:new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
      loadingText.anchor.set(0.5); loadingText.position.set(app.screen.width / 2, app.screen.height / 2);
      loadingText.isAxisTextElement = true; chartArea.addChild(loadingText); updateAxesTextScale(chartArea); return;
    }

    const { width: currentCanvasWidth, height: currentCanvasHeight } = app.screen;
    const drawableWidth = currentCanvasWidth - 2 * PADDING;
    const drawableHeight = currentCanvasHeight - 2 * PADDING;
    if (drawableWidth <= 0 || drawableHeight <= 0) return;

    const commonDotLogic = (track, screenX, screenY) => {
        let fillColor = DEFAULT_DOT_COLOR;
        if (isSimilarityMode && clusterSettings.enabled && track.clusterId !== undefined && clusters.length > 0) {
            const cluster = clusters.find(c => c.id === track.clusterId);
            fillColor = cluster ? cluster.color : DEFAULT_DOT_COLOR; // Use cluster.color which is now from dynamic palette
        } else {
            const happinessVal = track.parsedFeatures?.happiness;
            const aggressiveVal = track.parsedFeatures?.aggressive;
            const relaxedVal = track.parsedFeatures?.relaxed;
            if (typeof happinessVal === 'number' && happinessVal > 0.7) fillColor = HAPPINESS_COLOR;
            else if (typeof aggressiveVal === 'number' && aggressiveVal > 0.6) fillColor = AGGRESSIVE_COLOR;
            else if (typeof relaxedVal === 'number' && relaxedVal > 0.7) fillColor = RELAXED_COLOR;
        }

        const dotContainer = new PIXI.Container();
        dotContainer.position.set(screenX, screenY);
        dotContainer.eventMode = 'static'; dotContainer.cursor = 'pointer';
        const dataDot = new PIXI.Graphics().circle(0, 0, DOT_RADIUS).fill({ color: fillColor });
        dotContainer.addChild(dataDot);
        const hitAreaSize = DOT_RADIUS * 1.8; // Slightly larger hit area
        const hitArea = new PIXI.Graphics().circle(0,0, hitAreaSize).fill({color: 0xFFFFFF, alpha: 0.001}); // For better pointer events
        dotContainer.addChild(hitArea);


        dotContainer.on('pointerover', (event) => {
          event.stopPropagation(); dataDot.scale.set(DOT_RADIUS_HOVER / DOT_RADIUS);
          setCurrentHoverTrack(track);
          if (tooltipTimeoutRef.current) { clearTimeout(tooltipTimeoutRef.current); tooltipTimeoutRef.current = null; }
          const mousePosition = event.global; const tooltipWidth = 300; const tooltipHeight = 200; // Tooltip dimensions
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
      const pcaBounds = calculateVisualizationBounds(tsneData, 0.05); // Slightly less padding for PCA
      if (!pcaBounds) return;

      const graphics = new PIXI.Graphics();
      graphics.moveTo(PADDING, currentCanvasHeight - PADDING).lineTo(currentCanvasWidth - PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
      graphics.moveTo(PADDING, PADDING).lineTo(PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});

      const activeCats = Object.entries(clusterSettings)
        .filter(([,val]) => typeof val === 'number' && val > 0 && val <=1) // check if weight is active
        .map(([key]) => FEATURE_CATEGORIES[Object.keys(FEATURE_CATEGORIES).find(k=>FEATURE_CATEGORIES[k] === key)] || key) // Get the label
        .join('/') || "Selected";
      const xTitleText = clusterSettings.enabled ? `PC1 (${activeCats} Features)` : "Principal Component 1";
      const yTitleText = clusterSettings.enabled ? `PC2 (${activeCats} Features)` : "Principal Component 2";

      const titleTextStyle = { fontFamily: 'Arial', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center' };
      const xTitle = new PIXI.Text({text:xTitleText, style: titleTextStyle});
      xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5,0); xTitle.position.set(PADDING + drawableWidth/2, currentCanvasHeight - PADDING + 25); chartArea.addChild(xTitle);
      const yTitle = new PIXI.Text({text:yTitleText, style: titleTextStyle});
      yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5,1); yTitle.rotation = -Math.PI/2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight/2); chartArea.addChild(yTitle);
      chartArea.addChild(graphics);

      if (clusterSettings.enabled && clusters.length > 0) {
        clusters.forEach(cluster => {
            const pointsInCluster = tsneData.filter(p => tracks.find(t => t.id === p.trackId)?.clusterId === cluster.id);
            if (pointsInCluster.length > 0) {
                const avgX = pointsInCluster.reduce((sum, p) => sum + p.x, 0) / pointsInCluster.length;
                const avgY = pointsInCluster.reduce((sum, p) => sum + p.y, 0) / pointsInCluster.length;
                const screenX = PADDING + ((avgX - pcaBounds.xMin) / pcaBounds.xRange) * drawableWidth;
                const screenY = PADDING + (1 - ((avgY - pcaBounds.yMin) / pcaBounds.yRange)) * drawableHeight;

                const labelText = new PIXI.Text({
                    text:cluster.label,
                    style:{ fontFamily: 'Arial', fontSize: 10, fill: cluster.color, align: 'center', stroke: {color:0x000000, width:2, join:"round"}, miterLimit:10, wordWrap: true, wordWrapWidth: drawableWidth / clusters.length * 0.8 }
                });
                labelText.isAxisTextElement = true;
                labelText.anchor.set(0.5);
                labelText.position.set(screenX, screenY);
                chartArea.addChild(labelText);
            }
        });
      }

      tsneData.forEach(({ x: pcaX, y: pcaY, trackId }) => {
        const track = tracks.find(t => t.id === trackId);
        if (!track) return;
        const screenX = PADDING + ((pcaX - pcaBounds.xMin) / pcaBounds.xRange) * drawableWidth;
        const screenY = PADDING + (1 - ((pcaY - pcaBounds.yMin) / pcaBounds.yRange)) * drawableHeight;
        commonDotLogic(track, screenX, screenY);
      });

    } else if (!isSimilarityMode) {
      if (!axisMinMax.x || !axisMinMax.y || !axisMinMax.x.hasData || !axisMinMax.y.hasData) {
        const msgText = new PIXI.Text({text:"Select features or wait for axis range calculation.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
        msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
        msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea); return;
      }
      const { x: xRange, y: yRange } = axisMinMax;
      drawAxes(chartArea, xAxisFeature, yAxisFeature, xRange, yRange, {width: currentCanvasWidth, height: currentCanvasHeight});
      tracks.forEach((track) => {
        const rawXVal = track.parsedFeatures?.[xAxisFeature];
        const rawYVal = track.parsedFeatures?.[yAxisFeature];
        if (typeof rawXVal !== 'number' || isNaN(rawXVal) || typeof rawYVal !== 'number' || isNaN(rawYVal)) return;
        const screenX = PADDING + ((rawXVal - xRange.min) / xRange.range) * drawableWidth;
        const screenY = PADDING + (1 - ((rawYVal - yRange.min) / yRange.range)) * drawableHeight;
        commonDotLogic(track, screenX, screenY);
      });
    }
    updateAxesTextScale(chartArea);
  }, [
      isPixiAppReady, tracks, axisMinMax, xAxisFeature, yAxisFeature,
      isLoadingTracks, error, drawAxes, canvasSize, updateAxesTextScale,
      selectableFeatures,
      tsneData, isPcaCalculating, isClusteringCalculating, isSimilarityMode, clusterSettings, clusters, // Clustering states
      calculateVisualizationBounds, clusterColors // Added clusterColors
  ]);

  useEffect(() => {
    return () => { if (tooltipTimeoutRef.current) clearTimeout(tooltipTimeoutRef.current); };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (pixiAppRef.current && pixiCanvasContainerRef.current) {
        const { clientWidth, clientHeight } = pixiCanvasContainerRef.current;
        if (clientWidth > 0 && clientHeight > 0) {
            pixiAppRef.current.renderer.resize(clientWidth, clientHeight);
            setCanvasSize({ width: clientWidth, height: clientHeight });
        }
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, [isPixiAppReady]);


  const handleClusterSettingChange = (category) => {
    setClusterSettings(prev => ({ ...prev, [category]: !prev[category] }));
  };

  const handleNumClustersChange = (e) => {
    const val = parseInt(e.target.value, 10);
    if (val > 0 && val <= 50) { // Max 50 clusters, min 1
        setNumClustersControl(val);
    } else if (e.target.value === "") { // Allow empty to type
        setNumClustersControl(1); // Or some temp state
    }
  };


  return (
    <div className="visualization-outer-container">
      <div className="controls-panel" style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 1000, backgroundColor: "rgba(30,30,30,0.85)", padding: "15px", borderRadius: "8px", color: '#E0E0E0', maxHeight: '90vh', overflowY: 'auto' }}>
        <div className="mode-toggle" style={{ marginBottom: '15px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '1.1em' }}>
            <input type="checkbox" checked={isSimilarityMode}
              onChange={(e) => {
                setIsSimilarityMode(e.target.checked);
                if(chartAreaRef.current) { // Reset zoom/pan on mode switch
                    chartAreaRef.current.scale.set(MIN_ZOOM);
                    chartAreaRef.current.position.set(0,0);
                }
              }}
              style={{ width: '18px', height: '18px' }}/>
            Similarity Mode (Clustering)
          </label>
        </div>

        {isSimilarityMode && (
          <div className="clustering-controls" style={{ marginBottom: '15px', borderTop: '1px solid #555', paddingTop: '15px'}}>
            <div style={{fontWeight: 'bold', marginBottom: '10px', fontSize: '1.05em'}}>Cluster Configuration:</div>

            <div style={{marginBottom: '10px'}}>
                <label htmlFor="numClustersInput" style={{display: 'block', marginBottom: '3px'}}>Number of Clusters (k):</label>
                <input
                    type="number"
                    id="numClustersInput"
                    value={numClustersControl}
                    onChange={handleNumClustersChange}
                    min="1"
                    max="50"
                    style={{width: '60px', padding: '5px'}}
                />
            </div>

            <div style={{fontWeight: 'bold', marginBottom: '5px'}}>Feature Categories:</div>
            {[FEATURE_CATEGORIES.MOOD, FEATURE_CATEGORIES.SPECTRAL, FEATURE_CATEGORIES.TECHNICAL, FEATURE_CATEGORIES.STYLE, FEATURE_CATEGORIES.INSTRUMENT].map(category => (
              <div key={category} style={{marginBottom: '8px'}}>
                <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer'}}>
                  <input
                    type="checkbox"
                    checked={clusterSettings[category]}
                    onChange={() => handleClusterSettingChange(category)}
                    style={{ width: '16px', height: '16px' }}
                  />
                  <span style={{textTransform: 'capitalize'}}>{category.toLowerCase()}</span>
                </label>
              </div>
            ))}

            {/* Add Cluster Legend */}
            {clusters.length > 0 && (
              <div style={{marginTop: '15px', borderTop: '1px solid #555', paddingTop: '15px'}}>
                <div style={{fontWeight: 'bold', marginBottom: '10px'}}>Cluster Legend:</div>
                <div style={{maxHeight: '200px', overflowY: 'auto', paddingRight: '5px'}}>
                  {clusters.map((cluster) => (
                    <div key={cluster.id} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      marginBottom: '8px',
                      fontSize: '0.9em'
                    }}>
                      <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        backgroundColor: `#${cluster.color.toString(16).padStart(6, '0')}`,
                        flexShrink: 0
                      }} />
                      <span style={{
                        color: '#ccc',
                        wordBreak: 'break-word',
                        lineHeight: '1.2'
                      }}>
                        {cluster.label}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="style-threshold" style={{ marginBottom: '15px', borderTop: isSimilarityMode ? 'none' : '1px solid #555', paddingTop: isSimilarityMode ? '0' : '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>
            Dynamic Feature Min. Prob: {(styleThreshold * 100).toFixed(0)}%
          </label>
          <input type="range" min="0" max="100" value={styleThreshold * 100}
            onChange={(e) => setStyleThreshold(parseInt(e.target.value) / 100)}
            style={{ width: '100%' }} />
        </div>

        {!isSimilarityMode && (
          <div className="axis-selectors" style={{display: "flex", flexDirection:"column", gap: "10px"}}>
            <div className="axis-selector">
              <label htmlFor="xAxisSelect" style={{color: "#ccc", marginRight:"5px", display:'block', marginBottom:'3px'}}>X-Axis:</label>
              <select id="xAxisSelect" value={xAxisFeature} onChange={(e) => setXAxisFeature(e.target.value)} style={{padding:"5px", width: '100%'}}>
                {selectableFeatures.map((feature) => (<option key={`x-${feature.value}`} value={feature.value}>{feature.label}</option>))}
              </select>
            </div>
            <div className="axis-selector">
              <label htmlFor="yAxisSelect" style={{color: "#ccc", marginRight:"5px", display:'block', marginBottom:'3px'}}>Y-Axis:</label>
              <select id="yAxisSelect" value={yAxisFeature} onChange={(e) => setYAxisFeature(e.target.value)} style={{padding:"5px", width: '100%'}}>
                {selectableFeatures.map((feature) => (<option key={`y-${feature.value}`} value={feature.value}>{feature.label}</option>))}
              </select>
            </div>
          </div>
        )}
      </div>
      <div className="canvas-wrapper">
        <div ref={pixiCanvasContainerRef} className="pixi-canvas-target" />
        <div ref={wavesurferContainerRef} className="wavesurfer-container-hidden" style={{ display: 'none' }}></div>
        {(isLoadingTracks || (isSimilarityMode && (isClusteringCalculating || isPcaCalculating))) &&
            <div className="loading-overlay">
                {isLoadingTracks ? "Loading tracks..." :
                 isClusteringCalculating ? "Calculating clusters..." :
                 isPcaCalculating ? "Calculating similarity projection..." : ""}
            </div>
        }
        {error && <div className="error-overlay">{error}</div>}
      </div>
    </div>
  );
};

export default VisualizationCanvas;