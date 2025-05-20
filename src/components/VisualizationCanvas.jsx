import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import 'pixi.js/unsafe-eval';
import * as PIXI from 'pixi.js';
import WaveSurfer from 'wavesurfer.js';
import './VisualizationCanvas.scss'; // Ensure this path is correct
import defaultArtwork from "../../assets/default-artwork.png"; // Ensure this path is correct
import Waveform from './Waveform'; // Ensure this path is correct
import ReactDOM from 'react-dom/client';
import { PlaybackContext } from '../context/PlaybackContext'; // Ensure this path is correct
import { PCA } from 'ml-pca';
import { kmeans } from 'ml-kmeans';

/*
* IMPORTANT NOTE FOR ELECTRON USERS (Content Security Policy - CSP):
* (Original CSP note is preserved)
*/

// --- FEATURE CATEGORIES ---
const FEATURE_CATEGORIES = {
  MOOD: 'Mood',
  SPECTRAL: 'Spectral',
  TECHNICAL: 'Technical',
  STYLE: 'Style', // Genre and Subgenre will likely fall under this
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

const DYNAMIC_FEATURE_MIN_PROBABILITY = 0.3;
const PADDING = 70; const AXIS_COLOR = 0xAAAAAA; const TEXT_COLOR = 0xE0E0E0;
const DOT_RADIUS = 5; const DOT_RADIUS_HOVER = 7; const DEFAULT_DOT_COLOR = 0x00A9FF;
const HAPPINESS_COLOR = 0xFFD700; const AGGRESSIVE_COLOR = 0xFF4136; const RELAXED_COLOR = 0x2ECC40;
const TOOLTIP_BG_COLOR = 0x333333; const TOOLTIP_TEXT_COLOR = 0xFFFFFF;
const TOOLTIP_PADDING = 10; const COVER_ART_SIZE = 80;
const MIN_ZOOM = 1; const MAX_ZOOM = 5; const ZOOM_SENSITIVITY = 0.0005;
const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;

// --- HIERARCHICAL CLUSTERING CONSTANTS ---
// K values are kConfig: {min, max}. Adjusted based on previous feedback and reinforced by analysis.
const HIERARCHY_LEVELS = [
  { name: 'Spectral', kConfig: {min: 2, max: 3}, category: FEATURE_CATEGORIES.SPECTRAL, labelPrefix: 'Spec' },
  { name: 'Mood',     kConfig: {min: 2, max: 3}, category: FEATURE_CATEGORIES.MOOD,     labelPrefix: 'Mood' },
  { name: 'Instrument', kConfig: {min: 2, max: 2}, category: FEATURE_CATEGORIES.INSTRUMENT, labelPrefix: 'Inst' },
];
// --- END HIERARCHICAL CLUSTERING CONSTANTS ---

const generateClusterColors = (numColors) => {
  const colors = [];
  if (numColors <= 0) return [0xCCCCCC];
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

// --- HELPER FUNCTIONS FOR CLUSTERING ---
function euclideanDistance(point1, point2) {
    if (!point1 || !point2 || point1.length !== point2.length) {
        return Infinity;
    }
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
        sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
}

function calculateSilhouetteScore(points, assignments, numClusters) {
    if (numClusters <= 1 || points.length < numClusters || points.length === 0 || !assignments) {
        return -1;
    }
    let totalSilhouette = 0;
    let validPointsForScore = 0;

    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        const clusterIdx = assignments[i];
        let a_i = 0;
        let sameClusterPointsCount = 0;
        for (let j = 0; j < points.length; j++) {
            if (i === j) continue;
            if (assignments[j] === clusterIdx) {
                a_i += euclideanDistance(point, points[j]);
                sameClusterPointsCount++;
            }
        }
        if (sameClusterPointsCount > 0) a_i /= sameClusterPointsCount;
        else { validPointsForScore++; continue; }

        let b_i = Infinity;
        for (let k = 0; k < numClusters; k++) {
            if (k === clusterIdx) continue;
            let avgDistToOtherClusterK = 0;
            let otherClusterKPointsCount = 0;
            for (let j = 0; j < points.length; j++) {
                if (assignments[j] === k) {
                    avgDistToOtherClusterK += euclideanDistance(point, points[j]);
                    otherClusterKPointsCount++;
                }
            }
            if (otherClusterKPointsCount > 0) {
                avgDistToOtherClusterK /= otherClusterKPointsCount;
                b_i = Math.min(b_i, avgDistToOtherClusterK);
            }
        }
        if (b_i === Infinity) { validPointsForScore++; continue; }

        const s_i = (b_i - a_i) / Math.max(a_i, b_i);
        if (!isNaN(s_i)) totalSilhouette += s_i;
        validPointsForScore++;
    }
    if (validPointsForScore === 0) return 0;
    return totalSilhouette / validPointsForScore;
}

function calculateCentroid(pointsArray) {
    if (!pointsArray || pointsArray.length === 0) return [];
    const numDimensions = pointsArray[0].length;
    const centroid = new Array(numDimensions).fill(0);
    pointsArray.forEach(p => {
        for (let d = 0; d < numDimensions; d++) centroid[d] += p[d];
    });
    for (let d = 0; d < numDimensions; d++) centroid[d] /= pointsArray.length;
    return centroid;
}

const VisualizationCanvas = () => {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  const [isSimilarityMode, setIsSimilarityMode] = useState(false);
  const [styleThreshold, setStyleThreshold] = useState(DYNAMIC_FEATURE_MIN_PROBABILITY);

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

  const [currentHoverTrack, setCurrentHoverTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const tooltipTimeoutRef = useRef(null);
  const playButtonRef = useRef(null);
  const playIconRef = useRef(null);

  const [clusterSettings, setClusterSettings] = useState({
    [FEATURE_CATEGORIES.MOOD]: true,
    [FEATURE_CATEGORIES.SPECTRAL]: true,
    [FEATURE_CATEGORIES.TECHNICAL]: true,
    [FEATURE_CATEGORIES.STYLE]: true,
    [FEATURE_CATEGORIES.INSTRUMENT]: true,
  });

  const [clusters, setClusters] = useState([]);
  const [isClusteringCalculating, setIsClusteringCalculating] = useState(false);

  const dynamicClusterColors = useMemo(() => {
      return generateClusterColors(clusters.length > 0 ? clusters.length : 1);
  }, [clusters.length]);

  const [tsneData, setTsneData] = useState(null); // PCA data
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
        const dynamicFeatureSourceMap = new Map();

        const processedTracks = rawTracks.map(track => {
          const currentParsedFeatures = {};
          coreFeaturesConfig.forEach(coreFeatureConf => {
            const featureKey = coreFeatureConf.value;
            if (track[featureKey] !== undefined && track[featureKey] !== null) {
              const val = parseFloat(track[featureKey]);
              if (!isNaN(val)) {
                currentParsedFeatures[featureKey] = val;
                allDiscoveredFeatureKeys.add(featureKey);
                featureFrequencies[featureKey] = (featureFrequencies[featureKey] || 0) + 1;
              }
            }
          });

          const processDynamicFeaturesLocal = (featureObject, category) => {
              if (featureObject) {
                try {
                  const parsedObj = typeof featureObject === 'string' ? JSON.parse(featureObject) : featureObject;
                  Object.entries(parsedObj).forEach(([key, value]) => {
                    const val = parseFloat(value);
                    if (!isNaN(val) && val >= styleThreshold) {
                      currentParsedFeatures[key] = val;
                      allDiscoveredFeatureKeys.add(key);
                      dynamicFeatureSourceMap.set(key, category);
                      featureFrequencies[key] = (featureFrequencies[key] || 0) + 1;
                    }
                  });
                } catch (e) { console.error(`Error parsing ${category} features for track:`, track.id, e); }
              }
          };

          processDynamicFeaturesLocal(track.features, FEATURE_CATEGORIES.STYLE);
          processDynamicFeaturesLocal(track.instrument_features, FEATURE_CATEGORIES.INSTRUMENT);

          let mainGenre = 'Unknown Genre';
          let mainSubgenre = 'Unknown Subgenre';
          let highestProbGenreSubgenre = -1;

          for (const key in currentParsedFeatures) {
              if (dynamicFeatureSourceMap.get(key) === FEATURE_CATEGORIES.STYLE && key.includes('---')) {
                const value = currentParsedFeatures[key];
                if (value >= styleThreshold && value > highestProbGenreSubgenre) {
                    highestProbGenreSubgenre = value;
                    const parts = key.split('---');
                    if (parts.length >= 2) {
                      mainGenre = parts[0].replace(/_/g, ' ').trim();
                      mainSubgenre = parts[1].replace(/_/g, ' ').trim();
                    } else if (parts.length === 1 && key.length > 0) {
                      mainGenre = parts[0].replace(/_/g, ' ').trim();
                      mainSubgenre = 'N/A';
                    }
                }
              }
          }
          mainGenre = mainGenre.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');
          mainSubgenre = mainSubgenre.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');

          return {
            ...track,
            parsedFeatures: currentParsedFeatures,
            id: track.id.toString(),
            mainGenre,
            mainSubgenre
          };
        });

        const coreFeatureValues = new Set(coreFeaturesConfig.map(f => f.value));
        const dynamicFeatureConfigs = [];

        Array.from(allDiscoveredFeatureKeys).forEach(featureKey => {
          if (coreFeatureValues.has(featureKey)) return;
          let label = featureKey;
          if (featureKey.includes("---") && dynamicFeatureSourceMap.get(featureKey) === FEATURE_CATEGORIES.STYLE) {
            label = featureKey.split("---").map(part => part.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')).join(' / ');
          } else if (featureKey.includes("---")) {
            label = featureKey.substring(featureKey.indexOf("---") + 3);
            label = label.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          } else {
            label = label.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          }
          const frequency = featureFrequencies[featureKey] || 0;
          const determinedCategory = dynamicFeatureSourceMap.get(featureKey) || FEATURE_CATEGORIES.STYLE;

          if (frequency > 0) {
            dynamicFeatureConfigs.push({
              value: featureKey,
              label: `${label} (${frequency})`,
              isNumeric: true,
              axisTitleStyle: { fill: 0x95a5a6 },
              frequency: frequency,
              category: determinedCategory
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
        if (finalSelectableFeatures.length > 1 && (!finalSelectableFeatures.find(f => f.value === yAxisFeature) || yAxisFeature === xAxisFeature)) {
            const yCandidate = finalSelectableFeatures.find(f => f.value !== xAxisFeature) || finalSelectableFeatures[0];
            setYAxisFeature(yCandidate.value);
        } else if (finalSelectableFeatures.length === 1 && xAxisFeature !== finalSelectableFeatures[0].value) {
            setYAxisFeature(finalSelectableFeatures[0].value);
        }
        setTracks(processedTracks);
      } catch (fetchError) {
        console.error("Error fetching or processing tracks:", fetchError);
        setError(`Failed to load tracks: ${fetchError.message}`);
      } finally {
        setIsLoadingTracks(false);
      }
    };
    fetchTracksAndPrepareFeatures();
  }, [styleThreshold]);

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

 useEffect(() => {
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
        const tooltipBg = new PIXI.Graphics().roundRect(0, 0, 300, 200, 8).fill({ color: TOOLTIP_BG_COLOR });
        tooltipContainerRef.current.addChild(tooltipBg);
        coverArtSpriteRef.current = new PIXI.Sprite(PIXI.Texture.EMPTY);
        coverArtSpriteRef.current.position.set(TOOLTIP_PADDING, TOOLTIP_PADDING);
        coverArtSpriteRef.current.width = COVER_ART_SIZE;
        coverArtSpriteRef.current.height = COVER_ART_SIZE;
        tooltipContainerRef.current.addChild(coverArtSpriteRef.current);
        trackTitleTextRef.current = new PIXI.Text({ text: '', style: { fontFamily: 'Arial', fontSize: 16, fontWeight: 'bold', fill: TOOLTIP_TEXT_COLOR, wordWrap: true, wordWrapWidth: 200 }});
        trackTitleTextRef.current.position.set(COVER_ART_SIZE + 2 * TOOLTIP_PADDING, TOOLTIP_PADDING);
        tooltipContainerRef.current.addChild(trackTitleTextRef.current);
        trackFeaturesTextRef.current = new PIXI.Text({ text: '', style: { fontFamily: 'Arial', fontSize: 14, fill: 0xAAAAAA, wordWrap: true, wordWrapWidth: 200, lineHeight: 18 }});
        trackFeaturesTextRef.current.position.set(COVER_ART_SIZE + 2 * TOOLTIP_PADDING, TOOLTIP_PADDING + 30);
        tooltipContainerRef.current.addChild(trackFeaturesTextRef.current);

        playButtonRef.current = new PIXI.Graphics().circle(0, 0, PLAY_BUTTON_SIZE / 2).fill({ color: PLAY_BUTTON_COLOR });
        playButtonRef.current.position.set(300 - TOOLTIP_PADDING - PLAY_BUTTON_SIZE / 2, TOOLTIP_PADDING + PLAY_BUTTON_SIZE / 2);
        playButtonRef.current.eventMode = 'static';
        playButtonRef.current.cursor = 'pointer';
        tooltipContainerRef.current.addChild(playButtonRef.current);
        playIconRef.current = new PIXI.Graphics();
        playButtonRef.current.addChild(playIconRef.current);

        waveformContainerRef.current = new PIXI.Container();
        waveformContainerRef.current.position.set(TOOLTIP_PADDING, COVER_ART_SIZE + 2 * TOOLTIP_PADDING);
        tooltipContainerRef.current.addChild(waveformContainerRef.current);

        playButtonRef.current.on('pointerover', () => { playButtonRef.current.clear().circle(0, 0, PLAY_BUTTON_SIZE/2).fill({ color: PLAY_BUTTON_HOVER_COLOR }); playButtonRef.current.addChild(playIconRef.current);});
        playButtonRef.current.on('pointerout', () => { playButtonRef.current.clear().circle(0, 0, PLAY_BUTTON_SIZE/2).fill({ color: PLAY_BUTTON_COLOR }); playButtonRef.current.addChild(playIconRef.current); });
        playButtonRef.current.on('pointerdown', async (event) => {
          event.stopPropagation();
          const trackToPlay = currentTooltipTrackRef.current;
          if (trackToPlay && wavesurferRef.current) {
            if (wavesurferRef.current.isPlaying() && activeAudioUrlRef.current === trackToPlay.path) {
              wavesurferRef.current.pause();
            } else {
              if (activeAudioUrlRef.current !== trackToPlay.path) {
                activeAudioUrlRef.current = trackToPlay.path;
                await wavesurferRef.current.load(trackToPlay.path);
              } else {
                wavesurferRef.current.play().catch(e => console.error("Play error", e));
              }
            }
          }
        });

        if (wavesurferContainerRef.current && !wavesurferRef.current) {
            const wsInstance = WaveSurfer.create({
                container: wavesurferContainerRef.current,
                waveColor: '#6A82FB', progressColor: '#3B4D9A', height: 40, barWidth: 1,
                barGap: 1, cursorWidth: 0, interact: false,
                backend: 'MediaElement', normalize: true, autoCenter: true, partialRender: true, responsive: false,
            });
            wavesurferRef.current = wsInstance;
            console.log("🌊 Global Wavesurfer instance created.");
            wsInstance.on('error', (err) => console.error('🌊 Global WS Error:', err, "URL:", activeAudioUrlRef.current));
            wsInstance.on('ready', () => {
                const tooltipTrack = currentTooltipTrackRef.current;
                if (tooltipTrack && tooltipTrack.path === activeAudioUrlRef.current && tooltipContainerRef.current?.visible) {
                    wsInstance.play().catch(e => console.error("🌊 Error auto-playing on ready:", e));
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
                wavesurferRef.current.pause();
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
  }, []); // Removed updateAxesTextScale from dependency array as it's stable

  const formatTickValue = useCallback((value) => { // Removed unused isGenreAxis
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number' || isNaN(value)) return String(value);
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
    if (Math.abs(value) >= 10000) return value.toExponential(1);
    const numStr = value.toFixed(2);
    return parseFloat(numStr).toString();
  }, []);

  // Tooltip Update Logic (Largely unchanged, but ensure cluster label logic is correct)
  useEffect(() => {
    // ... (Tooltip update logic from previous turn, ensure track.clusterLabel and cluster.silhouetteScore are used correctly)
    // For brevity, this section is condensed. It should handle:
    // - Clearing old waveform instances
    // - Setting text for title and features (including cluster label if in similarity mode)
    // - Loading artwork
    // - Updating play/pause icon
    // - Positioning and showing the tooltip
    // - Rendering the Waveform React component inside the tooltip.
     if (!currentHoverTrack || !tooltipContainerRef.current || !pixiAppRef.current) {
      if (tooltipContainerRef.current) tooltipContainerRef.current.visible = false;
      const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
      existingReactContainers?.forEach(container => {
        const root = container._reactRoot;
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

        if (isSimilarityMode && currentHoverTrack.clusterLabel) {
            featuresText = `Cluster: ${currentHoverTrack.clusterLabel}\n`;
            if (currentHoverTrack.clusterSilhouetteScore !== undefined) { // Check if score exists
                 featuresText += `(Sil: ${currentHoverTrack.clusterSilhouetteScore > -Infinity ? currentHoverTrack.clusterSilhouetteScore.toFixed(2) : 'N/A'})\n`;
            }
        } else if (isSimilarityMode && currentHoverTrack.clusterId !== undefined && clusters.length > 0) {
            // Fallback for older structure or if track.clusterLabel isn't directly on track
            const cluster = clusters.find(c => c.id === currentHoverTrack.clusterId);
            featuresText = `Cluster: ${cluster ? cluster.label : 'N/A'}\n`;
             if (cluster && cluster.silhouetteScore !== undefined) {
                 featuresText += `(Sil: ${cluster.silhouetteScore > -Infinity ? cluster.silhouetteScore.toFixed(2) : 'N/A'})\n`;
            }
        }


        const xFeat = selectableFeatures.find(f => f.value === xAxisFeature);
        const yFeat = selectableFeatures.find(f => f.value === yAxisFeature);
        const xFeatureLabel = xFeat?.label.split('(')[0].trim() || xAxisFeature;
        const yFeatureLabel = yFeat?.label.split('(')[0].trim() || yAxisFeature;

        featuresText +=
          `${xFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[xAxisFeature])}\n` +
          `${yFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[yAxisFeature])}`;
        trackFeaturesTextRef.current.text = featuresText;

        const artworkPath = currentHoverTrack.artwork_thumbnail_path || defaultArtwork;
        coverArtSpriteRef.current.texture = await PIXI.Assets.load(artworkPath).catch(() => PIXI.Texture.from(defaultArtwork));

        if (playIconRef.current) {
            playIconRef.current.clear();
            if (isPlaying && activeAudioUrlRef.current === currentHoverTrack.path) {
              playIconRef.current.fill({ color: 0xFFFFFF })
                  .rect(-5, -6, 4, 12)
                  .rect(1, -6, 4, 12);
            } else {
              playIconRef.current.fill({ color: 0xFFFFFF }).moveTo(-4, -6).lineTo(-4, 6).lineTo(6, 0);
            }
        }

        tooltipContainerRef.current.visible = true;

        const waveformHostElement = document.createElement('div');
        waveformHostElement.className = 'waveform-react-container';
        waveformHostElement.style.width = '280px';
        waveformHostElement.style.height = '40px';
        waveformHostElement.style.position = 'absolute';
        waveformHostElement.style.pointerEvents = 'auto';

        const tooltipGlobalPos = tooltipContainerRef.current.getGlobalPosition(new PIXI.Point());
        const canvasRect = pixiCanvasContainerRef.current.getBoundingClientRect();

        waveformHostElement.style.left = `${tooltipGlobalPos.x - canvasRect.left + waveformContainerRef.current.x}px`;
        waveformHostElement.style.top = `${tooltipGlobalPos.y - canvasRect.top + waveformContainerRef.current.y}px`;
        pixiCanvasContainerRef.current.appendChild(waveformHostElement);

        const root = ReactDOM.createRoot(waveformHostElement);
        waveformHostElement._reactRoot = root;

        const playbackContextValue = { /* ... */ };

        root.render(
          <PlaybackContext.Provider value={playbackContextValue}>
            <Waveform
              key={`${currentHoverTrack.id}-tooltip-global`}
              trackId={currentHoverTrack.id.toString()}
              audioPath={currentHoverTrack.path}
              isInteractive={true}
              wavesurferInstanceRef={wavesurferRef}
              onPlay={() => { activeAudioUrlRef.current = currentHoverTrack.path; }}
              onReadyToPlay={(wsInstance) => { /* Autoplay handled by global */ }}
            />
          </PlaybackContext.Provider>
        );
        return () => {
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
    return () => {
      if (cleanupPromise && typeof cleanupPromise.then === 'function') {
        cleanupPromise.then(actualCleanup => {
          if (typeof actualCleanup === 'function') actualCleanup();
        }).catch(e => console.warn("Error in async tooltip cleanup:", e));
      } else if (typeof cleanupPromise === 'function') {
        cleanupPromise();
      }
    };
  }, [currentHoverTrack, xAxisFeature, yAxisFeature, selectableFeatures, formatTickValue, isSimilarityMode, clusters, isPlaying, activeAudioUrlRef]);


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


  const prepareFeatureData = useCallback((tracksToProcess, allSelectableFeaturesForScope, currentFeatureSettings) => {
    if (!tracksToProcess?.length || !allSelectableFeaturesForScope?.length) return null;

    const activeFeatureConfigs = allSelectableFeaturesForScope.filter(featureConf =>
      featureConf.isNumeric && currentFeatureSettings[featureConf.category]
    );

    if (!activeFeatureConfigs.length) {
      console.warn("DataPrep: No numeric features selected for the current settings/scope.");
      return null;
    }

    const featureVectors = [];
    const trackIdsForProcessing = [];
    const validTracksForProcessing = [];
    const vectorLength = activeFeatureConfigs.length;
    const tempVector = new Array(vectorLength);

    tracksToProcess.forEach(track => {
      let hasAnyValidData = false;
      for (let i = 0; i < vectorLength; i++) {
        const value = track.parsedFeatures?.[activeFeatureConfigs[i].value];
        tempVector[i] = typeof value === 'number' && !isNaN(value) ? value : 0; // Impute missing/NaN with 0
        if (tempVector[i] !== 0 && !isNaN(tempVector[i])) hasAnyValidData = true; // Ensure it's a valid number
      }

      if (hasAnyValidData || vectorLength === 0) {
        featureVectors.push([...tempVector]);
        trackIdsForProcessing.push(track.id);
        validTracksForProcessing.push(track);
      }
    });

    if (!featureVectors.length) return null;

    const numFeatures = vectorLength;
    const numSamples = featureVectors.length;

    const featureWeights = new Array(numFeatures).fill(1e-6);
    if (numSamples > 1) {
        for (let j = 0; j < numFeatures; j++) {
            let sum = 0;
            for (let i = 0; i < numSamples; i++) sum += featureVectors[i][j];
            const mean = sum / numSamples;
            let sumSquaredDiff = 0;
            for (let i = 0; i < numSamples; i++) sumSquaredDiff += Math.pow(featureVectors[i][j] - mean, 2);
            const variance = sumSquaredDiff / (numSamples - 1);
            if (variance > 1e-6) featureWeights[j] = variance;
        }
    }

    for (let i = 0; i < numSamples; i++) {
        for (let j = 0; j < numFeatures; j++) {
            featureVectors[i][j] *= Math.sqrt(featureWeights[j]);
        }
    }

    const normalizationParams = new Array(numFeatures);
    for (let j = 0; j < numFeatures; j++) {
      let minVal = Infinity; let maxVal = -Infinity;
      for (let i = 0; i < featureVectors.length; i++) {
        const val = featureVectors[i][j];
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
      const range = maxVal - minVal;
      normalizationParams[j] = { min: minVal, range: range === 0 ? 1 : range };
      if (range === 0) {
        for (let i = 0; i < featureVectors.length; i++) featureVectors[i][j] = 0.5;
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

  const runKMeansClustering = useCallback((data, kConfig) => {
    if (!data || !data.features || data.features.length === 0) return null;
    const nSamples = data.features.length;
    const actualMinK = Math.max(kConfig.min, 2);
    const actualMaxK = Math.min(kConfig.max, nSamples -1); // k must be < nSamples for silhouette

    if (nSamples < 2 || actualMinK > actualMaxK || nSamples < actualMinK) {
        const assignments = new Array(nSamples).fill(0);
        const trackIdToClusterId = {};
        data.trackIds.forEach((trackId) => trackIdToClusterId[trackId] = 0);
        return { assignments, centroids: nSamples > 0 ? [calculateCentroid(data.features)] : [], trackIdToClusterId, actualKUsed: 1, silhouetteScore: 0 };
    }

    let bestK = actualMinK, bestSilhouetteScore = -Infinity, bestKmeansResult = null;
    for (let k_to_test = actualMinK; k_to_test <= actualMaxK; k_to_test++) {
        try {
            const currentKmeansResult = kmeans(data.features, k_to_test, { seed: 42 });
            const assignments = currentKmeansResult.clusters;
            const numResultingClusters = currentKmeansResult.centroids.length;
            if (numResultingClusters < 2) {
                if (bestSilhouetteScore === -Infinity && !bestKmeansResult) {
                    bestKmeansResult = currentKmeansResult; bestK = numResultingClusters; bestSilhouetteScore = 0;
                }
                continue;
            }
            const score = calculateSilhouetteScore(data.features, assignments, numResultingClusters);
            if (score > bestSilhouetteScore) {
                bestSilhouetteScore = score; bestK = numResultingClusters; bestKmeansResult = currentKmeansResult;
            }
        } catch (error) { console.error(`Error during K-Means for k=${k_to_test}:`, error); }
    }

    if (!bestKmeansResult) {
        const fallbackK = Math.min(actualMinK, nSamples > 0 ? nSamples : 1); // Ensure fallbackK is valid
        bestKmeansResult = kmeans(data.features, fallbackK, { seed: 42 });
        bestK = bestKmeansResult.centroids.length;
        bestSilhouetteScore = calculateSilhouetteScore(data.features, bestKmeansResult.clusters, bestK);
        if (bestSilhouetteScore === -1 && bestK <=1 ) bestSilhouetteScore = 0;
    }

    const trackIdToClusterId = {};
    bestKmeansResult.clusters.forEach((clusterIndex, trackKmeansIdx) => {
      trackIdToClusterId[data.trackIds[trackKmeansIdx]] = clusterIndex;
    });
    return { assignments: bestKmeansResult.clusters, centroids: bestKmeansResult.centroids, trackIdToClusterId, actualKUsed: bestK, silhouetteScore: bestSilhouetteScore };
  }, []);

  const performHierarchicalClustering = useCallback((allTracks, allSelectableFeatures, hierarchyLevelsConfig) => {
    if (!allTracks || allTracks.length === 0) return { updatedTracks: [], finalClustersList: [] };
    console.log("🚀 Starting Hierarchical Grouping...");
    let currentTrackObjects = allTracks.map(t => ({ ...t, hierarchicalPath: [], clusterLabel: '', clusterSilhouetteScore: undefined }));

    const genreSubgenreGroups = currentTrackObjects.reduce((acc, track) => {
      const genreKey = track.mainGenre || 'Unknown Genre';
      const subgenreKey = track.mainSubgenre || 'Unknown Subgenre';
      track.hierarchicalPath.push(genreKey, subgenreKey);
      const groupKey = `${genreKey}__${subgenreKey}`;
      if (!acc[groupKey]) acc[groupKey] = [];
      acc[groupKey].push(track);
      return acc;
    }, {});

    let processedTracks = [];
    Object.values(genreSubgenreGroups).forEach(tracksInGsGroup => {
      let currentLevelTrackGroups = { "initial": tracksInGsGroup };
      hierarchyLevelsConfig.forEach(levelConf => {
        const nextLevelTrackGroups = {};
        Object.entries(currentLevelTrackGroups).forEach(([/* prevPathKey */, tracksInCurrentSubGroup]) => {
          if (tracksInCurrentSubGroup.length === 0) return;
          if (tracksInCurrentSubGroup.length < levelConf.kConfig.min || tracksInCurrentSubGroup.length < 2) {
            tracksInCurrentSubGroup.forEach(track => {
              track.hierarchicalPath.push(`${levelConf.labelPrefix}_0`);
              track.clusterSilhouetteScore = 0;
              const pathKey = track.hierarchicalPath.join(' > ');
              if (!nextLevelTrackGroups[pathKey]) nextLevelTrackGroups[pathKey] = [];
              nextLevelTrackGroups[pathKey].push(track);
            });
          } else {
            const featuresForThisLevel = allSelectableFeatures.filter(f => f.category === levelConf.category);
            const tempFeatureSettings = { [levelConf.category]: true };
            const dataForKMeans = prepareFeatureData(tracksInCurrentSubGroup, featuresForThisLevel, tempFeatureSettings);

            if (dataForKMeans && dataForKMeans.features.length >= levelConf.kConfig.min && dataForKMeans.features.length >=2) {
              const kmeansResult = runKMeansClustering(dataForKMeans, levelConf.kConfig);
              if (kmeansResult && kmeansResult.assignments) {
                dataForKMeans.originalTracks.forEach((processedTrack, idx) => {
                  const trackInSubGroup = tracksInCurrentSubGroup.find(t => t.id === processedTrack.id);
                  if(trackInSubGroup){
                    trackInSubGroup.hierarchicalPath.push(`${levelConf.labelPrefix}_${kmeansResult.assignments[idx]}`);
                    trackInSubGroup.clusterSilhouetteScore = kmeansResult.silhouetteScore;
                    const pathKey = trackInSubGroup.hierarchicalPath.join(' > ');
                    if (!nextLevelTrackGroups[pathKey]) nextLevelTrackGroups[pathKey] = [];
                    nextLevelTrackGroups[pathKey].push(trackInSubGroup);
                  }
                });
              } else {
                tracksInCurrentSubGroup.forEach(track => {
                  track.hierarchicalPath.push(`${levelConf.labelPrefix}_Err`);
                  track.clusterSilhouetteScore = undefined;
                  const pathKey = track.hierarchicalPath.join(' > ');
                  if (!nextLevelTrackGroups[pathKey]) nextLevelTrackGroups[pathKey] = [];
                  nextLevelTrackGroups[pathKey].push(track);
                });
              }
            } else {
              tracksInCurrentSubGroup.forEach(track => {
                track.hierarchicalPath.push(`${levelConf.labelPrefix}_FewData`);
                track.clusterSilhouetteScore = 0;
                const pathKey = track.hierarchicalPath.join(' > ');
                if (!nextLevelTrackGroups[pathKey]) nextLevelTrackGroups[pathKey] = [];
                nextLevelTrackGroups[pathKey].push(track);
              });
            }
          }
        });
        currentLevelTrackGroups = nextLevelTrackGroups;
      });
      Object.values(currentLevelTrackGroups).forEach(finalSubGroupTracks => processedTracks.push(...finalSubGroupTracks));
    });

    let finalClusterIdCounter = 0;
    const finalClustersMap = new Map();
    const finalClustersList = [];

    processedTracks.forEach(track => {
      track.clusterLabel = track.hierarchicalPath.join(' / ');
      if (!finalClustersMap.has(track.clusterLabel)) {
        const newClusterId = finalClusterIdCounter++;
        finalClustersMap.set(track.clusterLabel, newClusterId);
        finalClustersList.push({
          id: newClusterId,
          label: track.clusterLabel,
          trackIds: [],
          silhouetteScore: track.clusterSilhouetteScore // Store the silhouette from its formation
        });
      }
      track.clusterId = finalClustersMap.get(track.clusterLabel);
      const clusterForTrack = finalClustersList.find(c => c.id === track.clusterId);
      if(clusterForTrack) {
        clusterForTrack.trackIds.push(track.id);
        // Potentially average or update silhouette score for the cluster here if desired
      }
    });
    finalClustersList.sort((a,b) => a.label.localeCompare(b.label));
    console.log("✅ Hierarchical Grouping Completed. Found", finalClustersList.length, "leaf clusters.");
    return { updatedTracks: processedTracks, finalClustersList };
  }, [prepareFeatureData, runKMeansClustering]);

  const calculatePca = useCallback((data, forClusteringVis = false) => {
    if (!data || !data.features || data.features.length === 0) {
      if (forClusteringVis) setIsPcaCalculating(false);
      setTsneData(null); return;
    }
    const nActualFeatures = data.features[0]?.length || 0;
    if (nActualFeatures < 2) {
      console.warn("PCA: Not enough features (<2) for 2D projection. Features:", nActualFeatures);
      if (forClusteringVis) setIsPcaCalculating(false);
      setTsneData(null); return;
    }
    requestAnimationFrame(() => {
        try {
            console.log(`🔄 Starting PCA with ${data.features.length} tracks, ${nActualFeatures} features.`);
            const pca = new PCA(data.features);
            const result = pca.predict(data.features, { nComponents: 2 });
            let resultArray = result.data || result;

            if (!resultArray?.length || !resultArray[0]?.length || resultArray[0].length < 2) {
                console.warn("PCA result is not in expected format or has < 2 components. Actual components:", resultArray[0]?.length);
                  if (resultArray?.length && resultArray[0]?.length === 1) {
                    resultArray = resultArray.map(point => [point[0], Math.random() * 0.1 - 0.05]);
                  } else {
                    if (forClusteringVis) setIsPcaCalculating(false);
                    setTsneData(null); return;
                  }
            }
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            resultArray.forEach(point => {
                xMin = Math.min(xMin, point[0]); xMax = Math.max(xMax, point[0]);
                yMin = Math.min(yMin, point[1]); yMax = Math.max(yMax, point[1]);
            });
            const xRange = xMax - xMin || 1; const yRange = yMax - yMin || 1;
            const normalizedResult = resultArray.map((point, index) => ({
                x: (point[0] - xMin) / xRange,
                y: (point[1] - yMin) / yRange,
                trackId: data.trackIds[index]
            }));
            setTsneData(normalizedResult);
            console.log("✅ PCA completed.");
        } catch (error) {
            console.error("❌ Error calculating PCA:", error);
            setTsneData(null);
        } finally {
            if (forClusteringVis) setIsPcaCalculating(false);
        }
    });
  }, []);

  useEffect(() => {
    if (!isSimilarityMode || !tracks.length || !selectableFeatures.length) {
      setClusters([]); setTsneData(null);
      if (!isSimilarityMode && tracks.some(t => t.clusterId !== undefined)) {
          setTracks(prevTracks => prevTracks.map(t => ({...t, clusterId: undefined, clusterLabel: undefined, clusterSilhouetteScore: undefined })));
      }
      setIsClusteringCalculating(false); setIsPcaCalculating(false);
      return;
    }

    let animationFrameId;
    const timeoutId = setTimeout(() => {
      setIsClusteringCalculating(true); setIsPcaCalculating(true);
      animationFrameId = requestAnimationFrame(() => {
        const { updatedTracks, finalClustersList } = performHierarchicalClustering(
          tracks, selectableFeatures, HIERARCHY_LEVELS
        );
        setIsClusteringCalculating(false);
        if (updatedTracks.length > 0 && finalClustersList.length > 0) {
          setTracks(updatedTracks); setClusters(finalClustersList);
          const dataForPca = prepareFeatureData(updatedTracks, selectableFeatures, clusterSettings);
          if (dataForPca) calculatePca(dataForPca, true);
          else { setTsneData(null); setIsPcaCalculating(false); }
        } else {
          setClusters([]); setTsneData(null); setIsPcaCalculating(false);
        }
      });
    }, 500);
    return () => { clearTimeout(timeoutId); if (animationFrameId) cancelAnimationFrame(animationFrameId); };
  }, [
    isSimilarityMode, tracks.length, selectableFeatures.length,
    performHierarchicalClustering, calculatePca, prepareFeatureData,
    JSON.stringify(clusterSettings) // Stringify to ensure effect runs on deep changes
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

  // Main PIXI Rendering useEffect (handles drawing logic)
  useEffect(() => {
    // ... (Full Pixi rendering logic from previous turn)
    // This includes: clearing chart, drawing axes, drawing dots based on mode (scatter plot or PCA)
    // For brevity, this section is condensed. It uses commonDotLogic.
    // Ensure it correctly uses `dynamicClusterColors` for PCA plot dot colors.
     if (!isPixiAppReady || !pixiAppRef.current || !chartAreaRef.current || isLoadingTracks || error || !tracks || !canvasSize.width || !canvasSize.height) return;

    const app = pixiAppRef.current;
    const chartArea = chartAreaRef.current;
    chartArea.removeChildren();

    if (tracks.length === 0 && !isLoadingTracks) {
      const msgText = new PIXI.Text({text:"No tracks to display.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
      msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
      msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea); return;
    }

    let currentLoadingMessage = "";
    if (isSimilarityMode) {
        if (isClusteringCalculating) currentLoadingMessage = "Calculating hierarchical clusters...";
        else if (isPcaCalculating) currentLoadingMessage = "Calculating similarity projection (PCA)...";
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
        if (isSimilarityMode && track.clusterId !== undefined && clusters.length > 0) {
            const clusterIndex = clusters.findIndex(c => c.id === track.clusterId);
            if(clusterIndex !== -1) {
                fillColor = dynamicClusterColors[clusterIndex % dynamicClusterColors.length];
            }
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
        const hitArea = new PIXI.Graphics().circle(0,0, DOT_RADIUS * 1.8).fill({color:0xFFFFFF, alpha:0.001});
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
      const pcaBounds = calculateVisualizationBounds(tsneData, 0.05);
      if (!pcaBounds) return;

      const graphics = new PIXI.Graphics();
      graphics.moveTo(PADDING, currentCanvasHeight - PADDING).lineTo(currentCanvasWidth - PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
      graphics.moveTo(PADDING, PADDING).lineTo(PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});

      const activeCatsForPca = Object.entries(clusterSettings)
        .filter(([,isActive]) => isActive)
        .map(([key]) => FEATURE_CATEGORIES[Object.keys(FEATURE_CATEGORIES).find(k=>FEATURE_CATEGORIES[k] === key)] || key)
        .join('/') || "Selected";
      const xTitleText = `PC1 (Features: ${activeCatsForPca})`;
      const yTitleText = `PC2 (Features: ${activeCatsForPca})`;

      const titleTextStyle = { fontFamily: 'Arial', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center' };
      const xTitle = new PIXI.Text({text:xTitleText, style: titleTextStyle});
      xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5,0); xTitle.position.set(PADDING + drawableWidth/2, currentCanvasHeight - PADDING + 25); chartArea.addChild(xTitle);
      const yTitle = new PIXI.Text({text:yTitleText, style: titleTextStyle});
      yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5,1); yTitle.rotation = -Math.PI/2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight/2); chartArea.addChild(yTitle);
      chartArea.addChild(graphics);

      // Optional: Draw cluster labels on map (if desired, like C0, C1...)
      // The user's screenshot had a legend on the side, not labels directly on the map points.
      // If you want map labels, uncomment and adapt this section from your original code.

      tsneData.forEach(({ x: pcaX, y: pcaY, trackId }) => {
        const track = tracks.find(t => t.id === trackId);
        if (!track) return;
        const screenX = PADDING + ((pcaX - pcaBounds.xMin) / pcaBounds.xRange) * drawableWidth;
        const screenY = PADDING + (1 - ((pcaY - pcaBounds.yMin) / pcaBounds.yRange)) * drawableHeight;
        commonDotLogic(track, screenX, screenY);
      });

    } else if (isSimilarityMode && (!tsneData || tsneData.length === 0) && !isPcaCalculating && !isClusteringCalculating) {
        const msgText = new PIXI.Text({text:"PCA projection not available or no data.", style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
        msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
        msgText.isAxisTextElement = true; chartArea.addChild(msgText); updateAxesTextScale(chartArea);

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
      tsneData, isPcaCalculating, isClusteringCalculating, isSimilarityMode, clusterSettings, clusters,
      calculateVisualizationBounds, dynamicClusterColors
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

  return (
    <div className="visualization-outer-container">
      <div className="controls-panel" style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 1000, backgroundColor: "rgba(30,30,30,0.85)", padding: "15px", borderRadius: "8px", color: '#E0E0E0', maxHeight: '90vh', overflowY: 'auto', width: '280px' }}>
        <div className="mode-toggle" style={{ marginBottom: '15px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '1.1em' }}>
            <input type="checkbox" checked={isSimilarityMode}
              onChange={(e) => {
                setIsSimilarityMode(e.target.checked);
                if(chartAreaRef.current) {
                    chartAreaRef.current.scale.set(MIN_ZOOM);
                    chartAreaRef.current.position.set(0,0);
                }
              }}
              style={{ width: '18px', height: '18px' }}/>
            Similarity Mode (Hierarchical)
          </label>
        </div>

        {isSimilarityMode && (
          <div className="clustering-controls" style={{ marginBottom: '15px', borderTop: '1px solid #555', paddingTop: '15px'}}>
            <div style={{fontWeight: 'bold', marginBottom: '10px', fontSize: '1.05em'}}>Hierarchical Clustering:</div>
            <p style={{fontSize: '0.9em', color: '#bbb', marginBottom: '15px'}}>
              Tracks grouped by: Genre → Subgenre → Optimal K for (Spectral → Mood → Instruments) using Silhouette Score & Variance Weighted Features.
              PCA projection features below:
            </p>
            <div style={{fontWeight: 'bold', marginBottom: '5px'}}>Feature Categories for PCA:</div>
            {[FEATURE_CATEGORIES.MOOD, FEATURE_CATEGORIES.SPECTRAL, FEATURE_CATEGORIES.TECHNICAL, FEATURE_CATEGORIES.STYLE, FEATURE_CATEGORIES.INSTRUMENT].map(category => (
              <div key={category} style={{marginBottom: '8px'}}>
                <label style={{display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer'}}>
                  <input
                    type="checkbox"
                    checked={!!clusterSettings[category]}
                    onChange={() => handleClusterSettingChange(category)}
                    style={{ width: '16px', height: '16px' }}
                  />
                  <span style={{textTransform: 'capitalize'}}>{category.toLowerCase()}</span>
                </label>
              </div>
            ))}

            {clusters.length > 0 && (
              <div style={{marginTop: '15px', borderTop: '1px solid #555', paddingTop: '15px'}}>
                <div style={{fontWeight: 'bold', marginBottom: '10px'}}>Cluster Legend ({clusters.length}):</div>
                <div style={{maxHeight: '150px', overflowY: 'auto', paddingRight: '5px', fontSize: '0.85em'}}>
                  {clusters.map((cluster, index) => (
                    <div key={cluster.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <div style={{
                        width: '12px', height: '12px', borderRadius: '50%',
                        backgroundColor: `#${dynamicClusterColors[index % dynamicClusterColors.length].toString(16).padStart(6, '0')}`,
                        flexShrink: 0
                      }} />
                      <span style={{ color: '#ccc', wordBreak: 'break-word', lineHeight: '1.2' }}>
                        {cluster.label} ({cluster.trackIds.length})
                        {cluster.silhouetteScore !== undefined &&
                          ` (Sil: ${cluster.silhouetteScore > -Infinity ? cluster.silhouetteScore.toFixed(2) : 'N/A'})`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="style-threshold" style={{ marginBottom: '15px', borderTop: '1px solid #555', paddingTop: '15px' }}>
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
              <select id="xAxisSelect" value={xAxisFeature} onChange={(e) => setXAxisFeature(e.target.value)} style={{padding:"5px", width: '100%', backgroundColor: "#333", color:"#fff", border:"1px solid #555"}}>
                {selectableFeatures.map((feature) => (<option key={`x-${feature.value}`} value={feature.value}>{feature.label}</option>))}
              </select>
            </div>
            <div className="axis-selector">
              <label htmlFor="yAxisSelect" style={{color: "#ccc", marginRight:"5px", display:'block', marginBottom:'3px'}}>Y-Axis:</label>
              <select id="yAxisSelect" value={yAxisFeature} onChange={(e) => setYAxisFeature(e.target.value)} style={{padding:"5px", width: '100%', backgroundColor: "#333", color:"#fff", border:"1px solid #555"}}>
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
              isClusteringCalculating ? "Calculating hierarchical clusters..." :
              isPcaCalculating ? "Calculating similarity projection (PCA)..." : ""}
          </div>
        }
        {error && <div className="error-overlay">{error}</div>}
      </div>
    </div>
  );
};

export default VisualizationCanvas;