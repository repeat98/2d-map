import React, { useEffect, useState, useRef, useCallback } from 'react';
import 'pixi.js/unsafe-eval'; // For PixiJS v7+, often needed for certain features/performance.
import * as PIXI from 'pixi.js';
import WaveSurfer from 'wavesurfer.js'; // Ensure Wavesurfer.js is installed
import './VisualizationCanvas.scss';
import defaultArtwork from "../../assets/default-artwork.png";

/*
* IMPORTANT NOTE FOR ELECTRON USERS (Content Security Policy - CSP):
* If you see errors related to "blob:" URLs, "worker-src", or "Content Security Policy"
* blocking images (covers) or potentially other assets, you likely need to adjust
* your Electron app's Content Security Policy in your main process file (e.g., main.js).
* PixiJS's asset loader and other libraries might use Web Workers, which Electron's
* default CSP can block.
*
* Example CSP modification in your Electron main process:
*
* const { session } = require('electron');
* // After app is ready:
* session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
* callback({
* responseHeaders: {
* ...details.responseHeaders,
* // THIS IS AN EXAMPLE - TIGHTEN IT FOR PRODUCTION AS MUCH AS POSSIBLE
* 'Content-Security-Policy': [
* "default-src 'self';" +
* "script-src 'self' 'unsafe-inline' 'unsafe-eval';" + // 'unsafe-eval' for Pixi/libs
* "style-src 'self' 'unsafe-inline';" +
* "img-src 'self' data: http://localhost:3000 blob:;" +  // Allow blob: for images
* "media-src 'self' http://localhost:3000 blob:;" + // Allow blob: for media
* "worker-src 'self' blob:;" + // CRITICAL for Pixi Assets & potentially others
* "connect-src 'self' http://localhost:3000;" // For API calls
* ]
* }
* });
* });
*
* Ensure this is correctly integrated into your Electron application's lifecycle.
*/


// Default numerical features
const defaultNumericFeatures = [
  { value: 'bpm', label: 'BPM', axisTitleStyle: { fill: 0xe74c3c, fontWeight: 'bold' }, isNumeric: true },
  { value: 'danceability', label: 'Danceability', axisTitleStyle: { fill: 0x3498db }, isNumeric: true },
  { value: 'happiness', label: 'Happiness', axisTitleStyle: { fill: 0xf1c40f }, isNumeric: true },
  { value: 'party', label: 'Party Vibe', isNumeric: true },
  { value: 'aggressive', label: 'Aggressiveness', axisTitleStyle: { fill: 0xc0392b }, isNumeric: true },
  { value: 'relaxed', label: 'Relaxed Vibe', axisTitleStyle: { fill: 0x2ecc71 }, isNumeric: true },
  { value: 'sad', label: 'Sadness', isNumeric: true },
  { value: 'engagement', label: 'Engagement', isNumeric: true },
  { value: 'approachability', label: 'Approachability', isNumeric: true },
  { value: 'rms', label: 'RMS (Loudness)', isNumeric: true },
  { value: 'spectral_centroid', label: 'Spectral Centroid (Brightness)', isNumeric: true },
  { value: 'spectral_bandwidth', label: 'Spectral Bandwidth', isNumeric: true },
  { value: 'spectral_rolloff', label: 'Spectral Rolloff', isNumeric: true },
  { value: 'spectral_contrast', label: 'Spectral Contrast (Peakiness)', isNumeric: true },
  { value: 'spectral_flatness', label: 'Spectral Flatness (Noisiness)', isNumeric: true },
];

// Style constants
const PADDING = 70; const AXIS_COLOR = 0xAAAAAA; const TEXT_COLOR = 0xE0E0E0;
const DOT_RADIUS = 5; const DOT_RADIUS_HOVER = 7; const DEFAULT_DOT_COLOR = 0x00A9FF;
const HAPPINESS_COLOR = 0xFFD700; const AGGRESSIVE_COLOR = 0xFF4136; const RELAXED_COLOR = 0x2ECC40;
const TOOLTIP_BG_COLOR = 0x333333; const TOOLTIP_TEXT_COLOR = 0xFFFFFF;
const TOOLTIP_PADDING = 10; const COVER_ART_SIZE = 80;
const MIN_ZOOM = 0.5; const MAX_ZOOM = 3; const ZOOM_SENSITIVITY = 0.001;
const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;

const VisualizationCanvas = () => {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  
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

  const [selectableFeatures, setSelectableFeatures] = useState([...defaultNumericFeatures]);
  const [xAxisFeature, setXAxisFeature] = useState(defaultNumericFeatures[0]?.value || '');
  const [yAxisFeature, setYAxisFeature] = useState(defaultNumericFeatures[1]?.value || '');
  
  const [axisMinMax, setAxisMinMax] = useState({ x: null, y: null });
  const [isPixiAppReady, setIsPixiAppReady] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0});
  const onWheelZoomRef = useRef(null);

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [currentHoverTrack, setCurrentHoverTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const playButtonRef = useRef(null);
  const playIconRef = useRef(null);

  useEffect(() => {
    const fetchTracksAndPrepareFeatures = async () => {
      console.log("🚀 Fetching tracks and preparing features...");
      setIsLoadingTracks(true); setError(null); setAxisMinMax({ x: null, y: null });
      try {
        const response = await fetch('http://localhost:3000/tracks');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        let data = await response.json();
        
        console.log(`📊 Received ${data.length} tracks from API.`);
        if (data.length > 0) {
            console.log("🔍 First track data sample (for features, audioUrl, artwork):", {
                id: data[0].id,
                features_type: typeof data[0].features,
                features_value_preview: data[0].features ? (typeof data[0].features === 'string' ? data[0].features.substring(0,100) + "..." : Object.keys(data[0].features).slice(0,5).join(', ') + "...") : "N/A",
                audioUrl: data[0].audioUrl, // CRITICAL FOR WAVESURFER
                artwork_thumbnail_path: data[0].artwork_thumbnail_path, // For covers
                coverArtUrl: data[0].coverArtUrl // Alternative for covers
            });
        }

        const genreFrequencies = {}; const allGenreKeys = new Set();
        const processedTracks = data.map((track) => {
          let parsedFeatures = {};
          if (track.features) {
            let featuresSource = track.features;
            if (typeof featuresSource === 'string') {
              try { featuresSource = JSON.parse(featuresSource); } 
              catch (e) { console.warn(`⚠️ Could not parse features JSON for track ${track.id || track.path}`); featuresSource = {}; }
            }
            if (typeof featuresSource === 'object' && featuresSource !== null) {
              parsedFeatures = featuresSource;
              Object.keys(parsedFeatures).forEach(key => {
                const value = parsedFeatures[key];
                if (typeof value === 'number' && value > 0.01) {
                  allGenreKeys.add(key);
                  genreFrequencies[key] = (genreFrequencies[key] || 0) + 1;
                }
              });
            }
          }
          // Ensure audioUrl is set from path if not present
          const audioUrl = track.audioUrl || (track.path ? `file://${track.path}` : '');
          return { ...track, parsedFeatures, audioUrl };
        });
        setTracks(processedTracks);
        console.log("✅ Processed tracks. Genre Frequencies:", genreFrequencies);

        const sortedGenreKeys = Array.from(allGenreKeys).sort((a, b) => genreFrequencies[b] - genreFrequencies[a]);
        console.log("🎶 Sorted Genre Keys by Frequency:", sortedGenreKeys);

        let currentSelectableFeatures = [...defaultNumericFeatures];
        let newXAxisFeature = defaultNumericFeatures[0]?.value || '';
        let newYAxisFeature = defaultNumericFeatures[1]?.value || '';

        if (sortedGenreKeys.length > 0) { /* ... (same logic for setting selectable features and default axes) ... */ 
          const genreFeatures = sortedGenreKeys.map(genreKey => ({
            value: genreKey,
            label: `${genreKey.charAt(0).toUpperCase() + genreKey.slice(1).replace(/_/g, ' ')} (${genreFrequencies[genreKey]})`,
            isNumeric: false,
            axisTitleStyle: { fill: 0x95a5a6 } 
          }));
          currentSelectableFeatures = [...defaultNumericFeatures, ...genreFeatures];
          if (genreFeatures.length >= 2) {
            newXAxisFeature = genreFeatures[0].value; newYAxisFeature = genreFeatures[1].value;
          } else if (genreFeatures.length === 1) {
            newXAxisFeature = genreFeatures[0].value; newYAxisFeature = defaultNumericFeatures[0]?.value || genreFeatures[0].value;
          }
        } else { console.log("🤷 No significant genre keys found."); }
        
        setSelectableFeatures(currentSelectableFeatures);
        setXAxisFeature(newXAxisFeature); setYAxisFeature(newYAxisFeature);
        console.log(`🎯 Initial X-axis: ${newXAxisFeature}, Y-axis: ${newYAxisFeature}`);
      } catch (e) { console.error("💥 Failed to fetch/process tracks:", e); setError(e.message); /* Reset to defaults */ } 
      finally { setIsLoadingTracks(false); console.log("🏁 Finished fetching and preparing features.");}
    };
    fetchTracksAndPrepareFeatures();
  }, []);

  useEffect(() => { // MinMax calculation
    if (isLoadingTracks || !tracks || tracks.length === 0 || !xAxisFeature || !yAxisFeature || selectableFeatures.length === 0) return;
    const getFeatureInfo = (featureKey) => selectableFeatures.find(f => f.value === featureKey);
    const calculateMinMax = (featureKey) => { /* ... (same logic) ... */ 
      const featureInfo = getFeatureInfo(featureKey);
      if (!featureInfo) { console.warn(`⚠️ Feature info not found for key: ${featureKey}`); return { min: 0, max: 1, range: 1, hasData: false, count: 0 };}
      if (featureInfo.isNumeric) {
        const values = tracks.map(t => t[featureKey]).filter(v => typeof v === 'number' && !isNaN(v));
        if (values.length === 0) return { min: 0, max: 1, range: 1, hasData: false, count: 0 };
        const min = Math.min(...values); const max = Math.max(...values);
        const range = (max - min === 0) ? 1 : (max - min);
        return { min, max, range, hasData: true, count: values.length };
      } else { return { min: 0, max: 1, range: 1, hasData: true, count: tracks.length }; }
    };
    setAxisMinMax({ x: calculateMinMax(xAxisFeature), y: calculateMinMax(yAxisFeature) });
  }, [tracks, xAxisFeature, yAxisFeature, isLoadingTracks, selectableFeatures]);

  const updateAxesTextScale = useCallback((chartArea) => { /* ... (same logic) ... */ 
    if (!chartArea || !chartArea.scale) return;
    const currentChartScale = chartArea.scale.x; const inverseScale = 1 / currentChartScale;
    for (const child of chartArea.children) { if (child.isAxisTextElement) { child.scale.set(inverseScale); } }
  }, []);

  useEffect(() => { // Pixi App Setup, Tooltip event listeners, Wavesurfer, Zoom
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
        
        // Create chart area
        chartAreaRef.current = new PIXI.Container();
        app.stage.addChild(chartAreaRef.current);

        // Create tooltip container
        tooltipContainerRef.current = new PIXI.Container();
        tooltipContainerRef.current.visible = false;
        tooltipContainerRef.current.eventMode = 'static';
        tooltipContainerRef.current.cursor = 'default';
        app.stage.addChild(tooltipContainerRef.current);

        // Create tooltip background
        const tooltipBg = new PIXI.Graphics()
          .roundRect(0, 0, 300, 200, 8)
          .fill({ color: 0x333333 });
        tooltipContainerRef.current.addChild(tooltipBg);

        // Create cover art sprite
        coverArtSpriteRef.current = new PIXI.Sprite(PIXI.Texture.EMPTY);
        coverArtSpriteRef.current.position.set(10, 10);
        coverArtSpriteRef.current.width = 80;
        coverArtSpriteRef.current.height = 80;
        tooltipContainerRef.current.addChild(coverArtSpriteRef.current);

        // Create title text
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

        // Create features text
        trackFeaturesTextRef.current = new PIXI.Text({
          text: '',
          style: {
            fontFamily: 'Arial',
            fontSize: 14,
            fill: 0xAAAAAA,
            wordWrap: true,
            wordWrapWidth: 200
          }
        });
        trackFeaturesTextRef.current.position.set(100, 40);
        tooltipContainerRef.current.addChild(trackFeaturesTextRef.current);

        // Create play button
        playButtonRef.current = new PIXI.Graphics()
          .circle(0, 0, 12)
          .fill({ color: 0x6A82FB });
        playButtonRef.current.position.set(280, 30);
        playButtonRef.current.eventMode = 'static';
        playButtonRef.current.cursor = 'pointer';
        tooltipContainerRef.current.addChild(playButtonRef.current);

        // Create play icon
        playIconRef.current = new PIXI.Graphics();
        playIconRef.current.fill({ color: 0xFFFFFF })
          .moveTo(-4, -6)
          .lineTo(-4, 6)
          .lineTo(6, 0);
        playButtonRef.current.addChild(playIconRef.current);

        // Create waveform container
        waveformContainerRef.current = new PIXI.Container();
        waveformContainerRef.current.position.set(10, 100);
        tooltipContainerRef.current.addChild(waveformContainerRef.current);

        // Add hover effect for play button
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

        // Add click handler for play button
        playButtonRef.current.on('pointerdown', async (event) => {
          event.stopPropagation(); // Prevent event from bubbling to stage
          if (currentHoverTrack && wavesurferRef.current) {
            if (wavesurferRef.current.isPlaying()) {
              wavesurferRef.current.pause();
              setIsPlaying(false);
            } else {
              if (activeAudioUrlRef.current !== currentHoverTrack.audioUrl) {
                await wavesurferRef.current.load(currentHoverTrack.audioUrl);
                activeAudioUrlRef.current = currentHoverTrack.audioUrl;
              }
              wavesurferRef.current.play();
              setIsPlaying(true);
            }
          }
        });

        // Add wheel event listener for zooming
        onWheelZoomRef.current = (event) => {
          event.preventDefault();
          const chart = chartAreaRef.current;
          const currentApp = pixiAppRef.current;
          if (!chart || !currentApp) return;

          const point = new PIXI.Point(event.offsetX, event.offsetY);
          const prevScale = chart.scale.x;
          
          // Calculate zoom factor with reduced sensitivity
          const zoomFactor = 1 - (event.deltaY * ZOOM_SENSITIVITY);
          let newScale = prevScale * zoomFactor;
          
          // Clamp zoom level
          newScale = Math.max(MIN_ZOOM, Math.min(newScale, MAX_ZOOM));
          if (prevScale === newScale) return;

          // Calculate zoom center point
          const localPoint = chart.toLocal(point, currentApp.stage);
          
          // Apply new scale
          chart.scale.set(newScale);
          
          // Calculate new position to keep zoom centered on mouse
          const newGlobalPointOfLocal = chart.toGlobal(localPoint, currentApp.stage);
          const dx = newGlobalPointOfLocal.x - point.x;
          const dy = newGlobalPointOfLocal.y - point.y;
          
          // Update chart position
          chart.x -= dx;
          chart.y -= dy;
          
          // Keep chart within bounds
          const bounds = chart.getBounds();
          const screenBounds = new PIXI.Rectangle(0, 0, currentApp.screen.width, currentApp.screen.height);
          
          if (bounds.left > screenBounds.left) {
            chart.x += screenBounds.left - bounds.left;
          }
          if (bounds.right < screenBounds.right) {
            chart.x += screenBounds.right - bounds.right;
          }
          if (bounds.top > screenBounds.top) {
            chart.y += screenBounds.top - bounds.top;
          }
          if (bounds.bottom < screenBounds.bottom) {
            chart.y += screenBounds.bottom - bounds.bottom;
          }
          
          updateAxesTextScale(chart);
        };

        app.canvas.addEventListener('wheel', onWheelZoomRef.current, { passive: false });

        // Add drag handlers
        app.stage.eventMode = 'static';
        app.stage.cursor = 'grab';

        app.stage.on('pointerdown', (event) => {
          if (event.target === app.stage) {
            setIsDragging(true);
            setDragStart({ x: event.global.x, y: event.global.y });
            app.stage.cursor = 'grabbing';
          }
        });

        app.stage.on('pointermove', (event) => {
          if (isDragging && chartAreaRef.current) {
            const dx = event.global.x - dragStart.x;
            const dy = event.global.y - dragStart.y;
            
            chartAreaRef.current.x += dx;
            chartAreaRef.current.y += dy;
            
            setDragStart({ x: event.global.x, y: event.global.y });
          }
        });

        app.stage.on('pointerup', () => {
          setIsDragging(false);
          app.stage.cursor = 'grab';
        });

        app.stage.on('pointerupoutside', () => {
          setIsDragging(false);
          app.stage.cursor = 'grab';
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
  }, [isDragging, dragStart, updateAxesTextScale]);

  // Update tooltip content when track changes
  useEffect(() => {
    if (!currentHoverTrack || !tooltipContainerRef.current) return;

    const updateTooltip = async () => {
      try {
        // Update title and features
        trackTitleTextRef.current.text = currentHoverTrack.title || 'Unknown Title';
        const xFeatureLabel = selectableFeatures.find(f => f.value === xAxisFeature)?.label || xAxisFeature;
        const yFeatureLabel = selectableFeatures.find(f => f.value === yAxisFeature)?.label || yAxisFeature;
        trackFeaturesTextRef.current.text = `${xFeatureLabel}: ${formatTickValue(currentHoverTrack[xAxisFeature])}\n${yFeatureLabel}: ${formatTickValue(currentHoverTrack[yAxisFeature])}`;

        // Load cover art
        const artworkPath = currentHoverTrack.artwork_thumbnail_path || currentHoverTrack.coverArtUrl || defaultArtwork;
        console.log("🎨 Loading cover art from:", artworkPath);
        const texture = await PIXI.Texture.from(artworkPath);
        coverArtSpriteRef.current.texture = texture;

        // Load waveform
        waveformContainerRef.current.removeChildren();
        
        // Add background
        const bg = new PIXI.Graphics()
          .rect(0, 0, 150, 40)
          .fill({ color: 0x1A1A1A });
        waveformContainerRef.current.addChild(bg);

        if (currentHoverTrack.audioUrl) {
          console.log("🎵 Loading waveform for track:", currentHoverTrack.id);
          const waveformResponse = await fetch(`http://localhost:3000/tracks/waveform/${currentHoverTrack.id}`);
          if (waveformResponse.ok) {
            const waveformData = await waveformResponse.json();
            
            // Draw waveform
            const waveformGraphics = new PIXI.Graphics();
            waveformGraphics.setStrokeStyle({ width: 1, color: 0x6A82FB });
            
            const peaks = waveformData.waveform;
            const width = 150;
            const height = 40;
            const centerY = height / 2;
            const step = width / peaks.length;
            
            waveformGraphics.moveTo(0, centerY);
            
            for (let i = 0; i < peaks.length; i++) {
              const x = i * step;
              const y = centerY + (peaks[i] * centerY);
              waveformGraphics.lineTo(x, y);
            }
            
            waveformContainerRef.current.addChild(waveformGraphics);
          }
        }
      } catch (error) {
        console.error("💥 Error updating tooltip:", error);
        // Set default artwork if cover art loading fails
        coverArtSpriteRef.current.texture = PIXI.Texture.from(defaultArtwork);
        
        // Show error message in waveform container
        const errorText = new PIXI.Text({
          text: 'Error loading waveform',
          style: {
            fontFamily: 'Arial',
            fontSize: 12,
            fill: 0xFF0000,
            align: 'center'
          }
        });
        errorText.anchor.set(0.5);
        errorText.position.set(75, 20);
        waveformContainerRef.current.addChild(errorText);
      }
    };

    updateTooltip();
  }, [currentHoverTrack, xAxisFeature, yAxisFeature, selectableFeatures, formatTickValue]);

  const formatTickValue = useCallback((value, isGenreAxis) => {
    if (value === null || value === undefined) return 'N/A';
    if (isGenreAxis) return parseFloat(value.toFixed(1)).toString();
    if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(1);
    if (Math.abs(value) > 10000) return value.toExponential(1);
    return parseFloat(value.toFixed(1)).toString();
  }, []);
  const drawAxes = useCallback((chartArea, currentXAxisFeatureKey, currentYAxisFeatureKey, xRange, yRange, currentCanvasSize) => { /* ... (same) ... */ 
    if (!chartArea || !xRange || !yRange || !currentCanvasSize.width || !currentCanvasSize.height || selectableFeatures.length === 0) return;
    const graphics = new PIXI.Graphics();
    const { width: canvasWidth, height: canvasHeight } = currentCanvasSize;
    const drawableWidth = canvasWidth - 2 * PADDING; const drawableHeight = canvasHeight - 2 * PADDING;
    if (drawableWidth <=0 || drawableHeight <=0) return;
    const defaultAxisTextStyle = { fontFamily: 'Arial, sans-serif', fontSize: 12, fill: TEXT_COLOR, align: 'center' };
    const defaultTitleTextStyle = { fontFamily: 'Arial, sans-serif', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center'};
    const xFeatureInfo = selectableFeatures.find(f => f.value === currentXAxisFeatureKey);
    const yFeatureInfo = selectableFeatures.find(f => f.value === currentYAxisFeatureKey);
    const xTitleStyle = {...defaultTitleTextStyle, ...(xFeatureInfo?.axisTitleStyle || {})};
    const yTitleStyle = {...defaultTitleTextStyle, ...(yFeatureInfo?.axisTitleStyle || {})};
    const isXAxisGenre = xFeatureInfo && !xFeatureInfo.isNumeric;
    const isYAxisGenre = yFeatureInfo && !yFeatureInfo.isNumeric;
    graphics.moveTo(PADDING, canvasHeight - PADDING).lineTo(canvasWidth - PADDING, canvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
    graphics.moveTo(PADDING, PADDING).lineTo(PADDING, canvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR});
    const xTitle = new PIXI.Text({text:(xFeatureInfo?.label || currentXAxisFeatureKey), style:xTitleStyle});
    xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5, 0); xTitle.position.set(PADDING + drawableWidth / 2, canvasHeight - PADDING + 25);
    chartArea.addChild(xTitle);
    const yTitle = new PIXI.Text({text:(yFeatureInfo?.label || currentYAxisFeatureKey), style:yTitleStyle});
    yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5, 1); yTitle.rotation = -Math.PI / 2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight / 2);
    chartArea.addChild(yTitle);
    const numTicks = isXAxisGenre || isYAxisGenre ? 4 : 5;
    for (let i = 0; i <= numTicks; i++) {
        const xVal = xRange.min + (xRange.range / numTicks) * i;
        const xTickPos = PADDING + (i / numTicks) * drawableWidth;
        graphics.moveTo(xTickPos, canvasHeight - PADDING).lineTo(xTickPos, canvasHeight - PADDING + 5).stroke({width:1, color:AXIS_COLOR});
        const xLabel = new PIXI.Text({text:formatTickValue(xVal, isXAxisGenre), style:defaultAxisTextStyle});
        xLabel.isAxisTextElement = true; xLabel.anchor.set(0.5, 0); xLabel.position.set(xTickPos, canvasHeight - PADDING + 8);
        chartArea.addChild(xLabel);
        const yVal = yRange.min + (yRange.range / numTicks) * i;
        const yTickPos = canvasHeight - PADDING - (i / numTicks) * drawableHeight;
        graphics.moveTo(PADDING, yTickPos).lineTo(PADDING - 5, yTickPos).stroke({width:1, color:AXIS_COLOR});
        const yLabel = new PIXI.Text({text:formatTickValue(yVal, isYAxisGenre), style:defaultAxisTextStyle});
        yLabel.isAxisTextElement = true; yLabel.anchor.set(1, 0.5); yLabel.position.set(PADDING - 8, yTickPos);
        chartArea.addChild(yLabel);
    }
    chartArea.addChild(graphics);
  }, [formatTickValue, selectableFeatures]);

  useEffect(() => { // Drawing logic
    if (!isPixiAppReady || !pixiAppRef.current || !chartAreaRef.current || isLoadingTracks || error || !tracks || !canvasSize.width || !canvasSize.height || !axisMinMax.x || !axisMinMax.y || selectableFeatures.length === 0 ) return;
    const app = pixiAppRef.current; const chartArea = chartAreaRef.current;
    chartArea.removeChildren();
    if (tracks.length === 0 || !axisMinMax.x.hasData || !axisMinMax.y.hasData) { /* ... (message handling) ... */ 
        const msg = tracks.length === 0 ? "No tracks to display." : `Axis data not fully ready. X: ${axisMinMax.x?.count ?? 'N/A'}, Y: ${axisMinMax.y?.count ?? 'N/A'}`;
        const msgText = new PIXI.Text({text:msg, style:new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
        msgText.anchor.set(0.5); msgText.position.set(app.screen.width / 2, app.screen.height / 2);
        msgText.isAxisTextElement = true; chartArea.addChild(msgText);
        updateAxesTextScale(chartArea); return;
    }
    const { x: xRange, y: yRange } = axisMinMax;
    const { width: currentCanvasWidth, height: currentCanvasHeight } = app.screen;
    const drawableWidth = currentCanvasWidth - 2 * PADDING; const drawableHeight = currentCanvasHeight - 2 * PADDING;
    if (drawableWidth <= 0 || drawableHeight <= 0) return;
    drawAxes(chartArea, xAxisFeature, yAxisFeature, xRange, yRange, {width: currentCanvasWidth, height: currentCanvasHeight});
    const xFeatureInfo = selectableFeatures.find(f => f.value === xAxisFeature);
    const yFeatureInfo = selectableFeatures.find(f => f.value === yAxisFeature);

    tracks.forEach((track) => {
      let rawXVal, rawYVal; /* ... (raw value calculation) ... */ 
      if (xFeatureInfo && !xFeatureInfo.isNumeric) { rawXVal = track.parsedFeatures?.[xAxisFeature] || 0; } 
      else { rawXVal = track[xAxisFeature]; if (typeof rawXVal !== 'number' || isNaN(rawXVal)) rawXVal = xRange.min; }
      if (yFeatureInfo && !yFeatureInfo.isNumeric) { rawYVal = track.parsedFeatures?.[yAxisFeature] || 0; } 
      else { rawYVal = track[yAxisFeature]; if (typeof rawYVal !== 'number' || isNaN(rawYVal)) rawYVal = yRange.min; }
      const nX = xRange.range === 0 ? 0.5 : (rawXVal - xRange.min) / xRange.range;
      const nY = yRange.range === 0 ? 0.5 : (rawYVal - yRange.min) / yRange.range;
      const x = PADDING + nX * drawableWidth; const y = PADDING + (1 - nY) * drawableHeight;
      let fillColor = DEFAULT_DOT_COLOR; /* ... (color logic) ... */ 
      const happinessVal = track.parsedFeatures?.happiness ?? track.happiness;
      const aggressiveVal = track.parsedFeatures?.aggressive ?? track.aggressive;
      const relaxedVal = track.parsedFeatures?.relaxed ?? track.relaxed;
      if (typeof happinessVal === 'number' && happinessVal > 0.75) fillColor = HAPPINESS_COLOR;
      else if (typeof aggressiveVal === 'number' && aggressiveVal > 0.65) fillColor = AGGRESSIVE_COLOR;
      else if (typeof relaxedVal === 'number' && relaxedVal > 0.75) fillColor = RELAXED_COLOR;
      const dataDot = new PIXI.Graphics()
        .circle(0, 0, DOT_RADIUS)
        .fill({ color: fillColor });
      dataDot.position.set(x, y);
      dataDot.eventMode = 'static';
      dataDot.cursor = 'pointer';

      dataDot.on('pointerover', async (event) => {
        event.stopPropagation(); // Prevent event from bubbling to stage
        console.log("🎯 Dot hovered:", event.global);
        dataDot.scale.set(DOT_RADIUS_HOVER / DOT_RADIUS);
        setCurrentHoverTrack(track);
        
        // Position the tooltip
        const mousePosition = event.global;
        const tooltipWidth = 300;
        const tooltipHeight = 200;
        
        let x = mousePosition.x + 15;
        let y = mousePosition.y - tooltipHeight / 2;
        
        if (x + tooltipWidth > app.screen.width) {
          x = mousePosition.x - tooltipWidth - 15;
        }
        if (y + tooltipHeight > app.screen.height) {
          y = app.screen.height - tooltipHeight - 10;
        }
        if (y < 0) {
          y = 10;
        }
        
        if (tooltipContainerRef.current) {
          tooltipContainerRef.current.position.set(x, y);
          tooltipContainerRef.current.visible = true;
        }
      });

      dataDot.on('pointerout', (event) => {
        event.stopPropagation(); // Prevent event from bubbling to stage
        console.log("🎯 Dot unhovered");
        dataDot.scale.set(1.0);
        setCurrentHoverTrack(null);
        if (tooltipContainerRef.current) {
          tooltipContainerRef.current.visible = false;
        }
        if (wavesurferRef.current && wavesurferRef.current.isPlaying()) {
          wavesurferRef.current.pause();
          setIsPlaying(false);
        }
      });
      chartArea.addChild(dataDot);
    });
    updateAxesTextScale(chartArea);
  }, [isPixiAppReady, tracks, axisMinMax, xAxisFeature, yAxisFeature, isLoadingTracks, error, drawAxes, formatTickValue, canvasSize, updateAxesTextScale, selectableFeatures, setCurrentHoverTrack, setIsPlaying]);

  useEffect(() => { // Window Resize
    const handleResize = () => { /* ... */ }; // same
    window.addEventListener('resize', handleResize);
    if (isPixiAppReady && pixiAppRef.current && pixiCanvasContainerRef.current) { /* ... */ } // same
    return () => window.removeEventListener('resize', handleResize);
  }, [isPixiAppReady]);

  return ( 
    <div className="visualization-outer-container">
      <div className="controls-panel" style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 1000 }}>
        <div className="axis-selectors">
          <div className="axis-selector">
            <label>X-Axis:</label>
            <select value={xAxisFeature} onChange={(e) => setXAxisFeature(e.target.value)}>
              {selectableFeatures.map((feature) => (
                <option key={feature.value} value={feature.value}>
                  {feature.label}
                </option>
              ))}
            </select>
          </div>
          <div className="axis-selector">
            <label>Y-Axis:</label>
            <select value={yAxisFeature} onChange={(e) => setYAxisFeature(e.target.value)}>
              {selectableFeatures.map((feature) => (
                <option key={feature.value} value={feature.value}>
                  {feature.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
      <div className="canvas-wrapper">
        <div ref={pixiCanvasContainerRef} className="pixi-canvas-target" />
        <div ref={wavesurferContainerRef} className="wavesurfer-container-hidden"></div>
        {isLoadingTracks && <div className="loading-overlay">Loading tracks...</div>}
        {error && <div className="error-overlay">{error}</div>}
      </div>
    </div>
  );
};

export default VisualizationCanvas;