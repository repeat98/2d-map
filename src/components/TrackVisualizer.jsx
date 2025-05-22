import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import './TrackVisualizer.scss';
// import WaveSurfer from 'wavesurfer.js'; // WaveSurfer is used by the Waveform component, not directly here usually
import defaultArtwork from "../../assets/default-artwork.png";
import Waveform from './Waveform'; // Assuming Waveform.jsx is in the same directory or correctly pathed
import { PlaybackContext } from '../context/PlaybackContext'; // Assuming PlaybackContext.js is in ../context/

// --- Dark Mode Theme Variables (mirroring SCSS for JS logic if needed) ---
const DARK_MODE_TEXT_PRIMARY = '#e0e0e0';
const DARK_MODE_TEXT_SECONDARY = '#b0b0b0';
const DARK_MODE_SURFACE_ALT = '#3a3a3a';
const DARK_MODE_BORDER = '#4a4a4a';
// const DARK_MODE_ACCENT = '#00bcd4'; // Not directly used in JS logic shown, but good for reference


// --- Constants ---
const PADDING = 50;
const PCA_N_COMPONENTS = 2;
const HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE = 5;
const HDBSCAN_DEFAULT_MIN_SAMPLES = 3;
const TOOLTIP_OFFSET = 15;
const NOISE_CLUSTER_ID = -1;
const NOISE_CLUSTER_COLOR = '#555555';
const DEFAULT_CLUSTER_COLORS = [
  '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
  '#008080', '#e6beff', '#9A6324', '#fffac8', '#800000',
  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
  '#54A0FF', '#F4D03F', '#1ABC9C', '#E74C3C', '#8E44AD'
];
// const PLACEHOLDER_IMAGE = '/placeholder.png'; // Not used, defaultArtwork is used

const CATEGORY_WEIGHTS = {
  'genre': 4.0,
  'style': 4.0,
  'spectral': 2.0,
  'mood': 1.0,
  'instrument': 0.5,
  'default': 0.2,
};

const SPECTRAL_KEYWORDS = [
  'noisy', 'tonal', 'dark', 'bright', 'percussive', 'smooth', 'lufs'
];

const MOOD_KEYWORDS = [
  'happiness', 'party', 'aggressive', 'danceability', 'relaxed', 'sad', 'engagement', 'approachability'
];

const CATEGORY_BASE_COLORS = {
    'genre': '#F44336',
    'style': '#4CAF50',
    'spectral': '#2196F3',
    'mood': '#FF9800',
    'instrument': '#9C27B0',
};

const LUMINANCE_INCREMENT = 0.3;
const MAX_LUM_OFFSET = 0.5;


// --- Helper Functions ---

function calculateDistance(vec1, vec2) {
  if (!vec1 || !vec2) return Infinity;
  if (vec1.length !== vec2.length) return Infinity;
  let sumOfSquares = 0;
  for (let i = 0; i < vec1.length; i++) {
    const val1 = vec1[i] || 0;
    const val2 = vec2[i] || 0;
    const diff = val1 - val2;
    sumOfSquares += diff * diff;
  }
  return Math.sqrt(sumOfSquares);
}

function getAllFeatureKeysAndCategories(tracks) {
  const featuresWithCategories = new Map();
  const determineFinalCategory = (keyName, sourceCategory) => {
    const lowerKeyName = keyName.toLowerCase();
    if (SPECTRAL_KEYWORDS.includes(lowerKeyName)) return 'spectral';
    if (MOOD_KEYWORDS.includes(lowerKeyName)) return 'mood';
    return sourceCategory;
  };

  const processFeatureSource = (featureObj, sourceCategory, trackId) => {
    if (!featureObj) return;
    try {
      const parsed = typeof featureObj === 'string' ? JSON.parse(featureObj) : featureObj;
      if (typeof parsed === 'object' && parsed !== null) {
        Object.keys(parsed).forEach(key => {
          const existingCategory = featuresWithCategories.get(key);
          const finalCategory = determineFinalCategory(key, sourceCategory);
          if (!existingCategory || (existingCategory !== 'spectral' && finalCategory === 'spectral') || (existingCategory !== 'spectral' && existingCategory !== 'mood' && finalCategory === 'mood')) {
             featuresWithCategories.set(key, finalCategory);
          } else if (!existingCategory) {
             featuresWithCategories.set(key, finalCategory);
          }
        });
      }
    } catch (e) {
      // console.warn(`Failed to parse features for track ${trackId} (source: ${sourceCategory}) while getting keys:`, e, featureObj);
    }
  };

  tracks.forEach(track => {
    if (!track || !track.id) return;
    processFeatureSource(track.features, 'genre', track.id);
    processFeatureSource(track.style_features, 'style', track.id);
    processFeatureSource(track.instrument_features, 'instrument', track.id);
  });

  SPECTRAL_KEYWORDS.forEach(key => featuresWithCategories.set(key, 'spectral'));
  MOOD_KEYWORDS.forEach(key => featuresWithCategories.set(key, 'mood'));

  return Array.from(featuresWithCategories.entries())
    .map(([name, category]) => ({ name, category }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function mergeFeatureVectors(track, allFeatureNames) {
  const mergedFeatures = {};
  allFeatureNames.forEach(key => {
    mergedFeatures[key] = 0;
  });

  const parseAndMerge = (featureObj) => {
    if (!featureObj) return;
    try {
      const parsed = typeof featureObj === 'string' ? JSON.parse(featureObj) : featureObj;
      if (typeof parsed === 'object' && parsed !== null) {
        Object.entries(parsed).forEach(([key, value]) => {
          if (allFeatureNames.includes(key)) {
            const num = parseFloat(value);
            if (!isNaN(num)) mergedFeatures[key] = num;
          }
        });
      }
    } catch (e) {
      // console.warn(`Failed to parse features for track ${track?.id} during merge:`, e, featureObj);
    }
  };

  parseAndMerge(track.features);
  parseAndMerge(track.style_features);
  parseAndMerge(track.instrument_features);

  const directFeatures = {
    ...Object.fromEntries(SPECTRAL_KEYWORDS.map(k => [k, track[k]])),
    ...Object.fromEntries(MOOD_KEYWORDS.map(k => [k, track[k]]))
  };

  Object.entries(directFeatures).forEach(([key, value]) => {
    if (allFeatureNames.includes(key)) {
      const num = parseFloat(value);
      if (!isNaN(num)) mergedFeatures[key] = num;
    }
  });
  return allFeatureNames.map(key => mergedFeatures[key]);
}

function normalizeFeatures(featureVectors, featureCategories) {
  if (!featureVectors || featureVectors.length === 0) return [];
  const numSamples = featureVectors.length;
  const numFeatures = featureVectors[0]?.length || 0;
  if (numFeatures === 0) return featureVectors.map(() => []);

  let categories = featureCategories;
  if (categories.length !== numFeatures) {
    categories = Array(numFeatures).fill('default');
  }

  const means = new Array(numFeatures).fill(0);
  const stdDevs = new Array(numFeatures).fill(0);

  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) means[j] += vector[j] || 0;
  }
  for (let j = 0; j < numFeatures; j++) means[j] /= numSamples;

  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) stdDevs[j] += Math.pow((vector[j] || 0) - means[j], 2);
  }
  for (let j = 0; j < numFeatures; j++) stdDevs[j] = Math.sqrt(stdDevs[j] / numSamples);

  return featureVectors.map(vector =>
    vector.map((value, j) => {
      const std = stdDevs[j];
      const mean = means[j];
      const normalizedValue = (std < 1e-10) ? 0 : ((value || 0) - mean) / std;
      const category = (j < categories.length && categories[j]) ? categories[j] : 'default';
      const weight = CATEGORY_WEIGHTS[category] || CATEGORY_WEIGHTS['default'];
      return normalizedValue * weight;
    })
  );
}

function pca(processedData, nComponents = PCA_N_COMPONENTS) {
  if (!processedData || processedData.length === 0) return [];
  const numSamples = processedData.length;
  let numFeatures = processedData[0]?.length || 0;

  if (numFeatures === 0) return processedData.map(() => Array(nComponents).fill(0.5));
  nComponents = Math.min(nComponents, numFeatures > 0 ? numFeatures : nComponents);
  if (nComponents <= 0) return processedData.map(() => []);
  if (numSamples <= 1) return processedData.map(() => Array(nComponents).fill(0.5));

  const means = processedData[0].map((_, colIndex) => processedData.reduce((sum, row) => sum + row[colIndex], 0) / numSamples);
  const centeredData = processedData.map(row => row.map((val, colIndex) => val - means[colIndex]));

  const covarianceMatrix = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
  for (let i = 0; i < numFeatures; i++) {
    for (let j = 0; j < numFeatures; j++) {
      let sum = 0;
      for (let k = 0; k < numSamples; k++) sum += centeredData[k][i] * centeredData[k][j];
      covarianceMatrix[i][j] = sum / (numSamples - 1);
    }
  }

  const powerIteration = (matrix, numIterations = 100) => {
    const n = matrix.length;
    if (n === 0 || !matrix[0] || matrix[0].length === 0) return [];
    let vector = Array(n).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(vector.reduce((s, v) => s + v * v, 0));
    if (norm < 1e-10) vector = Array(n).fill(0); else vector = vector.map(v => v / norm);
    if (vector.every(v => v === 0) && n > 0) vector[0] = 1;

    for (let iter = 0; iter < numIterations; iter++) {
      let newVector = Array(n).fill(0);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) newVector[r] += (matrix[r]?.[c] || 0) * vector[c];
      }
      norm = Math.sqrt(newVector.reduce((s, val) => s + val * val, 0));
      if (norm < 1e-10) return Array(n).fill(0);
      vector = newVector.map(val => val / norm);
    }
    return vector;
  };

  const principalComponents = [];
  let tempCovarianceMatrix = covarianceMatrix.map(row => [...row]);

  for (let k = 0; k < nComponents; k++) {
    if (tempCovarianceMatrix.length === 0 || tempCovarianceMatrix.every(row => row.every(val => isNaN(val) || val === 0))) {
        const fallbackPc = Array(numFeatures).fill(0); if (k < numFeatures) fallbackPc[k] = 1;
        principalComponents.push(fallbackPc); continue;
    }
    const pc = powerIteration(tempCovarianceMatrix);
    if (pc.length === 0 || pc.every(v => v === 0)) {
      const fallbackPc = Array(numFeatures).fill(0); if (k < numFeatures) fallbackPc[k] = 1;
      principalComponents.push(fallbackPc); continue;
    }
    principalComponents.push(pc);

    if (k < nComponents - 1 && pc.length > 0) {
      let lambda = 0; const C_v = Array(numFeatures).fill(0);
      for (let i = 0; i < numFeatures; i++) {
        for (let j = 0; j < numFeatures; j++) C_v[i] += (tempCovarianceMatrix[i]?.[j] || 0) * pc[j];
        lambda += pc[i] * C_v[i];
      }
      const newTempCovMatrix = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
      for (let i = 0; i < numFeatures; i++) {
        for (let j = 0; j < numFeatures; j++) newTempCovMatrix[i][j] = (tempCovarianceMatrix[i]?.[j] || 0) - lambda * pc[i] * pc[j];
      }
      tempCovarianceMatrix = newTempCovMatrix;
    }
  }
  while (principalComponents.length < nComponents && numFeatures > 0) {
    const fallbackPc = Array(numFeatures).fill(0);
    if (principalComponents.length < numFeatures) fallbackPc[principalComponents.length] = 1;
    principalComponents.push(fallbackPc);
  }
  if (numFeatures === 0 && principalComponents.length < nComponents) {
    while(principalComponents.length < nComponents) principalComponents.push([]);
  }

  const projected = centeredData.map(row =>
    principalComponents.map(pcVector => {
      if (pcVector.length !== row.length) return 0;
      return row.reduce((sum, val, i) => sum + val * (pcVector[i] || 0), 0);
    })
  );

  if (projected.length === 0 || nComponents === 0) return projected.map(() => Array(nComponents).fill(0.5));
  const actualNumOutputComponents = projected[0]?.length || 0;
  if (actualNumOutputComponents === 0) return projected.map(() => Array(nComponents).fill(0.5));

  const minMax = Array(actualNumOutputComponents).fill(null).map((_, i) => ({
    min: Math.min(...projected.map(p => p[i])),
    max: Math.max(...projected.map(p => p[i])),
  }));

  return projected.map(p => p.map((val, i) => {
    if (i >= minMax.length || minMax[i] === null) return 0.5;
    const range = minMax[i].max - minMax[i].min;
    return (range > 1e-10) ? (val - minMax[i].min) / range : 0.5;
  }));
}

function hdbscan(data, minClusterSize = HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE, minSamples = HDBSCAN_DEFAULT_MIN_SAMPLES) {
  if (!data || data.length === 0) return [];
  const n = data.length;
  if (n === 0) return [];

  minClusterSize = Math.max(1, Math.min(minClusterSize, n));
  minSamples = Math.max(1, Math.min(minSamples, n > 1 ? n - 1 : 1));

  if (n < minClusterSize && n > 0) return Array(n).fill(NOISE_CLUSTER_ID);

  function computeMutualReachabilityDistance() {
    const distances = Array(n).fill(null).map(() => Array(n).fill(0));
    const coreDistances = Array(n).fill(Infinity);
    if (n === 0) return { distances, coreDistances };

    for (let i = 0; i < n; i++) {
      if (n <= 1 || minSamples >= n) { coreDistances[i] = Infinity; continue; }
      const pointDistances = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        pointDistances.push(calculateDistance(data[i], data[j]));
      }
      pointDistances.sort((a, b) => a - b);
      coreDistances[i] = pointDistances[minSamples - 1] ?? Infinity;
    }

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const directDist = calculateDistance(data[i], data[j]);
        const mrDist = Math.max(coreDistances[i], coreDistances[j], directDist);
        distances[i][j] = mrDist; distances[j][i] = mrDist;
      }
    }
    return distances;
  }

   function buildMST(mutualReachabilityDistances) {
    if (n === 0) return [];
    const mstEdges = []; const visited = new Array(n).fill(false);
    const minEdgeWeight = new Array(n).fill(Infinity); const edgeToVertex = new Array(n).fill(-1);
    if (n > 0) minEdgeWeight[0] = 0;

    for (let count = 0; count < n; count++) {
        let u = -1, currentMin = Infinity;
        for (let v = 0; v < n; v++) {
            if (!visited[v] && minEdgeWeight[v] < currentMin) { currentMin = minEdgeWeight[v]; u = v; }
        }
        if (u === -1) break;
        visited[u] = true;
        if (edgeToVertex[u] !== -1) mstEdges.push([u, edgeToVertex[u], minEdgeWeight[u]]);
        for (let v = 0; v < n; v++) {
            if (!visited[v]) {
                const weightUV = mutualReachabilityDistances[u]?.[v] ?? Infinity;
                if (weightUV < minEdgeWeight[v]) { minEdgeWeight[v] = weightUV; edgeToVertex[v] = u;}
            }
        }
    }
    return mstEdges;
  }

  function extractClustersSimplified(mst) {
    const labels = Array(n).fill(NOISE_CLUSTER_ID);
    if (n === 0 || (mst.length === 0 && n > 0 && minClusterSize > 1)) return labels;
    if (n > 0 && minClusterSize === 1) return Array(n).fill(0).map((_,i)=>i);

    let currentClusterId = 0;
    const parent = Array(n).fill(0).map((_, i) => i);
    const componentSize = Array(n).fill(1);

    function findSet(i) { if (parent[i] === i) return i; return parent[i] = findSet(parent[i]); }
    function uniteSets(i, j) {
      let rootI = findSet(i), rootJ = findSet(j);
      if (rootI !== rootJ) {
        if (componentSize[rootI] < componentSize[rootJ]) [rootI, rootJ] = [rootJ, rootI];
        parent[rootJ] = rootI; componentSize[rootI] += componentSize[rootJ]; return true;
      } return false;
    }
    const sortedMSTEdges = mst.sort((a, b) => a[2] - b[2]);
    for (const edge of sortedMSTEdges) uniteSets(edge[0], edge[1]);

    const rootToClusterId = new Map();
    for(let i = 0; i < n; i++){
        const root = findSet(i);
        if(componentSize[root] >= minClusterSize){
            if(!rootToClusterId.has(root)) rootToClusterId.set(root, currentClusterId++);
            labels[i] = rootToClusterId.get(root);
        } else { labels[i] = NOISE_CLUSTER_ID; }
    }
    return labels;
  }

  const mutualReachabilityDistances = computeMutualReachabilityDistance();
  const mst = buildMST(mutualReachabilityDistances);
  return extractClustersSimplified(mst);
}


// --- React Component ---
const TrackVisualizer = () => {
  const [tracks, setTracks] = useState([]);
  const [plotData, setPlotData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tooltip, setTooltip] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('genre');
  const [featureMetadata, setFeatureMetadata] = useState({ names: [], categories: [] });
  const [styleColors, setStyleColors] = useState(new Map());
  const [topNThreshold, setTopNThreshold] = useState(10);

  const [svgDimensions, setSvgDimensions] = useState({ width: window.innerWidth, height: window.innerHeight - 150 });
  const viewModeRef = React.useRef(null);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const VIEW_BOX_VALUE = `0 0 ${svgDimensions.width} ${svgDimensions.height}`;

  // Ref to hold the currently active/playing WaveSurfer instance from a tooltip
  const wavesurferRef = useRef(null);
  // const activeAudioUrlRef = useRef(null); // This ref seems unused for tooltip waveform playback in the current setup

  const hoverTimeoutRef = useRef(null);
  const isHoveringRef = useRef(false);
  const tooltipRef = useRef(null);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const velocityRef = useRef({ x: 0, y: 0 });
  const lastTimeRef = useRef(Date.now());
  const animationFrameRef = useRef(null);
  const lastPinchDistanceRef = useRef(null);
  const lastPinchCenterRef = useRef(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (viewModeRef.current) {
        setSvgDimensions({
          width: viewModeRef.current.clientWidth,
          height: viewModeRef.current.clientHeight,
        });
      } else {
        setSvgDimensions({ width: window.innerWidth, height: window.innerHeight - 180 });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);


  const handleWheel = useCallback((e) => {
    e.preventDefault();
    
    // Handle trackpad pinch-to-zoom
    if (e.ctrlKey) {
      const delta = e.deltaY;
      const zoomFactor = Math.pow(0.9, Math.sign(delta));
      const svgRect = e.currentTarget.getBoundingClientRect();
      const mouseX = e.clientX - svgRect.left;
      const mouseY = e.clientY - svgRect.top;

      setZoom(prevZoom => {
        const newZoom = Math.min(Math.max(prevZoom * zoomFactor, 1), 200);
        setPan(prevPan => ({
          x: prevPan.x - (mouseX - prevPan.x) * (newZoom / prevZoom - 1),
          y: prevPan.y - (mouseY - prevPan.y) * (newZoom / prevZoom - 1)
        }));
        return newZoom;
      });
      return;
    }

    // Handle trackpad two-finger scroll for panning
    if (Math.abs(e.deltaX) > 0 || Math.abs(e.deltaY) > 0) {
      setPan(prevPan => ({
        x: prevPan.x - e.deltaX,
        y: prevPan.y - e.deltaY
      }));
      return;
    }

    // Fallback to regular wheel zoom
    const delta = e.deltaY;
    const zoomFactor = Math.pow(0.9, Math.sign(delta));
    const svgRect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;
    const mouseY = e.clientY - svgRect.top;

    setZoom(prevZoom => {
      const newZoom = Math.min(Math.max(prevZoom * zoomFactor, 1), 200);
      setPan(prevPan => ({
        x: prevPan.x - (mouseX - prevPan.x) * (newZoom / prevZoom - 1),
        y: prevPan.y - (mouseY - prevPan.y) * (newZoom / prevZoom - 1)
      }));
      return newZoom;
    });
  }, []);

  const handleTouchStart = useCallback((e) => {
    if (e.touches.length === 2) {
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      lastPinchDistanceRef.current = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      );
      lastPinchCenterRef.current = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2
      };
    }
  }, []);

  const handleTouchMove = useCallback((e) => {
    if (e.touches.length === 2) {
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      const currentDistance = Math.hypot(
        touch2.clientX - touch1.clientX,
        touch2.clientY - touch1.clientY
      );
      const currentCenter = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2
      };

      if (lastPinchDistanceRef.current !== null) {
        const scale = currentDistance / lastPinchDistanceRef.current;
        const svgRect = e.currentTarget.getBoundingClientRect();
        const centerX = currentCenter.x - svgRect.left;
        const centerY = currentCenter.y - svgRect.top;

        setZoom(prevZoom => {
          const newZoom = Math.min(Math.max(prevZoom * scale, 1), 200);
          setPan(prevPan => ({
            x: prevPan.x - (centerX - prevPan.x) * (newZoom / prevZoom - 1),
            y: prevPan.y - (centerY - prevPan.y) * (newZoom / prevZoom - 1)
          }));
          return newZoom;
        });
      }

      // Handle panning during pinch
      if (lastPinchCenterRef.current) {
        const deltaX = currentCenter.x - lastPinchCenterRef.current.x;
        const deltaY = currentCenter.y - lastPinchCenterRef.current.y;
        setPan(prevPan => ({
          x: prevPan.x + deltaX,
          y: prevPan.y + deltaY
        }));
      }

      lastPinchDistanceRef.current = currentDistance;
      lastPinchCenterRef.current = currentCenter;
    }
  }, []);

  const handleTouchEnd = useCallback(() => {
    lastPinchDistanceRef.current = null;
    lastPinchCenterRef.current = null;
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      lastTimeRef.current = Date.now();
      velocityRef.current = { x: 0, y: 0 };
      
      // Cancel any ongoing animation
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (isDragging) {
      const currentTime = Date.now();
      const deltaTime = currentTime - lastTimeRef.current;
      
      if (deltaTime > 0) {
        const deltaX = e.clientX - lastMousePosRef.current.x;
        const deltaY = e.clientY - lastMousePosRef.current.y;
        
        // Calculate velocity (pixels per millisecond)
        velocityRef.current = {
          x: deltaX / deltaTime,
          y: deltaY / deltaTime
        };
        
        // Update pan with reduced sensitivity
        setPan({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y
        });
        
        lastMousePosRef.current = { x: e.clientX, y: e.clientY };
        lastTimeRef.current = currentTime;
      }
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      
      // Apply momentum if velocity is significant
      if (Math.abs(velocityRef.current.x) > 0.1 || Math.abs(velocityRef.current.y) > 0.1) {
        const applyMomentum = () => {
          const currentTime = Date.now();
          const deltaTime = currentTime - lastTimeRef.current;
          
          // Decay velocity
          velocityRef.current = {
            x: velocityRef.current.x * Math.pow(0.95, deltaTime / 16),
            y: velocityRef.current.y * Math.pow(0.95, deltaTime / 16)
          };
          
          // Apply velocity to pan
          setPan(prevPan => ({
            x: prevPan.x + velocityRef.current.x * deltaTime,
            y: prevPan.y + velocityRef.current.y * deltaTime
          }));
          
          lastTimeRef.current = currentTime;
          
          // Continue animation if velocity is still significant
          if (Math.abs(velocityRef.current.x) > 0.01 || Math.abs(velocityRef.current.y) > 0.01) {
            animationFrameRef.current = requestAnimationFrame(applyMomentum);
          }
        };
        
        animationFrameRef.current = requestAnimationFrame(applyMomentum);
      }
    }
  }, [isDragging]);

  const handleReset = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, []);

  const adjustLuminance = (hex, lum) => {
    hex = String(hex).replace(/[^0-9a-f]/gi, '');
    if (hex.length < 6) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    lum = lum || 0;
    let rgb = "#", c, i;
    for (i = 0; i < 3; i++) {
      c = parseInt(hex.substr(i*2, 2), 16);
      c = Math.round(Math.min(Math.max(0, c * (1 + lum)), 255));
      rgb += ("00"+c.toString(16)).substr(c.toString(16).length);
    }
    return rgb;
  };

  useEffect(() => {
    if (!tracks || tracks.length === 0 || !selectedCategory || !featureMetadata.names || featureMetadata.names.length === 0) {
      setStyleColors(new Map()); return;
    }

    const baseColorForCategory = CATEGORY_BASE_COLORS[selectedCategory] || NOISE_CLUSTER_COLOR;
    const featureFrequencies = new Map();

    tracks.forEach(track => {
      if (!track) return;
      let featuresToParse = null;
      let keyExtractor = (key) => key;

      if (selectedCategory === 'genre' || selectedCategory === 'style') {
        featuresToParse = track.features;
        keyExtractor = (key) => {
            const [genrePart, stylePart] = key.split('---');
            return selectedCategory === 'genre' ? genrePart : stylePart;
        };
      } else if (selectedCategory === 'instrument') {
        featuresToParse = track.instrument_features;
      } else if (selectedCategory === 'mood') {
        MOOD_KEYWORDS.forEach(key => {
            const value = track[key];
            if (typeof value === 'number' && !isNaN(value)) {
                featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + value);
            }
        });
      } else if (selectedCategory === 'spectral') {
        SPECTRAL_KEYWORDS.forEach(key => {
            const value = track[key];
            if (typeof value === 'number' && !isNaN(value)) {
                featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + value);
            }
        });
      }

      if (featuresToParse) {
        try {
          const parsed = typeof featuresToParse === 'string' ? JSON.parse(featuresToParse) : featuresToParse;
          if (typeof parsed === 'object' && parsed !== null) {
            Object.entries(parsed).forEach(([key, value]) => {
              const probability = parseFloat(value);
              if (isNaN(probability) || probability <= 0) return;
              const featureName = keyExtractor(key);
              if (featureName) {
                featureFrequencies.set(featureName, (featureFrequencies.get(featureName) || 0) + probability);
              }
            });
          }
        } catch (e) { /* console.warn(...) */ }
      }
    });

    const sortedFeatures = Array.from(featureFrequencies.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topNThreshold);

    const newStyleColors = new Map();
    sortedFeatures.forEach(([featureName], index) => {
      const luminanceFactor = index === 0 ? 0 :
        (index % 2 === 0 ?
          -Math.min(index * LUMINANCE_INCREMENT, MAX_LUM_OFFSET) :
          Math.min(index * LUMINANCE_INCREMENT, MAX_LUM_OFFSET));
      const shadedColor = adjustLuminance(baseColorForCategory, luminanceFactor);
      newStyleColors.set(featureName, shadedColor);
    });
    setStyleColors(newStyleColors);
  }, [tracks, selectedCategory, featureMetadata, topNThreshold]);

  const getDominantFeature = (track, selectedCategory, styleColorsMap) => {
    let dominantFeature = null;
    let maxValue = 0;

    if (selectedCategory === 'genre' || selectedCategory === 'style') {
      try {
        const features = typeof track.features === 'string' ? JSON.parse(track.features) : track.features;
        Object.entries(features || {}).forEach(([key, value]) => {
          const [genrePart, stylePart] = key.split('---');
          const featureName = selectedCategory === 'genre' ? genrePart : stylePart;
          const score = parseFloat(value);
          if (!isNaN(score) && score > maxValue && styleColorsMap.has(featureName)) {
            maxValue = score;
            dominantFeature = featureName;
          }
        });
      } catch (e) {}
    } else if (selectedCategory === 'instrument') {
      try {
        const features = typeof track.instrument_features === 'string' ? JSON.parse(track.instrument_features) : track.instrument_features;
        Object.entries(features || {}).forEach(([key, value]) => {
          const score = parseFloat(value);
          if (!isNaN(score) && score > maxValue && styleColorsMap.has(key)) {
            maxValue = score;
            dominantFeature = key;
          }
        });
      } catch (e) {}
    } else if (selectedCategory === 'mood') {
      MOOD_KEYWORDS.forEach(key => {
        const value = track[key];
        if (typeof value === 'number' && !isNaN(value) && value > maxValue && styleColorsMap.has(key)) {
          maxValue = value;
          dominantFeature = key;
        }
      });
    } else if (selectedCategory === 'spectral') {
      SPECTRAL_KEYWORDS.forEach(key => {
        const value = track[key];
        if (typeof value === 'number' && !isNaN(value) && value > maxValue && styleColorsMap.has(key)) {
          maxValue = value;
          dominantFeature = key;
        }
      });
    }
    return dominantFeature;
  };

  const trackColors = useMemo(() => {
    return plotData.map(track => {
      const dominantFeature = getDominantFeature(track, selectedCategory, styleColors);
      return {
        id: track.id,
        color: dominantFeature ? styleColors.get(dominantFeature) : NOISE_CLUSTER_COLOR,
        dominantFeature
      };
    });
  }, [plotData, selectedCategory, styleColors]);

  const fetchTracksData = useCallback(async () => {
    try {
      setLoading(true); setError(null);
      const response = await fetch('http://localhost:3000/tracks');
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      const rawData = await response.json();
      if (!Array.isArray(rawData)) throw new Error("Invalid data: Expected array.");

      const keysWithCats = getAllFeatureKeysAndCategories(rawData);
      const featureNames = keysWithCats.map(kc => kc.name);
      const featureCats = keysWithCats.map(kc => kc.category);
      setFeatureMetadata({ names: featureNames, categories: featureCats });

      const parsedTracks = rawData.map(track => {
        if (!track || typeof track !== 'object' || !track.id) return null;
        try {
          const featureVector = mergeFeatureVectors(track, featureNames);
          return { ...track, featureVector };
        } catch (e) { return null; }
      }).filter(Boolean);
      setTracks(parsedTracks);
    } catch (err) {
      setError(err.message || 'Unknown error.');
      setTracks([]); setFeatureMetadata({ names: [], categories: [] });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTracksData();
  }, [fetchTracksData]);

  useEffect(() => {
    if (loading || tracks.length === 0 || svgDimensions.width === 0 || svgDimensions.height === 0) {
      setPlotData([]); return;
    }
    if (featureMetadata.names.length === 0 && tracks.length > 0) return;

    const validTracksForProcessing = tracks.filter(t => t.featureVector && t.featureVector.length === featureMetadata.names.length);
    if (validTracksForProcessing.length === 0) {
      if (tracks.length > 0) setError("No tracks have valid features for processing.");
      setPlotData([]); return;
    }

    const featureVectors = validTracksForProcessing.map(t => t.featureVector);
    const processedFeatureData = normalizeFeatures(featureVectors, featureMetadata.categories);
    const clusterLabels = hdbscan(processedFeatureData);
    const projectedData = pca(processedFeatureData);

    const newPlotData = validTracksForProcessing.map((track, index) => {
      const p_coords = (projectedData && index < projectedData.length && projectedData[index]?.length === PCA_N_COMPONENTS)
                ? projectedData[index] : [0.5, 0.5];
      return {
        ...track,
        originalX: p_coords[0],
        originalY: p_coords[1],
        x: PADDING + p_coords[0] * (svgDimensions.width - 2 * PADDING),
        y: PADDING + p_coords[1] * (svgDimensions.height - 2 * PADDING),
        cluster: clusterLabels[index] ?? NOISE_CLUSTER_ID,
      };
    });
    setPlotData(newPlotData);
  }, [tracks, featureMetadata, loading, svgDimensions]);

  const handleMouseOver = useCallback((trackData, event) => {
    // Clear any existing timeout
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }

    isHoveringRef.current = true;

    const audioPath = trackData.audioUrl || (trackData.path ? `http://localhost:3000/audio/${trackData.id}` : null);
    const tooltipWidth = 300;
    const tooltipHeight = audioPath ? 200 : 150;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let x = event.clientX + TOOLTIP_OFFSET;
    let y = event.clientY + TOOLTIP_OFFSET;

    if (x + tooltipWidth > viewportWidth) {
      x = event.clientX - tooltipWidth - TOOLTIP_OFFSET;
    }
    if (y + tooltipHeight > viewportHeight) {
      y = event.clientY - tooltipHeight - TOOLTIP_OFFSET;
    }
    if (x < 0) x = TOOLTIP_OFFSET;
    if (y < 0) y = TOOLTIP_OFFSET;

    setTooltip({
      content: (
        <div>
          <img
            src={trackData.artwork_thumbnail_path || defaultArtwork}
            alt={`${trackData.artist || 'Unknown'} - ${trackData.title || 'Unknown'}`}
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = defaultArtwork;
              e.target.style.opacity = '0.7';
            }}
            style={{
              width: '80px',
              height: '80px',
              objectFit: 'cover',
              marginRight: '10px',
              float: 'left',
              borderRadius: '4px',
              transition: 'opacity 0.2s ease'
            }}
          />
          <div style={{ overflow: 'hidden'}}>
            <div><strong>{trackData.title || 'Unknown Title'}</strong></div>
            <div>{trackData.artist || 'Unknown Artist'}</div>
            <div><em>{trackData.album || 'Unknown Album'} ({trackData.year || 'N/A'})</em></div>
            <div>BPM: {trackData.bpm?.toFixed(1) || 'N/A'}, Key: {trackData.key || 'N/A'}</div>
            {trackData.tag1 && <div>Genre: {trackData.tag1} ({trackData.tag1_prob?.toFixed(2) || 'N/A'})</div>}
          </div>
          {audioPath && (
            <div className="waveform-container" style={{ clear: 'both', marginTop: '10px', width: '100%', height: '40px' }}>
              <PlaybackContext.Provider value={{
                setPlayingWaveSurfer: (newlyPlayingWavesurfer) => {
                  if (wavesurferRef.current && wavesurferRef.current !== newlyPlayingWavesurfer) {
                    try {
                      wavesurferRef.current.stop();
                    } catch (e) {
                      console.warn("Error stopping previous wavesurfer:", e);
                    }
                  }
                  wavesurferRef.current = newlyPlayingWavesurfer;
                },
                currentTrack: trackData,
                setCurrentTrack: () => {}
              }}>
                <Waveform
                  key={`waveform-tooltip-${trackData.id}`}
                  trackId={trackData.id.toString()}
                  audioPath={audioPath}
                  isInteractive={true}
                  onPlay={() => {}}
                />
              </PlaybackContext.Provider>
            </div>
          )}
        </div>
      ),
      x,
      y,
    });
  }, [wavesurferRef]);

  const handleMouseOut = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    
    // Check if we're actually hovering over the tooltip
    const tooltipElement = tooltipRef.current;
    if (tooltipElement) {
      const rect = tooltipElement.getBoundingClientRect();
      const mouseX = event.clientX;
      const mouseY = event.clientY;
      
      if (mouseX >= rect.left && mouseX <= rect.right && 
          mouseY >= rect.top && mouseY <= rect.bottom) {
        isHoveringRef.current = true;
        return;
      }
    }
    
    isHoveringRef.current = false;
    hoverTimeoutRef.current = setTimeout(() => {
      if (!isHoveringRef.current) {
        setTooltip(null);
      }
    }, 500);
  }, []);

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  const handleDotClick = useCallback((trackData) => console.log("Clicked track:", trackData.id, trackData.title), []);

  // Effect to clean up the main wavesurferRef when the component unmounts
  useEffect(() => {
    return () => {
      if (wavesurferRef.current) {
        try {
          wavesurferRef.current.stop();
          // wavesurferRef.current.destroy(); // The instance is owned by Waveform.jsx, it will destroy it.
        } catch(e) {
            // console.warn("Error stopping wavesurfer on TrackVisualizer unmount", e);
        }
        wavesurferRef.current = null;
      }
    };
  }, []);

  // Clean up animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  if (loading) return <div className="track-visualizer-loading">Loading tracks and features...</div>;
  if (error) return <div className="track-visualizer-error">Error: {error} <button onClick={fetchTracksData}>Try Reload</button></div>;
  if (plotData.length === 0 && !loading && tracks.length > 0) return <div className="track-visualizer-empty">Data processed, but no points to visualize. Check feature processing.</div>;
  if (plotData.length === 0 && !loading && tracks.length === 0) return <div className="track-visualizer-empty">No tracks data loaded.</div>;

  return (
    <div className="track-visualizer-container">
      <h3>Track Similarity Visualization</h3>
      <div className="controls-panel">
        <div className="category-toggle">
          <label htmlFor="categorySelect">Color by:</label>
          {Object.entries(CATEGORY_BASE_COLORS).map(([categoryKey, colorValue]) => (
            <button
              key={categoryKey}
              onClick={() => setSelectedCategory(categoryKey)}
              className={selectedCategory === categoryKey ? 'active' : ''}
              style={{
                backgroundColor: selectedCategory === categoryKey ? colorValue : DARK_MODE_SURFACE_ALT,
                color: selectedCategory === categoryKey ? (parseInt(colorValue.slice(1,3),16)*0.299 + parseInt(colorValue.slice(3,5),16)*0.587 + parseInt(colorValue.slice(5,7),16)*0.114 > 160 ? '#000000' : DARK_MODE_TEXT_PRIMARY) : DARK_MODE_TEXT_SECONDARY,
                border: `2px solid ${selectedCategory === categoryKey ? adjustLuminance(colorValue, -0.2) : DARK_MODE_BORDER}`,
              }}
            >
              {categoryKey.charAt(0).toUpperCase() + categoryKey.slice(1)}
            </button>
          ))}
        </div>
        <div className="threshold-control">
          <label htmlFor="topNInput">Top N Features:</label>
          <input
            id="topNInput" type="number" min="1" max="50" value={topNThreshold}
            onChange={(e) => setTopNThreshold(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
          />
        </div>
        <button onClick={handleReset} className="reset-button">Reset View</button>
      </div>
      <p className="info-text">
        Tracks clustered by audio feature similarity. Colors represent top {topNThreshold} dominant '{selectedCategory}' features. Shades indicate prominence. Gray dots are noise.
        <small>Scroll to zoom, drag to pan.</small>
      </p>
      <div className="visualization-area" ref={viewModeRef}>
        {svgDimensions.width > 0 && svgDimensions.height > 0 && (
            <svg
                className="track-plot"
                viewBox={VIEW_BOX_VALUE}
                preserveAspectRatio="xMidYMid meet"
                aria-labelledby="plotTitle" role="graphics-document"
                onWheel={handleWheel} onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart} onTouchMove={handleTouchMove} onTouchEnd={handleTouchEnd}
            >
            <title id="plotTitle">Track Similarity Plot</title>
            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                {plotData.map((track, index) => {
                  const colorInfo = trackColors[index] || { color: NOISE_CLUSTER_COLOR, dominantFeature: 'N/A' };
                  // Calculate screen-space radius (in pixels)
                  const screenRadius = 4; // Fixed screen-space radius in pixels
                  // Convert to SVG space
                  const svgRadius = screenRadius / zoom;
                  return (
                    <circle
                      key={track.id || `track-${index}`}
                      cx={track.x}
                      cy={track.y}
                      r={svgRadius}
                      fill={colorInfo.color}
                      onMouseMove={(e) => handleMouseOver(track, e)}
                      onMouseOut={handleMouseOut}
                      onClick={() => handleDotClick(track)}
                      className="track-dot"
                      style={{ transition: 'none' }} // Prevent any size transitions
                      tabIndex={0}
                      aria-label={`Track: ${track.title || 'Unknown'} by ${track.artist || 'Unknown'}, Feature: ${colorInfo.dominantFeature || 'None'}`}
                    />
                  );
                })}
            </g>
            </svg>
        )}
        {tooltip && (
          <div 
            ref={tooltipRef}
            className="track-tooltip" 
            style={{ 
              top: tooltip.y, 
              left: tooltip.x,
              position: 'fixed',
              zIndex: 1000,
              backgroundColor: DARK_MODE_SURFACE_ALT,
              color: DARK_MODE_TEXT_PRIMARY,
              padding: '10px',
              borderRadius: '4px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
              pointerEvents: 'auto',
              border: `1px solid ${DARK_MODE_BORDER}`
            }} 
            role="tooltip"
            onMouseEnter={() => {
              isHoveringRef.current = true;
              if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current);
                hoverTimeoutRef.current = null;
              }
            }}
            onMouseLeave={(e) => {
              // Check if we're moving to a child element
              const relatedTarget = e.relatedTarget;
              if (tooltipRef.current && tooltipRef.current.contains(relatedTarget)) {
                return;
              }
              isHoveringRef.current = false;
              handleMouseOut();
            }}
          >
            {tooltip.content}
          </div>
        )}
        <div className="legend">
          <h4>{selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)} Legend (Top {Math.min(topNThreshold, styleColors.size, 15)})</h4>
          <div className="style-legend">
            {Array.from(styleColors.entries())
              .sort((a,b) => a[0].localeCompare(b[0])) // Sort legend items alphabetically
              .slice(0,15) // Limit to 15 items in legend
              .map(([feature, color]) => (
                <div key={feature} className="legend-item">
                  <div className="color-box" style={{ backgroundColor: color }}></div>
                  <span className="feature-name">{feature}</span>
                </div>
            ))}
            {styleColors.size > 15 && <div className="legend-item">...and {styleColors.size - 15} more</div>}
            {styleColors.size === 0 && selectedCategory && <div className="legend-item">No dominant '{selectedCategory}' features found.</div>}
          </div>
          {plotData.some(p => p.cluster === NOISE_CLUSTER_ID || trackColors.some(tc => tc.color === NOISE_CLUSTER_COLOR)) && (
            <div className="legend-item noise-legend">
              <div className="color-box" style={{ backgroundColor: NOISE_CLUSTER_COLOR }}></div>
              <span className="feature-name">Noise / Other</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrackVisualizer;