import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import './TrackVisualizer.scss';
// import WaveSurfer from 'wavesurfer.js'; // WaveSurfer is used by the Waveform component, not directly here usually
import defaultArtwork from "../../assets/default-artwork.png";
import Waveform from './Waveform';
import FilterPanel from './FilterPanel';
import { PlaybackContext } from '../context/PlaybackContext';
import * as d3 from 'd3';

// --- Dark Mode Theme Variables (mirroring SCSS for JS logic if needed) ---
const DARK_MODE_TEXT_PRIMARY = '#e0e0e0';
const DARK_MODE_TEXT_SECONDARY = '#b0b0b0';
const DARK_MODE_SURFACE_ALT = '#3a3a3a';
const DARK_MODE_BORDER = '#4a4a4a';
// const DARK_MODE_ACCENT = '#00bcd4'; // Not directly used in JS logic shown, but good for reference


// --- Constants ---
const PADDING = 50;
const PCA_N_COMPONENTS = 2;
const HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE = 3;
const HDBSCAN_DEFAULT_MIN_SAMPLES = 2;
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
  'genre': 0.2,
  'style': 1,
  'spectral': 0,
  'mood': 0.1,
  'instrument': 0,
  'default': 0,
};

const SPECTRAL_KEYWORDS = [
  'atonal', 'tonal', 'dark', 'bright', 'percussive', 'smooth', 'lufs'
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
const HIGHLIGHT_COLOR = '#FF5A16';


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
    if (sourceCategory === 'mood') return 'mood';
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
    processFeatureSource(track.style_features, 'style', track.id);
    processFeatureSource(track.instrument_features, 'instrument', track.id);
    processFeatureSource(track.mood_features, 'mood', track.id);
  });

  SPECTRAL_KEYWORDS.forEach(key => featuresWithCategories.set(key, 'spectral'));

  return Array.from(featuresWithCategories.entries())
    .map(([name, category]) => ({ name, category }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function mergeFeatureVectors(track, allFeatureNames) {
  const mergedFeatures = {};
  allFeatureNames.forEach(key => {
    mergedFeatures[key] = 0;
  });

  const parseAndMerge = (featureObj, category) => {
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
      // console.warn(`Failed to parse ${category} features for track ${track?.id} during merge:`, e, featureObj);
    }
  };

  parseAndMerge(track.style_features, 'style');
  parseAndMerge(track.instrument_features, 'instrument');
  parseAndMerge(track.mood_features, 'mood');

  // Add direct spectral features
  SPECTRAL_KEYWORDS.forEach(key => {
    const value = track[key];
    if (typeof value === 'number' && !isNaN(value)) {
      mergedFeatures[key] = value;
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

  // First pass: Calculate robust statistics
  const medians = new Array(numFeatures).fill(0);
  const madValues = new Array(numFeatures).fill(0); // Median Absolute Deviation

  // Calculate medians
  for (let j = 0; j < numFeatures; j++) {
    const values = featureVectors.map(v => v[j] || 0).sort((a, b) => a - b);
    medians[j] = values[Math.floor(values.length / 2)];
  }

  // Calculate MAD values
  for (let j = 0; j < numFeatures; j++) {
    const deviations = featureVectors.map(v => Math.abs((v[j] || 0) - medians[j]));
    madValues[j] = deviations.sort((a, b) => a - b)[Math.floor(deviations.length / 2)] * 1.4826; // Scale factor for normal distribution
  }

  // Second pass: Apply robust normalization
  return featureVectors.map(vector =>
    vector.map((value, j) => {
      const mad = madValues[j];
      const median = medians[j];
      const normalizedValue = (mad < 1e-10) ? 0 : ((value || 0) - median) / mad;
      
      // Apply category weights with improved scaling
      const category = (j < categories.length && categories[j]) ? categories[j] : 'default';
      const weight = CATEGORY_WEIGHTS[category] || CATEGORY_WEIGHTS['default'];
      
      // Apply sigmoid function to bound the values
      const sigmoid = (x) => 2 / (1 + Math.exp(-x)) - 1;
      return sigmoid(normalizedValue * weight);
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

  // Center the data
  const means = processedData[0].map((_, colIndex) => 
    processedData.reduce((sum, row) => sum + (row[colIndex] || 0), 0) / numSamples
  );
  const centeredData = processedData.map(row => 
    row.map((val, colIndex) => (val || 0) - means[colIndex])
  );

  // Calculate covariance matrix with improved numerical stability
  const covarianceMatrix = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
  for (let i = 0; i < numFeatures; i++) {
    for (let j = i; j < numFeatures; j++) {
      let sum = 0;
      for (let k = 0; k < numSamples; k++) {
        sum += centeredData[k][i] * centeredData[k][j];
      }
      covarianceMatrix[i][j] = sum / (numSamples - 1);
      if (i !== j) covarianceMatrix[j][i] = covarianceMatrix[i][j];
    }
  }

  // Power iteration with improved convergence and robust sign consistency
  const powerIteration = (matrix, numIterations = 100) => {
    const n = matrix.length;
    if (n === 0 || !matrix[0] || matrix[0].length === 0) return [];
    
    // Initialize with a random vector
    let vector = Array(n).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(vector.reduce((s, v) => s + v * v, 0));
    if (norm < 1e-10) vector = Array(n).fill(0);
    else vector = vector.map(v => v / norm);
    
    if (vector.every(v => v === 0) && n > 0) vector[0] = 1;

    // Improved convergence with adaptive iterations
    let prevVector = null;
    let iter = 0;
    const maxIter = numIterations;
    const convergenceThreshold = 1e-10;

    while (iter < maxIter) {
      let newVector = Array(n).fill(0);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          newVector[r] += (matrix[r]?.[c] || 0) * vector[c];
        }
      }
      
      norm = Math.sqrt(newVector.reduce((s, val) => s + val * val, 0));
      if (norm < 1e-10) return Array(n).fill(0);
      
      newVector = newVector.map(val => val / norm);
      
      // Check convergence
      if (prevVector) {
        const diff = Math.sqrt(newVector.reduce((s, v, i) => s + Math.pow(v - prevVector[i], 2), 0));
        if (diff < convergenceThreshold) break;
      }
      
      prevVector = [...newVector];
      vector = newVector;
      iter++;
    }
    
    return vector;
  };

  const principalComponents = [];
  let tempCovarianceMatrix = covarianceMatrix.map(row => [...row]);

  // Calculate reference points for sign consistency
  const referencePoints = [];
  for (let i = 0; i < numFeatures; i++) {
    const values = centeredData.map(row => row[i]);
    const sortedValues = [...values].sort((a, b) => a - b);
    const q1 = sortedValues[Math.floor(sortedValues.length * 0.25)];
    const q3 = sortedValues[Math.floor(sortedValues.length * 0.75)];
    referencePoints.push((q1 + q3) / 2); // Use median of quartiles as reference
  }

  for (let k = 0; k < nComponents; k++) {
    if (tempCovarianceMatrix.length === 0 || tempCovarianceMatrix.every(row => row.every(val => isNaN(val) || val === 0))) {
      const fallbackPc = Array(numFeatures).fill(0);
      if (k < numFeatures) fallbackPc[k] = 1;
      principalComponents.push(fallbackPc);
      continue;
    }
    
    const pc = powerIteration(tempCovarianceMatrix);
    if (pc.length === 0 || pc.every(v => v === 0)) {
      const fallbackPc = Array(numFeatures).fill(0);
      if (k < numFeatures) fallbackPc[k] = 1;
      principalComponents.push(fallbackPc);
      continue;
    }

    // Ensure sign consistency using reference points
    const projection = referencePoints.reduce((sum, val, i) => sum + val * pc[i], 0);
    if (projection < 0) {
      pc.forEach((_, i) => pc[i] = -pc[i]);
    }
    
    principalComponents.push(pc);

    if (k < nComponents - 1 && pc.length > 0) {
      // Deflate the matrix
      const lambda = pc.reduce((sum, val, i) => 
        sum + val * tempCovarianceMatrix[i].reduce((s, v, j) => s + v * pc[j], 0), 0
      );
      
      const newTempCovMatrix = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
      for (let i = 0; i < numFeatures; i++) {
        for (let j = 0; j < numFeatures; j++) {
          newTempCovMatrix[i][j] = tempCovarianceMatrix[i][j] - lambda * pc[i] * pc[j];
        }
      }
      tempCovarianceMatrix = newTempCovMatrix;
    }
  }

  // Project the data
  const projected = centeredData.map(row =>
    principalComponents.map(pcVector => {
      if (pcVector.length !== row.length) return 0;
      return row.reduce((sum, val, i) => sum + val * (pcVector[i] || 0), 0);
    })
  );

  // Normalize the projection to better utilize the canvas space
  const minMax = Array(nComponents).fill(null).map((_, i) => ({
    min: Math.min(...projected.map(p => p[i])),
    max: Math.max(...projected.map(p => p[i])),
  }));

  // Apply sigmoid-like scaling for better distribution
  return projected.map(p => p.map((val, i) => {
    if (i >= minMax.length || minMax[i] === null) return 0.5;
    const range = minMax[i].max - minMax[i].min;
    if (range < 1e-10) return 0.5;
    
    // Center and scale
    const centered = (val - minMax[i].min) / range;
    // Apply sigmoid-like transformation
    const sigmoid = (x) => 2 / (1 + Math.exp(-4 * (x - 0.5))) - 1;
    return (sigmoid(centered) + 1) / 2;
  }));
}

function hdbscan(data, minClusterSize = HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE, minSamples = HDBSCAN_DEFAULT_MIN_SAMPLES) {
  if (!data || data.length === 0) return [];
  const n = data.length;
  if (n === 0) return [];

  // Adaptive parameters based on dataset size and density
  const adaptiveMinClusterSize = Math.max(2, Math.min(minClusterSize, Math.floor(n * 0.03))); // 3% of dataset size
  const adaptiveMinSamples = Math.max(2, Math.min(minSamples, Math.floor(n * 0.01))); // 1% of dataset size

  if (n < adaptiveMinClusterSize && n > 0) return Array(n).fill(NOISE_CLUSTER_ID);

  function computeMutualReachabilityDistance() {
    const distances = Array(n).fill(null).map(() => Array(n).fill(0));
    const coreDistances = Array(n).fill(Infinity);
    if (n === 0) return { distances, coreDistances };

    // Calculate core distances with adaptive k
    for (let i = 0; i < n; i++) {
      if (n <= 1 || adaptiveMinSamples >= n) { coreDistances[i] = Infinity; continue; }
      const pointDistances = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        pointDistances.push(calculateDistance(data[i], data[j]));
      }
      pointDistances.sort((a, b) => a - b);
      coreDistances[i] = pointDistances[adaptiveMinSamples - 1] ?? Infinity;
    }

    // Calculate mutual reachability distances with improved distance metric
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const directDist = calculateDistance(data[i], data[j]);
        // Use geometric mean for better balance
        const mrDist = Math.sqrt(coreDistances[i] * coreDistances[j]) * directDist;
        distances[i][j] = mrDist;
        distances[j][i] = mrDist;
      }
    }
    return distances;
  }

  function buildMST(mutualReachabilityDistances) {
    if (n === 0) return [];
    const mstEdges = [];
    const visited = new Array(n).fill(false);
    const minEdgeWeight = new Array(n).fill(Infinity);
    const edgeToVertex = new Array(n).fill(-1);
    if (n > 0) minEdgeWeight[0] = 0;

    for (let count = 0; count < n; count++) {
      let u = -1, currentMin = Infinity;
      for (let v = 0; v < n; v++) {
        if (!visited[v] && minEdgeWeight[v] < currentMin) {
          currentMin = minEdgeWeight[v];
          u = v;
        }
      }
      if (u === -1) break;
      visited[u] = true;
      if (edgeToVertex[u] !== -1) {
        mstEdges.push([u, edgeToVertex[u], minEdgeWeight[u]]);
      }
      for (let v = 0; v < n; v++) {
        if (!visited[v]) {
          const weightUV = mutualReachabilityDistances[u]?.[v] ?? Infinity;
          if (weightUV < minEdgeWeight[v]) {
            minEdgeWeight[v] = weightUV;
            edgeToVertex[v] = u;
          }
        }
      }
    }
    return mstEdges;
  }

  function extractClustersSimplified(mst) {
    const labels = Array(n).fill(NOISE_CLUSTER_ID);
    if (n === 0 || (mst.length === 0 && n > 0 && adaptiveMinClusterSize > 1)) return labels;
    if (n > 0 && adaptiveMinClusterSize === 1) return Array(n).fill(0).map((_,i)=>i);

    let currentClusterId = 0;
    const parent = Array(n).fill(0).map((_, i) => i);
    const componentSize = Array(n).fill(1);
    const edgeWeights = new Map();

    function findSet(i) {
      if (parent[i] === i) return i;
      return parent[i] = findSet(parent[i]);
    }

    function uniteSets(i, j, weight) {
      let rootI = findSet(i), rootJ = findSet(j);
      if (rootI !== rootJ) {
        if (componentSize[rootI] < componentSize[rootJ]) [rootI, rootJ] = [rootJ, rootI];
        parent[rootJ] = rootI;
        componentSize[rootI] += componentSize[rootJ];
        edgeWeights.set(rootI, Math.max(edgeWeights.get(rootI) || 0, weight));
        return true;
      }
      return false;
    }

    const sortedMSTEdges = mst.sort((a, b) => a[2] - b[2]);
    for (const edge of sortedMSTEdges) {
      uniteSets(edge[0], edge[1], edge[2]);
    }

    const rootToClusterId = new Map();
    for(let i = 0; i < n; i++) {
      const root = findSet(i);
      if(componentSize[root] >= adaptiveMinClusterSize) {
        if(!rootToClusterId.has(root)) {
          rootToClusterId.set(root, currentClusterId++);
        }
        labels[i] = rootToClusterId.get(root);
      } else {
        labels[i] = NOISE_CLUSTER_ID;
      }
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
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchSuggestions, setSearchSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [featureMetadata, setFeatureMetadata] = useState({ names: [], categories: [] });
  const [styleColors, setStyleColors] = useState(new Map());
  const [featureThresholds, setFeatureThresholds] = useState(new Map());
  const [thresholdMultiplier, setThresholdMultiplier] = useState(1.0);
  const [selectedFeatures, setSelectedFeatures] = useState({
    genre: [],
    style: [],
    mood: [],
    instrument: [],
    spectral: []
  });
  const [filterLogicMode, setFilterLogicMode] = useState('intersection');
  const [highlightThreshold, setHighlightThreshold] = useState(0.1);
  const [tempThreshold, setTempThreshold] = useState(0.1);
  const thresholdDebounceRef = useRef();

  const containerRef = useRef(null);
  const visualizationRef = useRef(null);
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

  const searchInputRef = useRef(null);
  const suggestionsRef = useRef(null);

  const svgRef = useRef(null);
  const d3ContainerRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  const [featureMinMax, setFeatureMinMax] = useState({});

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
    
    // Handle trackpad pinch-to-zoom (both Ctrl and Cmd/Meta key)
    if (e.ctrlKey || e.metaKey) {
      // Use a smaller zoom factor for smoother zooming
      const zoomFactor = Math.pow(0.95, Math.sign(e.deltaY));
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
      // Increase panning speed for trackpad
      const panSpeed = 1.5; // Increased from 0.5
      setPan(prevPan => ({
        x: prevPan.x - e.deltaX * panSpeed,
        y: prevPan.y - e.deltaY * panSpeed
      }));
      return;
    }

    // Fallback to regular wheel zoom with smoother factor
    const zoomFactor = Math.pow(0.95, Math.sign(e.deltaY));
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
        
        // Update pan with increased sensitivity
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

  const handleReset = useCallback(() => {
    if (d3ContainerRef.current?.svg && zoomBehaviorRef.current) {
      d3ContainerRef.current.svg
        .transition()
        .duration(750)
        .call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
    }
  }, []);

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
      setStyleColors(new Map());
      setFeatureThresholds(new Map());
      setFeatureMinMax({});
      return;
    }

    const baseColorForCategory = CATEGORY_BASE_COLORS[selectedCategory] || NOISE_CLUSTER_COLOR;
    const featureFrequencies = new Map();
    const featureValues = new Map();

    tracks.forEach(track => {
      if (!track) return;
      let featuresToParse = null;

      if (selectedCategory === 'genre' || selectedCategory === 'style') {
        featuresToParse = track.style_features;
        try {
          const parsed = typeof featuresToParse === 'string' ? JSON.parse(featuresToParse) : featuresToParse;
          if (typeof parsed === 'object' && parsed !== null) {
            Object.entries(parsed).forEach(([key, value]) => {
              const probability = parseFloat(value);
              if (isNaN(probability) || probability <= 0) return;
              
              // Split the key into genre and style parts
              const [genrePart, stylePart] = key.split('---');
              
              // Store the appropriate part based on selected category
              const featureKey = selectedCategory === 'genre' ? genrePart : stylePart;
              if (featureKey) {
                featureFrequencies.set(featureKey, (featureFrequencies.get(featureKey) || 0) + probability);
                if (!featureValues.has(featureKey)) featureValues.set(featureKey, []);
                featureValues.get(featureKey).push(probability);
              }
            });
          }
        } catch (e) { /* console.warn(...) */ }
      } else if (selectedCategory === 'instrument') {
        featuresToParse = track.instrument_features;
        try {
          const parsed = typeof featuresToParse === 'string' ? JSON.parse(featuresToParse) : featuresToParse;
          if (typeof parsed === 'object' && parsed !== null) {
            Object.entries(parsed).forEach(([key, value]) => {
              const probability = parseFloat(value);
              if (isNaN(probability) || probability <= 0) return;
              featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + probability);
              if (!featureValues.has(key)) featureValues.set(key, []);
              featureValues.get(key).push(probability);
            });
          }
        } catch (e) { /* console.warn(...) */ }
      } else if (selectedCategory === 'mood') {
        try {
          const features = typeof track.mood_features === 'string' ? JSON.parse(track.mood_features) : track.mood_features;
          if (features && typeof features === 'object') {
            Object.entries(features).forEach(([key, value]) => {
              const numValue = parseFloat(value);
              if (!isNaN(numValue)) {
                featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + numValue);
                if (!featureValues.has(key)) featureValues.set(key, []);
                featureValues.get(key).push(numValue);
              }
            });
          }
        } catch (e) { /* console.warn(...) */ }
      } else if (selectedCategory === 'spectral') {
        SPECTRAL_KEYWORDS.forEach(key => {
          const value = track[key];
          if (typeof value === 'number' && !isNaN(value)) {
            featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + value);
            if (!featureValues.has(key)) featureValues.set(key, []);
            featureValues.get(key).push(value);
          }
        });
      }
    });

    // Calculate thresholds based on feature variance
    const newThresholds = new Map();
    featureValues.forEach((values, feature) => {
      if (values.length > 0) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        newThresholds.set(feature, mean + (0.5 * stdDev));
      }
    });

    const sortedFeatures = Array.from(featureFrequencies.entries())
      .sort((a, b) => b[1] - a[1]); // Sort by frequency, no limit

    const newStyleColors = new Map();
    sortedFeatures.forEach(([featureName]) => {
      newStyleColors.set(featureName, baseColorForCategory);
    });

    setStyleColors(newStyleColors);
    setFeatureThresholds(newThresholds);

    // Compute min/max for each feature after loading tracks
    const minMax = {};
    tracks.forEach(track => {
      // Style features
      try {
        const styleFeatures = typeof track.style_features === 'string' ? JSON.parse(track.style_features) : track.style_features;
        if (styleFeatures) {
          Object.entries(styleFeatures).forEach(([key, value]) => {
            const v = parseFloat(value);
            if (!isNaN(v)) {
              if (!(key in minMax)) minMax[key] = { min: v, max: v };
              else {
                minMax[key].min = Math.min(minMax[key].min, v);
                minMax[key].max = Math.max(minMax[key].max, v);
              }
            }
          });
        }
      } catch (e) {}
      // Mood features
      try {
        const moodFeatures = typeof track.mood_features === 'string' ? JSON.parse(track.mood_features) : track.mood_features;
        if (moodFeatures) {
          Object.entries(moodFeatures).forEach(([key, value]) => {
            const v = parseFloat(value);
            if (!isNaN(v)) {
              if (!(key in minMax)) minMax[key] = { min: v, max: v };
              else {
                minMax[key].min = Math.min(minMax[key].min, v);
                minMax[key].max = Math.max(minMax[key].max, v);
              }
            }
          });
        }
      } catch (e) {}
      // Instrument features
      try {
        const instrumentFeatures = typeof track.instrument_features === 'string'
          ? JSON.parse(track.instrument_features)
          : track.instrument_features;
        if (instrumentFeatures) {
          Object.entries(instrumentFeatures).forEach(([key, value]) => {
            const v = parseFloat(value);
            if (!isNaN(v)) {
              if (!(key in minMax)) minMax[key] = { min: v, max: v };
              else {
                minMax[key].min = Math.min(minMax[key].min, v);
                minMax[key].max = Math.max(minMax[key].max, v);
              }
            }
          });
        }
      } catch (e) {}
      // Spectral features
      SPECTRAL_KEYWORDS.forEach(key => {
        const v = track[key];
        if (typeof v === 'number' && !isNaN(v)) {
          if (!(key in minMax)) minMax[key] = { min: v, max: v };
          else {
            minMax[key].min = Math.min(minMax[key].min, v);
            minMax[key].max = Math.max(minMax[key].max, v);
          }
        }
      });
    });
    setFeatureMinMax(minMax);
  }, [tracks, selectedCategory, featureMetadata]);

  // Generate dynamic colors for selected features
  const getFeatureColors = useCallback(() => {
    const allSelectedFeatures = Object.values(selectedFeatures).flat();
    console.log('Selected features:', allSelectedFeatures);
    
    if (allSelectedFeatures.length === 0) return {};
    
    const colors = {};
    allSelectedFeatures.forEach((feature, index) => {
      const colorStr = DEFAULT_CLUSTER_COLORS[index % DEFAULT_CLUSTER_COLORS.length];
      colors[feature] = colorStr;
      console.log(`Assigned color ${colorStr} to feature ${feature}`);
    });
    return colors;
  }, [selectedFeatures]);

  // Helper function to safely parse a color
  const safeParseColor = useCallback((colorStr) => {
    if (!colorStr) {
      console.warn('No color string provided to safeParseColor');
      return null;
    }
    try {
      const color = d3.color(colorStr);
      if (!color) {
        console.warn('d3.color returned null for:', colorStr);
        return null;
      }
      return { r: color.r, g: color.g, b: color.b };
    } catch (e) {
      console.warn('Failed to parse color:', colorStr, e);
      return null;
    }
  }, []);

  // Helper function to safely create a color string
  const safeCreateColor = useCallback((r, g, b) => {
    // Ensure RGB values are within valid range
    r = Math.max(0, Math.min(255, Math.round(r)));
    g = Math.max(0, Math.min(255, Math.round(g)));
    b = Math.max(0, Math.min(255, Math.round(b)));
    
    try {
      // Create hex color string directly
      const toHex = (n) => {
        const hex = n.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
      };
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    } catch (e) {
      console.warn('Error creating color:', { r, g, b }, e);
      return NOISE_CLUSTER_COLOR;
    }
  }, []);

  // Modify the dot color calculation to use normalized feature values for thresholding
  const getDotColor = useCallback((track) => {
    const selectedFeaturesList = Object.values(selectedFeatures).flat();
    if (selectedFeaturesList.length === 0) {
      return NOISE_CLUSTER_COLOR;
    }
    // Check if track has any of the selected features above normalized threshold
    const matchingFeatures = selectedFeaturesList.filter(feature => {
      // Style features
      try {
        const styleFeatures = typeof track.style_features === 'string' 
          ? JSON.parse(track.style_features) 
          : track.style_features;
        if (styleFeatures && styleFeatures[feature]) {
          const v = parseFloat(styleFeatures[feature]);
          const minmax = featureMinMax[feature];
          if (minmax && minmax.max > minmax.min) {
            const norm = (v - minmax.min) / (minmax.max - minmax.min);
            if (norm >= highlightThreshold) return true;
          }
        }
      } catch (e) {}
      // Mood features
      try {
        const moodFeatures = typeof track.mood_features === 'string'
          ? JSON.parse(track.mood_features)
          : track.mood_features;
        if (moodFeatures && moodFeatures[feature]) {
          const v = parseFloat(moodFeatures[feature]);
          const minmax = featureMinMax[feature];
          if (minmax && minmax.max > minmax.min) {
            const norm = (v - minmax.min) / (minmax.max - minmax.min);
            if (norm >= highlightThreshold) return true;
          }
        }
      } catch (e) {}
      // Instrument features
      try {
        const instrumentFeatures = typeof track.instrument_features === 'string'
          ? JSON.parse(track.instrument_features)
          : track.instrument_features;
        if (instrumentFeatures && instrumentFeatures[feature]) {
          const v = parseFloat(instrumentFeatures[feature]);
          const minmax = featureMinMax[feature];
          if (minmax && minmax.max > minmax.min) {
            const norm = (v - minmax.min) / (minmax.max - minmax.min);
            if (norm >= highlightThreshold) return true;
          }
        }
      } catch (e) {}
      // Spectral features
      if (SPECTRAL_KEYWORDS.includes(feature) && track[feature] !== undefined) {
        const v = track[feature];
        const minmax = featureMinMax[feature];
        if (minmax && minmax.max > minmax.min) {
          const norm = (v - minmax.min) / (minmax.max - minmax.min);
          if (norm >= highlightThreshold) return true;
        }
      }
      // Genre (no normalization, just match)
      if (track.genre === feature) {
        return true;
      }
      return false;
    });
    if (matchingFeatures.length === 0) {
      return NOISE_CLUSTER_COLOR;
    }
    if (filterLogicMode === 'intersection' && matchingFeatures.length !== selectedFeaturesList.length) {
      return NOISE_CLUSTER_COLOR;
    }
    return HIGHLIGHT_COLOR;
  }, [selectedFeatures, filterLogicMode, highlightThreshold, featureMinMax]);

  // Update trackColors to use the new getDotColor function
  const trackColors = useMemo(() => {
    return plotData.map(track => {
      // Get the display title (filename without extension if title is unknown)
      const displayTitle = track.title === 'Unknown Title' && track.path ? 
        track.path.split('/').pop().replace(/\.[^/.]+$/, '') : 
        (track.title || 'Unknown Title');

      // Get the filename from path if available
      const filename = track.path ? track.path.split('/').pop().replace(/\.[^/.]+$/, '') : '';

      // More comprehensive search with exact matching
      const isSearchMatch = searchQuery && (
        // Exact match for display title
        displayTitle.toLowerCase() === searchQuery.toLowerCase() ||
        // Exact match for original title
        (track.title && track.title.toLowerCase() === searchQuery.toLowerCase()) ||
        // Exact match for filename
        filename.toLowerCase() === searchQuery.toLowerCase() ||
        // Exact match for artist
        (track.artist && track.artist.toLowerCase() === searchQuery.toLowerCase()) ||
        // Exact match for album
        (track.album && track.album.toLowerCase() === searchQuery.toLowerCase()) ||
        // Exact match for key
        (track.key && track.key.toLowerCase() === searchQuery.toLowerCase()) ||
        // Partial matches as fallback
        displayTitle.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (track.title && track.title.toLowerCase().includes(searchQuery.toLowerCase())) ||
        filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (track.artist && track.artist.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (track.album && track.album.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (track.key && track.key.toLowerCase().includes(searchQuery.toLowerCase()))
      );

      // If there's a search match, highlight in gold
      if (isSearchMatch) {
        return {
          id: track.id,
          color: '#FFD700',
          dominantFeature: null,
          isSearchMatch: true
        };
      }

      // Use the new getDotColor function for feature-based coloring
      return {
        id: track.id,
        color: getDotColor(track),
        dominantFeature: null,
        isSearchMatch: false
      };
    });
  }, [plotData, searchQuery, getDotColor]);

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

    // Position tooltip vertically aligned with the cursor
    let x = event.clientX + 5; // Minimal horizontal offset
    let y = event.clientY - tooltipHeight - 5; // Position above the cursor by default

    // If there's not enough space above, position below
    if (y < 10) {
      y = event.clientY + 5;
    }

    // Ensure tooltip stays within viewport bounds horizontally
    if (x + tooltipWidth > viewportWidth) {
      x = Math.max(10, event.clientX - tooltipWidth - 5);
    }

    // Calculate cursor position relative to tooltip width for waveform centering
    const cursorPositionRelative = (event.clientX - x) / tooltipWidth;

    // Get display title - use filename without suffix if title is "Unknown Title"
    const displayTitle = trackData.title === 'Unknown Title' && trackData.path ? 
      trackData.path.split('/').pop().replace(/\.[^/.]+$/, '') : 
      (trackData.title || 'Unknown Title');

    setTooltip({
      content: (
        <div style={{ maxWidth: tooltipWidth }}>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
            <img
              src={trackData.artwork_thumbnail_path || defaultArtwork}
              alt={`${trackData.artist || 'Unknown'} - ${displayTitle}`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = defaultArtwork;
                e.target.style.opacity = '0.7';
              }}
              style={{
                width: '80px',
                height: '80px',
                objectFit: 'cover',
                borderRadius: '4px',
                transition: 'opacity 0.2s ease',
                flexShrink: 0
              }}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{displayTitle}</div>
              <div style={{ marginBottom: '4px' }}>{trackData.artist || 'Unknown Artist'}</div>
              <div style={{ fontStyle: 'italic', marginBottom: '4px' }}>{trackData.album || 'Unknown Album'} ({trackData.year || 'N/A'})</div>
              <div style={{ marginBottom: '4px' }}>BPM: {trackData.bpm?.toFixed(1) || 'N/A'}, Key: {trackData.key || 'N/A'}</div>
              {selectedFeature && (
                <div style={{ marginBottom: '4px' }}>
                  {selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)}: {selectedFeature} 
                  {(() => {
                    let probability = 0;
                    if (selectedCategory === 'genre' || selectedCategory === 'style') {
                      try {
                        const features = typeof trackData.style_features === 'string' ? JSON.parse(trackData.style_features) : trackData.style_features;
                        if (features && typeof features === 'object') {
                          const matchingKey = Object.keys(features).find(key => {
                            const [genrePart, stylePart] = key.split('---');
                            return selectedCategory === 'genre' ? 
                              genrePart === selectedFeature : 
                              stylePart === selectedFeature;
                          });
                          if (matchingKey) {
                            probability = parseFloat(features[matchingKey]);
                          }
                        }
                      } catch (e) {}
                    } else if (selectedCategory === 'instrument') {
                      try {
                        const features = typeof trackData.instrument_features === 'string' ? JSON.parse(trackData.instrument_features) : trackData.instrument_features;
                        if (features && typeof features === 'object') {
                          probability = parseFloat(features[selectedFeature]);
                        }
                      } catch (e) {}
                    } else if (selectedCategory === 'mood' || selectedCategory === 'spectral') {
                      probability = trackData[selectedFeature];
                    }
                    return ` (${(probability * 100).toFixed(1)}%)`;
                  })()}
                </div>
              )}
              {trackData.tag1 && <div>Genre: {trackData.tag1} ({trackData.tag1_prob?.toFixed(2) || 'N/A'})</div>}
            </div>
          </div>
          {audioPath && (
            <div className="waveform-container" style={{ width: '100%', height: '40px' }}>
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
                  initialPosition={cursorPositionRelative}
                  seekTo={cursorPositionRelative}
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

  // Function to generate search suggestions
  const generateSuggestions = useCallback((query) => {
    if (!query || query.length < 2) {
      setSearchSuggestions([]);
      return;
    }

    const queryLower = query.toLowerCase();
    const suggestions = new Set();

    plotData.forEach(track => {
      // Get the display title and filename
      const displayTitle = track.title === 'Unknown Title' && track.path ? 
        track.path.split('/').pop().replace(/\.[^/.]+$/, '') : 
        (track.title || 'Unknown Title');
      const filename = track.path ? track.path.split('/').pop().replace(/\.[^/.]+$/, '') : '';

      // Add matching suggestions
      if (displayTitle.toLowerCase().includes(queryLower)) {
        suggestions.add(displayTitle);
      }
      if (track.title && track.title.toLowerCase().includes(queryLower)) {
        suggestions.add(track.title);
      }
      if (filename.toLowerCase().includes(queryLower)) {
        suggestions.add(filename);
      }
      if (track.artist && track.artist.toLowerCase().includes(queryLower)) {
        suggestions.add(track.artist);
      }
      if (track.album && track.album.toLowerCase().includes(queryLower)) {
        suggestions.add(track.album);
      }
      if (track.tag1 && track.tag1.toLowerCase().includes(queryLower)) {
        suggestions.add(track.tag1);
      }
      if (track.key && track.key.toLowerCase().includes(queryLower)) {
        suggestions.add(track.key);
      }
    });

    // Convert to array and sort by relevance (exact matches first, then partial matches)
    const sortedSuggestions = Array.from(suggestions)
      .sort((a, b) => {
        const aStartsWith = a.toLowerCase().startsWith(queryLower);
        const bStartsWith = b.toLowerCase().startsWith(queryLower);
        if (aStartsWith && !bStartsWith) return -1;
        if (!aStartsWith && bStartsWith) return 1;
        return a.localeCompare(b);
      })
      .slice(0, 5); // Limit to 5 suggestions

    setSearchSuggestions(sortedSuggestions);
  }, [plotData]);

  // Handle search input changes
  const handleSearchChange = useCallback((e) => {
    const newQuery = e.target.value;
    setSearchQuery(newQuery);
    generateSuggestions(newQuery);
    setShowSuggestions(true);
    setSelectedSuggestionIndex(-1);
  }, [generateSuggestions]);

  // Handle suggestion selection
  const handleSuggestionClick = useCallback((suggestion) => {
    setSearchQuery(suggestion);
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e) => {
    if (!showSuggestions) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedSuggestionIndex(prev => 
          prev < searchSuggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedSuggestionIndex(prev => prev > -1 ? prev - 1 : -1);
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedSuggestionIndex > -1) {
          handleSuggestionClick(searchSuggestions[selectedSuggestionIndex]);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
        break;
    }
  }, [showSuggestions, searchSuggestions, selectedSuggestionIndex, handleSuggestionClick]);

  // Handle clicks outside the search box
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchInputRef.current && !searchInputRef.current.contains(event.target) &&
          suggestionsRef.current && !suggestionsRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current || !plotData.length) return;

    // Clear any existing visualization
    d3.select(svgRef.current).selectAll("*").remove();

    // Create SVG container
    const svg = d3.select(svgRef.current)
      .attr("width", svgDimensions.width)
      .attr("height", svgDimensions.height)
      .attr("viewBox", `0 0 ${svgDimensions.width} ${svgDimensions.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    // Create main group for all elements
    const g = svg.append("g");

    // Initialize zoom behavior
    zoomBehaviorRef.current = d3.zoom()
      .scaleExtent([1, 200])
      .on("zoom", (event) => {
        setZoom(event.transform.k);
        setPan({ x: event.transform.x, y: event.transform.y });
        g.attr("transform", event.transform);
      });

    // Apply zoom behavior to SVG
    svg.call(zoomBehaviorRef.current);

    // Create dots
    const dots = g.selectAll("circle")
      .data(plotData)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", 4)
      .attr("fill", (d, i) => trackColors[i]?.color || NOISE_CLUSTER_COLOR)
      .attr("class", "track-dot")
      .style("transition", "none")
      .on("mouseover", (event, d) => handleMouseOver(d, event))
      .on("mouseout", handleMouseOut)
      .on("click", (event, d) => handleDotClick(d));

    // Store D3 container reference
    d3ContainerRef.current = { svg, g, dots };

    // Cleanup function
    return () => {
      if (d3ContainerRef.current) {
        d3ContainerRef.current.svg.selectAll("*").remove();
      }
    };
  }, [plotData, svgDimensions, trackColors]);

  // Update dots when trackColors change
  useEffect(() => {
    if (!d3ContainerRef.current?.dots) return;

    d3ContainerRef.current.dots
      .attr("fill", (d, i) => trackColors[i]?.color || NOISE_CLUSTER_COLOR);
  }, [trackColors]);

  // Handle feature selection
  const handleFeatureToggle = useCallback((category, feature) => {
    setSelectedFeatures(prev => {
      const newFeatures = { ...prev };
      const categoryFeatures = [...prev[category]];
      const index = categoryFeatures.indexOf(feature);
      
      if (index === -1) {
        categoryFeatures.push(feature);
      } else {
        categoryFeatures.splice(index, 1);
      }
      
      newFeatures[category] = categoryFeatures;
      return newFeatures;
    });
  }, []);

  // Get filter options from tracks
  const filterOptions = useMemo(() => {
    const options = {
      genre: [],
      style: [],
      mood: [],
      instrument: [],
      spectral: []
    };

    tracks.forEach(track => {
      // Process genre
      if (track.genre) {
        const genre = { name: track.genre, count: 1 };
        const existingGenre = options.genre.find(g => g.name === genre.name);
        if (existingGenre) existingGenre.count++;
        else options.genre.push(genre);
      }

      // Process style features
      try {
        const styleFeatures = typeof track.style_features === 'string' 
          ? JSON.parse(track.style_features) 
          : track.style_features;
        if (styleFeatures) {
          Object.entries(styleFeatures).forEach(([name, value]) => {
            const feature = { name, count: 1 };
            const existing = options.style.find(f => f.name === name);
            if (existing) existing.count++;
            else options.style.push(feature);
          });
        }
      } catch (e) {}

      // Process mood features
      try {
        const moodFeatures = typeof track.mood_features === 'string'
          ? JSON.parse(track.mood_features)
          : track.mood_features;
        if (moodFeatures) {
          Object.entries(moodFeatures).forEach(([name, value]) => {
            const feature = { name, count: 1 };
            const existing = options.mood.find(f => f.name === name);
            if (existing) existing.count++;
            else options.mood.push(feature);
          });
        }
      } catch (e) {}

      // Process instrument features
      try {
        const instrumentFeatures = typeof track.instrument_features === 'string'
          ? JSON.parse(track.instrument_features)
          : track.instrument_features;
        if (instrumentFeatures) {
          Object.entries(instrumentFeatures).forEach(([name, value]) => {
            const feature = { name, count: 1 };
            const existing = options.instrument.find(f => f.name === name);
            if (existing) existing.count++;
            else options.instrument.push(feature);
          });
        }
      } catch (e) {}

      // Process spectral features
      SPECTRAL_KEYWORDS.forEach(key => {
        if (track[key] !== undefined) {
          const feature = { name: key, count: 1 };
          const existing = options.spectral.find(f => f.name === key);
          if (existing) existing.count++;
          else options.spectral.push(feature);
        }
      });
    });

    // Sort all options by count (descending) and then by name
    Object.keys(options).forEach(category => {
      options[category].sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return a.name.localeCompare(b.name);
      });
    });

    return options;
  }, [tracks]);

  // Debounce threshold update for smooth slider
  useEffect(() => {
    if (thresholdDebounceRef.current) clearTimeout(thresholdDebounceRef.current);
    thresholdDebounceRef.current = setTimeout(() => {
      setHighlightThreshold(tempThreshold);
    }, 40); // 40ms debounce for smoothness
    return () => clearTimeout(thresholdDebounceRef.current);
  }, [tempThreshold]);

  if (loading) return <div className="track-visualizer-loading">Loading tracks and features...</div>;
  if (loading) return <div className="track-visualizer-loading">Loading tracks and features...</div>;
  if (error) return <div className="track-visualizer-error">Error: {error} <button onClick={fetchTracksData}>Try Reload</button></div>;
  if (plotData.length === 0 && !loading && tracks.length > 0) return <div className="track-visualizer-empty">Data processed, but no points to visualize. Check feature processing.</div>;
  if (plotData.length === 0 && !loading && tracks.length === 0) return <div className="track-visualizer-empty">No tracks data loaded.</div>;

  return (
    <div className="TrackVisualizer" ref={containerRef}>
      <div className="VisualizationContainer" ref={visualizationRef}>
        <div className="controls-panel sticky-controls">
          <div className="search-box" ref={searchInputRef}>
            <label htmlFor="trackSearch">Search Tracks:</label>
            <div style={{ position: 'relative' }}>
              <input
                id="trackSearch"
                type="text"
                value={searchQuery}
                onChange={handleSearchChange}
                onKeyDown={handleKeyDown}
                onFocus={() => setShowSuggestions(true)}
                placeholder="Search by title, filename, artist, album, genre, or key..."
                style={{
                  backgroundColor: DARK_MODE_SURFACE_ALT,
                  color: DARK_MODE_TEXT_PRIMARY,
                  border: `1px solid ${DARK_MODE_BORDER}`,
                  padding: '4px 8px',
                  borderRadius: '4px',
                  width: '300px'
                }}
              />
              {showSuggestions && searchSuggestions.length > 0 && (
                <div
                  ref={suggestionsRef}
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    backgroundColor: DARK_MODE_SURFACE_ALT,
                    border: `1px solid ${DARK_MODE_BORDER}`,
                    borderRadius: '4px',
                    marginTop: '4px',
                    zIndex: 1000,
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                  }}
                >
                  {searchSuggestions.map((suggestion, index) => (
                    <div
                      key={suggestion}
                      onClick={() => handleSuggestionClick(suggestion)}
                      style={{
                        padding: '8px 12px',
                        cursor: 'pointer',
                        backgroundColor: index === selectedSuggestionIndex ? 
                          adjustLuminance(DARK_MODE_SURFACE_ALT, 0.1) : 
                          'transparent',
                        color: DARK_MODE_TEXT_PRIMARY,
                        ':hover': {
                          backgroundColor: adjustLuminance(DARK_MODE_SURFACE_ALT, 0.1)
                        }
                      }}
                    >
                      {suggestion}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="threshold-slider">
            <label htmlFor="highlightThreshold">Confidence: {highlightThreshold}</label>
            <input
              id="highlightThreshold"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={highlightThreshold}
              onChange={e => setHighlightThreshold(Number(e.target.value))}
              style={{ marginLeft: '10px', width: '120px', accentColor: HIGHLIGHT_COLOR }}
            />
          </div>
          <div className="control-buttons">
            <button 
              onClick={handleReset} 
              className="reset-button"
              title="Reset zoom and pan to default view"
            >
              Reset View
            </button>
            <button
              onClick={() => setFilterLogicMode(prev => prev === 'intersection' ? 'union' : 'intersection')}
              className="filter-logic-button"
              title={`Current mode: ${filterLogicMode}. Click to switch between AND/OR logic`}
              style={{
                backgroundColor: filterLogicMode === 'intersection' ? HIGHLIGHT_COLOR : '#2196F3',
                color: 'white',
                border: 'none',
                padding: '4px 8px',
                borderRadius: '4px',
                cursor: 'pointer',
                marginLeft: '10px'
              }}
            >
              {filterLogicMode === 'intersection' ? 'AND Mode' : 'OR Mode'}
            </button>
          </div>
        </div>
        <div className="visualization-area" ref={viewModeRef}>
          <svg
            ref={svgRef}
            className="track-plot"
            aria-labelledby="plotTitle"
            role="graphics-document"
          >
            <title id="plotTitle">Track Similarity Plot</title>
          </svg>
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
        </div>
      </div>
      <div className="FilterPanelWrapper">
        <FilterPanel
          filterOptions={filterOptions}
          activeFilters={selectedFeatures}
          onToggleFilter={handleFeatureToggle}
          filterLogicMode={filterLogicMode}
          onToggleFilterLogicMode={() => setFilterLogicMode(prev => 
            prev === 'intersection' ? 'union' : 'intersection'
          )}
        />
      </div>
    </div>
  );
};

export default TrackVisualizer;