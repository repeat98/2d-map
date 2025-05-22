import React, { useState, useEffect, useMemo, useCallback } from 'react';
import './TrackVisualizer.scss'; // Make sure this SCSS file exists and is styled appropriately

// --- Constants ---
const SVG_WIDTH = window.innerWidth;
const SVG_HEIGHT = window.innerHeight;
const PADDING = 50;
const PCA_N_COMPONENTS = 2;
const HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE = 5;
const HDBSCAN_DEFAULT_MIN_SAMPLES = 3;
const TOOLTIP_OFFSET = 15;
const NOISE_CLUSTER_ID = -1;
const NOISE_CLUSTER_COLOR = '#cccccc';
const DEFAULT_CLUSTER_COLORS = [ // Kept for potential fallback or other uses
  '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
  '#008080', '#e6beff', '#9A6324', '#fffac8', '#800000',
  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
  '#54A0FF', '#F4D03F', '#1ABC9C', '#E74C3C', '#8E44AD'
];
const VIEW_BOX_VALUE = `0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`;
const PLACEHOLDER_IMAGE = '/placeholder.png';

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
    'genre': '#e6194B',      // Red
    'style': '#3cb44b',      // Green
    'spectral': '#4363d8',   // Blue
    'mood': '#f58231',       // Orange
    'instrument': '#911eb4', // Purple
};

const LUMINANCE_INCREMENT = 0.15;
const MAX_LUM_OFFSET = 0.6;


// --- Helper Functions ---

function calculateDistance(vec1, vec2) {
  if (!vec1 || !vec2) {
    console.warn('Missing vectors for distance calculation');
    return Infinity;
  }
  if (vec1.length !== vec2.length) {
    console.warn('Vectors have different lengths for distance calculation:', { len1: vec1.length, len2: vec2.length });
    return Infinity;
  }
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
    if (SPECTRAL_KEYWORDS.includes(lowerKeyName)) {
      return 'spectral';
    }
    if (MOOD_KEYWORDS.includes(lowerKeyName)) {
      return 'mood';
    }
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
      console.warn(`Failed to parse features for track ${trackId} (source: ${sourceCategory}) while getting keys:`, e, featureObj);
    }
  };

  // Process genre/style features
  tracks.forEach(track => {
    if (!track || !track.id) return;
    processFeatureSource(track.features, 'genre', track.id);
    processFeatureSource(track.style_features, 'style', track.id);
    processFeatureSource(track.instrument_features, 'instrument', track.id);
  });

  // Add spectral and mood features directly
  SPECTRAL_KEYWORDS.forEach(key => {
    featuresWithCategories.set(key, 'spectral');
  });

  MOOD_KEYWORDS.forEach(key => {
    featuresWithCategories.set(key, 'mood');
  });

  const categorizedFeatures = Array.from(featuresWithCategories.entries())
    .map(([name, category]) => ({ name, category }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return categorizedFeatures;
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
            if (!isNaN(num)) {
              mergedFeatures[key] = num;
            }
          }
        });
      }
    } catch (e) {
      console.warn(`Failed to parse features for track ${track?.id} during merge:`, e, featureObj);
    }
  };

  // Handle genre/style features
  parseAndMerge(track.features);
  parseAndMerge(track.style_features);
  parseAndMerge(track.instrument_features);

  // Handle spectral features
  const spectralFeatures = {
    'noisy': track.noisy,
    'tonal': track.tonal,
    'dark': track.dark,
    'bright': track.bright,
    'percussive': track.percussive,
    'smooth': track.smooth,
    'lufs': track.lufs
  };

  // Handle mood features
  const moodFeatures = {
    'happiness': track.happiness,
    'party': track.party,
    'aggressive': track.aggressive,
    'danceability': track.danceability,
    'relaxed': track.relaxed,
    'sad': track.sad,
    'engagement': track.engagement,
    'approachability': track.approachability
  };

  // Merge spectral features
  Object.entries(spectralFeatures).forEach(([key, value]) => {
    if (allFeatureNames.includes(key)) {
      const num = parseFloat(value);
      if (!isNaN(num)) {
        mergedFeatures[key] = num;
      }
    }
  });

  // Merge mood features
  Object.entries(moodFeatures).forEach(([key, value]) => {
    if (allFeatureNames.includes(key)) {
      const num = parseFloat(value);
      if (!isNaN(num)) {
        mergedFeatures[key] = num;
      }
    }
  });

  return allFeatureNames.map(key => mergedFeatures[key]);
}

function normalizeFeatures(featureVectors, featureCategories) {
  if (!featureVectors || featureVectors.length === 0) return [];

  const numSamples = featureVectors.length;
  if (numSamples === 0) return [];
  const numFeatures = featureVectors[0]?.length || 0;
  if (numFeatures === 0) return featureVectors.map(() => []);

  if (featureCategories.length !== numFeatures) {
    console.error(
        `Normalization Error: Mismatch between number of features (${numFeatures}) and categories (${featureCategories.length}). Providing defaults for categories.`
    );
    // Fallback if categories are mismatched, though this indicates a deeper issue.
    featureCategories = Array(numFeatures).fill('default');
  }


  const means = new Array(numFeatures).fill(0);
  const stdDevs = new Array(numFeatures).fill(0);

  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) {
      means[j] += vector[j] || 0;
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    means[j] /= numSamples;
  }

  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) {
      stdDevs[j] += Math.pow((vector[j] || 0) - means[j], 2);
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    stdDevs[j] = Math.sqrt(stdDevs[j] / numSamples);
  }

  return featureVectors.map(vector =>
    vector.map((value, j) => {
      const std = stdDevs[j];
      const mean = means[j];
      const normalizedValue = (std < 1e-10) ? 0 : ((value || 0) - mean) / std;
      const category = (j < featureCategories.length && featureCategories[j]) ? featureCategories[j] : 'default';
      const weight = CATEGORY_WEIGHTS[category] || CATEGORY_WEIGHTS['default'];
      return normalizedValue * weight;
    })
  );
}

function pca(processedData, nComponents = PCA_N_COMPONENTS) {
  if (!processedData || processedData.length === 0) return [];
  const numSamples = processedData.length;
  let numFeatures = processedData[0]?.length || 0;

  if (numFeatures === 0) {
    console.warn('PCA: Input data has no features.');
    return processedData.map(() => Array(nComponents).fill(0.5));
  }
  if (nComponents > numFeatures) {
    nComponents = numFeatures;
  }
   if (nComponents <= 0) {
    nComponents = numFeatures > 0 ? 1 : 0;
    if (nComponents === 0) return processedData.map(() => []);
  }
  if (numSamples <= 1) {
    return processedData.map(() => Array(nComponents).fill(0.5));
  }

  const means = processedData[0].map((_, colIndex) => {
    let sum = 0;
    for (let i = 0; i < numSamples; i++) sum += processedData[i][colIndex];
    return sum / numSamples;
  });
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
    if (norm < 1e-10) vector = Array(n).fill(0);
    else vector = vector.map(v => v / norm);
    if (vector.every(v => v === 0) && n > 0) vector[0] = 1;

    for (let iter = 0; iter < numIterations; iter++) {
      let newVector = Array(n).fill(0);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          newVector[r] += (matrix[r]?.[c] || 0) * vector[c];
        }
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
    principalComponents.push(pc);

    if (k < nComponents - 1 && pc.length > 0) {
      let lambda = 0;
      const C_v = Array(numFeatures).fill(0);
      for (let i = 0; i < numFeatures; i++) {
        for (let j = 0; j < numFeatures; j++) C_v[i] += (tempCovarianceMatrix[i]?.[j] || 0) * pc[j];
        lambda += pc[i] * C_v[i];
      }
      const newTempCovMatrix = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
      for (let i = 0; i < numFeatures; i++) {
        for (let j = 0; j < numFeatures; j++) {
          newTempCovMatrix[i][j] = (tempCovarianceMatrix[i]?.[j] || 0) - lambda * pc[i] * pc[j];
        }
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

  if (projected.length === 0 || nComponents === 0) {
    return projected.map(() => Array(nComponents).fill(0.5));
  }
  const actualNumOutputComponents = projected[0]?.length || 0;
  if (actualNumOutputComponents === 0) {
    return projected.map(() => Array(nComponents).fill(0.5));
  }

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

  if (n < minClusterSize && n > 0) {
    return Array(n).fill(NOISE_CLUSTER_ID);
  }

  function computeMutualReachabilityDistance() {
    const distances = Array(n).fill(null).map(() => Array(n).fill(0));
    const coreDistances = Array(n).fill(Infinity);
    if (n === 0) return { distances, coreDistances };

    for (let i = 0; i < n; i++) {
      if (n <= 1 || minSamples >= n) {
        coreDistances[i] = Infinity;
        continue;
      }
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
    if (n === 0 || (mst.length === 0 && n > 0 && minClusterSize > 1)) return labels;
    if (n > 0 && minClusterSize === 1) return Array(n).fill(0).map((_,i)=>i);

    let currentClusterId = 0;
    const parent = Array(n).fill(0).map((_, i) => i);
    const componentSize = Array(n).fill(1);

    function findSet(i) {
      if (parent[i] === i) return i;
      return parent[i] = findSet(parent[i]);
    }
    function uniteSets(i, j) {
      let rootI = findSet(i), rootJ = findSet(j);
      if (rootI !== rootJ) {
        if (componentSize[rootI] < componentSize[rootJ]) [rootI, rootJ] = [rootJ, rootI];
        parent[rootJ] = rootI;
        componentSize[rootI] += componentSize[rootJ];
        return true;
      }
      return false;
    }
    const sortedMSTEdges = mst.sort((a, b) => a[2] - b[2]);
    for (const edge of sortedMSTEdges) uniteSets(edge[0], edge[1]);

    const rootToClusterId = new Map();
    for(let i = 0; i < n; i++){
        const root = findSet(i);
        if(componentSize[root] >= minClusterSize){
            if(!rootToClusterId.has(root)) rootToClusterId.set(root, currentClusterId++);
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
  const [featureMetadata, setFeatureMetadata] = useState({ names: [], categories: [] });
  const [styleColors, setStyleColors] = useState(new Map());
  const [topNThreshold, setTopNThreshold] = useState(10);
  
  // Add zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setPlotData(prevData => {
        if (!prevData || prevData.length === 0) return prevData;
        return prevData.map(point => ({
          ...point,
          x: PADDING + point.x * (window.innerWidth - 2 * PADDING),
          y: PADDING + point.y * (window.innerHeight - 2 * PADDING)
        }));
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle zoom with mouse wheel
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY;
    const zoomFactor = delta > 0 ? 0.9 : 1.1;
    const newZoom = Math.min(Math.max(zoom * zoomFactor, 0.1), 10);
    
    // Calculate mouse position relative to SVG
    const svgRect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;
    const mouseY = e.clientY - svgRect.top;
    
    // Calculate new pan to zoom towards mouse position
    const newPan = {
      x: pan.x - (mouseX - pan.x) * (zoomFactor - 1),
      y: pan.y - (mouseY - pan.y) * (zoomFactor - 1)
    };
    
    setZoom(newZoom);
    setPan(newPan);
  }, [zoom, pan]);

  // Handle panning
  const handleMouseDown = useCallback((e) => {
    if (e.button === 0) { // Left mouse button
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Reset zoom and pan
  const handleReset = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const adjustLuminance = (hex, lum) => {
    hex = String(hex).replace(/[^0-9a-f]/gi, '');
    if (hex.length < 6) {
      hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    lum = lum || 0;
    let rgb = "#", c, i;
    for (i = 0; i < 3; i++) {
      c = parseInt(hex.substr(i * 2, 2), 16);
      c = Math.round(Math.min(Math.max(0, c * (1 + lum)), 255));
      rgb += ("00" + c.toString(16)).substr(c.toString(16).length);
    }
    return rgb;
  };

  useEffect(() => {
    if (!tracks || tracks.length === 0 || !selectedCategory || !featureMetadata.names || featureMetadata.names.length === 0) {
      setStyleColors(new Map());
      return;
    }

    const baseColorForCategory = CATEGORY_BASE_COLORS[selectedCategory] || NOISE_CLUSTER_COLOR;
    const featureFrequencies = new Map();

    tracks.forEach(track => {
      if (!track) return;

      switch(selectedCategory) {
        case 'genre':
        case 'style':
          const featuresGS = track.features;
          if (featuresGS) {
            try {
              const parsed = typeof featuresGS === 'string' ? JSON.parse(featuresGS) : featuresGS;
              if (typeof parsed === 'object' && parsed !== null) {
                Object.entries(parsed).forEach(([key, value]) => {
                  const probability = parseFloat(value);
                  if (isNaN(probability) || probability <= 0) return;

                  const [genrePart, stylePart] = key.split('---');
                  const featureName = selectedCategory === 'genre' ? genrePart : stylePart;
                  if (featureName) {
                    featureFrequencies.set(featureName, (featureFrequencies.get(featureName) || 0) + probability);
                  }
                });
              }
            } catch (e) {
              console.warn(`Failed to parse genre/style features for track ${track.id}:`, e, featuresGS);
            }
          }
          break;

        case 'mood':
          MOOD_KEYWORDS.forEach(key => {
            const value = track[key];
            if (typeof value === 'number' && !isNaN(value)) {
              featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + value);
            }
          });
          break;

        case 'spectral':
          SPECTRAL_KEYWORDS.forEach(key => {
            const value = track[key];
            if (typeof value === 'number' && !isNaN(value)) {
              featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + value);
            }
          });
          break;

        case 'instrument':
          const instrumentF = track.instrument_features;
          if (instrumentF) {
            try {
              const parsed = typeof instrumentF === 'string' ? JSON.parse(instrumentF) : instrumentF;
              if (typeof parsed === 'object' && parsed !== null) {
                Object.entries(parsed).forEach(([key, value]) => {
                  const score = parseFloat(value);
                  if (!isNaN(score) && score > 0) {
                    featureFrequencies.set(key, (featureFrequencies.get(key) || 0) + score);
                  }
                });
              }
            } catch (e) {
              console.warn(`Failed to parse instrument features for track ${track.id}:`, e, instrumentF);
            }
          }
          break;
      }
    });

    const sortedFeatures = Array.from(featureFrequencies.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topNThreshold);

    const newStyleColors = new Map();
    sortedFeatures.forEach(([featureName], index) => {
      let luminanceFactor = 0;
      if (index > 0) {
        const magnitude = Math.min(Math.ceil(index / 2) * LUMINANCE_INCREMENT, MAX_LUM_OFFSET);
        luminanceFactor = (index % 2 === 1) ? magnitude : -magnitude;
      }
      const shadedColor = adjustLuminance(baseColorForCategory, luminanceFactor);
      newStyleColors.set(featureName, shadedColor);
    });

    setStyleColors(newStyleColors);
  }, [tracks, selectedCategory, featureMetadata, topNThreshold]);


  const clusterColors = useMemo(() => {
    if (!plotData || plotData.length === 0 || styleColors.size === 0) {
      const fallbackMap = {};
       if(plotData.length > 0) {
        const uniqueClusters = [...new Set(plotData.map(p => p.cluster))];
        uniqueClusters.forEach(id => fallbackMap[id] = NOISE_CLUSTER_COLOR);
       }
      return fallbackMap;
    }

    const uniqueClusters = [...new Set(plotData.map(p => p.cluster))].sort((a,b) => a-b);
    const newClusterColors = {};

    uniqueClusters.forEach(clusterId => {
      if (clusterId === NOISE_CLUSTER_ID) {
        newClusterColors[clusterId] = NOISE_CLUSTER_COLOR;
        return;
      }

      const clusterTracks = plotData.filter(p => p.cluster === clusterId);
      const clusterFeatureValues = new Map(); // featureName -> sum of values in this cluster

      clusterTracks.forEach(track => {
        if (!track) return;
        switch(selectedCategory) {
          case 'genre':
          case 'style':
            const featuresGS = track.features;
            if (featuresGS) {
              try {
                const parsed = typeof featuresGS === 'string' ? JSON.parse(featuresGS) : featuresGS;
                if (typeof parsed === 'object' && parsed !== null) {
                  Object.entries(parsed).forEach(([key, value]) => {
                    const [genrePart, stylePart] = key.split('---');
                    const featureName = selectedCategory === 'genre' ? genrePart : stylePart;
                    if (featureName && styleColors.has(featureName)) {
                      const score = parseFloat(value);
                      if (!isNaN(score) && score > 0) {
                        clusterFeatureValues.set(featureName, (clusterFeatureValues.get(featureName) || 0) + score);
                      }
                    }
                  });
                }
              } catch (e) { /* console.warn(...) */ }
            }
            break;

          case 'mood':
            MOOD_KEYWORDS.forEach(key => {
              if (styleColors.has(key)) {
                const value = track[key];
                if (typeof value === 'number' && !isNaN(value)) {
                  clusterFeatureValues.set(key, (clusterFeatureValues.get(key) || 0) + value);
                }
              }
            });
            break;

          case 'spectral':
            SPECTRAL_KEYWORDS.forEach(key => {
              if (styleColors.has(key)) {
                const value = track[key];
                if (typeof value === 'number' && !isNaN(value)) {
                  clusterFeatureValues.set(key, (clusterFeatureValues.get(key) || 0) + value);
                }
              }
            });
            break;

          case 'instrument':
            const instrumentF = track.instrument_features;
            if (instrumentF) {
              try {
                const parsed = typeof instrumentF === 'string' ? JSON.parse(instrumentF) : instrumentF;
                if (typeof parsed === 'object' && parsed !== null) {
                  Object.entries(parsed).forEach(([key, value]) => {
                    if (styleColors.has(key)) {
                      const score = parseFloat(value);
                      if (!isNaN(score) && score > 0) {
                        clusterFeatureValues.set(key, (clusterFeatureValues.get(key) || 0) + score);
                      }
                    }
                  });
                }
              } catch (e) { /* console.warn(...) */ }
            }
            break;
        }
      });

      // Find the dominant feature and its relative strength
      let dominantFeature = null;
      let maxValue = 0;
      let totalValue = 0;

      clusterFeatureValues.forEach((value, feature) => {
        totalValue += value;
        if (value > maxValue) {
          maxValue = value;
          dominantFeature = feature;
        }
      });

      if (dominantFeature) {
        // Calculate relative strength (0 to 1) of the dominant feature
        const relativeStrength = totalValue > 0 ? maxValue / totalValue : 0;
        // Map relative strength to a luminance offset (-0.3 to 0.3)
        const luminanceOffset = (relativeStrength - 0.5) * 0.6;
        const baseColor = styleColors.get(dominantFeature) || NOISE_CLUSTER_COLOR;
        newClusterColors[clusterId] = adjustLuminance(baseColor, luminanceOffset);
      } else {
        newClusterColors[clusterId] = DEFAULT_CLUSTER_COLORS[clusterId % DEFAULT_CLUSTER_COLORS.length] || NOISE_CLUSTER_COLOR;
      }
    });

    return newClusterColors;
  }, [plotData, styleColors, selectedCategory, tracks, featureMetadata, topNThreshold]);

  useEffect(() => {
    const fetchTracks = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://localhost:3000/tracks');
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        const rawData = await response.json();
        if (!Array.isArray(rawData)) {
            console.error("Fetched data is not an array:", rawData);
            throw new Error("Invalid data format: Expected an array of tracks.");
        }

        const keysWithCats = getAllFeatureKeysAndCategories(rawData);
        const featureNames = keysWithCats.map(kc => kc.name);
        const featureCats = keysWithCats.map(kc => kc.category);

        if (featureNames.length === 0) {
            console.warn("No features found across all tracks.");
        }
        setFeatureMetadata({ names: featureNames, categories: featureCats });

        const parsedTracks = rawData.map(track => {
          if (!track || typeof track !== 'object' || !track.id) {
            console.warn('Skipping invalid track data entry:', track);
            return null;
          }
          try {
            const featureVector = mergeFeatureVectors(track, featureNames);
            return { ...track, featureVector };
          } catch (e) {
            console.warn(`Error processing features for track ${track.id}:`, e);
            return null;
          }
        }).filter(Boolean);

        setTracks(parsedTracks);

      } catch (err) {
        console.error("Failed to fetch or process tracks:", err);
        setError(err.message || 'An unknown error occurred.');
        setTracks([]);
        setFeatureMetadata({ names: [], categories: [] });
      } finally {
        setLoading(false);
      }
    };
    fetchTracks();
  }, []);

  useEffect(() => {
    if (loading || tracks.length === 0) {
      setPlotData([]);
      return;
    }
    if (featureMetadata.names.length === 0 && tracks.length > 0) {
        console.warn("Tracks available, but feature metadata missing. Plotting may be incorrect.");
        // Potentially set an error or allow processing with defaults if featureMetadata is crucial and missing
        return;
    }
    if (tracks.some(t => !t.featureVector)) {
        console.warn("Some tracks are missing feature vectors. Plotting may be incomplete or incorrect.");
        // Potentially filter these tracks out or handle them before normalization
    }


    // console.log("Processing tracks for visualization...");
    const validTracksForProcessing = tracks.filter(t => t.featureVector && t.featureVector.length === featureMetadata.names.length);

    if (validTracksForProcessing.length !== tracks.length) {
        console.warn(`Filtered out ${tracks.length - validTracksForProcessing.length} tracks due to missing or mismatched feature vectors.`);
    }
    if (validTracksForProcessing.length === 0 && tracks.length > 0) {
        setError("No tracks have valid feature vectors for processing.");
        setPlotData([]);
        return;
    }
     if (validTracksForProcessing.length === 0) {
        setPlotData([]);
        return;
    }


    const featureVectors = validTracksForProcessing.map(t => t.featureVector);

    // Ensure featureMetadata.categories corresponds to featureMetadata.names
    // This should be guaranteed by getAllFeatureKeysAndCategories and mergeFeatureVectors if implemented correctly
    const processedFeatureData = normalizeFeatures(featureVectors, featureMetadata.categories);
    const clusterLabels = hdbscan(processedFeatureData, HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE, HDBSCAN_DEFAULT_MIN_SAMPLES);
    const projectedData = pca(processedFeatureData, PCA_N_COMPONENTS);

    const newPlotData = validTracksForProcessing.map((track, index) => {
      const p = (projectedData && index < projectedData.length && projectedData[index]?.length === PCA_N_COMPONENTS)
                ? projectedData[index]
                : [0.5, 0.5]; // Fallback if PCA fails for a point
      return {
        ...track,
        x: PADDING + p[0] * (SVG_WIDTH - 2 * PADDING),
        y: PADDING + p[1] * (SVG_HEIGHT - 2 * PADDING),
        cluster: clusterLabels[index] ?? NOISE_CLUSTER_ID,
      };
    });

    setPlotData(newPlotData);
    // console.log("Plot data prepared.");

  }, [tracks, featureMetadata, loading]); // Removed PCA_N_COMPONENTS, HDBSCAN_... as they are constants

  const handleMouseOver = useCallback((trackData, event) => {
    setTooltip({
      content: (
        <>
          <img
            src={trackData.artwork_thumbnail_path ? `file://${trackData.artwork_thumbnail_path}` : PLACEHOLDER_IMAGE}
            alt={`${trackData.artist || 'Unknown Artist'} - ${trackData.title || 'Unknown Title'}`}
            onError={(e) => { e.target.onerror = null; e.target.src = PLACEHOLDER_IMAGE; }}
            style={{ width: '80px', height: '80px', objectFit: 'cover', marginRight: '10px', float: 'left', border: '1px solid #ccc' }}
          />
          <div style={{ overflow: 'hidden'}}>
            <div><strong>{trackData.title || 'Unknown Title'}</strong></div>
            <div>{trackData.artist || 'Unknown Artist'}</div>
            <div><em>{trackData.album || 'Unknown Album'} ({trackData.year || 'N/A'})</em></div>
            <div>BPM: {trackData.bpm?.toFixed(1) || 'N/A'}, Key: {trackData.key || 'N/A'}</div>
            {trackData.tag1 && <div>Genre: {trackData.tag1} ({trackData.tag1_prob?.toFixed(2) || 'N/A'})</div>}
          </div>
        </>
      ),
      x: event.clientX + TOOLTIP_OFFSET,
      y: event.clientY + TOOLTIP_OFFSET,
    });
  }, []);

  const handleMouseOut = useCallback(() => setTooltip(null), []);
  const handleDotClick = useCallback((trackData) => console.log("Clicked track:", trackData.id, trackData.title), []);

  if (loading) return <div className="track-visualizer-loading">Loading tracks and features...</div>;
  if (error) return <div className="track-visualizer-error">Error: {error} <button onClick={() => { setError(null); setLoading(true); fetchTracks(); /* or window.location.reload() */ }}>Try Reload</button></div>;
  if (plotData.length === 0 && !loading && tracks.length > 0) return <div className="track-visualizer-empty">Data processed, but no points to visualize. Check console for errors or warnings.</div>;
  if (plotData.length === 0 && !loading && tracks.length === 0) return <div className="track-visualizer-empty">No tracks data loaded.</div>;


  return (
    <div className="track-visualizer-container" style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <h3>Track Similarity Visualization (HDBSCAN & PCA)</h3>
      <div className="controls-panel">
        <div className="category-toggle">
          <label>Color by Category: </label>
          {Object.entries(CATEGORY_BASE_COLORS).map(([categoryKey, colorValue]) => (
            <button
              key={categoryKey}
              onClick={() => setSelectedCategory(categoryKey)}
              className={selectedCategory === categoryKey ? 'active' : ''}
              style={{
                backgroundColor: selectedCategory === categoryKey ? colorValue : '#f0f0f0',
                color: selectedCategory === categoryKey ? (parseInt(colorValue.slice(1,3),16)*0.299 + parseInt(colorValue.slice(3,5),16)*0.587 + parseInt(colorValue.slice(5,7),16)*0.114 > 186 ? '#000000' : '#ffffff'): '#333333',
                border: `2px solid ${selectedCategory === categoryKey ? adjustLuminance(colorValue, -0.3) : colorValue}`,
                margin: '0 5px',
                padding: '5px 10px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: selectedCategory === categoryKey ? 'bold' : 'normal',
              }}
            >
              {categoryKey.charAt(0).toUpperCase() + categoryKey.slice(1)}
            </button>
          ))}
        </div>
        <div className="threshold-control">
          <label htmlFor="topNInput">Top N Features: </label>
          <input
            id="topNInput"
            type="number"
            min="1"
            max="50"
            value={topNThreshold}
            onChange={(e) => setTopNThreshold(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
            style={{
              width: '60px',
              padding: '4px',
              marginLeft: '5px',
              borderRadius: '4px',
              border: '1px solid #ccc'
            }}
          />
        </div>
        <button 
          onClick={handleReset}
          style={{
            marginLeft: '10px',
            padding: '5px 10px',
            borderRadius: '4px',
            border: '1px solid #ccc',
            backgroundColor: '#f0f0f0',
            cursor: 'pointer'
          }}
        >
          Reset View
        </button>
      </div>
      <p className="info-text">
        Tracks clustered by audio feature similarity. Colors represent the top {topNThreshold} dominant features
        of the '{selectedCategory}' category within each cluster. Shades indicate feature prominence. Gray dots are noise.
        <br />
        <small>Use mouse wheel to zoom, drag to pan, and click Reset View to return to default view.</small>
      </p>
      <div className="visualization-area" style={{ width: '100%', height: 'calc(100vh - 150px)', position: 'relative' }}>
        <svg 
          className="track-plot" 
          viewBox={VIEW_BOX_VALUE} 
          preserveAspectRatio="xMidYMid meet" 
          aria-labelledby="plotTitle" 
          role="graphics-document"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ 
            width: '100%', 
            height: '100%',
            cursor: isDragging ? 'grabbing' : 'grab'
          }}
        >
          <title id="plotTitle">Track Similarity Plot</title>
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {plotData.map(track => (
              <circle
                key={track.id || String(Math.random())}
                cx={track.x}
                cy={track.y}
                r={6}
                fill={clusterColors[track.cluster] || NOISE_CLUSTER_COLOR}
                onMouseMove={(e) => handleMouseOver(track, e)}
                onMouseOut={handleMouseOut}
                onClick={() => handleDotClick(track)}
                className="track-dot"
                tabIndex={0}
                aria-label={`Track: ${track.title || 'Unknown'} by ${track.artist || 'Unknown'}, Cluster: ${track.cluster === NOISE_CLUSTER_ID ? 'Noise' : 'C' + (track.cluster + 1)}`}
              />
            ))}
          </g>
        </svg>
        {tooltip && (
          <div className="track-tooltip" style={{ top: tooltip.y, left: tooltip.x }} role="tooltip">
            {tooltip.content}
          </div>
        )}
        <div className="legend" style={{ position: 'absolute', right: '20px', top: '20px', backgroundColor: 'rgba(255, 255, 255, 0.9)', padding: '10px', borderRadius: '4px', maxHeight: 'calc(100% - 40px)', overflowY: 'auto' }}>
          <h4>{selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)} Feature Legend (Top {Math.min(topNThreshold, styleColors.size, 15)})</h4>
          <div className="style-legend">
            {Array.from(styleColors.entries())
              .sort((a, b) => a[0].localeCompare(b[0]))
              .slice(0, 15)
              .map(([feature, color]) => (
                <div key={feature} className="legend-item">
                  <span className="legend-color-swatch" style={{ backgroundColor: color }}></span>
                  {feature}
                </div>
              ))}
              {styleColors.size > 15 && <div className="legend-item">...and {styleColors.size - 15} more</div>}
              {styleColors.size === 0 && <div className="legend-item">No dominant features found for this category.</div>}
          </div>
          <div className="cluster-legend">
            <h4>Clusters (Dominant Feature Color)</h4>
            {Object.entries(clusterColors)
              .filter(([clusterIdString]) => parseInt(clusterIdString) !== NOISE_CLUSTER_ID)
              .sort(([idA], [idB]) => parseInt(idA) - parseInt(idB))
              .map(([clusterIdString, color]) => (
                <div key={clusterIdString} className="legend-item">
                  <span className="legend-color-swatch" style={{ backgroundColor: color }}></span>
                  Cluster {parseInt(clusterIdString) + 1}
                </div>
              ))}
            {plotData.some(p => p.cluster === NOISE_CLUSTER_ID) && (
              <div className="legend-item">
                <span className="legend-color-swatch" style={{ backgroundColor: NOISE_CLUSTER_COLOR }}></span>
                Noise
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrackVisualizer;