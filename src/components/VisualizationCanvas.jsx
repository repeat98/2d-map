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
import { UMAP } from 'umap-js'; // Updated import to use named import
import { kmeans } from 'ml-kmeans';

// --- FEATURE CATEGORIES ---
const FEATURE_CATEGORIES = {
  MOOD: 'Mood',
  SPECTRAL: 'Spectral',
  STYLE: 'Style',
  INSTRUMENT: 'Instrument'
};

// Fixed colors for each feature category
const CATEGORY_COLORS = {
  [FEATURE_CATEGORIES.MOOD]: 0x3498db,      // Blue
  [FEATURE_CATEGORIES.SPECTRAL]: 0x2ecc71,  // Green
  [FEATURE_CATEGORIES.STYLE]: 0xe74c3c,     // Red
  [FEATURE_CATEGORIES.INSTRUMENT]: 0xf1c40f  // Yellow
};

// Add improved feature weighting constants
const CATEGORY_WEIGHTS = {
  'genre': 4.0,        // High weight
  'style': 4.0,        // High weight
  'spectral': 2.0,     // Medium weight
  'mood': 1.0,         // Lower weight
  'instrument': 0.5,   // Even lower weight
  'default': 0.2,      // Fallback for uncategorized or less important features
};

// Keywords to identify spectral features. Case-insensitive.
const SPECTRAL_KEYWORDS = [
  'spectral', 'mfcc', 'chroma', 'tempo', 'bpm', 'loudness',
  'energy', 'zcr', 'rms', 'flux', 'rolloff', 'centroid',
  'bandwidth', 'contrast', 'flatness', 'onset', 'harmonic', 'percussive'
];

// HDBSCAN constants
const HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE = 3; // Reduced from 5 to allow smaller clusters
const HDBSCAN_DEFAULT_MIN_SAMPLES = 2; // Reduced from 3 to be more sensitive to local structure
const NOISE_CLUSTER_ID = -1;
const NOISE_CLUSTER_COLOR = 0xcccccc;

const coreFeaturesConfig = [
  { value: 'bpm', label: 'BPM', axisTitleStyle: { fill: 0xe74c3c, fontWeight: 'bold' }, isNumeric: true, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'key', label: 'Key', axisTitleStyle: { fill: 0xe74c3c, fontWeight: 'bold' }, isNumeric: true, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'danceability', label: 'Danceability', axisTitleStyle: { fill: 0x3498db }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'happiness', label: 'Happiness', axisTitleStyle: { fill: 0xf1c40f }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'party', label: 'Party Vibe', isNumeric: true, axisTitleStyle: { fill: 0x9b59b6 }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'aggressive', label: 'Aggressiveness', axisTitleStyle: { fill: 0xc0392b }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'relaxed', label: 'Relaxed Vibe', axisTitleStyle: { fill: 0x2ecc71 }, isNumeric: true, category: FEATURE_CATEGORIES.MOOD },
  { value: 'sad', label: 'Sadness', isNumeric: true, axisTitleStyle: { fill: 0x7f8c8d }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'engagement', label: 'Engagement', isNumeric: true, axisTitleStyle: { fill: 0x1abc9c }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'approachability', label: 'Approachability', isNumeric: true, axisTitleStyle: { fill: 0x34495e }, category: FEATURE_CATEGORIES.MOOD },
  { value: 'noisy', label: 'Noisy', isNumeric: true, axisTitleStyle: { fill: 0x16a085 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'tonal', label: 'Tonal', isNumeric: true, axisTitleStyle: { fill: 0x27ae60 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'dark', label: 'Dark', isNumeric: true, axisTitleStyle: { fill: 0x2980b9 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'bright', label: 'Bright', isNumeric: true, axisTitleStyle: { fill: 0x8e44ad }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'percussive', label: 'Percussive', isNumeric: true, axisTitleStyle: { fill: 0xf39c12 }, category: FEATURE_CATEGORIES.SPECTRAL },
  { value: 'smooth', label: 'Smooth', isNumeric: true, axisTitleStyle: { fill: 0xd35400 }, category: FEATURE_CATEGORIES.SPECTRAL },
];

// For Similarity Mode feature preparation
const SIMILARITY_EXPECTED_FEATURES = [
    // Mood features (higher weight for emotional characteristics)
    { name: 'happiness', weight: 1.2, category: FEATURE_CATEGORIES.MOOD },
    { name: 'party', weight: 1.2, category: FEATURE_CATEGORIES.MOOD },
    { name: 'aggressive', weight: 1.2, category: FEATURE_CATEGORIES.MOOD },
    { name: 'danceability', weight: 0.8, category: FEATURE_CATEGORIES.MOOD }, // Lower weight as it's often biased
    { name: 'relaxed', weight: 1.2, category: FEATURE_CATEGORIES.MOOD },
    { name: 'sad', weight: 1.2, category: FEATURE_CATEGORIES.MOOD },
    { name: 'engagement', weight: 1.0, category: FEATURE_CATEGORIES.MOOD },
    { name: 'approachability', weight: 1.0, category: FEATURE_CATEGORIES.MOOD },
    // Spectral features (higher weight for timbral characteristics)
    { name: 'noisy', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'tonal', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'dark', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'bright', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'percussive', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'smooth', weight: 1.3, category: FEATURE_CATEGORIES.SPECTRAL },
    { name: 'key', weight: 1.0, category: FEATURE_CATEGORIES.SPECTRAL, isKey: true }, // Consider cyclical encoding later
    { name: 'bpm', weight: 1.0, category: FEATURE_CATEGORIES.SPECTRAL, isBpm: true }  // Consider log transform later
];


const FIXED_STYLE_THRESHOLD = 0.0;
const PADDING = 70; const AXIS_COLOR = 0xAAAAAA; const TEXT_COLOR = 0xE0E0E0;
const DOT_RADIUS = 5; const DOT_RADIUS_HOVER = 7; const DEFAULT_DOT_COLOR = 0xFFFFFF;
const HIGHLIGHT_COLORS = [0x87CEFA, 0x32CD32, 0xFF7F50, 0xDAA520, 0xBA55D3, 0xFF69B4, 0x00CED1]; // For X/Y mode highlighting

const TOOLTIP_BG_COLOR = 0x333333; const TOOLTIP_TEXT_COLOR = 0xFFFFFF;
const TOOLTIP_PADDING = 10; const COVER_ART_SIZE = 80;
const MIN_ZOOM = 1; const MAX_ZOOM = 5; const ZOOM_SENSITIVITY = 0.0005;
const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;

// HIERARCHY_LEVELS is no longer used for primary clustering with HDBSCAN
// const HIERARCHY_LEVELS = [
//   { name: 'Spectral', kConfig: {min: 2, max: 4}, category: FEATURE_CATEGORIES.SPECTRAL, labelPrefix: 'Spec' },
//   { name: 'Mood', kConfig: {min: 2, max: 4}, category: FEATURE_CATEGORIES.MOOD,     labelPrefix: 'Mood' },
//   { name: 'Instrument', kConfig: {min: 2, max: 3}, category: FEATURE_CATEGORIES.INSTRUMENT, labelPrefix: 'Inst' },
// ];

const PROJECTION_METHODS = {
    PCA: 'PCA',
    UMAP: 'UMAP'
};

const generateClusterColors = (numColors) => {
  const colors = [];
  if (numColors <= 0) return [0xCCCCCC]; // Default grey if no clusters
  // Special color for noise/outliers if clusterId is -1
  colors.push(0x808080); // Grey for noise (index 0, assuming noise id maps to this)

  for (let i = 0; i < numColors; i++) { // numColors here means distinct non-noise clusters
    const hue = (i * (360 / numColors)) % 360;
    const hex = hslToHex(hue, 75, 65); 
    colors.push(parseInt(hex.replace('#', ''), 16));
  }
  return colors;
};

function hslToHex(h, s, l) {
  l /= 100;
  const a = s * Math.min(l, 1 - l) / 100;
  const f = n => {
const generateClusterColors = (clusters) => {
  // Base colors for main genres
  const genreColors = {
    'Rock': 0xe74c3c,      // Red
    'Pop': 0x3498db,       // Blue
    'Electronic': 0x2ecc71, // Green
    'Hip Hop': 0xf1c40f,    // Yellow
    'Jazz': 0x9b59b6,      // Purple
    'Classical': 0x1abc9c,  // Turquoise
    'Folk': 0xe67e22,      // Orange
    'R&B': 0x34495e,       // Dark Blue
    'Country': 0x16a085,   // Dark Green
    'Metal': 0x7f8c8d,     // Grey
    'Blues': 0xc0392b,     // Dark Red
    'Reggae': 0x27ae60,    // Forest Green
    'Soul': 0x8e44ad,      // Deep Purple
    'Funk': 0xd35400,      // Burnt Orange
    'Punk': 0x2980b9,      // Steel Blue
    'Ambient': 0x95a5a6,   // Light Grey
    'Noise / Outliers': 0x808080, // Grey for noise
    'Unprocessed': 0x808080,      // Grey for unprocessed
  };

  // Create a map of cluster IDs to colors
  const colorMap = new Map();
  
  clusters.forEach(cluster => {
    if (cluster.id === -1) {
      colorMap.set(cluster.id, genreColors['Noise / Outliers']);
    } else if (cluster.id === -2) {
      colorMap.set(cluster.id, genreColors['Unprocessed']);
    } else {
      // Extract the main genre from the cluster label
      const label = cluster.label;
      const mainGenre = label.split(' (')[0]; // Get the part before the first parenthesis
      
      // Find the base color for this genre
      let baseColor = genreColors[mainGenre];
      
      // If genre not found in our predefined colors, generate a color based on the genre name
      if (!baseColor) {
        // Generate a deterministic color based on the genre name
        const hash = mainGenre.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        const hue = (hash * 137.5) % 360; // Golden ratio to spread colors
        baseColor = parseInt(hslToHex(hue, 75, 65).replace('#', ''), 16);
      }
      
      // For subgenres, create a slightly different shade
      if (label.includes('(')) {
        // Adjust the color slightly for subgenres
        const r = ((baseColor >> 16) & 0xFF) + 20;
        const g = ((baseColor >> 8) & 0xFF) + 20;
        const b = (baseColor & 0xFF) + 20;
        baseColor = (Math.min(r, 255) << 16) | (Math.min(g, 255) << 8) | Math.min(b, 255);
      }
      
      colorMap.set(cluster.id, baseColor);
    }
  });

  return colorMap;
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

// Euclidean distance and Silhouette Score might not be primary for HDBSCAN,
// but could be used for secondary analysis or if K-Means is used as a fallback.
// For now, `calculateSilhouetteScore` might not be directly used by the main HDBSCAN flow.
function euclideanDistance(point1, point2) {
    if (!point1 || !point2 || point1.length !== point2.length) return Infinity;
    let sum = 0;
    for (let i = 0; i < point1.length; i++) sum += Math.pow(point1[i] - point2[i], 2);
    return Math.sqrt(sum);
}

function calculateSilhouetteScore(points, assignments, numClusters) {
    if (numClusters <= 1 || points.length < numClusters || points.length === 0 || !assignments || assignments.length !== points.length) return -1; 
    let totalSilhouette = 0;
    let validPointsForScore = 0;

    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        if (!point || point.some(isNaN)) continue; 

        const clusterIdx = assignments[i];
        if (clusterIdx === -1) continue; // Skip noise points for silhouette score

        let a_i = 0;
        let sameClusterPointsCount = 0;

        for (let j = 0; j < points.length; j++) {
            if (i === j || !points[j] || points[j].some(isNaN) || assignments[j] === -1) continue;
            if (assignments[j] === clusterIdx) {
                a_i += euclideanDistance(point, points[j]);
                sameClusterPointsCount++;
            }
        }
        if (sameClusterPointsCount > 0) a_i /= sameClusterPointsCount;
        else a_i = 0; 


        let b_i = Infinity;
        for (let k = 0; k < numClusters; k++) { // Assuming numClusters is max clusterId + 1 (excluding -1)
            if (k === clusterIdx) continue;
            let avgDistToOtherClusterK = 0;
            let otherClusterKPointsCount = 0;
            for (let j = 0; j < points.length; j++) {
                if (!points[j] || points[j].some(isNaN) || assignments[j] === -1) continue;
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

        if (b_i === Infinity && a_i === 0) { 
            validPointsForScore++; 
            continue;
        }
        if (b_i === Infinity) { 
            validPointsForScore++;
            continue;
        }

        const s_i = (b_i - a_i) / Math.max(a_i, b_i);
        if (!isNaN(s_i)) {
            totalSilhouette += s_i;
        }
        validPointsForScore++;
    }
    if (validPointsForScore === 0) return 0; 
    return totalSilhouette / validPointsForScore;
}


// calculateCentroid might be useful for labeling HDBSCAN clusters
function calculateCentroid(pointsArray) {
    if (!pointsArray || pointsArray.length === 0) return [];
    const numDimensions = pointsArray[0].length;
    const centroid = new Array(numDimensions).fill(0);
    let validPointsCount = 0;
    pointsArray.forEach(p => {
        if (p && p.length === numDimensions && !p.some(isNaN)) {
            for (let d = 0; d < numDimensions; d++) centroid[d] += p[d];
            validPointsCount++;
        }
    });
    if (validPointsCount === 0) return new Array(numDimensions).fill(0);
    for (let d = 0; d < numDimensions; d++) centroid[d] /= validPointsCount;
    return centroid;
}

/**
 * Calculates Euclidean distance between two feature vectors.
 * @param {number[]} vec1 - The first feature vector.
 * @param {number[]} vec2 - The second feature vector.
 * @returns {number} The Euclidean distance, or Infinity if inputs are invalid.
 */
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
    const val1 = vec1[i] || 0; // Default to 0 if undefined/NaN
    const val2 = vec2[i] || 0; // Default to 0 if undefined/NaN
    const diff = val1 - val2;
    sumOfSquares += diff * diff;
  }
  return Math.sqrt(sumOfSquares);
}

/**
 * Normalizes features using Z-score normalization and applies category-based weighting.
 * @param {number[][]} featureVectors - Array of feature vectors (arrays of numbers).
 * @param {string[]} featureCategories - Array of category strings, corresponding to the order of features in vectors.
 * @returns {number[][]} Normalized and category-weighted feature vectors.
 */
function normalizeFeatures(featureVectors, featureCategories) {
  if (!featureVectors || featureVectors.length === 0) return [];

  const numSamples = featureVectors.length;
  if (numSamples === 0) return [];
  const numFeatures = featureVectors[0]?.length || 0;
  if (numFeatures === 0) return featureVectors.map(() => []); // Return empty vectors if features are empty

  if (featureCategories.length !== numFeatures) {
    console.error(
        `Normalization Error: Mismatch between number of features (${numFeatures}) and categories (${featureCategories.length}). ` +
        `This will lead to incorrect weighting. Ensure featureCategories array matches feature vector structure.`
    );
  }

  const means = new Array(numFeatures).fill(0);
  const stdDevs = new Array(numFeatures).fill(0);

  // Calculate means for each feature
  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) {
      means[j] += vector[j] || 0; // Treat undefined/NaN as 0 for sum
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    means[j] /= numSamples;
  }

  // Calculate standard deviations for each feature
  for (const vector of featureVectors) {
    for (let j = 0; j < numFeatures; j++) {
      stdDevs[j] += Math.pow((vector[j] || 0) - means[j], 2);
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    stdDevs[j] = Math.sqrt(stdDevs[j] / numSamples); // Population standard deviation
  }

  // Apply Z-score normalization and category-based weighting
  return featureVectors.map(vector =>
    vector.map((value, j) => {
      const std = stdDevs[j];
      const mean = means[j];
      // Z-score normalize
      const normalizedValue = (std < 1e-10) ? 0 : ((value || 0) - mean) / std; // Avoid division by zero; if std is tiny, feature is constant

      // Apply category weight
      const category = (j < featureCategories.length && featureCategories[j]) ? featureCategories[j] : 'default';
      const weight = CATEGORY_WEIGHTS[category] || CATEGORY_WEIGHTS['default']; // Fallback to default weight

      return normalizedValue * weight;
    })
  );
}

/**
 * Simplified HDBSCAN clustering implementation.
 * @param {number[][]} data - Array of data points (feature vectors).
 * @param {number} [minClusterSize=HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE] - Minimum points for a cluster.
 * @param {number} [minSamples=HDBSCAN_DEFAULT_MIN_SAMPLES] - Parameter for core distance (like minPts).
 * @returns {number[]} Array of cluster labels for each data point (-1 for noise).
 */
function hdbscan(data, minClusterSize = HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE, minSamples = HDBSCAN_DEFAULT_MIN_SAMPLES) {
  if (!data || data.length === 0) return [];
  const n = data.length;

  if (n === 0) return []; // No data, no labels

  // Adaptive parameters based on dataset size
  const adaptiveMinClusterSize = Math.max(2, Math.min(minClusterSize, Math.floor(n * 0.05))); // 5% of dataset size, min 2
  const adaptiveMinSamples = Math.max(2, Math.min(minSamples, Math.floor(n * 0.02))); // 2% of dataset size, min 2

  if (n < adaptiveMinClusterSize && n > 0) { // Not enough points to form any cluster of minClusterSize
    return Array(n).fill(NOISE_CLUSTER_ID);
  }

  // 1. Compute mutual reachability distance
  function computeMutualReachabilityDistance() {
    const distances = Array(n).fill(null).map(() => Array(n).fill(0)); // Mutual reachability distance matrix
    const coreDistances = Array(n).fill(Infinity); // Core distance for each point

    if (n === 0) return { distances, coreDistances };

    // Calculate core distances with adaptive k
    for (let i = 0; i < n; i++) {
      if (n <= 1 || adaptiveMinSamples >= n) { // If n=1 or minSamples too high, core_k is effectively infinite
        coreDistances[i] = Infinity;
        continue;
      }
      const pointDistances = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        pointDistances.push(calculateDistance(data[i], data[j]));
      }
      pointDistances.sort((a, b) => a - b);
      // Core distance is the k-th nearest neighbor distance (minSamples-th as arrays are 0-indexed)
      coreDistances[i] = pointDistances[adaptiveMinSamples - 1] ?? Infinity;
    }

    // Calculate mutual reachability distances with improved distance metric
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) { // Iterate only upper triangle for efficiency
        const directDist = calculateDistance(data[i], data[j]);
        // mutual_reach_dist(a,b) = max(core_k(a), core_k(b), dist(a,b))
        const mrDist = Math.max(coreDistances[i], coreDistances[j], directDist);
        distances[i][j] = mrDist;
        distances[j][i] = mrDist; // Symmetric matrix
      }
    }
    return distances; // This is effectively the graph for MST
  }

  // 2. Build minimum spanning tree (MST) using Prim's algorithm
  function buildMST(mutualReachabilityDistances) {
    if (n === 0) return [];
    const mstEdges = []; // Stores [u, v, weight] for each edge in MST
    const visited = new Array(n).fill(false);
    const minEdgeWeight = new Array(n).fill(Infinity); // Min weight to connect this vertex to current MST
    const edgeToVertex = new Array(n).fill(-1); // Stores the other vertex of the edge connecting to MST

    if (n > 0) minEdgeWeight[0] = 0; // Start MST construction from vertex 0

    for (let count = 0; count < n; count++) {
      let u = -1; // Vertex with min weight not yet in MST
      let currentMin = Infinity;
      for (let v = 0; v < n; v++) {
        if (!visited[v] && minEdgeWeight[v] < currentMin) {
          currentMin = minEdgeWeight[v];
          u = v;
        }
      }

      if (u === -1) { // No more reachable vertices, graph might be disconnected
        break;
      }

      visited[u] = true; // Add u to MST

      // If u is not the starting vertex (which has no edgeTo), add the edge to MST
      if (edgeToVertex[u] !== -1) {
        mstEdges.push([u, edgeToVertex[u], minEdgeWeight[u]]);
      }

      // Update minEdgeWeight for adjacent vertices of u
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

  // 3. Extract clusters from MST
  function extractClusters(mst) {
    const labels = Array(n).fill(NOISE_CLUSTER_ID);
    if (n === 0 || mst.length === 0 && n > 0 && minClusterSize > 1) return labels;
    if (n > 0 && minClusterSize === 1) return Array(n).fill(0).map((_,i)=>i);

    let currentClusterId = 0;

    // Disjoint Set Union (DSU) data structure for finding connected components
    const parent = Array(n).fill(0).map((_, i) => i);
    const componentSize = Array(n).fill(1);

    function findSet(i) {
      if (parent[i] === i) return i;
      return parent[i] = findSet(parent[i]); // Path compression
    }

    function uniteSets(i, j) {
      let rootI = findSet(i);
      let rootJ = findSet(j);
      if (rootI !== rootJ) {
        if (componentSize[rootI] < componentSize[rootJ]) [rootI, rootJ] = [rootJ, rootI];
        parent[rootJ] = rootI;
        componentSize[rootI] += componentSize[rootJ];
        return true;
      }
      return false;
    }

    // Sort MST edges by weight
    const sortedMSTEdges = mst.sort((a, b) => a[2] - b[2]);

    for (const edge of sortedMSTEdges) {
      const [u, v, weight] = edge;
      uniteSets(u, v);
    }

    // Assign cluster IDs to components that meet minClusterSize
    const rootToClusterId = new Map();
    for(let i = 0; i < n; i++){
      const root = findSet(i);
      if(componentSize[root] >= minClusterSize){
        if(!rootToClusterId.has(root)){
          rootToClusterId.set(root, currentClusterId++);
        }
        labels[i] = rootToClusterId.get(root);
      } else {
        labels[i] = NOISE_CLUSTER_ID;
      }
    }
    return labels;
  }

  // --- HDBSCAN Main Algorithm Flow ---
  const mutualReachabilityDistances = computeMutualReachabilityDistance();
  const mst = buildMST(mutualReachabilityDistances);
  const labels = extractClusters(mst);

  return labels;
}

const VisualizationCanvas = () => {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [isLoadingTracks, setIsLoadingTracks] = useState(true);
  const [isSimilarityMode, setIsSimilarityMode] = useState(false);

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

  const waveformContainerRef = useRef(null); // PIXI container for React waveform

  const [selectableFeatures, setSelectableFeatures] = useState([...coreFeaturesConfig]);
  const [xAxisFeature, setXAxisFeature] = useState(coreFeaturesConfig[0]?.value || '');
  const [yAxisFeature, setYAxisFeature] = useState(coreFeaturesConfig[1]?.value || '');

  const [axisMinMax, setAxisMinMax] = useState({ x: null, y: null });
  const [isPixiAppReady, setIsPixiAppReady] = useState(false)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0});
  const onWheelZoomRef = useRef(null);

  const [currentHoverTrack, setCurrentHoverTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const tooltipTimeoutRef = useRef(null);
  const playButtonRef = useRef(null);
  const playIconRef = useRef(null);

  const [clusterSettings, setClusterSettings] = useState({ // For projection feature selection
    [FEATURE_CATEGORIES.MOOD]: true,
    [FEATURE_CATEGORIES.SPECTRAL]: true,
    [FEATURE_CATEGORIES.STYLE]: true,
    [FEATURE_CATEGORIES.INSTRUMENT]: true,
  });

  const [clusters, setClusters] = useState([]); // Stores HDBSCAN cluster results
  const [clusterDisplayColors, setClusterDisplayColors] = useState([]);
  const [isClusteringCalculating, setIsClusteringCalculating] = useState(false);

  const [selectedIndividualFeatures, setSelectedIndividualFeatures] = useState({});
  const [highlightPrecisionThreshold, setHighlightPrecisionThreshold] = useState(0.1);

  const [projectionData, setProjectionData] = useState(null); 
  const [isProjectionCalculating, setIsProjectionCalculating] = useState(false);
  const [projectionMethod, setProjectionMethod] = useState(PROJECTION_METHODS.UMAP); 

  // Create a ref to store the dynamicFeatureSourceMap
  const dynamicFeatureSourceMapRef = useRef(new Map());

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
        // Reset the map for this fetch
        dynamicFeatureSourceMapRef.current.clear();

        // Add core feature keys to allDiscoveredFeatureKeys and dynamicFeatureSourceMap
        coreFeaturesConfig.forEach(coreFeat => {
            allDiscoveredFeatureKeys.add(coreFeat.value);
            dynamicFeatureSourceMapRef.current.set(coreFeat.value, coreFeat.category);
        });

        const processedTracks = rawTracks.map(track => {
          const currentParsedFeatures = {};
          coreFeaturesConfig.forEach(coreFeatureConf => {
            const featureKey = coreFeatureConf.value;
            if (track[featureKey] !== undefined && track[featureKey] !== null) {
              // For key, store original string; for BPM store number; others parse to float.
              let val;
              if (featureKey === 'key') {
                val = track[featureKey]; // Keep as string for now, will be mapped later
              } else {
                val = parseFloat(track[featureKey]);
              }

              if (featureKey === 'key' || !isNaN(val)) {
                currentParsedFeatures[featureKey] = val;
                featureFrequencies[featureKey] = (featureFrequencies[featureKey] || 0) + 1;
              }
            }
          });

          const processDynamicFeaturesLocal = (featureObject, category) => {
              if (featureObject) {
                try {
                    // Handle both string and object types
                    const parsedObj = typeof featureObject === 'string' ? JSON.parse(featureObject) : featureObject;
                    if (parsedObj && typeof parsedObj === 'object') {
                        Object.entries(parsedObj).forEach(([key, value]) => {
                            const val = parseFloat(value);
                            if (!isNaN(val) && val >= FIXED_STYLE_THRESHOLD) {
                                currentParsedFeatures[key] = val;
                                allDiscoveredFeatureKeys.add(key);
                                dynamicFeatureSourceMapRef.current.set(key, category);
                                featureFrequencies[key] = (featureFrequencies[key] || 0) + 1;
                            }
                        });
                    }
                } catch (e) { 
                    console.warn(`Warning parsing ${category} features for track:`, track.id, e); 
                }
              }
          };

          processDynamicFeaturesLocal(track.features, FEATURE_CATEGORIES.STYLE);
          processDynamicFeaturesLocal(track.instrument_features, FEATURE_CATEGORIES.INSTRUMENT);

          let mainGenre = 'Unknown Genre';
          let mainSubgenre = 'Unknown Subgenre';
          let highestProbGenreSubgenre = -1;

          for (const key in currentParsedFeatures) {
              if (dynamicFeatureSourceMapRef.current.get(key) === FEATURE_CATEGORIES.STYLE && key.includes('---')) {
                const value = currentParsedFeatures[key];
                if (value >= FIXED_STYLE_THRESHOLD && value > highestProbGenreSubgenre) {
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
          if (coreFeatureValues.has(featureKey)) return; // Skip core features, they are already in coreFeaturesConfig
          let label = featureKey;
          if (featureKey.includes("---") && dynamicFeatureSourceMapRef.current.get(featureKey) === FEATURE_CATEGORIES.STYLE) {
            label = featureKey.split("---").map(part => part.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')).join(' / ');
          } else if (featureKey.includes("---")) {
            label = featureKey.substring(featureKey.indexOf("---") + 3);
            label = label.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          } else {
            label = label.replace(/_/g, ' ').split(/[\s-]+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          }
          const frequency = featureFrequencies[featureKey] || 0;
          const determinedCategory = dynamicFeatureSourceMapRef.current.get(featureKey) || FEATURE_CATEGORIES.STYLE; // Default, though should be set

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
            setYAxisFeature(finalSelectableFeatures[0].value); // Should be finalSelectableFeatures[0].value too if only one
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
  }, []); // Ensure this runs only once

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

  // PIXI App Initialization (largely unchanged)
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
        trackFeaturesTextRef.current.position.set(COVER_ART_SIZE + 2 * TOOLTIP_PADDING, TOOLTIP_PADDING + 30); // Adjusted Y for title
        tooltipContainerRef.current.addChild(trackFeaturesTextRef.current);

        playButtonRef.current = new PIXI.Graphics().circle(0, 0, PLAY_BUTTON_SIZE / 2).fill({ color: PLAY_BUTTON_COLOR });
        playButtonRef.current.position.set(300 - TOOLTIP_PADDING - PLAY_BUTTON_SIZE / 2, TOOLTIP_PADDING + PLAY_BUTTON_SIZE / 2); // Top right
        playButtonRef.current.eventMode = 'static';
        playButtonRef.current.cursor = 'pointer';
        tooltipContainerRef.current.addChild(playButtonRef.current);
        playIconRef.current = new PIXI.Graphics();
        playButtonRef.current.addChild(playIconRef.current);

        waveformContainerRef.current = new PIXI.Container(); // PIXI container
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
                  // wavesurferRef.current.play().catch(e => console.error("🌊 Error auto-playing on ready:", e));
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
  }, []); 

  const formatTickValue = useCallback((value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number' || isNaN(value)) return String(value);
    if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
    if (Math.abs(value) >= 10000) return value.toExponential(1);
    const numStr = value.toFixed(2);
    return parseFloat(numStr).toString();
  }, []);

  // Tooltip Update Effect (largely unchanged, ensure clusterSilhouetteScore is handled or omitted gracefully)
  useEffect(() => {
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
            featuresText = `Group: ${currentHoverTrack.clusterLabel}\n`;
            // HDBSCAN doesn't directly give silhouette, so we might omit or use a different metric
            // if (currentHoverTrack.clusterSilhouetteScore !== undefined) { 
            //   featuresText += `(Validity: ${currentHoverTrack.clusterSilhouetteScore > -Infinity ? currentHoverTrack.clusterSilhouetteScore.toFixed(2) : 'N/A'})\n`;
            // }
        }


        if (!isSimilarityMode || (isSimilarityMode && projectionData)) {
            const xFeatKey = isSimilarityMode ? (projectionMethod === PROJECTION_METHODS.PCA ? 'PC1' : 'UMAP1') : xAxisFeature;
            const yFeatKey = isSimilarityMode ? (projectionMethod === PROJECTION_METHODS.PCA ? 'PC2' : 'UMAP2') : yAxisFeature;

            const xFeatConf = selectableFeatures.find(f => f.value === xAxisFeature);
            const yFeatConf = selectableFeatures.find(f => f.value === yAxisFeature);

            const xFeatureLabel = isSimilarityMode ? xFeatKey : (xFeatConf?.label.split('(')[0].trim() || xAxisFeature);
            const yFeatureLabel = isSimilarityMode ? yFeatKey : (yFeatConf?.label.split('(')[0].trim() || yAxisFeature);
            
            if(isSimilarityMode && projectionData) {
                const pointData = projectionData.find(p => p.trackId === currentHoverTrack.id);
                if(pointData) {
                    featuresText += `${xFeatureLabel}: ${formatTickValue(pointData.x)}\n`;
                    featuresText += `${yFeatureLabel}: ${formatTickValue(pointData.y)}`;
                }
            } else if (!isSimilarityMode) {
                featuresText +=
                `${xFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[xAxisFeature])}\n` +
                `${yFeatureLabel}: ${formatTickValue(currentHoverTrack.parsedFeatures?.[yAxisFeature])}`;
            }
        }
        trackFeaturesTextRef.current.text = featuresText.trim();


        const artworkPath = currentHoverTrack.artwork_thumbnail_path || defaultArtwork;
        coverArtSpriteRef.current.texture = await PIXI.Assets.load(artworkPath).catch(() => PIXI.Texture.from(defaultArtwork));

        if (playIconRef.current) {
            playIconRef.current.clear();
            const isCurrentlyPlayingThisTrack = isPlaying && activeAudioUrlRef.current === currentHoverTrack.path;
            if (isCurrentlyPlayingThisTrack) {
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

        const waveformPixiContainerGlobalPos = waveformContainerRef.current.getGlobalPosition(new PIXI.Point());
        const canvasRect = pixiCanvasContainerRef.current.getBoundingClientRect();

        waveformHostElement.style.left = `${waveformPixiContainerGlobalPos.x - canvasRect.left}px`;
        waveformHostElement.style.top = `${waveformPixiContainerGlobalPos.y - canvasRect.top}px`;

        pixiCanvasContainerRef.current.appendChild(waveformHostElement);

        const root = ReactDOM.createRoot(waveformHostElement);
        waveformHostElement._reactRoot = root; 

        const playbackContextValue = { wavesurferRef, activeAudioUrlRef, setIsPlaying };

        root.render(
          <PlaybackContext.Provider value={playbackContextValue}>
            <Waveform
              key={`${currentHoverTrack.id}-tooltip-global`} 
              trackId={currentHoverTrack.id.toString()}
              audioPath={currentHoverTrack.path}
              isInteractive={true}
              wavesurferInstanceRef={wavesurferRef} 
              onPlay={() => {
                activeAudioUrlRef.current = currentHoverTrack.path;
              }}
              onReadyToPlay={(wsInstance) => {
                if (activeAudioUrlRef.current === currentHoverTrack.path && !wsInstance.isPlaying()) {
                  // wsInstance.play().catch(e => console.error("Tooltip WS play error:", e));
                }
              }}
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
        const existingReactContainers = pixiCanvasContainerRef.current?.querySelectorAll('.waveform-react-container');
        existingReactContainers?.forEach(container => {
            const root = container._reactRoot;
            if (root) root.unmount();
            container.remove();
        });
      }
    };

    let asyncCleanupFunction = null;
    updateTooltipVisuals().then(cleanupFunc => {
        asyncCleanupFunction = cleanupFunc;
    });

    return () => {
      if (typeof asyncCleanupFunction === 'function') {
        asyncCleanupFunction();
      }
    };
  }, [currentHoverTrack, xAxisFeature, yAxisFeature, selectableFeatures, formatTickValue, isSimilarityMode, clusters, projectionData, projectionMethod, isPlaying, activeAudioUrlRef.current]);


  const updateAxesTextScale = useCallback((chartArea) => {
    if (!chartArea || !chartArea.scale) return;
    const currentChartScale = chartArea.scale.x; const inverseScale = 1 / currentChartScale;
    for (const child of chartArea.children) { if (child.isAxisTextElement) { child.scale.set(inverseScale); } }
  }, []);

  // DrawAxes (for X/Y mode, largely unchanged)
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


  // prepareFeatureData: Modified for Similarity Mode (weighting)
  const prepareFeatureData = useCallback((tracksToProcess, allSelectableFeaturesForScope, settingsForProjection) => {
    if (!tracksToProcess?.length || !allSelectableFeaturesForScope?.length) return null;

    if (isSimilarityMode) {
        const trackIdsForProcessing = [];
        const validTracksForProcessing = [];
        const featureValuesByName = new Map(); 
        const featureStatsByName = new Map();  

        const canonicalFeatureOrder = [];
        SIMILARITY_EXPECTED_FEATURES.forEach(f => canonicalFeatureOrder.push({ ...f }));

        const allDynamicFeatureNames = new Set();
        tracksToProcess.forEach(track => {
            if (track.features) {
                try {
                    const styleFeats = typeof track.features === 'string' ? JSON.parse(track.features) : track.features;
                    Object.keys(styleFeats).forEach(key => allDynamicFeatureNames.add(key));
                } catch (e) { /* ignore */ }
            }
            if (track.instrument_features) {
                try {
                    const instFeats = typeof track.instrument_features === 'string' ? JSON.parse(track.instrument_features) : track.instrument_features;
                    Object.keys(instFeats).forEach(key => allDynamicFeatureNames.add(key));
                } catch (e) { /* ignore */ }
            }
        });
        
        const sortedDynamicFeatureNames = Array.from(allDynamicFeatureNames).sort();
        sortedDynamicFeatureNames.forEach(name => {
            const coreFeature = SIMILARITY_EXPECTED_FEATURES.find(f => f.name === name);
            if (!coreFeature) { 
                const selectableRef = allSelectableFeaturesForScope.find(sf => sf.value === name);
                canonicalFeatureOrder.push({
                    name,
                    weight: 0.75,
                    category: selectableRef ? selectableRef.category : FEATURE_CATEGORIES.STYLE, 
                });
            }
        });

        canonicalFeatureOrder.forEach(featureInfo => {
            featureValuesByName.set(featureInfo.name, []);
            featureStatsByName.set(featureInfo.name, {
                min: Infinity, max: -Infinity, sum: 0, count: 0,
                baseWeight: featureInfo.weight || 1.0,
                isKey: !!featureInfo.isKey,
                isBpm: !!featureInfo.isBpm,
                category: featureInfo.category,
            });
        });

        tracksToProcess.forEach(track => {
            canonicalFeatureOrder.forEach(featureInfo => {
                let rawValue;
                const pf = track.parsedFeatures; 

                if (pf && pf[featureInfo.name] !== undefined && pf[featureInfo.name] !== null) {
                    rawValue = pf[featureInfo.name];
                } else if (track.features && dynamicFeatureSourceMapRef.current.get(featureInfo.name) === FEATURE_CATEGORIES.STYLE) {
                    try {
                        const styleFeats = typeof track.features === 'string' ? JSON.parse(track.features) : track.features;
                        if (styleFeats && typeof styleFeats === 'object' && styleFeats[featureInfo.name] !== undefined) {
                            rawValue = parseFloat(styleFeats[featureInfo.name]);
                        }
                    } catch (e) { /* ignore */ }
                } else if (track.instrument_features && dynamicFeatureSourceMapRef.current.get(featureInfo.name) === FEATURE_CATEGORIES.INSTRUMENT) {
                    try {
                        const instFeats = typeof track.instrument_features === 'string' ? JSON.parse(track.instrument_features) : track.instrument_features;
                        if (instFeats && typeof instFeats === 'object' && instFeats[featureInfo.name] !== undefined) {
                            rawValue = parseFloat(instFeats[featureInfo.name]);
                        }
                    } catch (e) { /* ignore */ }
                }

                let value = 0; 
                if (rawValue !== undefined) {
                    if (featureInfo.isKey) {
                        const keyMap = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11, 'CM':0,'C#M':1,'DM':2,'D#M':3,'EM':4,'FM':5,'F#M':6,'GM':7,'G#M':8,'AM':9,'A#M':10,'BM':11, 'CMIN':0,'C#MIN':1,'DMIN':2,'D#MIN':3,'EMIN':4,'FMIN':5,'F#MIN':6,'GMIN':7,'G#MIN':8,'AMIN':9,'A#MIN':10,'BMIN':11};
                        const keyStr = String(rawValue).toUpperCase().replace(/\s+/g, '');
                        value = keyMap[keyStr] !== undefined ? keyMap[keyStr] / 11 : 0; 
                    } else if (featureInfo.isBpm) {
                        const bpmNum = parseFloat(rawValue);
                        // Consider log transform + normalization here in future
                        value = !isNaN(bpmNum) ? Math.min(Math.max(bpmNum, 50), 250) / 200 : 0.5; // Range 50-250 to 0-1
                    } else {
                        const num = parseFloat(rawValue);
                        value = !isNaN(num) ? num : 0;
                    }
                }
                
                featureValuesByName.get(featureInfo.name).push(value);
                const stats = featureStatsByName.get(featureInfo.name);
                stats.min = Math.min(stats.min, value);
                stats.max = Math.max(stats.max, value);
                stats.sum += value;
                stats.count++;
            });
        });

        const featureNormalizers = new Map();
        const combinedFeatureWeights = new Map();

        canonicalFeatureOrder.forEach(featureInfo => {
            const fName = featureInfo.name;
            const values = featureValuesByName.get(fName);
            const stats = featureStatsByName.get(fName);

            if (values.length > 0 && stats.count > 0) {
                stats.mean = stats.sum / stats.count;
                
                const uniqueValues = new Set(values.map(v => Math.round(v * 1000) / 1000)).size; // Increased precision for uniqueness
                const discriminativeWeight = values.length > 0 ? Math.max(0.1, uniqueValues / values.length) : 0.1; // Ensure non-zero, penalize constant features

                // Simplified weighting: base weight * discriminative power
                combinedFeatureWeights.set(fName, stats.baseWeight * discriminativeWeight);

                if (stats.isKey || stats.isBpm) { 
                    featureNormalizers.set(fName, (v) => v); // Already 0-1
                } else {
                    const range = stats.max - stats.min;
                    if (range < 1e-6) { 
                        featureNormalizers.set(fName, () => 0.5); 
                        combinedFeatureWeights.set(fName, stats.baseWeight * 0.05); // Heavily penalize constant features
                    } else {
                        featureNormalizers.set(fName, (v) => (v - stats.min) / range);
                    }
                }
            } else { 
                combinedFeatureWeights.set(fName, 0); 
                featureNormalizers.set(fName, () => 0); 
            }
        });
        
        const fullFeatureVectors = [];
        tracksToProcess.forEach(track => {
            const vector = [];
            let hasValidData = false;
            canonicalFeatureOrder.forEach(featureInfo => {
                const fName = featureInfo.name;
                let rawValue;
                const pf = track.parsedFeatures;
                if (pf && pf[fName] !== undefined && pf[fName] !== null) rawValue = pf[fName];
                else if (track.features && dynamicFeatureSourceMapRef.current.get(featureInfo.name) === FEATURE_CATEGORIES.STYLE) {
                     try {
                        const styleFeats = typeof track.features === 'string' ? JSON.parse(track.features) : track.features;
                        if (styleFeats && typeof styleFeats === 'object' && styleFeats[fName] !== undefined) rawValue = parseFloat(styleFeats[fName]);
                    } catch(e) {}
                } else if (track.instrument_features && dynamicFeatureSourceMapRef.current.get(featureInfo.name) === FEATURE_CATEGORIES.INSTRUMENT) {
                     try {
                        const instFeats = typeof track.instrument_features === 'string' ? JSON.parse(track.instrument_features) : track.instrument_features;
                        if (instFeats && typeof instFeats === 'object' && instFeats[fName] !== undefined) rawValue = parseFloat(instFeats[fName]);
                    } catch(e) {}
                }


                let preNormalizedValue = 0;
                if (rawValue !== undefined) {
                    const stats = featureStatsByName.get(fName); 
                    if (stats.isKey) {
                        const keyMap = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11, 'CM':0,'C#M':1,'DM':2,'D#M':3,'EM':4,'FM':5,'F#M':6,'GM':7,'G#M':8,'AM':9,'A#M':10,'BM':11, 'CMIN':0,'C#MIN':1,'DMIN':2,'D#MIN':3,'EMIN':4,'FMIN':5,'F#MIN':6,'GMIN':7,'G#MIN':8,'AMIN':9,'A#MIN':10,'BMIN':11};
                        const keyStr = String(rawValue).toUpperCase().replace(/\s+/g, '');
                        preNormalizedValue = keyMap[keyStr] !== undefined ? keyMap[keyStr] / 11 : 0;
                    } else if (stats.isBpm) {
                        const bpmNum = parseFloat(rawValue);
                        preNormalizedValue = !isNaN(bpmNum) ? Math.min(Math.max(bpmNum, 50), 250) / 200 : 0.5;
                    } else {
                        const num = parseFloat(rawValue);
                        preNormalizedValue = !isNaN(num) ? num : 0;
                    }
                }

                const normalizer = featureNormalizers.get(fName);
                const weight = combinedFeatureWeights.get(fName);
                const normalizedVal = normalizer(preNormalizedValue);
                vector.push(normalizedVal * weight);
                if (!isNaN(normalizedVal) && weight > 0) hasValidData = true;
            });

            if (hasValidData) {
                fullFeatureVectors.push(vector);
                trackIdsForProcessing.push(track.id);
                validTracksForProcessing.push(track);
            }
        });

        if (!fullFeatureVectors.length) return null;

        let vectorsForProjection = fullFeatureVectors;
        let featureConfigsForProjection = canonicalFeatureOrder.map(fInfo => ({
            value: fInfo.name, label: fInfo.name, category: fInfo.category 
        }));

        if (settingsForProjection) { 
            const projectionFeatureIndicesToKeep = [];
            const tempFeatureConfigsForProjection = [];

            canonicalFeatureOrder.forEach((featureInfo, index) => {
                if (settingsForProjection[featureInfo.category]) {
                    projectionFeatureIndicesToKeep.push(index);
                    const fullConfig = allSelectableFeaturesForScope.find(f => f.value === featureInfo.name);
                    tempFeatureConfigsForProjection.push(fullConfig || { value: featureInfo.name, label: featureInfo.name, category: featureInfo.category });
                }
            });

            if (projectionFeatureIndicesToKeep.length > 0) {
                vectorsForProjection = fullFeatureVectors.map(fullVector =>
                    projectionFeatureIndicesToKeep.map(idx => fullVector[idx])
                );
                featureConfigsForProjection = tempFeatureConfigsForProjection;
            } else { 
                vectorsForProjection = fullFeatureVectors.map(() => []); 
                featureConfigsForProjection = [];
            }
        }
        
        return {
            features: vectorsForProjection, 
            trackIds: trackIdsForProcessing,
            originalTracks: validTracksForProcessing,
            featureConfigsUsed: featureConfigsForProjection,
            _fullWeightedNormalizedFeatures: fullFeatureVectors, 
            _canonicalFeatureOrderRef: canonicalFeatureOrder, 
        };

    } else {
        // --- X/Y Scatter Plot Mode (largely unchanged) ---
        const activeFeatureConfigs = allSelectableFeaturesForScope.filter(featureConf =>
            featureConf.isNumeric 
        );
        if (!activeFeatureConfigs.length) return null;

        const featureVectors = [];
        const trackIdsForProcessing = [];
        const validTracksForProcessing = [];

        const relevantFeatureKeys = [xAxisFeature, yAxisFeature].filter(Boolean);
        if(relevantFeatureKeys.length === 0) return null;

        tracksToProcess.forEach(track => {
            const vector = [];
            relevantFeatureKeys.forEach(fKey => {
                const value = track.parsedFeatures?.[fKey];
                const numValue = (typeof value === 'number' && !isNaN(value)) ? value : 0; 
                vector.push(numValue);
            });
            if (vector.length === relevantFeatureKeys.length) { 
                featureVectors.push(vector);
                trackIdsForProcessing.push(track.id);
                validTracksForProcessing.push(track);
            }
        });

        if (!featureVectors.length) return null;
        
        return {
            features: featureVectors, 
            trackIds: trackIdsForProcessing,
            originalTracks: validTracksForProcessing,
            featureConfigsUsed: activeFeatureConfigs.filter(f => relevantFeatureKeys.includes(f.value)),
            normalizationParams: null 
        };
    }
  }, [isSimilarityMode, xAxisFeature, yAxisFeature, selectableFeatures]); // Added selectableFeatures


  // HDBSCAN Clustering (New)
  const performHDBSCANClustering = useCallback(async (allTracksToCluster, allSelectableFeaturesForClustering) => {
      if (!allTracksToCluster || allTracksToCluster.length === 0) {
          return { updatedTracks: [], finalClustersList: [] };
      }
      console.log("🚀 Starting HDBSCAN Grouping...");
      // Note: setLoading states are managed by the calling useEffect

      // 1. Prepare FULL feature vectors using the modified prepareFeatureData
      // settingsForProjection is null, so it uses/returns _fullWeightedNormalizedFeatures
      const globalFeatureData = prepareFeatureData(allTracksToCluster, allSelectableFeaturesForClustering, null);

      if (!globalFeatureData || !globalFeatureData._fullWeightedNormalizedFeatures || globalFeatureData._fullWeightedNormalizedFeatures.length === 0) {
          console.error("HDBSCAN Clustering: Could not prepare global features or features are empty.");
          const updated = allTracksToCluster.map(t => ({ ...t, clusterId: -1, clusterLabel: "Unclustered (feature prep failed)" }));
          return { updatedTracks: updated, finalClustersList: [{ id: -1, label: 'Unclustered (feature prep failed)', trackIds: allTracksToCluster.map(t => t.id) }] };
      }

      const { _fullWeightedNormalizedFeatures: featuresForHdbscan,
              trackIds: processedTrackIds, // track IDs corresponding to featuresForHdbscan rows
              originalTracks: tracksUsedInFeaturePrep // track objects corresponding to featuresForHdbscan rows
            } = globalFeatureData;

      if (featuresForHdbscan.length < 2 || featuresForHdbscan[0].length === 0) {
          console.warn("HDBSCAN: Not enough samples or features for clustering.");
          const updated = tracksUsedInFeaturePrep.map(track => ({ ...track, clusterId: -1, clusterLabel: 'Unclustered (few samples/features)' }));
          return { updatedTracks: updated, finalClustersList: [{ id: -1, label: 'Unclustered (few samples/features)', trackIds: processedTrackIds }] };
      }

      let clusterAssignments;
      try {
          // HDBSCAN parameters - these may need tuning!
          const minClusterSize = Math.max(5, Math.min(15, Math.floor(featuresForHdbscan.length * 0.02))); // e.g. 2% of dataset, capped
          const minPoints = Math.max(2, Math.min(minClusterSize, Math.floor(minClusterSize * 0.5)));      // smaller than or equal to minClusterSize

          if (featuresForHdbscan.length < minClusterSize) {
              console.warn(`HDBSCAN: Too few samples (${featuresForHdbscan.length}) for minClusterSize=${minClusterSize}. Treating all as noise.`);
              clusterAssignments = new Array(featuresForHdbscan.length).fill(-1);
          } else {
              console.log(`Running HDBSCAN with minPoints=${minPoints}, minClusterSize=${minClusterSize} on ${featuresForHdbscan.length} samples with ${featuresForHdbscan[0].length} dims.`);
              // hdbscan-js API: HDBSCAN(dataset, minClusterSize, minPoints, distanceFunction)
              clusterAssignments = hdbscan(featuresForHdbscan, minClusterSize, minPoints);
          }
      } catch (e) {
          console.error("Error during HDBSCAN execution:", e);
          clusterAssignments = new Array(featuresForHdbscan.length).fill(-1); // Assign all to noise on error
      }

      // Process assignments and generate labels
      const clusterDataMap = new Map(); // Map<clusterId, trackObjects[]>
      processedTrackIds.forEach((trackId, index) => {
          const assignedClusterId = clusterAssignments[index];
          if (!clusterDataMap.has(assignedClusterId)) {
              clusterDataMap.set(assignedClusterId, []);
          }
          // Find the original track object from tracksUsedInFeaturePrep
          const trackObject = tracksUsedInFeaturePrep[index]; // Assuming order is preserved
          if (trackObject) {
            clusterDataMap.get(assignedClusterId).push(trackObject);
          }
      });

      const finalClustersList = [];
      const updatedTracksWithClusters = [];

      for (const [clusterId, tracksInCluster] of clusterDataMap.entries()) {
          if (tracksInCluster.length === 0) continue;

          let label = `Group ${clusterId}`;
          if (clusterId === -1) {
              label = "Noise / Outliers";
          } else {
              const genreCounts = {};
              tracksInCluster.forEach(t => {
                  const key = `${t.mainGenre || 'Genre?'}${t.mainSubgenre && t.mainSubgenre !=='N/A' ? ` (${t.mainSubgenre})` : ''}`;
                  genreCounts[key] = (genreCounts[key] || 0) + 1;
              });
              if (Object.keys(genreCounts).length > 0) {
                  const dominantGenre = Object.entries(genreCounts).sort((a, b) => b[1] - a[1])[0][0];
                  label = `${dominantGenre} (Group ${clusterId})`;
              }
          }

          finalClustersList.push({
              id: clusterId,
              label: label,
              trackIds: tracksInCluster.map(t => t.id),
              silhouetteScore: undefined // Silhouette not standard for HDBSCAN
          });

          tracksInCluster.forEach(track => {
              updatedTracksWithClusters.push({ ...track, clusterId: clusterId, clusterLabel: label, clusterSilhouetteScore: undefined });
          });
      }
      
      // Ensure all original tracks are in updatedTracksWithClusters, even if missed by feature prep
      const tracksInUpdatedSet = new Set(updatedTracksWithClusters.map(t => t.id));
      allTracksToCluster.forEach(originalTrack => {
          if (!tracksInUpdatedSet.has(originalTrack.id)) {
              updatedTracksWithClusters.push({ ...originalTrack, clusterId: -2, clusterLabel: "Unprocessed", clusterSilhouetteScore: undefined });
              if(!finalClustersList.find(c => c.id === -2)){
                finalClustersList.push({ id: -2, label: "Unprocessed", trackIds: [originalTrack.id] });
              } else {
                finalClustersList.find(c => c.id === -2).trackIds.push(originalTrack.id);
              }
          }
      });


      finalClustersList.sort((a, b) => {
          if (a.id === -1 && b.id !== -1) return 1; // Noise last (or first if preferred)
          if (a.id !== -1 && b.id === -1) return -1;
          if (a.id === -2 && b.id !== -2) return 1; // Unprocessed after noise
          if (a.id !== -2 && b.id === -2) return -1;
          return a.id - b.id; // Then sort by ID
      });

      console.log("✅ HDBSCAN Grouping Completed. Found", finalClustersList.filter(c => c.id !== -1 && c.id !== -2).length, "meaningful groups.");
      return { updatedTracks: updatedTracksWithClusters, finalClustersList };

  }, [prepareFeatureData]);


  // Projection Calculation (largely unchanged, but called after HDBSCAN)
  const calculateProjection = useCallback(async (data, method, forClusteringVis = false) => {
    if (!data || !data.features || data.features.length === 0) {
        if (forClusteringVis) setIsProjectionCalculating(false);
        setProjectionData(null); return;
    }
    const nActualFeatures = data.features[0]?.length || 0;
    if (nActualFeatures < 2) {
        console.warn(`${method}: Not enough features (<2) for 2D projection. Features:`, nActualFeatures);
        if (nActualFeatures === 1 && data.features.length > 1) { 
            console.log(`${method} yielded 1 component, using it for X-axis and adding jitter for Y-axis.`);
            const jitteredResult = data.features.map((point, index) => ({
                x: point[0], 
                y: Math.random() * 0.1 - 0.05, 
                trackId: data.trackIds[index]
            }));
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            jitteredResult.forEach(p => { xMin = Math.min(xMin, p.x); xMax = Math.max(xMax, p.x); yMin = Math.min(yMin, p.y); yMax = Math.max(yMax, p.y); });
            const xRange = xMax - xMin || 1; const yRange = yMax - yMin || 1;
            const normalized = jitteredResult.map(p => ({
                x: (p.x - xMin) / xRange,
                y: (p.y - yMin) / yRange, 
                trackId: p.trackId
            }));
            setProjectionData(normalized);
        } else {
            setProjectionData(null);
        }
        if (forClusteringVis) setIsProjectionCalculating(false);
        return;
    }

    return new Promise((resolve) => {
        requestAnimationFrame(async () => {
            try {
                console.log(`🔄 Starting ${method} with ${data.features.length} tracks, ${nActualFeatures} features.`);
                let projectionResultMatrix;
                if (method === PROJECTION_METHODS.PCA) {
                    const pca = new PCA(data.features);
                    projectionResultMatrix = pca.predict(data.features, { nComponents: 2 }).data;
                } else if (method === PROJECTION_METHODS.UMAP) {
                    const nNeighbors = Math.min(15, data.features.length - 1);
                    if (data.features.length <= nNeighbors) {
                        console.warn(`UMAP: Not enough samples (${data.features.length}) for nNeighbors=${nNeighbors}. Using PCA instead.`);
                        const pca = new PCA(data.features);
                        projectionResultMatrix = pca.predict(data.features, { nComponents: 2 }).data;
                    } else {
                        const umap = new UMAP({
                            nComponents: 2,
                            nNeighbors: Math.min(30, Math.max(15, Math.floor(data.features.length * 0.1))), // Adaptive neighbors based on dataset size
                            minDist: 0.3, // Increased from 0.1 to spread points more
                            spread: 1.0, // Added spread parameter to control the scale of the embedding
                            random: () => 0.5 // For deterministic UMAP for now
                        });
                        const result = await umap.fit(data.features); // umap-js fit is async
                        projectionResultMatrix = result;
                    }
                }

                if (!projectionResultMatrix || !projectionResultMatrix.length || !projectionResultMatrix[0] || projectionResultMatrix[0].length < 2) {
                    console.warn(`${method} result is not in expected 2D format. Actual components:`, projectionResultMatrix && projectionResultMatrix[0] ? projectionResultMatrix[0].length : 'N/A');
                    setProjectionData(null);
                    if (forClusteringVis) setIsProjectionCalculating(false);
                    resolve();
                    return;
                }

                let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
                projectionResultMatrix.forEach(point => {
                    xMin = Math.min(xMin, point[0]); xMax = Math.max(xMax, point[0]);
                    yMin = Math.min(yMin, point[1]); yMax = Math.max(yMax, point[1]);
                });
                const xRange = xMax - xMin || 1; const yRange = yMax - yMin || 1;

                const normalizedResult = projectionResultMatrix.map((point, index) => ({
                    x: (point[0] - xMin) / xRange,
                    y: (point[1] - yMin) / yRange,
                    trackId: data.trackIds[index]
                }));
                setProjectionData(normalizedResult);
                console.log(`✅ ${method} completed.`);
            } catch (error) {
                console.error(`❌ Error calculating ${method}:`, error);
                setProjectionData(null);
            } finally {
                if (forClusteringVis) setIsProjectionCalculating(false);
                resolve();
            }
        });
    });
  }, []);


  // Similarity Mode: Clustering & Projection Trigger
  useEffect(() => {
    if (!isSimilarityMode || !tracks.length || !selectableFeatures.length) {
      setClusters([]); setProjectionData(null);
      if (!isSimilarityMode && tracks.some(t => t.clusterId !== undefined)) {
          setTracks(prevTracks => prevTracks.map(t => ({...t, clusterId: undefined, clusterLabel: undefined, clusterSilhouetteScore: undefined })));
      }
      setIsClusteringCalculating(false); setIsProjectionCalculating(false);
      return;
    }

    let animationFrameId;
    const timeoutId = setTimeout(async () => { // Make outer function async
      setIsClusteringCalculating(true);
      setIsProjectionCalculating(true); 

      // Run clustering (HDBSCAN)
      // `performHDBSCANClustering` is now async because it might do internal async work later
      // or simply to match `requestAnimationFrame` pattern if that was intended for long tasks.
      // However, hdbscan-js is synchronous.
      const { updatedTracks: tracksWithClusters, finalClustersList } = await performHDBSCANClustering(
        tracks, selectableFeatures // Pass original tracks
      );
      setIsClusteringCalculating(false); 

      if (tracksWithClusters.length > 0 && finalClustersList.length > 0) {
        setTracks(tracksWithClusters); // Update tracks with new cluster info
        setClusters(finalClustersList);

        // Generate colors using the new hierarchical color generation
        const newClusterColors = generateClusterColors(finalClustersList);
        setClusterDisplayColors(newClusterColors);

        // Prepare data for projection using CURRENT clusterSettings (UI checkboxes)
        // It's important that prepareFeatureData here uses tracksWithClusters if its internal logic relies on prior processing.
        // Or, it should always work from the base rawTracks if it re-does all parsing.
        // Given its current structure, passing tracksWithClusters (which are just enriched rawTracks) is fine.
        const dataForProjection = prepareFeatureData(tracksWithClusters, selectableFeatures, clusterSettings);
        
        if (dataForProjection && dataForProjection.features.length > 0 && dataForProjection.features[0].length > 0) {
          await calculateProjection(dataForProjection, projectionMethod, true); 
        } else {
          console.warn("Projection: No valid data prepared for projection after HDBSCAN.");
          setProjectionData(null);
          setIsProjectionCalculating(false);
        }
      } else {
        setClusters([]); setProjectionData(null);
        setIsProjectionCalculating(false); 
      }
    }, 500); 

    return () => { clearTimeout(timeoutId); if (animationFrameId) cancelAnimationFrame(animationFrameId); };
  }, [
    isSimilarityMode,
    tracks.length, // Only track count as a coarse trigger
    //selectableFeatures.length, // Avoid if selectableFeatures is stable struct but values change
    performHDBSCANClustering, // Stable callback
    calculateProjection,      // Stable callback
    prepareFeatureData,       // Stable callback
    JSON.stringify(clusterSettings), 
    projectionMethod,
  ]);


  const calculateVisualizationBounds = useCallback((points, padding = 0.05) => { 
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

  // Main PIXI Rendering useEffect (largely unchanged, but uses new clusterDisplayColors map)
  useEffect(() => {
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
        if (isClusteringCalculating) currentLoadingMessage = "Calculating similarity groups (HDBSCAN)...";
        else if (isProjectionCalculating) currentLoadingMessage = `Calculating similarity projection (${projectionMethod})...`;
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

        if (isSimilarityMode) {
            // Use the clusterDisplayColors map, which handles -1 for noise
            if (track.clusterId !== undefined && clusterDisplayColors[track.clusterId]) {
                fillColor = clusterDisplayColors[track.clusterId];
            } else if (track.clusterId !== undefined) {
                fillColor = 0xCCCCCC; // Fallback grey for unknown cluster IDs
            }
        } else { // X/Y Mode Highlighting
            const activeMenuSelections = Object.entries(selectedIndividualFeatures)
                .filter(([, isSelected]) => isSelected)
                .map(([key]) => key);

            if (activeMenuSelections.length > 0) {
                let matchCount = 0;
                let categoryColors = new Set();
                
                for (const featureValue of activeMenuSelections) {
                    const trackFeatureVal = track.parsedFeatures?.[featureValue];
                    if (typeof trackFeatureVal === 'number' && trackFeatureVal >= highlightPrecisionThreshold) {
                        matchCount++;
                        // Get the category color for this feature
                        const feature = selectableFeatures.find(f => f.value === featureValue);
                        if (feature && feature.category) {
                            categoryColors.add(CATEGORY_COLORS[feature.category]);
                        }
                    }
                }
                
                if (matchCount > 0) {
                    // If multiple categories are selected, blend their colors
                    if (categoryColors.size > 1) {
                        const colors = Array.from(categoryColors);
                        const r = Math.floor(colors.reduce((sum, color) => sum + ((color >> 16) & 0xFF), 0) / colors.length);
                        const g = Math.floor(colors.reduce((sum, color) => sum + ((color >> 8) & 0xFF), 0) / colors.length);
                        const b = Math.floor(colors.reduce((sum, color) => sum + (color & 0xFF), 0) / colors.length);
                        fillColor = (r << 16) | (g << 8) | b;
                    } else {
                        fillColor = categoryColors.values().next().value;
                    }
                }
            }
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

    if (isSimilarityMode && projectionData && projectionData.length > 0) {
      const projectionBounds = calculateVisualizationBounds(projectionData, 0.05); 
      if (!projectionBounds) return; 

      const graphics = new PIXI.Graphics(); 
      graphics.moveTo(PADDING, currentCanvasHeight - PADDING).lineTo(currentCanvasWidth - PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR}); 
      graphics.moveTo(PADDING, PADDING).lineTo(PADDING, currentCanvasHeight - PADDING).stroke({width:1, color:AXIS_COLOR}); 

      const activeCatsForProjection = Object.entries(clusterSettings)
        .filter(([,isActive]) => isActive)
        .map(([key]) => FEATURE_CATEGORIES[Object.keys(FEATURE_CATEGORIES).find(k=>FEATURE_CATEGORIES[k] === key)] || key)
        .join('/') || "Selected";
      const xTitleText = `${projectionMethod === PROJECTION_METHODS.PCA ? 'PC1' : 'UMAP Dim 1'} (Features: ${activeCatsForProjection})`;
      const yTitleText = `${projectionMethod === PROJECTION_METHODS.PCA ? 'PC2' : 'UMAP Dim 2'} (Features: ${activeCatsForProjection})`;

      const titleTextStyle = { fontFamily: 'Arial', fontSize: 14, fontWeight: 'bold', fill: TEXT_COLOR, align: 'center' };
      const xTitle = new PIXI.Text({text:xTitleText, style: titleTextStyle});
      xTitle.isAxisTextElement = true; xTitle.anchor.set(0.5,0); xTitle.position.set(PADDING + drawableWidth/2, currentCanvasHeight - PADDING + 25); chartArea.addChild(xTitle);
      const yTitle = new PIXI.Text({text:yTitleText, style: titleTextStyle});
      yTitle.isAxisTextElement = true; yTitle.anchor.set(0.5,1); yTitle.rotation = -Math.PI/2; yTitle.position.set(PADDING - 45, PADDING + drawableHeight/2); chartArea.addChild(yTitle);
      chartArea.addChild(graphics);


      projectionData.forEach(({ x: projX, y: projY, trackId }) => {
        const track = tracks.find(t => t.id === trackId);
        if (!track) return;
        const screenX = PADDING + projX * drawableWidth;
        const screenY = PADDING + (1 - projY) * drawableHeight; 
        commonDotLogic(track, screenX, screenY);
      });

    } else if (isSimilarityMode && (!projectionData || projectionData.length === 0) && !isProjectionCalculating && !isClusteringCalculating) {
        const msgText = new PIXI.Text({text:`${projectionMethod} projection not available or no data.`, style: new PIXI.TextStyle({ fill: 'orange', fontSize: 16, align: 'center'})});
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
        const normX = xRange.range === 0 ? 0.5 : (rawXVal - xRange.min) / xRange.range; 
        const normY = yRange.range === 0 ? 0.5 : (rawYVal - yRange.min) / yRange.range; 

        const screenX = PADDING + normX * drawableWidth;
        const screenY = PADDING + (1 - normY) * drawableHeight; 
        commonDotLogic(track, screenX, screenY);
      });
    }
    updateAxesTextScale(chartArea);
  }, [
    isPixiAppReady, tracks, axisMinMax, xAxisFeature, yAxisFeature,
    isLoadingTracks, error, drawAxes, canvasSize, updateAxesTextScale,
    selectableFeatures, selectedIndividualFeatures, highlightPrecisionThreshold,
    projectionData, isProjectionCalculating, isClusteringCalculating, isSimilarityMode,
    clusterSettings, clusters, clusterDisplayColors, projectionMethod,
    calculateVisualizationBounds 
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
            Similarity Mode
          </label>
        </div>

        <div className="feature-highlighting-controls" style={{ marginBottom: '15px', borderTop: '1px solid #555', paddingTop: '15px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '10px', fontSize: '1.05em' }}>Feature Highlighting</div>
            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>
                    Highlight Min. Probability: {(highlightPrecisionThreshold * 100).toFixed(0)}%
                </label>
                <input
                    type="range" min="0" max="100"
                    value={highlightPrecisionThreshold * 100}
                    onChange={(e) => setHighlightPrecisionThreshold(parseInt(e.target.value, 10) / 100)}
                    style={{ width: '100%' }}
                />
            </div>
            <div style={{fontWeight: 'bold', marginBottom: '10px'}}>Select Features to Highlight:</div>
            {Object.values(FEATURE_CATEGORIES).map(category => {
            const featuresInCategory = selectableFeatures.filter(f => f.category === category && f.isNumeric); 
            if (featuresInCategory.length === 0) return null;
            return (
                <div key={category} style={{ marginBottom: '10px' }}>
                <strong style={{ display: 'block', marginBottom: '5px', color: `#${CATEGORY_COLORS[category].toString(16).padStart(6, '0')}`, textTransform: 'capitalize' }}>{category.toLowerCase()}</strong>
                {featuresInCategory.map(feature => (
                    <div key={feature.value} style={{ marginLeft: '10px', marginBottom: '3px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.9em' }}>
                        <input
                        type="checkbox"
                        checked={!!selectedIndividualFeatures[feature.value]}
                        onChange={() => setSelectedIndividualFeatures(prev => ({ ...prev, [feature.value]: !prev[feature.value] }))}
                        style={{ width: '14px', height: '14px', flexShrink: 0 }}
                        />
                        {feature.label.split('(')[0].trim()}
                    </label>
                    </div>
                ))}
                </div>
            );
            })}
        </div>


        {isSimilarityMode && (
          <div className="clustering-controls" style={{ marginBottom: '15px', borderTop: '1px solid #555', paddingTop: '15px'}}>
            <div style={{fontWeight: 'bold', marginBottom: '10px', fontSize: '1.05em'}}>Similarity Mode Settings:</div>
             <div style={{ marginBottom: '10px' }}>
                <label htmlFor="projectionMethodSelect" style={{color: "#ccc", marginRight:"5px", display:'block', marginBottom:'3px'}}>Projection Method:</label>
                <select
                    id="projectionMethodSelect"
                    value={projectionMethod}
                    onChange={(e) => setProjectionMethod(e.target.value)}
                    style={{padding:"5px", width: '100%', backgroundColor: "#333", color:"#fff", border:"1px solid #555"}}
                >
                    <option value={PROJECTION_METHODS.UMAP}>UMAP (Recommended)</option>
                    <option value={PROJECTION_METHODS.PCA}>PCA</option>
                </select>
            </div>

            <p style={{fontSize: '0.9em', color: '#bbb', marginBottom: '15px'}}>
              Similarity grouping by HDBSCAN. Projection visualizes these groups using selected feature categories below.
            </p>
            <div style={{fontWeight: 'bold', marginBottom: '5px'}}>Feature Categories for {projectionMethod} Projection:</div>
            {[FEATURE_CATEGORIES.MOOD, FEATURE_CATEGORIES.SPECTRAL, FEATURE_CATEGORIES.STYLE, FEATURE_CATEGORIES.INSTRUMENT].map(category => (
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
                <div style={{fontWeight: 'bold', marginBottom: '10px'}}>Found Groups ({clusters.filter(c => c.id !== -1 && c.id !== -2).length}):</div>
                <div style={{maxHeight: '150px', overflowY: 'auto', paddingRight: '5px', fontSize: '0.85em'}}>
                  {clusters.map((cluster) => (
                    <div key={cluster.id} style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
                        <span style={{
                            display: 'inline-block',
                            width: '12px',
                            height: '12px',
                            borderRadius: '50%',
                            backgroundColor: `#${(clusterDisplayColors[cluster.id] || 0xCCCCCC).toString(16).padStart(6,'0')}`,
                            marginRight: '8px',
                            flexShrink: 0,
                        }}></span>
                      <span style={{ color: '#ccc', wordBreak: 'break-word', lineHeight: '1.2' }}>
                        {cluster.label} ({cluster.trackIds.length})
                        {/* Silhouette score not standard for HDBSCAN, remove or replace later if needed */}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!isSimilarityMode && (
          <div className="axis-selectors-container" style={{borderTop: '1px solid #555', paddingTop: '15px'}}>
              <div style={{fontWeight: 'bold', marginBottom: '10px', fontSize: '1.05em'}}>Scatter Plot Axes:</div>
            <div className="axis-selectors" style={{display: "flex", flexDirection:"column", gap: "10px", marginBottom: '15px'}}>
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
          </div>
        )}
      </div> {/* End Controls Panel */}

      <div className="canvas-wrapper">
        <div ref={pixiCanvasContainerRef} className="pixi-canvas-target" />
        <div ref={wavesurferContainerRef} className="wavesurfer-container-hidden" style={{ display: 'none' }}></div>

        {(isLoadingTracks || (isSimilarityMode && (isClusteringCalculating || isProjectionCalculating))) &&
          <div className="loading-overlay">
            {isLoadingTracks ? "Loading tracks..." :
              isClusteringCalculating ? "Calculating similarity groups (HDBSCAN)..." :
              isProjectionCalculating ? `Calculating similarity projection (${projectionMethod})...` : ""}
          </div>
        }
        {error && <div className="error-overlay">{error}</div>}
      </div>
    </div>
  );
};

export default VisualizationCanvas;