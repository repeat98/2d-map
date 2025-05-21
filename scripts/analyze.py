#!/usr/bin/env python
# analyze.py

import os, sys

# Attempt to disable Essentia warnings via environment variables.
os.environ["ESSENTIA_LOG_DISABLE"] = "1"
os.environ["ESSENTIA_ENABLE_WARNINGS"] = "0"
os.environ["ESSENTIA_LOG_LEVEL"] = "error"

import sqlite3
import json
import numpy as np
from threading import Lock
from tinytag import TinyTag
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
    RhythmExtractor2013,
    KeyExtractor,
    Resample
)

from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time
import threading

# Redirect the underlying C++ stderr (file descriptor 2) to /dev/null to suppress warnings.
try:
    if os.name != 'nt': # Not "nul" on Windows
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
    # For Windows, suppressing C++ level stderr is more complex and often not fully achievable this way.
except OSError:
    print("Could not redirect C++ stderr to /dev/null.", file=sys.__stderr__)


# Additional imports for artwork extraction
import mutagen
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.id3 import APIC
from mutagen.mp4 import MP4, MP4Cover
from mutagen.aac import AAC
from PIL import Image
import hashlib
import io
from multiprocessing import Process, freeze_support # Removed duplicate Process

# Import sklearn for similarity matrix computation (if needed elsewhere)
# from sklearn.preprocessing import StandardScaler # Not directly used in this script currently

# Spectral analysis imports
import librosa

# Attempt to import pyloudnorm for improved LUFS calculation
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Warning: pyloudnorm library not found. LUFS calculation will use RMS-based approximation.", file=sys.stderr)


# ------------------------------------------------------------------------------
# Setup error logging to a file
# ------------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

error_log_path = os.path.join(script_dir, "error.log")
# Ensure error log can be written, handle potential permission issues gracefully
try:
    error_log_file = open(error_log_path, 'a', buffering=1)  # line-buffered
    original_stderr = sys.stderr
    sys.stderr = error_log_file
except IOError as e:
    print(f"Warning: Could not open error log file {error_log_path}: {e}", file=sys.__stderr__)
    # Keep original_stderr if log file fails
    error_log_file = None # type: ignore
    original_stderr = sys.stderr


# ------------------------------------------------------------------------------
# Global Paths and Model Initialization
# ------------------------------------------------------------------------------
embedding_model_path = os.path.join(script_dir, "essentia_model/discogs-effnet-bs64-1.pb")
classification_model_path = os.path.join(script_dir, "essentia_model/genre_discogs400-discogs-effnet-1.pb")
happiness_model_path = os.path.join(script_dir, "essentia_model/mood_happy-discogs-effnet-1.pb")
party_model_path = os.path.join(script_dir, "essentia_model/mood_party-discogs-effnet-1.pb")
aggressive_model_path = os.path.join(script_dir, "essentia_model/mood_aggressive-discogs-effnet-1.pb")
danceability_model_path = os.path.join(script_dir, "essentia_model/danceability-discogs-effnet-1.pb")
relaxed_model_path = os.path.join(script_dir, "essentia_model/mood_relaxed-discogs-effnet-1.pb")
sad_model_path = os.path.join(script_dir, "essentia_model/mood_sad-discogs-effnet-1.pb")
engagement_model_path = os.path.join(script_dir, "essentia_model/engagement_2c-discogs-effnet-1.pb")
approachability_model_path = os.path.join(script_dir, "essentia_model/approachability_2c-discogs-effnet-1.pb")
instrument_model_path = os.path.join(script_dir, "essentia_model/mtg_jamendo_instrument-discogs-effnet-1.pb")

class_labels_path = os.path.join(script_dir, 'essentia_model/genre_discogs400-discogs-effnet-1.json')
instrument_labels_path = os.path.join(script_dir, 'essentia_model/mtg_jamendo_instrument-discogs-effnet-1.json')

db_name = "tracks.db" # Define DB name
db_folder = os.path.join(script_dir, "../db/") # Define DB folder
db_path = os.path.join(db_folder, db_name) # Construct full DB path

artworks_dir = os.path.join(script_dir, '../assets/artworks')
os.makedirs(artworks_dir, exist_ok=True)
os.makedirs(db_folder, exist_ok=True) # Ensure DB directory exists

# Load models
try:
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_model_path,
        output="PartitionedCall:1",
        patchHopSize=64
    )
except Exception as e:
    print(f"FATAL: Error loading Tensorflow embedding model: {embedding_model_path}. Exception: {e}", file=sys.stderr)
    sys.exit(1) # Critical error, exit

try:
    classification_model = TensorflowPredict2D(
        graphFilename=classification_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )
except Exception as e:
    print(f"FATAL: Error loading Tensorflow classification model: {classification_model_path}. Exception: {e}", file=sys.stderr)
    sys.exit(1) # Critical error, exit

# Helper to load optional models
def load_optional_model(path, name, output_name="model/Softmax"):
    try:
        model = TensorflowPredict2D(graphFilename=path, output=output_name)
        print(f"{name} model loaded successfully.")
        return model
    except Exception as e:
        print(f"Warning: Error loading {name} model ({path}): {e}. This analysis will be skipped.", file=sys.stderr)
        return None

happiness_model = load_optional_model(happiness_model_path, "Happiness")
party_model = load_optional_model(party_model_path, "Party")
aggressive_model = load_optional_model(aggressive_model_path, "Aggressive")
danceability_model = load_optional_model(danceability_model_path, "Danceability")
relaxed_model = load_optional_model(relaxed_model_path, "Relaxed")
sad_model = load_optional_model(sad_model_path, "Sad")
engagement_model = load_optional_model(engagement_model_path, "Engagement")
approachability_model = load_optional_model(approachability_model_path, "Approachability")
instrument_model = load_optional_model(instrument_model_path, "Instrument", output_name="model/Sigmoid") # Changed to use correct output node

# Load class labels for genre
try:
    with open(class_labels_path, 'r') as file:
        class_labels = json.load(file).get("classes", [])
    if not class_labels:
        print(f"Warning: Genre class labels file {class_labels_path} is empty or malformed.", file=sys.stderr)
except Exception as e:
    print(f"Error loading genre class labels from {class_labels_path}: {e}", file=sys.stderr)
    class_labels = []

# Load class labels for instruments
try:
    with open(instrument_labels_path, 'r') as file:
        instrument_labels = json.load(file).get("classes", [])
    if not instrument_labels:
        print(f"Warning: Instrument class labels file {instrument_labels_path} is empty or malformed.", file=sys.stderr)
except Exception as e:
    print(f"Error loading instrument class labels from {instrument_labels_path}: {e}", file=sys.stderr)
    instrument_labels = []


db_lock = Lock()

# ------------------------------------------------------------------------------
# Database Initialization and Integrity Check
# ------------------------------------------------------------------------------
def init_db():
    """Initialize the database and create the directory if it does not exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True) # Redundant if db_folder is created above, but safe
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    instrument_cols_sql_create = ""
    for i in range(1, 11):
        instrument_cols_sql_create += f",\n            instrument{i} TEXT DEFAULT NULL"
        instrument_cols_sql_create += f",\n            instrument{i}_prob REAL DEFAULT NULL"

    # --- MODIFIED SCHEMA ---
    # Using new spectral features directly in CREATE TABLE
    # Removed spectral_centroid, spectral_bandwidth, spectral_rolloff,
    # spectral_contrast, spectral_flatness, rms
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS classified_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            features BLOB NOT NULL,
            instrument_features BLOB NOT NULL, /* Stores all instrument probabilities as JSON blob */
            artist TEXT DEFAULT 'Unknown Artist',
            title TEXT DEFAULT 'Unknown Title',
            album TEXT DEFAULT 'Unknown Album',
            year TEXT DEFAULT 'Unknown Year',
            time TEXT DEFAULT '00:00',
            bpm REAL DEFAULT 0.00,
            key TEXT DEFAULT 'Unknown',
            date TEXT NOT NULL,
            tag1 TEXT, tag1_prob REAL DEFAULT NULL,
            tag2 TEXT, tag2_prob REAL DEFAULT NULL,
            tag3 TEXT, tag3_prob REAL DEFAULT NULL,
            tag4 TEXT, tag4_prob REAL DEFAULT NULL,
            tag5 TEXT, tag5_prob REAL DEFAULT NULL,
            tag6 TEXT, tag6_prob REAL DEFAULT NULL,
            tag7 TEXT, tag7_prob REAL DEFAULT NULL,
            tag8 TEXT, tag8_prob REAL DEFAULT NULL,
            tag9 TEXT, tag9_prob REAL DEFAULT NULL,
            tag10 TEXT, tag10_prob REAL DEFAULT NULL
            {instrument_cols_sql_create}, /* For top N instruments */
            artwork_path TEXT DEFAULT NULL,
            artwork_thumbnail_path TEXT DEFAULT NULL,
            -- New spectral features
            noisy REAL DEFAULT NULL,
            tonal REAL DEFAULT NULL,
            dark REAL DEFAULT NULL,
            bright REAL DEFAULT NULL,
            percussive REAL DEFAULT NULL,
            smooth REAL DEFAULT NULL,
            lufs TEXT DEFAULT NULL, -- Storing as text e.g., "-23.5" or "-23.5 LUFS"
            -- Mood and other high-level features
            happiness REAL DEFAULT NULL,
            party REAL DEFAULT NULL,
            aggressive REAL DEFAULT NULL,
            danceability REAL DEFAULT NULL,
            relaxed REAL DEFAULT NULL,
            sad REAL DEFAULT NULL,
            engagement REAL DEFAULT NULL,
            approachability REAL DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

def check_db_integrity():
    """Run PRAGMA integrity_check to confirm the DB is healthy."""
    try:
        with sqlite3.connect(db_path) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                print("Database integrity check: OK")
            else:
                print(f"Database integrity check failed: {result}", file=sys.stderr)
    except Exception as e:
        print(f"Error checking database integrity: {e}", file=sys.stderr)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
print_lock = threading.Lock()

def normalize_audio(audio_array): # Renamed for clarity
    """Normalize audio array to [-1, 1]."""
    denom = np.max(np.abs(audio_array)) # Simpler max abs
    if denom == 0:
        return audio_array
    return audio_array / denom

def md5_hash(data):
    return hashlib.md5(data).hexdigest()

# ------------------------------------------------------------------------------
# Audio Processing and Classification
# ------------------------------------------------------------------------------
def classify_track(filepath):
    """Classify the track using the integrated models."""
    try:
        # Unpack instrument_scores as well
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
        danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
        instrument_scores_dict = process_audio_file(filepath, embedding_model, classification_model, instrument_model)
        if features: # Check if features (genre) were successfully extracted
            return features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
                   danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
                   instrument_scores_dict
        return {}, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error classifying {filepath}: {e}", file=sys.stderr)
        return {}, None, None, None, None, None, None, None, None, None, None, None


def process_audio_file(audio_file_path, emb_model, class_model, instr_model):
    """Process an audio file and return genre features, mood scores, instrument scores, and audio data."""
    try:
        # Load and normalize audio
        audio_44k_orig = MonoLoader(filename=audio_file_path, sampleRate=44100, resampleQuality=4)()
        audio_44k = normalize_audio(audio_44k_orig) # Use the renamed normalization function

        resampler = Resample(inputSampleRate=44100, outputSampleRate=16000, quality=4)
        audio_16k = resampler(audio_44k) # Resample from normalized 44.1kHz audio

        embeddings = emb_model(audio_16k)

        # Genre predictions
        genre_predictions = class_model(embeddings)
        genre_predictions_mean = np.mean(genre_predictions, axis=0)
        genre_result = {class_labels[i]: float(genre_predictions_mean[i]) for i in range(len(class_labels)) if class_labels}


        # Helper for mood model predictions
        def predict_mood(model, embeddings_data, model_name):
            if model is None: return None
            try:
                predictions = model(embeddings_data)
                predictions_mean = np.mean(predictions, axis=0)
                # Assuming binary classification [negative_class_prob, positive_class_prob]
                # or single output [positive_class_prob]
                if predictions_mean.shape[0] >= 2:
                    return float(predictions_mean[1]) # Probability of the "positive" class (e.g. happy, party)
                return float(predictions_mean[0]) # If only one output, assume it's the positive class prob
            except Exception as e_mood:
                print(f"Error in {model_name} prediction for {audio_file_path}: {e_mood}", file=sys.stderr)
                return None

        # Using the helper for moods. Note: some models might need inversion (1.0 - score)
        # Happiness: often [happy_prob, not_happy_prob] or similar - adjust if model output is different
        # The original script had `1.0 - score` for happiness, aggressive, danceability. This depends on model training.
        # For now, let's assume the model directly outputs the positive class probability.
        # If "happy" is index 0, then score = predictions_mean[0]. If index 1, score = predictions_mean[1].
        # The original logic for happiness was: 1.0 - float(happiness_predictions_mean[1]) if ... else 1.0 - float(happiness_predictions_mean[0])
        # This implies the model might output "not_happy" probability or "happy" probability at different indices.
        # Let's assume direct positive probability for now, and user can adjust.
        # Example for happiness if it's inverted:
        # happiness_raw = predict_mood(happiness_model, embeddings, "happiness")
        # happiness_score = (1.0 - happiness_raw) if happiness_raw is not None else None

        happiness_score = predict_mood(happiness_model, embeddings, "happiness")
        party_score = predict_mood(party_model, embeddings, "party")
        aggressive_score = predict_mood(aggressive_model, embeddings, "aggressive")
        danceability_score = predict_mood(danceability_model, embeddings, "danceability")
        relaxed_score = predict_mood(relaxed_model, embeddings, "relaxed")
        sad_score = predict_mood(sad_model, embeddings, "sad")
        engagement_score = predict_mood(engagement_model, embeddings, "engagement")
        approachability_score = predict_mood(approachability_model, embeddings, "approachability")


        # Instrument prediction
        instrument_scores_dict = None
        if instr_model is not None and instrument_labels:
            try:
                instrument_predictions = instr_model(embeddings)
                instrument_predictions_mean = np.mean(instrument_predictions, axis=0)
                instrument_scores_dict = {instrument_labels[i]: float(instrument_predictions_mean[i]) for i in range(len(instrument_labels))}
            except Exception as ie:
                print(f"Error in instrument prediction for {audio_file_path}: {ie}", file=sys.stderr)

        return genre_result, audio_16k, audio_44k_orig, happiness_score, party_score, aggressive_score, \
               danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
               instrument_scores_dict
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {e}", file=sys.stderr)
        return None, None, None, None, None, None, None, None, None, None, None, None


def track_exists(filepath):
    """Check if the track is already in the database."""
    with db_lock: # Ensure thread-safe database access
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM classified_tracks WHERE path = ?', (filepath,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

def extract_metadata(filepath):
    """Extract metadata from audio file."""
    try:
        tag = TinyTag.get(filepath, image=False) # image=False might speed up if artwork handled separately
        return {
            'artist': tag.artist if tag.artist else 'Unknown Artist',
            'title': tag.title if tag.title else 'Unknown Title',
            'album': tag.album if tag.album else 'Unknown Album',
            'year': str(tag.year) if tag.year else 'Unknown Year', # Ensure year is string
            'duration': (f"{int(tag.duration // 60)}:{int(tag.duration % 60):02d}"
                         if tag.duration else '00:00')
        }
    except Exception as e:
        print(f"Error extracting metadata from {filepath}: {e}", file=sys.stderr)
        return {
            'artist': 'Unknown Artist',
            'title': 'Unknown Title',
            'album': 'Unknown Album',
            'year': 'Unknown Year',
            'duration': '00:00'
        }

def extract_bpm(audio_44k): # Takes 44.1kHz audio as input
    """Extract BPM from an audio file using Essentia."""
    try:
        rhythm_extractor = RhythmExtractor2013(method="degara")
        bpm, _, _, _, _ = rhythm_extractor(audio_44k) # Use the 44.1kHz audio passed in
        # Heuristic: if BPM is very low, try doubling (common for halftime/doubletime)
        # This is subjective and might need adjustment.
        if bpm > 0 and bpm < 80: # Example threshold
             bpm_candidate_double = bpm * 2
             if bpm_candidate_double < 220: # Avoid excessively high BPMs
                 bpm = bpm_candidate_double
        elif bpm > 200: # If very high, try halving
            bpm_candidate_half = bpm / 2
            if bpm_candidate_half > 70:
                bpm = bpm_candidate_half
        return round(bpm, 2) if bpm else 0.00
    except Exception as e:
        print(f"Error extracting BPM: {e}", file=sys.stderr)
        return 0.00

def extract_key(audio_16k): # Takes 16kHz audio as input (as per original logic)
    """Extract the musical key and scale (major/minor) from audio using Essentia."""
    try:
        key_extractor = KeyExtractor()
        key, scale, strength = key_extractor(audio_16k)
        return f"{key} {scale.capitalize()}" if key and scale else 'Unknown'
    except Exception as e:
        print(f"Error extracting key: {e}", file=sys.stderr)
        return 'Unknown'

# ------------------------------------------------------------------------------
# Artwork Extraction
# ------------------------------------------------------------------------------
def extract_artwork(audio_file, track_id_for_log):
    """Extract artwork from the audio file and return paths to original and thumbnail images."""
    artwork_path = None
    artwork_thumbnail_path = None

    try:
        audio = mutagen.File(audio_file)
        if audio is None:
            print(f"No audio metadata found for {audio_file}", file=sys.stderr)
            return None, None

        artwork_data = None
        artwork_extension = 'jpg'  # Default extension

        if isinstance(audio, MP3) and audio.tags:
            for tag_name in audio.tags:
                if tag_name.startswith('APIC'):
                    apic_tag = audio.tags[tag_name]
                    artwork_data = apic_tag.data
                    artwork_extension = 'jpg' if 'jpeg' in apic_tag.mime.lower() else 'png' if 'png' in apic_tag.mime.lower() else 'jpg'
                    break
        elif isinstance(audio, FLAC) and audio.pictures:
            pic = audio.pictures[0]
            artwork_data = pic.data
            artwork_extension = pic.mime.split('/')[-1].lower() if '/' in pic.mime else 'jpg'
        elif isinstance(audio, MP4) and audio.tags:
            covr_data = audio.tags.get('covr')
            if covr_data:
                artwork_data = bytes(covr_data[0])
                artwork_extension = 'jpg' if covr_data[0].imageformat == MP4Cover.FORMAT_JPEG else 'png'
        elif isinstance(audio, AAC) and audio.tags:
            apic_tag = audio.tags.get('APIC:Cover') or audio.tags.get('APIC')
            if apic_tag:
                artwork_data = apic_tag.data
                artwork_extension = 'jpg' if 'jpeg' in apic_tag.mime.lower() else 'png' if 'png' in apic_tag.mime.lower() else 'jpg'

        if artwork_data:
            artwork_hash = md5_hash(artwork_data)
            original_artwork_filename = f"{artwork_hash}.{artwork_extension}"
            artwork_path = os.path.join(artworks_dir, original_artwork_filename)

            if not os.path.exists(artwork_path):
                with open(artwork_path, 'wb') as f:
                    f.write(artwork_data)

            # Create thumbnail
            thumb_ext = 'jpg'  # Standardize thumbnail to JPG for smaller size
            if artwork_extension == 'png':
                thumb_ext = 'png'

            resized_artwork_filename = f"{artwork_hash}_128x128.{thumb_ext}"
            artwork_thumbnail_path = os.path.join(artworks_dir, resized_artwork_filename)

            if not os.path.exists(artwork_thumbnail_path):
                try:
                    with Image.open(io.BytesIO(artwork_data)) as img:
                        img_rgb = img.convert("RGB") if thumb_ext == 'jpg' else img.convert("RGBA")
                        img_rgb.thumbnail((128, 128), Image.Resampling.LANCZOS)
                        save_format = 'JPEG' if thumb_ext == 'jpg' else 'PNG'
                        img_rgb.save(artwork_thumbnail_path, format=save_format, quality=85 if save_format=='JPEG' else None)
                except Exception as img_error:
                    print(f"Error creating thumbnail for {audio_file}: {img_error}", file=sys.stderr)
                    artwork_thumbnail_path = None

            print(f"Successfully extracted artwork for {audio_file}", file=sys.stderr)
        else:
            print(f"No artwork found in {audio_file}", file=sys.stderr)

    except Exception as e:
        print(f"Error processing artwork for file {audio_file} (Track ID {track_id_for_log}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    
    return artwork_path, artwork_thumbnail_path


# ------------------------------------------------------------------------------
# (Incorporated and Refined) Spectral Analysis
# ------------------------------------------------------------------------------
def init_spectral_columns():
    """
    Ensures classified_tracks table has new spectral/mood/instrument columns.
    Removes old spectral columns if they exist (for migrating older DBs).
    If table is created by new init_db, this primarily checks and confirms schema.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(classified_tracks)")
    existing_cols_info = {row[1]: row[2] for row in cursor.fetchall()} # col_name: col_type

    # New columns that should exist (name, type)
    # (lufs is TEXT)
    new_feature_cols = [
        ("noisy", "REAL"), ("tonal", "REAL"), ("dark", "REAL"), ("bright", "REAL"),
        ("percussive", "REAL"), ("smooth", "REAL"), ("lufs", "TEXT"),
        ("happiness", "REAL"), ("party", "REAL"), ("aggressive", "REAL"),
        ("danceability", "REAL"), ("relaxed", "REAL"), ("sad", "REAL"),
        ("engagement", "REAL"), ("approachability", "REAL"),
        ("instrument_features", "BLOB") # Blob for all instrument scores
    ]
    for i in range(1, 11): # Top N instrument names and probabilities
        new_feature_cols.append((f"instrument{i}", "TEXT"))
        new_feature_cols.append((f"instrument{i}_prob", "REAL"))

    # Old columns to be removed if they exist
    old_spectral_columns = [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_contrast", "spectral_flatness", "rms"
    ]

    # Remove old columns
    for old_col in old_spectral_columns:
        if old_col in existing_cols_info:
            try:
                print(f"Attempting to remove old column: {old_col}")
                cursor.execute(f"ALTER TABLE classified_tracks DROP COLUMN {old_col}")
                conn.commit() # Commit each drop
                print(f"Removed old column {old_col} from classified_tracks.")
                existing_cols_info.pop(old_col) # Update our cache of existing columns
            except sqlite3.OperationalError as e:
                # This can happen if the column is part of an index or constraint not handled here
                print(f"Warning: Could not remove column {old_col} (it might be in use or already handled): {e}", file=sys.stderr)


    # Add new columns if they don't exist
    for col_name, col_type in new_feature_cols:
        if col_name not in existing_cols_info:
            try:
                default_clause = "DEFAULT NULL" # All new columns are nullable
                cursor.execute(f"ALTER TABLE classified_tracks ADD COLUMN {col_name} {col_type} {default_clause}")
                conn.commit() # Commit each add
                print(f"Added column {col_name} ({col_type}) to classified_tracks.")
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column {col_name}: {e}", file=sys.stderr)
    conn.close()

# --- MODIFIED/ENHANCED SPECTRAL FEATURES CALCULATION ---
def analyze_spectral_features(file_path):
    """
    Load audio with librosa and compute specific spectral features.
    Features are normalized or designed to be in a meaningful range (often 0-1).
    """
    try:
        # Load a segment of audio (e.g., first 30s or a middle segment)
        # Using a consistent sample rate for librosa features
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0, res_type='kaiser_fast')
        except Exception as audio_load_error:
            print(f"Error loading audio with librosa: {audio_load_error}. Trying alternative method...", file=sys.stderr)
            # Fallback to using soundfile directly
            import soundfile as sf
            y, sr = sf.read(file_path)
            if len(y.shape) > 1:  # Convert to mono if stereo
                y = np.mean(y, axis=1)
            # Resample if needed
            if sr != 22050:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=22050)
                sr = 22050
            # Take first 30 seconds
            y = y[:int(30.0 * sr)]

        if len(y) == 0:
            print(f"Warning: Audio loaded from {file_path} is empty. Skipping spectral analysis.", file=sys.stderr)
            return None

        # 1. Noisy / Tonal
        # Spectral flatness: 0 (tonal) to 1 (noisy)
        # Zero-crossing rate: Higher for noisy signals
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        mean_flatness = np.mean(spectral_flatness)
        mean_zcr = np.mean(zcr) # ZCR is already 0-1 scaled by librosa based on frame length

        # Combine: simple average, both features contribute to "noisiness"
        noisy_score = float(np.clip((mean_flatness + mean_zcr) / 2.0, 0.0, 1.0))
        tonal_score = 1.0 - noisy_score

        # 2. Dark / Bright
        # Spectral centroid: Higher for brighter sounds
        # MFCC[0] (after abs, mean): Relates to energy in lower mel bands. Higher might mean less "dark".
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # We only need the first MFCC

        norm_centroid = np.mean(spectral_centroid) / (sr / 2.0) # Normalize by Nyquist

        # Normalize MFCC[0] energy for the current segment to approx 0-1
        abs_mfcc0_frames = np.abs(mfccs[0,:])
        mfcc0_min = np.min(abs_mfcc0_frames)
        mfcc0_max = np.max(abs_mfcc0_frames)
        
        if (mfcc0_max - mfcc0_min) > 1e-6: # Avoid division by zero if flat
            norm_mfcc0_energy_frames = (abs_mfcc0_frames - mfcc0_min) / (mfcc0_max - mfcc0_min)
            mean_norm_mfcc0_energy = np.mean(norm_mfcc0_energy_frames)
        elif mfcc0_max > 1e-6 : # if not flat but min is same as max (single value)
            mean_norm_mfcc0_energy = 1.0 if mfcc0_max > np.median(abs_mfcc0_frames) else 0.0 # Crude guess
        else: # effectively zero
            mean_norm_mfcc0_energy = 0.0


        # Brighter = higher centroid, higher (normalized) MFCC0 energy (less low-freq dominance)
        # This combination is heuristic. Higher MFCC0 energy could also mean "powerful" rather than "bright".
        # Consider `(norm_centroid * 0.7 + mean_norm_mfcc0_energy * 0.3)` if centroid is more dominant.
        bright_score = float(np.clip((norm_centroid + mean_norm_mfcc0_energy) / 2.0, 0.0, 1.0))
        dark_score = 1.0 - bright_score

        # 3. Percussive / Smooth
        # Onset strength: Measures rapid changes, higher for percussive
        # We'll use a high-pass filtered RMS energy for a simpler percussiveness indicator
        # Or, stick to onset envelope standard deviation as a measure of "peakiness"
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Percussiveness can be related to the variance or peakiness of the onset envelope
        # High standard deviation of onset_env means more dynamic changes (more percussive)
        # Normalize std dev: A simple way is to divide by mean if mean is not zero.
        # Or clip based on typical observed ranges.
        onset_std = np.std(onset_env)
        onset_mean = np.mean(onset_env)
        if onset_mean > 1e-6:
            percussive_score = float(np.clip(onset_std / onset_mean, 0.0, 2.0) / 2.0) # Heuristic scaling to 0-1
        else:
            percussive_score = 0.0 # If no onsets, not percussive
        percussive_score = float(np.clip(percussive_score, 0.0, 1.0))
        smooth_score = 1.0 - percussive_score
        
        # 4. LUFS (Loudness Units Full Scale)
        lufs_value_str = "N/A"
        if PYLOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(rate=sr) # BS.1770 meter
                # Note: librosa.load returns float32 in [-1, 1] by default. pyloudnorm expects this.
                integrated_loudness = meter.integrated_loudness(y)
                lufs_value_str = f"{integrated_loudness:.1f}"
            except Exception as lufs_e:
                print(f"pyloudnorm error for {file_path}: {lufs_e}. Falling back to RMS.", file=sys.stderr)
                # Fallback to RMS-based if pyloudnorm fails
                rms_value = np.mean(librosa.feature.rms(y=y))
                if rms_value > 1e-10: # Avoid log(0)
                    lufs_db = 20 * np.log10(rms_value)
                    lufs_value_str = f"{lufs_db:.1f} (RMS)"
                else:
                    lufs_value_str = "-inf (RMS)"
        else: # Fallback if pyloudnorm not installed
            rms_value = np.mean(librosa.feature.rms(y=y))
            if rms_value > 1e-10:
                lufs_db = 20 * np.log10(rms_value)
                lufs_value_str = f"{lufs_db:.1f} (RMS)"
            else:
                lufs_value_str = "-inf (RMS)"

        return {
            "noisy": noisy_score,
            "tonal": tonal_score,
            "dark": dark_score,
            "bright": bright_score,
            "percussive": percussive_score,
            "smooth": smooth_score,
            "lufs": lufs_value_str
        }

    except Exception as e:
        print(f"Error extracting spectral features from {file_path}: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc(file=sys.stderr) # For debugging
        return None


def update_spectral_features_in_db(track_id, features_dict):
    """Update classified_tracks with spectral features for the given track ID."""
    if not features_dict: return

    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE classified_tracks
                SET noisy = ?, tonal = ?, dark = ?, bright = ?,
                    percussive = ?, smooth = ?, lufs = ?
                WHERE id = ?
            ''', (
                features_dict["noisy"], features_dict["tonal"], features_dict["dark"], features_dict["bright"],
                features_dict["percussive"], features_dict["smooth"], features_dict["lufs"],
                track_id
            ))
            conn.commit()
        except Exception as e:
            print(f"Error updating spectral features for track ID {track_id}: {e}", file=sys.stderr)
        finally:
            conn.close()


def get_unanalyzed_tracks_for_full_reprocessing(): # Renamed for clarity
    """Return (id, path) for tracks missing *any* key generated features."""
    # This query checks for spectral, mood, or primary instrument features being NULL.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = '''
        SELECT id, path
        FROM classified_tracks
        WHERE noisy IS NULL OR tonal IS NULL OR dark IS NULL OR bright IS NULL OR
              percussive IS NULL OR smooth IS NULL OR lufs IS NULL OR
              happiness IS NULL OR /* Add other mood features if they are always expected */
              instrument1 IS NULL /* Check if primary instrument tag is missing */
    '''
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        # This might happen if DB schema is still old and columns like 'noisy' don't exist.
        print(f"Error querying unanalyzed tracks (DB schema might be outdated, run init_db/init_spectral_columns): {e}", file=sys.stderr)
        rows = []
    conn.close()
    return rows

def process_single_track_for_spectral_reanalysis(track_tuple): # Renamed
    """Helper for parallel spectral re-analysis if only spectral features are missing."""
    track_id, file_path = track_tuple
    if not os.path.isfile(file_path):
        print(f"File not found, skipping spectral re-analysis for: {file_path} (ID: {track_id})", file=sys.stderr)
        return

    features = analyze_spectral_features(file_path)
    if features:
        update_spectral_features_in_db(track_id, features)
    # else: No features extracted, error already printed by analyze_spectral_features

def analyze_missing_spectral_features_only(num_workers): # Renamed for clarity
    """Re-analyzes *only* spectral features for tracks where they are missing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Query for tracks missing *only* spectral features (other main features might be present)
    cursor.execute('''
        SELECT id, path
        FROM classified_tracks
        WHERE (noisy IS NULL OR tonal IS NULL OR dark IS NULL OR bright IS NULL OR
               percussive IS NULL OR smooth IS NULL OR lufs IS NULL)
    ''')
    unanalyzed_spectral_tracks = cursor.fetchall()
    conn.close()

    total_unanalyzed = len(unanalyzed_spectral_tracks)
    if total_unanalyzed == 0:
        print("No tracks found missing only spectral features for re-analysis.")
        return

    print(f"Found {total_unanalyzed} track(s) missing spectral features. Processing for spectral re-analysis...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_track = {
            executor.submit(process_single_track_for_spectral_reanalysis, track_tuple): track_tuple
            for track_tuple in unanalyzed_spectral_tracks
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_track), total=total_unanalyzed, desc="Re-analyzing missing spectral features"):
            track_tuple_completed = future_to_track[future]
            try:
                future.result(timeout=90) # Timeout for spectral analysis of one file
            except TimeoutError:
                print(f"Timeout during spectral re-analysis of track {track_tuple_completed[1]} (ID: {track_tuple_completed[0]})", file=sys.stderr)
            except Exception as e_spec:
                print(f"Error in spectral re-processing for track {track_tuple_completed[1]} (ID: {track_tuple_completed[0]}): {e_spec}", file=sys.stderr)

    print("Missing spectral feature re-analysis completed.")

# ------------------------------------------------------------------------------
# End-to-End Processing of One Audio File
# ------------------------------------------------------------------------------
def process_audio_file_path(file_path_to_process):
    """Process a single audio file path: classification, metadata, spectral, DB insert/update."""
    try:
        # Essentia-based classification (genre, moods, instruments) and audio loading
        genre_features, audio_16k, audio_44k, happiness_sc, party_sc, aggressive_sc, \
        danceability_sc, relaxed_sc, sad_sc, engagement_sc, approachability_sc, \
        instrument_features_dict = classify_track(file_path_to_process)

        if not genre_features or audio_16k is None or audio_44k is None:
            print(f"Core classification failed for {file_path_to_process}. Skipping further processing.", file=sys.stderr)
            return

        # Metadata and basic audio features
        metadata = extract_metadata(file_path_to_process)
        key_val = extract_key(audio_16k)
        bpm_val = extract_bpm(audio_44k)
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare top N genre tags
        tags_sorted = sorted(genre_features.items(), key=lambda x: x[1], reverse=True)[:10]
        tag_names_map = {}
        tag_probs_map = {}
        for i, (genre, prob) in enumerate(tags_sorted):
            tag_names_map[f'tag{i+1}'] = genre
            tag_probs_map[f'tag{i+1}_prob'] = float(prob)
        genre_features_blob = json.dumps(genre_features).encode('utf-8')

        # Prepare ALL instrument features for blob and top N for individual columns
        all_instrument_scores_blob = b"{}"
        top_instrument_names_map = {}
        top_instrument_probs_map = {}
        if instrument_features_dict:
            significant_instrument_scores = {
                k: v for k, v in instrument_features_dict.items() if v > 0.01
            }
            all_instrument_scores_blob = json.dumps(significant_instrument_scores).encode('utf-8')

            sorted_instruments = sorted(instrument_features_dict.items(), key=lambda item: item[1], reverse=True)
            for i, (instrument, prob) in enumerate(sorted_instruments[:10]):
                top_instrument_names_map[f'instrument{i+1}'] = instrument
                top_instrument_probs_map[f'instrument{i+1}_prob'] = float(prob)

        # Librosa-based spectral analysis
        perceptual_spectral_features = analyze_spectral_features(file_path_to_process)
        if perceptual_spectral_features is None:
            perceptual_spectral_features = {
                "noisy": None, "tonal": None, "dark": None, "bright": None,
                "percussive": None, "smooth": None, "lufs": None
            }

        # Extract artwork first
        artwork_original_path, artwork_thumb_path = extract_artwork(file_path_to_process, None)

        # Database insertion/update
        with db_lock:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            instrument_col_names_list = []
            instrument_col_placeholders_list = []
            instrument_values_for_tuple_list = []
            for i in range(1, 11):
                instrument_col_names_list.append(f"instrument{i}")
                instrument_col_names_list.append(f"instrument{i}_prob")
                instrument_col_placeholders_list.append("?")
                instrument_col_placeholders_list.append("?")
                instrument_values_for_tuple_list.append(top_instrument_names_map.get(f'instrument{i}'))
                instrument_values_for_tuple_list.append(top_instrument_probs_map.get(f'instrument{i}_prob'))

            instrument_col_names_str = ", " + ", ".join(instrument_col_names_list) if instrument_col_names_list else ""
            instrument_col_placeholders_str = ", " + ", ".join(instrument_col_placeholders_list) if instrument_col_placeholders_list else ""

            sql_insert_query = f'''
                INSERT OR REPLACE INTO classified_tracks (
                    path, features, instrument_features, artist, title, album, year, time, bpm, key, date,
                    tag1, tag1_prob, tag2, tag2_prob, tag3, tag3_prob, tag4, tag4_prob, tag5, tag5_prob,
                    tag6, tag6_prob, tag7, tag7_prob, tag8, tag8_prob, tag9, tag9_prob, tag10, tag10_prob
                    {instrument_col_names_str},
                    artwork_path, artwork_thumbnail_path,
                    noisy, tonal, dark, bright, percussive, smooth, lufs,
                    happiness, party, aggressive, danceability, relaxed, sad, engagement, approachability
                ) VALUES ({", ".join(["?"] * (31 + len(instrument_values_for_tuple_list) + 2 + 7 + 8))})
            '''

            values_tuple = (
                file_path_to_process, genre_features_blob, all_instrument_scores_blob,
                metadata['artist'], metadata['title'], metadata['album'], metadata['year'], metadata['duration'],
                bpm_val, key_val, current_date,
                # Tags (20 values)
                *[val for i in range(1, 11) for val in (tag_names_map.get(f'tag{i}'), tag_probs_map.get(f'tag{i}_prob'))],
                # Instruments (N*2 values)
                *instrument_values_for_tuple_list,
                # Artwork paths
                artwork_original_path, artwork_thumb_path,
                # New spectral features (7 values)
                perceptual_spectral_features["noisy"], perceptual_spectral_features["tonal"],
                perceptual_spectral_features["dark"], perceptual_spectral_features["bright"],
                perceptual_spectral_features["percussive"], perceptual_spectral_features["smooth"],
                perceptual_spectral_features["lufs"],
                # Mood features (8 values)
                happiness_sc, party_sc, aggressive_sc, danceability_sc,
                relaxed_sc, sad_sc, engagement_sc, approachability_sc
            )

            try:
                cursor.execute(sql_insert_query, values_tuple)
                conn.commit()
                print(f"Successfully processed and stored {file_path_to_process}", file=sys.stderr)
            except Exception as db_error:
                print(f"Database error for {file_path_to_process}: {db_error}", file=sys.stderr)
                conn.rollback()
            finally:
                conn.close()

    except Exception as e_main_process:
        print(f"FATAL error processing file {file_path_to_process}: {e_main_process}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


# ------------------------------------------------------------------------------
# Process All Audio Files in a Folder (with concurrency + timeout)
# ------------------------------------------------------------------------------
def process_folder(folder_to_scan, num_concurrent_processes):
    """Process all audio files in the folder (recursively)."""
    audio_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.opus', '.wma', '.aac'}
    files_to_process_paths = []
    already_classified_count = 0

    print(f"Scanning folder: {folder_to_scan}...")
    # Collect all file paths first
    all_audio_file_paths_in_folder = []
    for root, _, files in os.walk(folder_to_scan):
        for file_name in files:
            if Path(file_name).suffix.lower() in audio_formats:
                all_audio_file_paths_in_folder.append(os.path.join(root, file_name))
    
    print(f"Found {len(all_audio_file_paths_in_folder)} audio files in scan.")

    # Check existence in DB
    for fp in tqdm(all_audio_file_paths_in_folder, desc="Checking existing tracks in DB"):
        if not track_exists(fp):
            files_to_process_paths.append(fp)
        else:
            already_classified_count += 1
    
    total_files_in_scan = len(files_to_process_paths) + already_classified_count
    print(f"Number of tracks already classified: {already_classified_count}")
    print(f"Number of new/updated tracks to process: {len(files_to_process_paths)}")
    print(f"Total audio tracks found in folder: {total_files_in_scan}")

    if not files_to_process_paths:
        print("No new tracks to process.")
        return

    # Use ProcessPoolExecutor for CPU-bound tasks like audio processing
    with ProcessPoolExecutor(max_workers=num_concurrent_processes) as executor:
        future_to_path_map = {
            executor.submit(process_audio_file_path, fp_single): fp_single
            for fp_single in files_to_process_paths
        }

        for future_item in tqdm(
            concurrent.futures.as_completed(future_to_path_map),
            total=len(future_to_path_map),
            desc="Processing audio files",
            file=sys.stdout # Ensure tqdm output goes to actual stdout
        ):
            path_processed = future_to_path_map[future_item]
            try:
                future_item.result(timeout=300) # Increased timeout per file (5 minutes)
            except TimeoutError:
                print(f"Timeout processing file ( ممکن است فایل خراب باشد یا بسیار طولانی): {path_processed}", file=sys.stderr)
            except Exception as e_folder_proc:
                print(f"Error processing file {path_processed} in pool: {e_folder_proc}", file=sys.stderr)
                # import traceback; traceback.print_exc(file=sys.stderr) # For detailed debugging

    print("Folder processing completed.")


# ------------------------------------------------------------------------------
# Vector Database for Similarity Search
# ------------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None # type: ignore
    FAISS_AVAILABLE = False
    print("Warning: Faiss library not installed. Vector indexing and similarity search will not be available.", file=sys.stderr)

def create_embedding_vector_from_genre_features(genre_features_dict): # Renamed
    """Create L2 normalized embedding vector from genre features using global class_labels order."""
    if not class_labels:
        # print("Genre class_labels not loaded. Cannot create genre embedding vector.", file=sys.stderr) # Too verbose
        return np.array([], dtype=np.float32)

    # Ensure all labels are present, defaulting to 0.0 for missing ones
    genre_vector = np.array([genre_features_dict.get(label, 0.0) for label in class_labels], dtype=np.float32)
    
    # L2 normalization
    norm = np.linalg.norm(genre_vector)
    if norm > 0:
        return genre_vector / norm
    return genre_vector # Return zero vector if norm is zero


def build_faiss_vector_index(): # Renamed for clarity
    """Build a FAISS vector index using stored 'features' (genre probabilities) from the database."""
    if not FAISS_AVAILABLE:
        print("Faiss not available. Skipping vector index build.", file=sys.stderr)
        return None, None
    if not class_labels:
        print("Genre class_labels are not loaded. Cannot build vector index.", file=sys.stderr)
        return None, None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, path, features FROM classified_tracks WHERE features IS NOT NULL')
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error reading features for vector index: {e}", file=sys.stderr)
        rows = []
    conn.close()

    if not rows:
        print("No tracks with features found in database to build index.", file=sys.stderr)
        return None, None

    embeddings_list = []
    track_ids_for_map = [] # Stores DB track IDs corresponding to index positions

    for track_id, path, features_blob in tqdm(rows, desc="Generating embeddings for FAISS index"):
        try:
            features_dict_from_db = json.loads(features_blob.decode('utf-8')) # Decode blob then parse
        except Exception as e_json:
            print(f"Error parsing features blob for track ID {track_id} ({path}): {e_json}", file=sys.stderr)
            continue

        emb_vector = create_embedding_vector_from_genre_features(features_dict_from_db)
        if emb_vector.size == 0: # Skip if embedding couldn't be created
            continue
        embeddings_list.append(emb_vector)
        track_ids_for_map.append(track_id) # Store the database ID

    if not embeddings_list:
        print("No valid embeddings generated. Cannot build FAISS index.", file=sys.stderr)
        return None, None

    embeddings_np_array = np.vstack(embeddings_list).astype('float32')
    dimension = embeddings_np_array.shape[1]

    if dimension == 0:
        print("Embedding dimension is 0. Cannot build FAISS index.", file=sys.stderr)
        return None, None

    faiss_index = faiss.IndexFlatL2(dimension) # Using L2 distance
    faiss_index.add(embeddings_np_array)
    print(f"FAISS index built with {faiss_index.ntotal} embeddings. Dimension: {dimension}")

    index_file_path = os.path.join(script_dir, "vector_index.faiss")
    map_file_path = os.path.join(script_dir, "track_index_to_id_map.json") # Map from FAISS index pos to DB track ID

    faiss.write_index(faiss_index, index_file_path)
    with open(map_file_path, "w") as f_map:
        json.dump(track_ids_for_map, f_map)

    print(f"FAISS index saved to: {index_file_path}")
    print(f"Track ID map saved to: {map_file_path}")

    return faiss_index, track_ids_for_map


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    """Main function to parse arguments and orchestrate audio processing."""
    parser = argparse.ArgumentParser(description="Process and classify audio files, storing results in a database.")
    parser.add_argument("path", help="Path to the folder containing audio files or a single audio file.")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Number of processes for concurrent file processing.")
    parser.add_argument("--build-index", action="store_true", help="Build or rebuild the FAISS vector index after processing.")
    parser.add_argument("--reanalyze-spectral", action="store_true", help="Re-analyze spectral features for tracks missing them in the DB.")
    args = parser.parse_args()

    print(f"Using database at: {db_path}")
    init_db() # Initialize DB schema (creates table if not exists with new schema)
    check_db_integrity() # Check DB health
    init_spectral_columns() # Add/remove columns for new/old spectral features (migration step)

    target_path = args.path
    num_processes = args.threads # Renamed for clarity

    if os.path.isdir(target_path):
        process_folder(target_path, num_processes)
    elif os.path.isfile(target_path):
        # if not track_exists(target_path): # Process even if it exists, to ensure all fields are up-to-date
        print(f"Processing single file: {target_path}", file=sys.stdout) # Use actual stdout
        process_audio_file_path(target_path) # This function handles DB insertion/replacement
        print(f"Processing completed for file: {target_path}", file=sys.stdout)
        # else:
            # print(f"Track already classified: {target_path}. It will be re-processed and updated.", file=sys.stdout)
            # process_audio_file_path(target_path) # Re-process to update
    else:
        print(f"Error: The path {target_path} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)


    if args.reanalyze_spectral:
        print("Starting re-analysis of missing spectral features...")
        analyze_missing_spectral_features_only(num_processes)

    if args.build_index:
        print("Building FAISS vector index...")
        build_faiss_vector_index()

    print("Main script execution finished.")


# ------------------------------------------------------------------------------
# Safe Entry for Multiprocessing on Windows
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support() # Needed for PyInstaller/cx_Freeze compatibility on Windows

    try:
        main()
    except Exception as unhandled_e_main:
        print(f"Unhandled exception in main execution: {unhandled_e_main}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        # Close the error log file if it was opened
        if error_log_file and not error_log_file.closed:
            sys.stderr.flush()
            error_log_file.close()
        # Restore original stderr if it was changed
        if original_stderr and sys.stderr != original_stderr:
            sys.stderr = original_stderr
        print("Script finished. Error log (if any) is at: " + error_log_path)