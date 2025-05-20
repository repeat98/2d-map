#!/usr/bin/env python
# analyze.py

import os, sys

# Attempt to disable Essentia warnings via environment variables.
os.environ["ESSENTIA_LOG_DISABLE"] = "1"
os.environ["ESSENTIA_ENABLE_WARNINGS"] = "0"
os.environ["ESSENTIA_LOG_LEVEL"] = "error"

import sys
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
# (On Windows, replace os.devnull with "nul".)
try:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
except OSError: # In some environments like GitHub Actions, this might fail or be unnecessary
    print("Could not redirect C++ stderr to /dev/null.", file=sys.__stderr__) # Use original stderr for this message


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
from multiprocessing import Process, freeze_support, Process

# Import sklearn for similarity matrix computation (if needed elsewhere)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# Setup error logging to a file
# ------------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

error_log_path = os.path.join(script_dir, "error.log")
error_log_file = open(error_log_path, 'a', buffering=1)  # line-buffered
original_stderr = sys.stderr
sys.stderr = error_log_file

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
# New MTG Jamendo Instrument model path
instrument_model_path = os.path.join(script_dir, "essentia_model/mtg_jamendo_instrument-discogs-effnet-1.pb") # ADJUST THIS PATH

class_labels_path = os.path.join(script_dir, 'essentia_model/genre_discogs400-discogs-effnet-1.json')
# New MTG Jamendo Instrument labels path
instrument_labels_path = os.path.join(script_dir, 'essentia_model/mtg_jamendo_instrument-discogs-effnet-1.json') # ADJUST THIS PATH

db_path = os.path.join(script_dir, "../db/tracks.db")

# Paths for artwork storage
artworks_dir = os.path.join(script_dir, '../assets/artworks')
os.makedirs(artworks_dir, exist_ok=True)

# Load models
try:
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_model_path,
        output="PartitionedCall:1",
        patchHopSize=64
    )
except Exception as e:
    print(f"Error while loading Tensorflow embedding model with path: {embedding_model_path}", file=sys.stderr)
    raise e

try:
    classification_model = TensorflowPredict2D(
        graphFilename=classification_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )
except Exception as e:
    print(f"Error while loading Tensorflow classification model with path: {classification_model_path}", file=sys.stderr)
    raise e

try:
    happiness_model = TensorflowPredict2D(
        graphFilename=happiness_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow happiness model with path: {happiness_model_path}", file=sys.stderr)
    happiness_model = None
    print("Happiness analysis will be skipped.")

try:
    party_model = TensorflowPredict2D(
        graphFilename=party_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow party model with path: {party_model_path}", file=sys.stderr)
    party_model = None
    print("Party mood analysis will be skipped.")

try:
    aggressive_model = TensorflowPredict2D(
        graphFilename=aggressive_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow aggressive model with path: {aggressive_model_path}", file=sys.stderr)
    aggressive_model = None
    print("Aggressive mood analysis will be skipped.")

try:
    danceability_model = TensorflowPredict2D(
        graphFilename=danceability_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow danceability model with path: {danceability_model_path}", file=sys.stderr)
    danceability_model = None
    print("Danceability analysis will be skipped.")

try:
    relaxed_model = TensorflowPredict2D(
        graphFilename=relaxed_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow relaxed model with path: {relaxed_model_path}", file=sys.stderr)
    relaxed_model = None
    print("Relaxed mood analysis will be skipped.")

try:
    sad_model = TensorflowPredict2D(
        graphFilename=sad_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow sad model with path: {sad_model_path}", file=sys.stderr)
    sad_model = None
    print("Sad mood analysis will be skipped.")

try:
    engagement_model = TensorflowPredict2D(
        graphFilename=engagement_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow engagement model with path: {engagement_model_path}", file=sys.stderr)
    engagement_model = None
    print("Engagement analysis will be skipped.")

try:
    approachability_model = TensorflowPredict2D(
        graphFilename=approachability_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow approachability model with path: {approachability_model_path}", file=sys.stderr)
    approachability_model = None
    print("Approachability analysis will be skipped.")

# New MTG Jamendo Instrument model loader
try:
    instrument_model = TensorflowPredict2D(
        graphFilename=instrument_model_path,
        # input=" Placeholder name if different from default", # Adjust if necessary
        # output=" Softmax output name if different from default", # Adjust if necessary
    )
except Exception as e:
    print(f"Error while loading Tensorflow instrument model with path: {instrument_model_path}", file=sys.stderr)
    instrument_model = None
    print("Instrument detection will be skipped.")


# Load class labels for genre
try:
    with open(class_labels_path, 'r') as file:
        class_labels = json.load(file).get("classes", [])
except Exception as e:
    print(f"Error loading genre class labels: {e}", file=sys.stderr)
    class_labels = []

# Load class labels for instruments
try:
    with open(instrument_labels_path, 'r') as file:
        instrument_labels = json.load(file).get("classes", []) # Assuming 'classes' key
except Exception as e:
    print(f"Error loading instrument class labels: {e}", file=sys.stderr)
    instrument_labels = []


db_lock = Lock()

# ------------------------------------------------------------------------------
# Database Initialization and Integrity Check
# ------------------------------------------------------------------------------
def init_db():
    """Initialize the database and create the directory if it does not exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    instrument_cols_sql_create = ""
    for i in range(1, 11):
        instrument_cols_sql_create += f",\n            instrument{i} TEXT DEFAULT NULL"
        instrument_cols_sql_create += f",\n            instrument{i}_prob REAL DEFAULT NULL"

    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS classified_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            features BLOB NOT NULL,
            instrument_features BLOB NOT NULL,
            artist TEXT DEFAULT 'Unknown Artist',
            title TEXT DEFAULT 'Unknown Title',
            album TEXT DEFAULT 'Unknown Album',
            year TEXT DEFAULT 'Unknown Year',
            time TEXT DEFAULT '00:00',
            bpm REAL DEFAULT 0.00,
            key TEXT DEFAULT 'Unknown',
            date TEXT NOT NULL,
            tag1 TEXT,
            tag1_prob REAL DEFAULT NULL,
            tag2 TEXT,
            tag2_prob REAL DEFAULT NULL,
            tag3 TEXT,
            tag3_prob REAL DEFAULT NULL,
            tag4 TEXT,
            tag4_prob REAL DEFAULT NULL,
            tag5 TEXT,
            tag5_prob REAL DEFAULT NULL,
            tag6 TEXT,
            tag6_prob REAL DEFAULT NULL,
            tag7 TEXT,
            tag7_prob REAL DEFAULT NULL,
            tag8 TEXT,
            tag8_prob REAL DEFAULT NULL,
            tag9 TEXT,
            tag9_prob REAL DEFAULT NULL,
            tag10 TEXT,
            tag10_prob REAL DEFAULT NULL
            {instrument_cols_sql_create},
            artwork_path TEXT DEFAULT NULL,
            artwork_thumbnail_path TEXT DEFAULT NULL,
            spectral_centroid REAL DEFAULT NULL,
            spectral_bandwidth REAL DEFAULT NULL,
            spectral_rolloff REAL DEFAULT NULL,
            spectral_contrast REAL DEFAULT NULL,
            spectral_flatness REAL DEFAULT NULL,
            rms REAL DEFAULT NULL,
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

def normalize(audio):
    denom = np.max(np.abs(audio), axis=0)
    if denom == 0:
        return audio
    return audio / denom

def md5_hash(data):
    return hashlib.md5(data).hexdigest()

# ------------------------------------------------------------------------------
# Audio Processing and Classification
# ------------------------------------------------------------------------------
def classify_track(filepath):
    """Classify the track using the integrated models."""
    try:
        # Updated to unpack instrument_scores as well
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
        danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
        instrument_scores = process_audio_file(filepath, embedding_model, classification_model, instrument_model) # Pass instrument_model
        if features:
            return features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
                   danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
                   instrument_scores
        return {}, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error classifying {filepath}: {e}", file=sys.stderr)
        return {}, None, None, None, None, None, None, None, None, None, None, None


def process_audio_file(audio_file_path, embedding_model, classification_model, current_instrument_model): # Added current_instrument_model
    """Process an audio file and return genre features, mood scores, instrument scores, and audio data."""
    try:
        audio_44k = MonoLoader(filename=audio_file_path, sampleRate=44100, resampleQuality=4)()
        audio_44k = normalize(audio_44k)

        resample = Resample(inputSampleRate=44100, outputSampleRate=16000, quality=4)
        audio_16k = resample(audio_44k)

        embeddings = embedding_model(audio_16k)

        genre_predictions = classification_model(embeddings)
        genre_predictions_mean = np.mean(genre_predictions, axis=0)
        genre_result = {class_labels[i]: float(genre_predictions_mean[i]) for i in range(len(class_labels))}

        happiness_score = None
        global happiness_model
        if happiness_model is not None:
            try:
                happiness_predictions = happiness_model(embeddings)
                happiness_predictions_mean = np.mean(happiness_predictions, axis=0)
                happiness_score = 1.0 - float(happiness_predictions_mean[1]) if happiness_predictions_mean.shape[0] >= 2 else 1.0 - float(happiness_predictions_mean[0])
            except Exception as he:
                print(f"Error in happiness prediction for {audio_file_path}: {he}", file=sys.stderr)

        party_score = None
        global party_model
        if party_model is not None:
            try:
                party_predictions = party_model(embeddings)
                party_predictions_mean = np.mean(party_predictions, axis=0)
                party_score = float(party_predictions_mean[1]) if party_predictions_mean.shape[0] >= 2 else float(party_predictions_mean[0])
            except Exception as pe:
                print(f"Error in party mood prediction for {audio_file_path}: {pe}", file=sys.stderr)

        aggressive_score = None
        global aggressive_model
        if aggressive_model is not None:
            try:
                aggressive_predictions = aggressive_model(embeddings)
                aggressive_predictions_mean = np.mean(aggressive_predictions, axis=0)
                aggressive_score = 1.0 - float(aggressive_predictions_mean[1]) if aggressive_predictions_mean.shape[0] >= 2 else 1.0 - float(aggressive_predictions_mean[0])
            except Exception as ae:
                print(f"Error in aggressive mood prediction for {audio_file_path}: {ae}", file=sys.stderr)

        danceability_score = None
        global danceability_model
        if danceability_model is not None:
            try:
                danceability_predictions = danceability_model(embeddings)
                danceability_predictions_mean = np.mean(danceability_predictions, axis=0)
                danceability_score = 1.0 - float(danceability_predictions_mean[1]) if danceability_predictions_mean.shape[0] >= 2 else 1.0 - float(danceability_predictions_mean[0])
            except Exception as de:
                print(f"Error in danceability prediction for {audio_file_path}: {de}", file=sys.stderr)

        relaxed_score = None
        global relaxed_model
        if relaxed_model is not None:
            try:
                relaxed_predictions = relaxed_model(embeddings)
                relaxed_predictions_mean = np.mean(relaxed_predictions, axis=0)
                relaxed_score = float(relaxed_predictions_mean[1]) if relaxed_predictions_mean.shape[0] >= 2 else float(relaxed_predictions_mean[0])
            except Exception as re:
                print(f"Error in relaxed mood prediction for {audio_file_path}: {re}", file=sys.stderr)

        sad_score = None
        global sad_model
        if sad_model is not None:
            try:
                sad_predictions = sad_model(embeddings)
                sad_predictions_mean = np.mean(sad_predictions, axis=0)
                sad_score = float(sad_predictions_mean[1]) if sad_predictions_mean.shape[0] >= 2 else float(sad_predictions_mean[0])
            except Exception as se:
                print(f"Error in sad mood prediction for {audio_file_path}: {se}", file=sys.stderr)

        engagement_score = None
        global engagement_model
        if engagement_model is not None:
            try:
                engagement_predictions = engagement_model(embeddings)
                engagement_predictions_mean = np.mean(engagement_predictions, axis=0)
                engagement_score = float(engagement_predictions_mean[1]) if engagement_predictions_mean.shape[0] >= 2 else float(engagement_predictions_mean[0])
            except Exception as ee:
                print(f"Error in engagement prediction for {audio_file_path}: {ee}", file=sys.stderr)

        approachability_score = None
        global approachability_model
        if approachability_model is not None:
            try:
                approachability_predictions = approachability_model(embeddings)
                approachability_predictions_mean = np.mean(approachability_predictions, axis=0)
                approachability_score = float(approachability_predictions_mean[1]) if approachability_predictions_mean.shape[0] >= 2 else float(approachability_predictions_mean[0])
            except Exception as ae_approach: # Changed from 'ae' to avoid conflict
                print(f"Error in approachability prediction for {audio_file_path}: {ae_approach}", file=sys.stderr)

        # Add instrument prediction if the model is available
        instrument_scores = None
        # global instrument_model # already using current_instrument_model passed as argument
        if current_instrument_model is not None and instrument_labels:
            try:
                instrument_predictions = current_instrument_model(embeddings) # Use the passed instrument_model
                instrument_predictions_mean = np.mean(instrument_predictions, axis=0)
                instrument_scores = {instrument_labels[i]: float(instrument_predictions_mean[i]) for i in range(len(instrument_labels))}
            except Exception as ie:
                print(f"Error in instrument prediction for {audio_file_path}: {ie}", file=sys.stderr)


        return genre_result, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
               danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
               instrument_scores
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}", file=sys.stderr)
        return None, None, None, None, None, None, None, None, None, None, None, None


def track_exists(filepath):
    """Check if the track is already in the database."""
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM classified_tracks WHERE path = ?', (filepath,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

def extract_metadata(filepath):
    """Extract metadata from audio file."""
    try:
        tag = TinyTag.get(filepath)
        return {
            'artist': tag.artist if tag.artist else 'Unknown Artist',
            'title': tag.title if tag.title else 'Unknown Title',
            'album': tag.album if tag.album else 'Unknown Album',
            'year': tag.year if tag.year else 'Unknown Year',
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

def extract_bpm(audio):
    """Extract BPM from an audio file using Essentia."""
    try:
        rhythm_extractor = RhythmExtractor2013(method="degara")
        bpm, _, _, _, _ = rhythm_extractor(audio)
        if bpm < 90: # Simple heuristic to prefer faster BPMs
            bpm *= 2
        return round(bpm, 2)
    except Exception as e:
        print(f"Error extracting BPM: {e}", file=sys.stderr)
        return 0.00

def extract_key(audio):
    """Extract the musical key and scale (major/minor) from audio using Essentia."""
    try:
        key_extractor = KeyExtractor()
        key, scale, strength = key_extractor(audio)
        return f"{key} {scale.capitalize()}"
    except Exception as e:
        print(f"Error extracting key: {e}", file=sys.stderr)
        return 'Unknown'

# ------------------------------------------------------------------------------
# Artwork Extraction
# ------------------------------------------------------------------------------
def extract_artwork(audio_file, track_id):
    """Extract artwork from the audio file and return paths to original and thumbnail images."""
    artwork_path = None
    artwork_thumbnail_path = None

    try:
        audio = mutagen.File(audio_file)

        if audio is None:
            return None, None

        artwork_data = None
        artwork_extension = None

        if isinstance(audio, MP3):
            if audio.tags is not None:
                for tag_name in audio.tags:
                    if tag_name.startswith('APIC'):
                        tag = audio.tags[tag_name]
                        artwork_data = tag.data
                        artwork_extension = 'jpg' if tag.mime == 'image/jpeg' else 'png'
                        break
        elif isinstance(audio, FLAC):
            if audio.pictures:
                pic = audio.pictures[0]
                artwork_data = pic.data
                artwork_extension = pic.mime.split('/')[1]
        elif isinstance(audio, MP4) or isinstance(audio, AAC): # AAC might not have 'covr' in all cases
            if 'covr' in audio.tags:
                covr_tag = audio.tags['covr'][0]
                artwork_data = bytes(covr_tag) # Ensure it's bytes
                if isinstance(covr_tag, MP4Cover):
                    format_type = covr_tag.imageformat
                    artwork_extension = 'jpg' if format_type == MP4Cover.FORMAT_JPEG else 'png'
                else: # Fallback for non-MP4Cover but still image data
                    artwork_extension = 'jpg' # Default or try to infer
            elif isinstance(audio, AAC) and audio.tags and 'APIC:Cover' in audio.tags: # Check for APIC in AAC
                 apic_tag = audio.tags['APIC:Cover']
                 artwork_data = apic_tag.data
                 artwork_extension = 'jpg' if apic_tag.mime == 'image/jpeg' else 'png'


        if artwork_data:
            artwork_hash = md5_hash(artwork_data)
            original_artwork_filename = f"{artwork_hash}.{artwork_extension}"
            artwork_path = os.path.join(artworks_dir, original_artwork_filename)

            if not os.path.exists(artwork_path):
                with open(artwork_path, 'wb') as f:
                    f.write(artwork_data)

            with Image.open(io.BytesIO(artwork_data)) as img:
                img = img.convert("RGB") # Convert to RGB to handle various formats including RGBA
                img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                # Ensure the correct extension for the saved thumbnail
                thumb_ext = artwork_extension if artwork_extension in ['jpg', 'png'] else 'jpg' # Default to jpg
                resized_artwork_filename = f"{artwork_hash}_128x128.{thumb_ext}"
                artwork_thumbnail_path = os.path.join(artworks_dir, resized_artwork_filename)
                if not os.path.exists(artwork_thumbnail_path):
                    img.save(artwork_thumbnail_path, format='JPEG' if thumb_ext == 'jpg' else 'PNG') # Specify format

            with print_lock:
                print(f"Artwork saved for track ID {track_id} to {artwork_path} and {artwork_thumbnail_path}")
        else:
            with print_lock:
                print(f"No artwork found for track ID {track_id}")

    except Exception as e:
        with print_lock:
            print(f"Error processing artwork for file {audio_file}: {e}", file=sys.stderr)

    return artwork_path, artwork_thumbnail_path

# ------------------------------------------------------------------------------
# End-to-End Processing of One Audio File
# ------------------------------------------------------------------------------
def process_audio_file_path(file_path):
    """Process a single audio file path."""
    try:
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
        danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
        instrument_scores = classify_track(file_path)

        if features and audio_16k is not None and audio_44k is not None:
            metadata = extract_metadata(file_path)
            key_val = extract_key(audio_16k)
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bpm = extract_bpm(audio_44k)

            # Process genre tags
            tags_sorted = sorted(features.items(), key=lambda x: x[1], reverse=True)[:10]
            tag_names = {}
            tag_probs = {}
            for i, (genre, prob) in enumerate(tags_sorted):
                tag_names[f'tag{i+1}'] = genre
                tag_probs[f'tag{i+1}_prob'] = float(prob)
            feature_blob = json.dumps(features).encode('utf-8') # Genre features blob

            # Process ALL instrument features for the instrument_features blob column
            all_instrument_features_dict = {}
            if instrument_scores:
                for instrument_name, prob in instrument_scores.items():
                    if prob > 0.01:  # Only store significant probabilities
                        all_instrument_features_dict[instrument_name] = prob
            instrument_features_blob = json.dumps(all_instrument_features_dict).encode('utf-8')

            # Process top 10 instrument data for individual columns
            top_instrument_names = {}
            top_instrument_probs = {}
            if instrument_scores:
                sorted_instruments = sorted(instrument_scores.items(), key=lambda item: item[1], reverse=True)
                for i, (instrument, prob) in enumerate(sorted_instruments[:10]): # Get top 10
                    top_instrument_names[f'instrument{i+1}'] = instrument
                    top_instrument_probs[f'instrument{i+1}_prob'] = float(prob)

            spectral_features = analyze_spectral_features(file_path)
            if spectral_features is None:
                spectral_features = {"spectral_centroid": None, "spectral_bandwidth": None, "spectral_rolloff": None, "spectral_contrast": None, "spectral_flatness": None, "rms": None}

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Prepare column names and placeholders for instruments
                instrument_col_names_str = ""
                instrument_col_placeholders_str = ""
                instrument_values_for_tuple = []
                for i in range(1, 11):
                    instrument_col_names_str += f", instrument{i}, instrument{i}_prob"
                    instrument_col_placeholders_str += ", ?, ?"
                    instrument_values_for_tuple.append(top_instrument_names.get(f'instrument{i}'))
                    instrument_values_for_tuple.append(top_instrument_probs.get(f'instrument{i}_prob'))

                sql_query = f'''
                    INSERT OR REPLACE INTO classified_tracks (
                        path, features, instrument_features, artist, title, album, year, time, bpm, key, date,
                        tag1, tag1_prob, tag2, tag2_prob, tag3, tag3_prob, tag4, tag4_prob, tag5, tag5_prob,
                        tag6, tag6_prob, tag7, tag7_prob, tag8, tag8_prob, tag9, tag9_prob, tag10, tag10_prob
                        {instrument_col_names_str},
                        artwork_path, artwork_thumbnail_path,
                        spectral_centroid, spectral_bandwidth, spectral_rolloff,
                        spectral_contrast, spectral_flatness, rms, happiness, party,
                        aggressive, danceability, relaxed, sad, engagement, approachability
                    ) VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?, /* 11 for path to date */
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,? /* 20 for tags */
                        {instrument_col_placeholders_str}, /* 20 for instruments */
                        ?,?, /* 2 for artwork */
                        ?,?,?,?,?,?, /* 6 for spectral */
                        ?,?,?,?,?,?,?,? /* 8 for moods */
                    )
                '''

                values_tuple = (
                    file_path, feature_blob, instrument_features_blob,
                    metadata['artist'], metadata['title'], metadata['album'], metadata['year'], metadata['duration'],
                    bpm, key_val, date,
                    # Tags (20 values)
                    tag_names.get('tag1'), tag_probs.get('tag1_prob'),
                    tag_names.get('tag2'), tag_probs.get('tag2_prob'),
                    tag_names.get('tag3'), tag_probs.get('tag3_prob'),
                    tag_names.get('tag4'), tag_probs.get('tag4_prob'),
                    tag_names.get('tag5'), tag_probs.get('tag5_prob'),
                    tag_names.get('tag6'), tag_probs.get('tag6_prob'),
                    tag_names.get('tag7'), tag_probs.get('tag7_prob'),
                    tag_names.get('tag8'), tag_probs.get('tag8_prob'),
                    tag_names.get('tag9'), tag_probs.get('tag9_prob'),
                    tag_names.get('tag10'), tag_probs.get('tag10_prob'),
                    # Instruments (20 values from instrument_values_for_tuple)
                    *instrument_values_for_tuple,
                    # Artwork placeholders
                    None, None,
                    # Spectral features (6 values)
                    spectral_features["spectral_centroid"],
                    spectral_features["spectral_bandwidth"],
                    spectral_features["spectral_rolloff"],
                    spectral_features["spectral_contrast"],
                    spectral_features["spectral_flatness"],
                    spectral_features["rms"],
                    # Mood features (8 values)
                    happiness_score, party_score, aggressive_score,
                    danceability_score, relaxed_score, sad_score, engagement_score, approachability_score
                )

                cursor.execute(sql_query, values_tuple)
                conn.commit()

            # Get track_id for artwork
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM classified_tracks WHERE path = ?', (file_path,))
                result = cursor.fetchone()
                track_id = result[0] if result else None

            if track_id:
                artwork_path_val, artwork_thumbnail_path_val = extract_artwork(file_path, track_id)
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE classified_tracks
                        SET artwork_path = ?, artwork_thumbnail_path = ?
                        WHERE id = ?
                    ''', (artwork_path_val, artwork_thumbnail_path_val, track_id))
                    conn.commit()

    except Exception as e:
        print(f"Failed to process {file_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


# ------------------------------------------------------------------------------
# Process All Audio Files in a Folder (with concurrency + timeout)
# ------------------------------------------------------------------------------
def process_folder(folder_path, num_processes):
    """Process all audio files in the folder (recursively)."""
    audio_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.opus', '.wma', '.aac'}
    file_paths = []
    already_classified_count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in audio_formats:
                file_path = os.path.join(root, file)
                if not track_exists(file_path):
                    file_paths.append(file_path)
                else:
                    already_classified_count += 1

    total_files = len(file_paths) + already_classified_count
    print(f"Number of tracks already classified: {already_classified_count}")
    print(f"Number of tracks to classify: {len(file_paths)}")
    print(f"Total number of tracks: {total_files}")

    if not file_paths:
        print("No new tracks to process.")
        return

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_path = {
            executor.submit(process_audio_file_path, fp): fp for fp in file_paths
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_path),
            total=len(future_to_path),
            file=sys.stdout,
            desc="Processing files"
        ):
            fp = future_to_path[future]
            try:
                future.result(timeout=180) # Increased timeout
            except TimeoutError:
                print(f"Timeout while processing {fp}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing file {fp}: {e}", file=sys.stderr)

    print("Processing completed.")

# ------------------------------------------------------------------------------
# (Incorporated) Spectral Analysis from spectral_analysis.py
# ------------------------------------------------------------------------------
import librosa  # needed for spectral analysis

def init_spectral_columns(): # Renamed to reflect its broader scope now
    """
    Ensure the 'classified_tracks' table has columns for spectral features,
    mood features, and instrument features. If not, add them.
    This function is primarily for updating older DB schemas.
    New schemas should be handled by init_db().
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define instrument tag columns again for checking
    instrument_tag_cols_definitions = []
    for i in range(1, 11): # For top 10 instruments
        instrument_tag_cols_definitions.append((f"instrument{i}", "TEXT"))
        instrument_tag_cols_definitions.append((f"instrument{i}_prob", "REAL"))

    cursor.execute("PRAGMA table_info(classified_tracks)")
    existing_cols = [row[1] for row in cursor.fetchall()]

    feature_cols_to_check = [
        ("spectral_centroid", "REAL"),
        ("spectral_bandwidth", "REAL"),
        ("spectral_rolloff", "REAL"),
        ("spectral_contrast", "REAL"),
        ("spectral_flatness", "REAL"),
        ("rms", "REAL"),
        ("happiness", "REAL"),
        ("party", "REAL"),
        ("aggressive", "REAL"),
        ("danceability", "REAL"),
        ("relaxed", "REAL"),
        ("sad", "REAL"),
        ("engagement", "REAL"),
        ("approachability", "REAL"),
        ("instrument_features", "BLOB") # Ensure this one exists too
    ] + instrument_tag_cols_definitions

    for col, col_type in feature_cols_to_check:
        if col not in existing_cols:
            # For BLOB NOT NULL, SQLite requires a default value for ADD COLUMN if table is not empty.
            # However, we set DEFAULT NULL for most, and instrument_features in init_db is NOT NULL.
            # Here, if adding instrument_features, we assume it can be NULL initially if altering.
            # For simplicity, all added columns here will default to NULL.
            # The `instrument_features BLOB NOT NULL` constraint is primarily for new table creation.
            default_clause = "DEFAULT NULL"
            if col == "instrument_features" and col_type == "BLOB": # Special handling if we decided it must be NOT NULL
                 # This could be an issue if altering an existing table with rows.
                 # init_db() handles it correctly for new tables.
                 # For ALTER, best to allow NULL then populate, or use a default empty blob.
                 # For now, sticking to DEFAULT NULL for all ALTER ops here.
                 pass


            alter_query = f"ALTER TABLE classified_tracks ADD COLUMN {col} {col_type} {default_clause}"
            try:
                cursor.execute(alter_query)
                conn.commit()
                print(f"Added column {col} to classified_tracks.")
            except sqlite3.OperationalError as e:
                print(f"Could not add column {col}: {e}", file=sys.stderr)

    conn.close()

def get_unanalyzed_tracks():
    """
    Return a list of (id, path) for tracks whose spectral_centroid, rms,
    or any mood/instrument primary score is NULL. (Instrument check based on instrument1)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check for missing instrument1 as a proxy for instrument analysis (for discrete columns)
    query = '''
        SELECT id, path
        FROM classified_tracks
        WHERE spectral_centroid IS NULL OR rms IS NULL OR
              happiness IS NULL OR party IS NULL OR aggressive IS NULL OR
              danceability IS NULL OR relaxed IS NULL OR sad IS NULL OR engagement IS NULL OR
              approachability IS NULL OR instrument1 IS NULL
    '''
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error querying unanalyzed tracks (likely missing columns, run init_db/init_spectral_columns): {e}", file=sys.stderr)
        rows = [] # Return empty if table schema is not ready
    conn.close()
    return rows

def analyze_spectral_features(file_path):
    """
    Load audio with librosa and compute relevant spectral features including RMS.
    Returns a dict with the mean values of each feature.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0) # Analyze first 30s
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = float(np.mean(cent))
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth = float(np.mean(bandwidth))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff = float(np.mean(rolloff))
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast = float(np.mean(contrast))
        flatness = librosa.feature.spectral_flatness(y=y)
        spectral_flatness = float(np.mean(flatness))
        rms_feat = librosa.feature.rms(y=y) # Renamed from rms to rms_feat to avoid conflict with rms parameter name
        rms_value = float(np.mean(rms_feat))
        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_contrast": spectral_contrast,
            "spectral_flatness": spectral_flatness,
            "rms": rms_value
        }
    except Exception as e:
        print(f"Error extracting spectral features from {file_path}: {e}", file=sys.stderr)
        return None

def update_spectral_features_in_db(track_id, features):
    """
    Update classified_tracks with the spectral features for the given track ID.
    """
    if not features:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE classified_tracks
            SET
              spectral_centroid = ?,
              spectral_bandwidth = ?,
              spectral_rolloff = ?,
              spectral_contrast = ?,
              spectral_flatness = ?,
              rms = ?
            WHERE id = ?
        ''', (
            features["spectral_centroid"],
            features["spectral_bandwidth"],
            features["spectral_rolloff"],
            features["spectral_contrast"],
            features["spectral_flatness"],
            features["rms"],
            track_id
        ))
        conn.commit()
    except Exception as e:
        print(f"Error updating spectral features for track ID {track_id}: {e}", file=sys.stderr)
    finally:
        conn.close()


def process_single_track_for_spectral(track_tuple):
    """
    Helper function for parallel execution of spectral analysis.
    track_tuple -> (track_id, file_path)
    """
    track_id, file_path = track_tuple
    if not os.path.isfile(file_path):
        print(f"File not found, skipping spectral analysis for: {file_path}", file=sys.stderr)
        return

    features = analyze_spectral_features(file_path)
    if features is None:
        return

    update_spectral_features_in_db(track_id, features)

def analyze_missing_spectral_features(num_workers):
    """
    If some tracks are missing spectral features, analyze them in parallel.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, path
        FROM classified_tracks
        WHERE spectral_centroid IS NULL OR rms IS NULL
    ''') # Only spectral for this re-analysis pass
    unanalyzed_tracks = cursor.fetchall()
    conn.close()

    total_unanalyzed = len(unanalyzed_tracks)
    if total_unanalyzed == 0:
        print("No tracks found missing *only* spectral features.")
        return

    print(f"Found {total_unanalyzed} track(s) missing spectral analysis. Processing...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_track_for_spectral, t): t for t in unanalyzed_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_unanalyzed, desc="Analyzing missing spectral"):
            try:
                future.result(timeout=60) # Timeout for spectral analysis
            except TimeoutError:
                track_tuple = futures[future]
                print(f"Timeout during spectral analysis of track {track_tuple[1]}", file=sys.stderr)
            except Exception as e:
                track_tuple = futures[future]
                print(f"Error in spectral processing track {track_tuple[1]}: {e}", file=sys.stderr)

    print("Missing spectral feature analysis completed.")


# ------------------------------------------------------------------------------
# New: Create Embeddings and Build a Vector Database for Similarity Search
# ------------------------------------------------------------------------------
try:
    import faiss
except ImportError:
    faiss = None
    print("Faiss is not installed. Vector indexing functionality will not be available.", file=sys.stderr)

def create_embedding(genre_features):
    """
    Create an embedding vector using only the genre features from the 'features' column.
    The vector is constructed based on the global class_labels order and then L2 normalized.
    """
    if not class_labels:
        print("Genre class_labels are not loaded. Cannot create embeddings.", file=sys.stderr)
        return np.array([])

    genre_vector = np.array([genre_features.get(label, 0.0) for label in class_labels], dtype=np.float32)
    norm = np.linalg.norm(genre_vector)
    if norm > 0:
        genre_vector = genre_vector / norm
    return genre_vector

def build_vector_index():
    """
    Build a FAISS vector index using only the stored 'features' (genre probabilities) from the database.
    The index (and a mapping of track paths) is saved to disk.
    """
    if faiss is None:
        print("Faiss is not installed. Cannot build vector index.", file=sys.stderr)
        return None, None
    if not class_labels:
        print("Genre class_labels are not loaded. Cannot build vector index.", file=sys.stderr)
        return None, None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, path, features
        FROM classified_tracks
        WHERE features IS NOT NULL
    ''')
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    track_ids_map = []

    for track_id, path, features_blob in tqdm(rows, desc="Generating embeddings for index"):
        try:
            features_dict = json.loads(features_blob)
        except Exception as e:
            print(f"Error parsing features for track ID {track_id} ({path}): {e}", file=sys.stderr)
            continue

        emb = create_embedding(features_dict)
        if emb.size == 0:
            continue
        embeddings.append(emb)
        track_ids_map.append(track_id)

    if not embeddings:
        print("No embeddings to index.", file=sys.stderr)
        return None, None

    embeddings_np = np.vstack(embeddings).astype('float32')
    d = embeddings_np.shape[1]

    if d == 0:
        print("Embedding dimension is 0. Cannot build index.", file=sys.stderr)
        return None, None

    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    print(f"Index built with {index.ntotal} embeddings. Dimension: {d}")

    index_path = os.path.join(script_dir, "vector_index.faiss")
    map_path = os.path.join(script_dir, "track_index_to_id_map.json")

    faiss.write_index(index, index_path)
    with open(map_path, "w") as f:
        json.dump(track_ids_map, f)

    print(f"FAISS index saved to {index_path}")
    print(f"Track ID map saved to {map_path}")

    return index, track_ids_map


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process and classify audio files.")
    parser.add_argument("path", help="Path to the folder containing audio files or a single audio file.")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1), help="Number of threads to use for processing.")
    parser.add_argument("--build-index", action="store_true", help="Build vector index after processing.")
    parser.add_argument("--reanalyze-spectral", action="store_true", help="Re-analyze spectral features for tracks missing them.")
    args = parser.parse_args()

    init_db()
    check_db_integrity()
    init_spectral_columns() # Ensures all analytical columns exist, including new instrument ones for older DBs.

    path = args.path
    num_threads = args.threads

    if os.path.isdir(path):
        process_folder(path, num_threads)
    elif os.path.isfile(path):
        # if not track_exists(path): # Process even if it exists, to update with new columns if necessary
        print(f"Processing single file: {path}", file=sys.stdout)
        process_audio_file_path(path)
        print(f"Processing completed for file: {path}", file=sys.stdout)
        # else:
        # print(f"Track already classified: {path}. Use --reanalyze-spectral to update missing spectral data or re-run to update all fields.", file=sys.stdout)
    else:
        print(f"The path {path} is neither a file nor a directory.", file=sys.stderr)


    if args.reanalyze_spectral:
        print("Starting re-analysis of missing spectral features...")
        analyze_missing_spectral_features(num_threads)

    if args.build_index:
        print("Building vector index...")
        build_vector_index()

    print("Main script execution finished.")


def f(): # Dummy function for cx_Freeze
    pass

# ------------------------------------------------------------------------------
# Safe Entry (to avoid hanging processes on Windows with multiprocessing)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    # Process(target=f).start() # This was likely a placeholder or specific workaround.
                                # If not essential for current freezing setup, can be removed.

    try:
        main()
    except Exception as unhandled_e:
        print(f"Unhandled exception in main: {unhandled_e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        if sys.stderr != original_stderr and sys.stderr is not None and not sys.stderr.closed:
            sys.stderr.flush()
            sys.stderr.close()
            sys.stderr = original_stderr
        print("Script finished and error log closed.")