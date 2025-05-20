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
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 2)

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
# New approachability model path
approachability_model_path = os.path.join(script_dir, "essentia_model/approachability_2c-discogs-effnet-1.pb")

class_labels_path = os.path.join(script_dir, 'essentia_model/genre_discogs400-discogs-effnet-1.json')
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

# New approachability model loader
try:
    approachability_model = TensorflowPredict2D(
        graphFilename=approachability_model_path,
        output="model/Softmax"
    )
except Exception as e:
    print(f"Error while loading Tensorflow approachability model with path: {approachability_model_path}", file=sys.stderr)
    approachability_model = None
    print("Approachability analysis will be skipped.")

# Load class labels for genre
try:
    with open(class_labels_path, 'r') as file:
        class_labels = json.load(file).get("classes", [])
except Exception as e:
    print(f"Error loading genre class labels: {e}", file=sys.stderr)
    class_labels = []

db_lock = Lock()

# ------------------------------------------------------------------------------
# Database Initialization and Integrity Check
# ------------------------------------------------------------------------------
def init_db():
    """Initialize the database and create the directory if it does not exist."""
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classified_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            features BLOB NOT NULL,
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
            tag10_prob REAL DEFAULT NULL,
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

    # Check if columns exist, add them if not
    cursor.execute("PRAGMA table_info(classified_tracks)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Ensure tag probability columns exist
    for i in range(1, 11):
        col_name = f"tag{i}_prob"
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE classified_tracks ADD COLUMN {col_name} REAL DEFAULT NULL")
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
    # Avoid division by zero (if audio is empty or something unusual)
    denom = np.max(np.abs(audio), axis=0)
    if denom == 0:
        return audio
    return audio / denom

def md5_hash(data):
    """Compute MD5 hash of data."""
    return hashlib.md5(data).hexdigest()

# ------------------------------------------------------------------------------
# Audio Processing and Classification
# ------------------------------------------------------------------------------
def classify_track(filepath):
    """Classify the track using the integrated models."""
    try:
        # Updated to unpack approachability_score as well
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, danceability_score, relaxed_score, sad_score, engagement_score, approachability_score = process_audio_file(filepath, embedding_model, classification_model)
        if features:
            return features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, danceability_score, relaxed_score, sad_score, engagement_score, approachability_score
        return {}, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error classifying {filepath}: {e}", file=sys.stderr)
        return {}, None, None, None, None, None, None, None, None, None, None

def process_audio_file(audio_file_path, embedding_model, classification_model):
    """Process an audio file and return genre features, mood scores, and audio data."""
    try:
        # Load the audio file with sampleRate=44100, resampleQuality=4
        audio_44k = MonoLoader(filename=audio_file_path, sampleRate=44100, resampleQuality=4)()
        audio_44k = normalize(audio_44k)
        
        # Resample to 16000 Hz for classification
        resample = Resample(inputSampleRate=44100, outputSampleRate=16000, quality=4)
        audio_16k = resample(audio_44k)

        # Compute embeddings
        embeddings = embedding_model(audio_16k)

        # Get genre predictions
        genre_predictions = classification_model(embeddings)
        genre_predictions_mean = np.mean(genre_predictions, axis=0)

        # Map genre predictions to class labels
        genre_result = {class_labels[i]: float(genre_predictions_mean[i]) for i in range(len(class_labels))}

        # Add happiness prediction if the model is available
        happiness_score = None
        global happiness_model
        if happiness_model is not None:
            try:
                happiness_predictions = happiness_model(embeddings)
                happiness_predictions_mean = np.mean(happiness_predictions, axis=0)
                if happiness_predictions_mean.shape[0] >= 2:
                    happiness_score = 1.0 - float(happiness_predictions_mean[1])
                else:
                    happiness_score = 1.0 - float(happiness_predictions_mean[0])
            except Exception as he:
                print(f"Error in happiness prediction for {audio_file_path}: {he}", file=sys.stderr)
        
        # Add party mood prediction if the model is available
        party_score = None
        global party_model
        if party_model is not None:
            try:
                party_predictions = party_model(embeddings)
                party_predictions_mean = np.mean(party_predictions, axis=0)
                if party_predictions_mean.shape[0] >= 2:
                    party_score = float(party_predictions_mean[1])
                else:
                    party_score = float(party_predictions_mean[0])
            except Exception as pe:
                print(f"Error in party mood prediction for {audio_file_path}: {pe}", file=sys.stderr)
                
        # Add aggressive mood prediction if the model is available
        aggressive_score = None
        global aggressive_model
        if aggressive_model is not None:
            try:
                aggressive_predictions = aggressive_model(embeddings)
                aggressive_predictions_mean = np.mean(aggressive_predictions, axis=0)
                if aggressive_predictions_mean.shape[0] >= 2:
                    aggressive_score = 1.0 - float(aggressive_predictions_mean[1])
                else:
                    aggressive_score = 1.0 - float(aggressive_predictions_mean[0])
            except Exception as ae:
                print(f"Error in aggressive mood prediction for {audio_file_path}: {ae}", file=sys.stderr)
                
        # Add danceability prediction if the model is available
        danceability_score = None
        global danceability_model
        if danceability_model is not None:
            try:
                danceability_predictions = danceability_model(embeddings)
                danceability_predictions_mean = np.mean(danceability_predictions, axis=0)
                if danceability_predictions_mean.shape[0] >= 2:
                    danceability_score = 1.0 - float(danceability_predictions_mean[1])
                else:
                    danceability_score = 1.0 - float(danceability_predictions_mean[0])
            except Exception as de:
                print(f"Error in danceability prediction for {audio_file_path}: {de}", file=sys.stderr)
                
        # Add relaxed mood prediction if the model is available
        relaxed_score = None
        global relaxed_model
        if relaxed_model is not None:
            try:
                relaxed_predictions = relaxed_model(embeddings)
                relaxed_predictions_mean = np.mean(relaxed_predictions, axis=0)
                if relaxed_predictions_mean.shape[0] >= 2:
                    relaxed_score = float(relaxed_predictions_mean[1])
                else:
                    relaxed_score = float(relaxed_predictions_mean[0])
            except Exception as re:
                print(f"Error in relaxed mood prediction for {audio_file_path}: {re}", file=sys.stderr)
                
        # Add sad mood prediction if the model is available
        sad_score = None
        global sad_model
        if sad_model is not None:
            try:
                sad_predictions = sad_model(embeddings)
                sad_predictions_mean = np.mean(sad_predictions, axis=0)
                if sad_predictions_mean.shape[0] >= 2:
                    sad_score = float(sad_predictions_mean[1])
                else:
                    sad_score = float(sad_predictions_mean[0])
            except Exception as se:
                print(f"Error in sad mood prediction for {audio_file_path}: {se}", file=sys.stderr)
        
        # Add engagement prediction if the model is available
        engagement_score = None
        global engagement_model
        if engagement_model is not None:
            try:
                engagement_predictions = engagement_model(embeddings)
                engagement_predictions_mean = np.mean(engagement_predictions, axis=0)
                if engagement_predictions_mean.shape[0] >= 2:
                    engagement_score = float(engagement_predictions_mean[1])
                else:
                    engagement_score = float(engagement_predictions_mean[0])
            except Exception as ee:
                print(f"Error in engagement prediction for {audio_file_path}: {ee}", file=sys.stderr)
        
        # Add approachability prediction if the model is available
        approachability_score = None
        global approachability_model
        if approachability_model is not None:
            try:
                approachability_predictions = approachability_model(embeddings)
                approachability_predictions_mean = np.mean(approachability_predictions, axis=0)
                if approachability_predictions_mean.shape[0] >= 2:
                    approachability_score = float(approachability_predictions_mean[1])
                else:
                    approachability_score = float(approachability_predictions_mean[0])
            except Exception as ae:
                print(f"Error in approachability prediction for {audio_file_path}: {ae}", file=sys.stderr)

        return genre_result, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, danceability_score, relaxed_score, sad_score, engagement_score, approachability_score
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}", file=sys.stderr)
        return None, None, None, None, None, None, None, None, None, None, None

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
        if bpm < 90:
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
            return None, None  # Unsupported file type
        
        artwork_data = None
        artwork_extension = None

        if isinstance(audio, MP3):
            if audio.tags is not None:
                for tag in audio.tags.values():
                    if isinstance(tag, APIC):
                        artwork_data = tag.data
                        artwork_extension = 'jpg' if tag.mime == 'image/jpeg' else 'png'
                        break
        elif isinstance(audio, FLAC):
            if audio.pictures:
                pic = audio.pictures[0]
                artwork_data = pic.data
                artwork_extension = pic.mime.split('/')[1]
        elif isinstance(audio, MP4) or isinstance(audio, AAC):
            if 'covr' in audio.tags:
                artwork_data = audio.tags['covr'][0]
                if isinstance(audio, MP4):
                    if isinstance(artwork_data, MP4Cover):
                        format_type = artwork_data.imageformat
                        artwork_extension = 'jpg' if format_type == MP4Cover.FORMAT_JPEG else 'png'
                else:
                    artwork_extension = 'jpg'

        if artwork_data:
            artwork_hash = md5_hash(artwork_data)
            original_artwork_filename = f"{artwork_hash}.{artwork_extension}"
            artwork_path = os.path.join(artworks_dir, original_artwork_filename)
            
            if not os.path.exists(artwork_path):
                with open(artwork_path, 'wb') as f:
                    f.write(artwork_data)
            
            with Image.open(io.BytesIO(artwork_data)) as img:
                img = img.convert("RGB")
                img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                resized_artwork_filename = f"{artwork_hash}_128x128.{artwork_extension}"
                artwork_thumbnail_path = os.path.join(artworks_dir, resized_artwork_filename)
                if not os.path.exists(artwork_thumbnail_path):
                    img.save(artwork_thumbnail_path, format=img.format)
            
            with print_lock:
                print(f"Artwork saved for track ID {track_id} to {artwork_path} and {artwork_thumbnail_path}")
        else:
            with print_lock:
                print(f"No artwork found for track ID {track_id}")
    
    except Exception as e:
        with print_lock:
            print(f"Error processing file {audio_file}: {e}")
    
    return artwork_path, artwork_thumbnail_path

# ------------------------------------------------------------------------------
# Database Insertion
# ------------------------------------------------------------------------------
def insert_into_db(path, features, metadata, key, date, tag_names, tag_probs, bpm,
                   artwork_path, artwork_thumbnail_path, happiness_score, party_score, 
                   aggressive_score, danceability_score, relaxed_score, sad_score, 
                   engagement_score, approachability_score):
    """Insert track information into the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            feature_blob = json.dumps(features).encode('utf-8')
            cursor.execute('''
                INSERT OR REPLACE INTO classified_tracks (
                    path, features, artist, title, album, year, time, bpm, key, date,
                    tag1, tag1_prob, tag2, tag2_prob, tag3, tag3_prob, tag4, tag4_prob, tag5, tag5_prob,
                    tag6, tag6_prob, tag7, tag7_prob, tag8, tag8_prob, tag9, tag9_prob, tag10, tag10_prob,
                    artwork_path, artwork_thumbnail_path,
                    spectral_centroid, spectral_bandwidth, spectral_rolloff,
                    spectral_contrast, spectral_flatness, rms, happiness, party, 
                    aggressive, danceability, relaxed, sad, engagement, approachability
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                path, feature_blob, metadata['artist'], metadata['title'], metadata['album'],
                metadata['year'], metadata['duration'], bpm, key, date,
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
                artwork_path, artwork_thumbnail_path,
                None, None, None, None, None, None,
                happiness_score, party_score, aggressive_score,
                danceability_score, relaxed_score, sad_score, engagement_score, approachability_score
            ))
            conn.commit()
    except Exception as e:
        print(f"Error inserting {path} into the database: {e}", file=sys.stderr)

# ------------------------------------------------------------------------------
# End-to-End Processing of One Audio File
# ------------------------------------------------------------------------------
def process_audio_file_path(file_path):
    """Process a single audio file path."""
    try:
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
        danceability_score, relaxed_score, sad_score, engagement_score, approachability_score = classify_track(file_path)
        if features and audio_16k is not None and audio_44k is not None:
            metadata = extract_metadata(file_path)
            key_val = extract_key(audio_16k)
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bpm = extract_bpm(audio_44k)
            tags_sorted = sorted(features.items(), key=lambda x: x[1], reverse=True)[:10]
            tag_names = {}
            tag_probs = {}
            for i, (genre, prob) in enumerate(tags_sorted):
                tag_names[f'tag{i+1}'] = genre
                tag_probs[f'tag{i+1}_prob'] = float(prob)  # force conversion to float
            feature_blob = json.dumps(features).encode('utf-8')
            
            # Compute spectral features directly for the file
            spectral_features = analyze_spectral_features(file_path)
            if spectral_features is None:
                spectral_features = {"spectral_centroid": None, "spectral_bandwidth": None, "spectral_rolloff": None, "spectral_contrast": None, "spectral_flatness": None, "rms": None}
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO classified_tracks (
                        path, features, artist, title, album, year, time, bpm, key, date,
                        tag1, tag1_prob, tag2, tag2_prob, tag3, tag3_prob, tag4, tag4_prob, tag5, tag5_prob,
                        tag6, tag6_prob, tag7, tag7_prob, tag8, tag8_prob, tag9, tag9_prob, tag10, tag10_prob,
                        artwork_path, artwork_thumbnail_path,
                        spectral_centroid, spectral_bandwidth, spectral_rolloff,
                        spectral_contrast, spectral_flatness, rms, happiness, party, 
                        aggressive, danceability, relaxed, sad, engagement, approachability
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    file_path, feature_blob, metadata['artist'], metadata['title'], metadata['album'],
                    metadata['year'], metadata['duration'], bpm, key_val, date,
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
                    None, None,
                    spectral_features["spectral_centroid"],
                    spectral_features["spectral_bandwidth"],
                    spectral_features["spectral_rolloff"],
                    spectral_features["spectral_contrast"],
                    spectral_features["spectral_flatness"],
                    spectral_features["rms"],
                    happiness_score, party_score, aggressive_score,
                    danceability_score, relaxed_score, sad_score, engagement_score, approachability_score
                ))
                conn.commit()
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM classified_tracks WHERE path = ?', (file_path,))
                result = cursor.fetchone()
                track_id = result[0] if result else None

            if track_id:
                artwork_path, artwork_thumbnail_path = extract_artwork(file_path, track_id)
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE classified_tracks
                        SET artwork_path = ?, artwork_thumbnail_path = ?
                        WHERE id = ?
                    ''', (artwork_path, artwork_thumbnail_path, track_id))
                    conn.commit()

    except Exception as e:
        print(f"Failed to process {file_path}: {e}", file=sys.stderr)

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
                future.result(timeout=120) 
            except TimeoutError:
                print(f"Timeout while processing {fp}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing file {fp}: {e}", file=sys.stderr)

    print("Processing completed.")

# ------------------------------------------------------------------------------
# (Incorporated) Spectral Analysis from spectral_analysis.py
# ------------------------------------------------------------------------------
import librosa  # needed for spectral analysis

def init_spectral_columns():
    """
    Ensure the 'classified_tracks' table has columns
    for spectral features and mood features. If not, add them.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classified_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            features BLOB NOT NULL,
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
            tag10_prob REAL DEFAULT NULL,
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

    cursor.execute("PRAGMA table_info(classified_tracks)")
    existing_cols = [row[1] for row in cursor.fetchall()]

    feature_cols = [
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
        ("approachability", "REAL")
    ]

    for col, col_type in feature_cols:
        if col not in existing_cols:
            alter_query = f"ALTER TABLE classified_tracks ADD COLUMN {col} {col_type}"
            cursor.execute(alter_query)
            conn.commit()

    conn.close()

def get_unanalyzed_tracks():
    """
    Return a list of (id, path) for tracks whose spectral_centroid, rms, 
    happiness, party, aggressive, danceability, relaxed, sad, or engagement is NULL.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, path
        FROM classified_tracks
        WHERE spectral_centroid IS NULL OR rms IS NULL OR 
              happiness IS NULL OR party IS NULL OR aggressive IS NULL OR 
              danceability IS NULL OR relaxed IS NULL OR sad IS NULL OR engagement IS NULL
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

def analyze_spectral_features(file_path):
    """
    Load only the first 30 seconds of the file with librosa
    and compute relevant spectral features including RMS.
    Returns a dict with the mean values of each feature.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0)
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
        rms = librosa.feature.rms(y=y)
        rms_value = float(np.mean(rms))
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
    conn.close()

def process_single_track_for_spectral(track_tuple):
    """
    Helper function for parallel execution.
    track_tuple -> (track_id, file_path)
    """
    track_id, file_path = track_tuple
    if not os.path.isfile(file_path):
        print(f"File not found, skipping: {file_path}", file=sys.stderr)
        return

    features = analyze_spectral_features(file_path)
    if features is None:
        return

    update_spectral_features_in_db(track_id, features)

def analyze_missing_spectral_features(num_workers):
    """
    If some tracks are missing spectral features, analyze them in parallel.
    """
    unanalyzed_tracks = get_unanalyzed_tracks()
    total_unanalyzed = len(unanalyzed_tracks)
    if total_unanalyzed == 0:
        print("No unanalyzed tracks found (spectral features).")
        return

    print(f"Found {total_unanalyzed} track(s) without spectral analysis. Processing...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_track_for_spectral, t): t for t in unanalyzed_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_unanalyzed, desc="Analyzing"):
            try:
                future.result()
            except Exception as e:
                track_tuple = futures[future]
                print(f"Error in processing track {track_tuple}: {e}", file=sys.stderr)

    print("Spectral feature analysis completed.")

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
    genre_vector = np.array([genre_features.get(label, 0.0) for label in class_labels], dtype=float)
    norm = np.linalg.norm(genre_vector)
    if norm > 0:
        genre_vector = genre_vector / norm
    return genre_vector

def build_vector_index():
    """
    Build a FAISS vector index using only the stored 'features' from the database.
    The index (and a mapping of track paths) is saved to disk.
    """
    if faiss is None:
        print("Faiss is not installed. Cannot build vector index.", file=sys.stderr)
        return None, None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT path, features
        FROM classified_tracks
        WHERE features IS NOT NULL
    ''')
    rows = cursor.fetchall()
    conn.close()
    embeddings = []
    track_paths = []
    for row in rows:
        path = row[0]
        features_blob = row[1]
        try:
            features_dict = json.loads(features_blob)
        except Exception as e:
            print(f"Error parsing features for {path}: {e}", file=sys.stderr)
            continue
        emb = create_embedding(features_dict)
        embeddings.append(emb)
        track_paths.append(path)
    if not embeddings:
        print("No embeddings to index.", file=sys.stderr)
        return None, None
    embeddings = np.vstack(embeddings).astype('float32')
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"Index built with {index.ntotal} embeddings.")
    
    # Save index to file and mapping to a JSON file
    faiss.write_index(index, "vector_index.faiss")
    with open("track_paths.json", "w") as f:
        json.dump(track_paths, f)
    
    return index, track_paths

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process and classify audio files.")
    parser.add_argument("path", help="Path to the folder containing audio files or a single audio file.")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Number of threads to use for processing.")
    parser.add_argument("--build-index", action="store_true", help="Build vector index after processing.")
    args = parser.parse_args()

    init_db()
    check_db_integrity()
    init_spectral_columns()

    path = args.path
    num_threads = args.threads

    if os.path.isdir(path):
        process_folder(path, num_threads)
    elif os.path.isfile(path):
        if not track_exists(path):
            process_audio_file_path(path)
            print(f"Processing completed for file: {path}", file=sys.stdout)
        else:
            print(f"Track already classified: {path}", file=sys.stderr)
    else:
        print(f"The path {path} is neither a file nor a directory.", file=sys.stderr)

    # Optionally, build the vector index for similarity search using only the features column
    if args.build_index:
        build_vector_index()

def f():
    print("Hello from cx_Freeze")

# ------------------------------------------------------------------------------
# Safe Entry (to avoid hanging processes on Windows with multiprocessing)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    Process(target=f).start()
    try:
        main()
    except Exception as unhandled_e:
        print(f"Unhandled exception in main: {unhandled_e}", file=sys.stderr)
    finally:
        sys.stderr = original_stderr
        error_log_file.close()