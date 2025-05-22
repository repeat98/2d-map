#!/usr/bin/env python
# analyze.py

import librosa
from multiprocessing import freeze_support
import io
import hashlib
from PIL import Image
from mutagen.aac import AAC
from mutagen.mp4 import MP4, MP4Cover
from mutagen.id3 import APIC
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
import mutagen
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from tqdm import tqdm
import argparse
from pathlib import Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
    RhythmExtractor2013,
    KeyExtractor,
    Resample
)
from tinytag import TinyTag
import numpy as np
import json
import sqlite3
import os
import sys

# Attempt to disable Essentia warnings via environment variables.
os.environ["ESSENTIA_LOG_DISABLE"] = "1"
os.environ["ESSENTIA_ENABLE_WARNINGS"] = "0"
os.environ["ESSENTIA_LOG_LEVEL"] = "error"

# Redirect the underlying C++ stderr (file descriptor 2) to /dev/null to suppress warnings.
_cpp_stderr_redirected = False
try:
    if os.name != 'nt':  # Not "nul" on Windows
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd) # Close the original file descriptor for /dev/null
        _cpp_stderr_redirected = True
except OSError:
    # Use sys.__stderr__ for pre-redirection output if normal stderr is already captured
    print("Warning: Could not redirect C++ stderr to /dev/null.", file=sys.__stderr__)

# Attempt to import pyloudnorm for improved LUFS calculation
try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    # Use sys.__stderr__ if regular stderr is redirected early
    print("Warning: pyloudnorm library not found. LUFS calculation will use RMS-based approximation.", file=sys.__stderr__)

# Conditional import for soundfile as a fallback
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile library not found. Fallback audio loading in spectral analysis might fail.", file=sys.__stderr__)


# ------------------------------------------------------------------------------
# Setup error logging to a file
# ------------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

error_log_path = os.path.join(script_dir, "error.log")
_actual_stderr = sys.stderr  # Store the initial stderr stream
_error_log_file_stream = None  # Initialize file stream variable

try:
    _error_log_file_stream = open(error_log_path, 'a', buffering=1)  # line-buffered
    sys.stderr = _error_log_file_stream  # Redirect stderr to the log file
except IOError as e:
    # If opening the log file fails, print the warning to the _actual_stderr.
    # sys.stderr would not have been changed in this case.
    print(f"Warning: Could not open error log file {error_log_path} for writing: {e}", file=_actual_stderr)
    _error_log_file_stream = None  # Ensure it's None if open failed

# ------------------------------------------------------------------------------
# Global Paths and Model Initialization
# ------------------------------------------------------------------------------
embedding_model_path = os.path.join(
    script_dir, "essentia_model/discogs-effnet-bs64-1.pb")
classification_model_path = os.path.join(
    script_dir, "essentia_model/genre_discogs400-discogs-effnet-1.pb")
happiness_model_path = os.path.join(
    script_dir, "essentia_model/mood_happy-discogs-effnet-1.pb")
party_model_path = os.path.join(
    script_dir, "essentia_model/mood_party-discogs-effnet-1.pb")
aggressive_model_path = os.path.join(
    script_dir, "essentia_model/mood_aggressive-discogs-effnet-1.pb")
danceability_model_path = os.path.join(
    script_dir, "essentia_model/danceability-discogs-effnet-1.pb")
relaxed_model_path = os.path.join(
    script_dir, "essentia_model/mood_relaxed-discogs-effnet-1.pb")
sad_model_path = os.path.join(
    script_dir, "essentia_model/mood_sad-discogs-effnet-1.pb")
engagement_model_path = os.path.join(
    script_dir, "essentia_model/engagement_2c-discogs-effnet-1.pb")
approachability_model_path = os.path.join(
    script_dir, "essentia_model/approachability_2c-discogs-effnet-1.pb")
instrument_model_path = os.path.join(
    script_dir, "essentia_model/mtg_jamendo_instrument-discogs-effnet-1.pb")

class_labels_path = os.path.join(
    script_dir, 'essentia_model/genre_discogs400-discogs-effnet-1.json')
instrument_labels_path = os.path.join(
    script_dir, 'essentia_model/mtg_jamendo_instrument-discogs-effnet-1.json')

db_name = "tracks.db"
db_folder = os.path.join(script_dir, "../db/")
db_path = os.path.join(db_folder, db_name)

artworks_dir = os.path.join(script_dir, '../assets/artworks')
os.makedirs(artworks_dir, exist_ok=True)
os.makedirs(db_folder, exist_ok=True)

# Load models
try:
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_model_path,
        output="PartitionedCall:1",
        patchHopSize=64
    )
except Exception as e:
    print(
        f"FATAL: Error loading Tensorflow embedding model: {embedding_model_path}. Exception: {e}", file=sys.stderr)
    sys.exit(1)

try:
    classification_model = TensorflowPredict2D(
        graphFilename=classification_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )
except Exception as e:
    print(
        f"FATAL: Error loading Tensorflow classification model: {classification_model_path}. Exception: {e}", file=sys.stderr)
    sys.exit(1)

def load_optional_model(path, name, output_name="model/Softmax"):
    try:
        model = TensorflowPredict2D(graphFilename=path, output=output_name)
        print(f"{name} model loaded successfully.", file=sys.stdout) # Progress to stdout
        return model
    except Exception as e:
        print(
            f"Warning: Error loading {name} model ({path}): {e}. This analysis will be skipped.", file=sys.stderr)
        return None

happiness_model = load_optional_model(happiness_model_path, "Happiness")
party_model = load_optional_model(party_model_path, "Party")
aggressive_model = load_optional_model(aggressive_model_path, "Aggressive")
danceability_model = load_optional_model(
    danceability_model_path, "Danceability")
relaxed_model = load_optional_model(relaxed_model_path, "Relaxed")
sad_model = load_optional_model(sad_model_path, "Sad")
engagement_model = load_optional_model(engagement_model_path, "Engagement")
approachability_model = load_optional_model(
    approachability_model_path, "Approachability")
instrument_model = load_optional_model(
    instrument_model_path, "Instrument", output_name="model/Sigmoid")

try:
    with open(class_labels_path, 'r') as file:
        class_labels = json.load(file).get("classes", [])
    if not class_labels:
        print(
            f"Warning: Genre class labels file {class_labels_path} is empty or malformed.", file=sys.stderr)
except Exception as e:
    print(
        f"Error loading genre class labels from {class_labels_path}: {e}", file=sys.stderr)
    class_labels = []

try:
    with open(instrument_labels_path, 'r') as file:
        instrument_labels = json.load(file).get("classes", [])
    if not instrument_labels:
        print(
            f"Warning: Instrument class labels file {instrument_labels_path} is empty or malformed.", file=sys.stderr)
except Exception as e:
    print(
        f"Error loading instrument class labels from {instrument_labels_path}: {e}", file=sys.stderr)
    instrument_labels = []

db_lock = Lock()

# ------------------------------------------------------------------------------
# Database Initialization and Integrity Check
# ------------------------------------------------------------------------------
def init_db():
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
            {instrument_cols_sql_create},
            artwork_path TEXT DEFAULT NULL,
            artwork_thumbnail_path TEXT DEFAULT NULL,
            noisy REAL DEFAULT NULL,
            tonal REAL DEFAULT NULL,
            dark REAL DEFAULT NULL,
            bright REAL DEFAULT NULL,
            percussive REAL DEFAULT NULL,
            smooth REAL DEFAULT NULL,
            lufs TEXT DEFAULT NULL,
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
    try:
        with sqlite3.connect(db_path) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                print("Database integrity check: OK", file=sys.stdout)
            else:
                print(
                    f"Database integrity check failed: {result}", file=sys.stderr)
    except Exception as e:
        print(f"Error checking database integrity: {e}", file=sys.stderr)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def normalize_audio(audio_array):
    denom = np.max(np.abs(audio_array))
    if denom == 0:
        return audio_array
    return audio_array / denom

def md5_hash(data):
    return hashlib.md5(data).hexdigest()

# ------------------------------------------------------------------------------
# Audio Processing and Classification
# ------------------------------------------------------------------------------
def classify_track(filepath):
    try:
        features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
        danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
        instrument_scores_dict = process_audio_file(
            filepath, embedding_model, classification_model, instrument_model)
        if features:
            return features, audio_16k, audio_44k, happiness_score, party_score, aggressive_score, \
                   danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
                   instrument_scores_dict
        return {}, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error classifying {filepath}: {e}", file=sys.stderr)
        return {}, None, None, None, None, None, None, None, None, None, None, None

def process_audio_file(audio_file_path, emb_model, class_model, instr_model):
    try:
        audio_44k_orig = MonoLoader(
            filename=audio_file_path, sampleRate=44100, resampleQuality=4)()
        audio_44k = normalize_audio(audio_44k_orig)

        resampler = Resample(inputSampleRate=44100,
                             outputSampleRate=16000, quality=4)
        audio_16k = resampler(audio_44k)
        embeddings = emb_model(audio_16k)

        genre_predictions = class_model(embeddings)
        genre_predictions_mean = np.mean(genre_predictions, axis=0)
        genre_result = {class_labels[i]: float(
            genre_predictions_mean[i]) for i in range(len(class_labels)) if class_labels}

        def predict_mood(model, embeddings_data, model_name):
            if model is None: return None
            try:
                predictions = model(embeddings_data)
                predictions_mean = np.mean(predictions, axis=0)
                if predictions_mean.shape[0] >= 2:
                    return float(predictions_mean[1])
                return float(predictions_mean[0])
            except Exception as e_mood:
                print(
                    f"Error in {model_name} prediction for {audio_file_path}: {e_mood}", file=sys.stderr)
                return None

        happiness_score = predict_mood(happiness_model, embeddings, "happiness")
        party_score = predict_mood(party_model, embeddings, "party")
        aggressive_score = predict_mood(aggressive_model, embeddings, "aggressive")
        danceability_score = predict_mood(danceability_model, embeddings, "danceability")
        relaxed_score = predict_mood(relaxed_model, embeddings, "relaxed")
        sad_score = predict_mood(sad_model, embeddings, "sad")
        engagement_score = predict_mood(engagement_model, embeddings, "engagement")
        approachability_score = predict_mood(approachability_model, embeddings, "approachability")

        instrument_scores_dict = None
        if instr_model is not None and instrument_labels:
            try:
                instrument_predictions = instr_model(embeddings)
                instrument_predictions_mean = np.mean(
                    instrument_predictions, axis=0)
                instrument_scores_dict = {instrument_labels[i]: float(
                    instrument_predictions_mean[i]) for i in range(len(instrument_labels))}
            except Exception as ie:
                print(
                    f"Error in instrument prediction for {audio_file_path}: {ie}", file=sys.stderr)

        return genre_result, audio_16k, audio_44k_orig, happiness_score, party_score, aggressive_score, \
               danceability_score, relaxed_score, sad_score, engagement_score, approachability_score, \
               instrument_scores_dict
    except Exception as e:
        print(
            f"Error processing audio file {audio_file_path}: {e}", file=sys.stderr)
        return None, None, None, None, None, None, None, None, None, None, None, None

def track_exists(filepath):
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT 1 FROM classified_tracks WHERE path = ?', (filepath,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

def extract_metadata(filepath):
    try:
        tag = TinyTag.get(filepath, image=False)
        return {
            'artist': tag.artist if tag.artist else 'Unknown Artist',
            'title': tag.title if tag.title else 'Unknown Title',
            'album': tag.album if tag.album else 'Unknown Album',
            'year': str(tag.year) if tag.year else 'Unknown Year',
            'duration': (f"{int(tag.duration // 60)}:{int(tag.duration % 60):02d}"
                         if tag.duration else '00:00')
        }
    except Exception as e:
        print(
            f"Error extracting metadata from {filepath}: {e}", file=sys.stderr)
        return {
            'artist': 'Unknown Artist',
            'title': 'Unknown Title',
            'album': 'Unknown Album',
            'year': 'Unknown Year',
            'duration': '00:00'
        }

def extract_bpm(audio_44k):
    try:
        rhythm_extractor = RhythmExtractor2013(method="degara")
        bpm, _, _, _, _ = rhythm_extractor(audio_44k)
        if 0 < bpm < 80:
             bpm_candidate_double = bpm * 2
             if bpm_candidate_double < 220:
                 bpm = bpm_candidate_double
        elif bpm > 200:
            bpm_candidate_half = bpm / 2
            if bpm_candidate_half > 70:
                bpm = bpm_candidate_half
        return round(bpm, 2) if bpm else 0.00
    except Exception as e:
        print(f"Error extracting BPM: {e}", file=sys.stderr)
        return 0.00

def extract_key(audio_16k):
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
def extract_artwork(audio_file, track_id_for_log_display): # param renamed for clarity
    artwork_path = None
    artwork_thumbnail_path = None

    try:
        audio = mutagen.File(audio_file)
        if audio is None:
            print(f"No audio metadata found for {audio_file}", file=sys.stderr)
            return None, None

        artwork_data = None
        artwork_extension = 'jpg'

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
        elif isinstance(audio, AAC) and audio.tags: # Requires mutagen to be new enough for .aac
            apic_tag = audio.tags.get('APIC:Cover') or audio.tags.get('APIC') # Common APIC tags
            if apic_tag and hasattr(apic_tag, 'data'): # Check if it's a valid APIC tag object
                 artwork_data = apic_tag.data
                 artwork_extension = 'jpg' if 'jpeg' in getattr(apic_tag, 'mime', '').lower() else \
                                     'png' if 'png' in getattr(apic_tag, 'mime', '').lower() else 'jpg'


        if artwork_data:
            artwork_hash = md5_hash(artwork_data)
            original_artwork_filename = f"{artwork_hash}.{artwork_extension}"
            artwork_path = os.path.join(artworks_dir, original_artwork_filename)

            if not os.path.exists(artwork_path):
                with open(artwork_path, 'wb') as f:
                    f.write(artwork_data)

            thumb_ext = 'jpg' if artwork_extension != 'png' else 'png'
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
            # Removed "Successfully extracted artwork..." print to reduce log noise
        # else: # Removed "No artwork found..." print to reduce log noise

    except Exception as e:
        log_id_str = f"(Track ID {track_id_for_log_display})" if track_id_for_log_display else ""
        print(f"Error processing artwork for file {audio_file} {log_id_str}: {e}", file=sys.stderr)
        # import traceback # Keep this commented unless deep debugging
        # traceback.print_exc(file=sys.stderr)
    return artwork_path, artwork_thumbnail_path

# ------------------------------------------------------------------------------
# Spectral Analysis and DB Column Management
# ------------------------------------------------------------------------------
def init_spectral_columns():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(classified_tracks)")
    existing_cols_info = {row[1]: row[2] for row in cursor.fetchall()}

    new_feature_cols = [
        ("noisy", "REAL"), ("tonal", "REAL"), ("dark", "REAL"), ("bright", "REAL"),
        ("percussive", "REAL"), ("smooth", "REAL"), ("lufs", "TEXT"),
        ("happiness", "REAL"), ("party", "REAL"), ("aggressive", "REAL"),
        ("danceability", "REAL"), ("relaxed", "REAL"), ("sad", "REAL"),
        ("engagement", "REAL"), ("approachability", "REAL"),
        ("instrument_features", "BLOB")
    ]
    for i in range(1, 11):
        new_feature_cols.append((f"instrument{i}", "TEXT"))
        new_feature_cols.append((f"instrument{i}_prob", "REAL"))

    old_spectral_columns = [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_contrast", "spectral_flatness", "rms", "idx", "x", "y",
        "combined_features", "mfcc_features", "chroma_features"
    ]

    for old_col in old_spectral_columns:
        if old_col in existing_cols_info:
            try:
                # print(f"Attempting to remove old column: {old_col}") # Optional: for debugging
                cursor.execute(f"ALTER TABLE classified_tracks DROP COLUMN {old_col}")
                conn.commit()
                # print(f"Removed old column {old_col} from classified_tracks.") # Optional
                existing_cols_info.pop(old_col)
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not remove column {old_col}: {e}", file=sys.stderr)

    for col_name, col_type in new_feature_cols:
        if col_name not in existing_cols_info:
            try:
                default_clause = "DEFAULT NULL"
                cursor.execute(f"ALTER TABLE classified_tracks ADD COLUMN {col_name} {col_type} {default_clause}")
                conn.commit()
                # print(f"Added column {col_name} ({col_type}) to classified_tracks.") # Optional
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column {col_name}: {e}", file=sys.stderr)
    conn.close()

def analyze_spectral_features(file_path):
    try:
        y, sr = None, None
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0, res_type='kaiser_fast')
        except Exception as audio_load_error:
            print(f"Librosa load error for {file_path}: {audio_load_error}. Trying soundfile...", file=sys.stderr)
            if SOUNDFILE_AVAILABLE:
                try:
                    data, samplerate = sf.read(file_path)
                    if len(data.shape) > 1: data = np.mean(data, axis=1)
                    if samplerate != 22050:
                        y = librosa.resample(y=data, orig_sr=samplerate, target_sr=22050, res_type='kaiser_fast')
                    else:
                        y = data
                    sr = 22050
                    y = y[:int(30.0 * sr)] # Ensure duration limit
                except Exception as sf_error:
                    print(f"Soundfile load error for {file_path}: {sf_error}", file=sys.stderr)
                    return None
            else:
                print(f"Soundfile not available, cannot load {file_path} for spectral analysis.", file=sys.stderr)
                return None


        if y is None or len(y) == 0:
            print(f"Warning: Audio loaded from {file_path} is empty. Skipping spectral analysis.", file=sys.stderr)
            return None

        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_flatness = np.mean(spectral_flatness)
        mean_zcr = np.mean(zcr)
        noisy_score = float(np.clip((mean_flatness + mean_zcr) / 2.0, 0.0, 1.0))
        tonal_score = 1.0 - noisy_score

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        norm_centroid = np.mean(spectral_centroid) / (sr / 2.0)
        abs_mfcc0_frames = np.abs(mfccs[0,:])
        mfcc0_min, mfcc0_max = np.min(abs_mfcc0_frames), np.max(abs_mfcc0_frames)
        mean_norm_mfcc0_energy = 0.0
        if (mfcc0_max - mfcc0_min) > 1e-6:
            norm_mfcc0_energy_frames = (abs_mfcc0_frames - mfcc0_min) / (mfcc0_max - mfcc0_min)
            mean_norm_mfcc0_energy = np.mean(norm_mfcc0_energy_frames)
        elif mfcc0_max > 1e-6 :
            mean_norm_mfcc0_energy = 1.0 if mfcc0_max > np.median(abs_mfcc0_frames) else 0.0

        bright_score = float(np.clip((norm_centroid + mean_norm_mfcc0_energy) / 2.0, 0.0, 1.0))
        dark_score = 1.0 - bright_score

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_std, onset_mean = np.std(onset_env), np.mean(onset_env)
        percussive_score = 0.0
        if onset_mean > 1e-6:
            percussive_score = float(np.clip(onset_std / onset_mean, 0.0, 2.0) / 2.0)
        percussive_score = float(np.clip(percussive_score, 0.0, 1.0))
        smooth_score = 1.0 - percussive_score
        
        lufs_value_str = "N/A"
        if PYLOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(rate=sr)
                integrated_loudness = meter.integrated_loudness(y)
                lufs_value_str = f"{integrated_loudness:.1f}"
            except Exception as lufs_e:
                print(f"pyloudnorm error for {file_path}: {lufs_e}. Falling back to RMS.", file=sys.stderr)
                rms_value = np.mean(librosa.feature.rms(y=y))
                lufs_value_str = f"{20 * np.log10(rms_value):.1f} (RMS)" if rms_value > 1e-10 else "-inf (RMS)"
        else:
            rms_value = np.mean(librosa.feature.rms(y=y))
            lufs_value_str = f"{20 * np.log10(rms_value):.1f} (RMS)" if rms_value > 1e-10 else "-inf (RMS)"

        return {
            "noisy": noisy_score, "tonal": tonal_score, "dark": dark_score, "bright": bright_score,
            "percussive": percussive_score, "smooth": smooth_score, "lufs": lufs_value_str
        }
    except Exception as e:
        print(f"Error extracting spectral features from {file_path}: {e}", file=sys.stderr)
        return None

def update_spectral_features_in_db(track_id, features_dict):
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
                features_dict["percussive"], features_dict["smooth"], features_dict["lufs"], track_id
            ))
            conn.commit()
        except Exception as e:
            print(f"Error updating spectral features for track ID {track_id}: {e}", file=sys.stderr)
        finally:
            conn.close()

def process_single_track_for_spectral_reanalysis(track_tuple):
    track_id, file_path = track_tuple
    if not os.path.isfile(file_path):
        print(f"File not found, skipping spectral re-analysis for: {file_path} (ID: {track_id})", file=sys.stderr)
        return

    features = analyze_spectral_features(file_path)
    if features:
        update_spectral_features_in_db(track_id, features)

def analyze_missing_spectral_features_only(num_workers):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id, path FROM classified_tracks
            WHERE (noisy IS NULL OR tonal IS NULL OR dark IS NULL OR bright IS NULL OR
                   percussive IS NULL OR smooth IS NULL OR lufs IS NULL)
        ''')
        unanalyzed_spectral_tracks = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"DB error querying for spectral re-analysis (schema might be old): {e}", file=sys.stderr)
        unanalyzed_spectral_tracks = []
    conn.close()

    total_unanalyzed = len(unanalyzed_spectral_tracks)
    if total_unanalyzed == 0:
        print("No tracks found missing only spectral features for re-analysis.", file=sys.stdout)
        return

    print(f"Found {total_unanalyzed} track(s) missing spectral features. Processing...", file=sys.stdout)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_track = {
            executor.submit(process_single_track_for_spectral_reanalysis, track_tuple): track_tuple
            for track_tuple in unanalyzed_spectral_tracks
        }
        for future in tqdm(as_completed(future_to_track), total=total_unanalyzed, desc="Re-analyzing spectral features", file=sys.stdout):
            track_tuple_completed = future_to_track[future]
            try:
                future.result(timeout=120) # Timeout for spectral analysis
            except TimeoutError:
                print(f"Timeout: Spectral re-analysis of {track_tuple_completed[1]} (ID: {track_tuple_completed[0]})", file=sys.stderr)
            except Exception as e_spec:
                print(f"Error: Spectral re-processing {track_tuple_completed[1]} (ID: {track_tuple_completed[0]}): {e_spec}", file=sys.stderr)
    print("Missing spectral feature re-analysis completed.", file=sys.stdout)

# ------------------------------------------------------------------------------
# End-to-End Processing of One Audio File
# ------------------------------------------------------------------------------
def process_audio_file_path(file_path_to_process):
    try:
        genre_features, audio_16k, audio_44k, happiness_sc, party_sc, aggressive_sc, \
        danceability_sc, relaxed_sc, sad_sc, engagement_sc, approachability_sc, \
        instrument_features_dict = classify_track(file_path_to_process)

        if not genre_features or audio_16k is None or audio_44k is None:
            print(f"Core classification failed for {file_path_to_process}. Skipping.", file=sys.stderr)
            return

        metadata = extract_metadata(file_path_to_process)
        key_val = extract_key(audio_16k)
        bpm_val = extract_bpm(audio_44k)
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        tags_sorted = sorted(genre_features.items(), key=lambda x: x[1], reverse=True)[:10]
        tag_names_map, tag_probs_map = {}, {}
        for i, (genre, prob) in enumerate(tags_sorted):
            tag_names_map[f'tag{i+1}'] = genre
            tag_probs_map[f'tag{i+1}_prob'] = float(prob)
        genre_features_blob = json.dumps(genre_features).encode('utf-8')

        all_instrument_scores_blob = b"{}"
        top_instrument_names_map, top_instrument_probs_map = {}, {}
        if instrument_features_dict:
            significant_instrument_scores = {
                k: v for k, v in instrument_features_dict.items() if v > 0.01 # Store only scores > 1%
            }
            all_instrument_scores_blob = json.dumps(significant_instrument_scores).encode('utf-8')
            sorted_instruments = sorted(instrument_features_dict.items(), key=lambda item: item[1], reverse=True)
            for i, (instrument, prob) in enumerate(sorted_instruments[:10]):
                top_instrument_names_map[f'instrument{i+1}'] = instrument
                top_instrument_probs_map[f'instrument{i+1}_prob'] = float(prob)

        perceptual_spectral_features = analyze_spectral_features(file_path_to_process)
        if perceptual_spectral_features is None:
            perceptual_spectral_features = {
                "noisy": None, "tonal": None, "dark": None, "bright": None,
                "percussive": None, "smooth": None, "lufs": None
            }
        
        artwork_original_path, artwork_thumb_path = extract_artwork(file_path_to_process, None) # No track ID available here yet

        with db_lock:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            instrument_col_names_list, instrument_col_placeholders_list, instrument_values_for_tuple_list = [], [], []
            for i in range(1, 11):
                instrument_col_names_list.extend([f"instrument{i}", f"instrument{i}_prob"])
                instrument_col_placeholders_list.extend(["?", "?"])
                instrument_values_for_tuple_list.extend([
                    top_instrument_names_map.get(f'instrument{i}'),
                    top_instrument_probs_map.get(f'instrument{i}_prob')
                ])

            instrument_col_names_str = ", " + ", ".join(instrument_col_names_list) if instrument_col_names_list else ""
            
            num_base_cols = 11 # path to date
            num_tag_cols = 20 # tag1/prob to tag10/prob
            num_artwork_cols = 2
            num_spectral_cols = 7
            num_mood_cols = 8
            total_placeholders = num_base_cols + num_tag_cols + len(instrument_values_for_tuple_list) + \
                                 num_artwork_cols + num_spectral_cols + num_mood_cols

            sql_insert_query = f'''
                INSERT OR REPLACE INTO classified_tracks (
                    path, features, instrument_features, artist, title, album, year, time, bpm, key, date,
                    {", ".join([f"tag{i}, tag{i}_prob" for i in range(1,11)])}
                    {instrument_col_names_str},
                    artwork_path, artwork_thumbnail_path,
                    noisy, tonal, dark, bright, percussive, smooth, lufs,
                    happiness, party, aggressive, danceability, relaxed, sad, engagement, approachability
                ) VALUES ({", ".join(["?"] * total_placeholders)})
            '''

            values_tuple = (
                file_path_to_process, genre_features_blob, all_instrument_scores_blob,
                metadata['artist'], metadata['title'], metadata['album'], metadata['year'], metadata['duration'],
                bpm_val, key_val, current_date,
                *[val for i in range(1, 11) for val in (tag_names_map.get(f'tag{i}'), tag_probs_map.get(f'tag{i}_prob'))],
                *instrument_values_for_tuple_list,
                artwork_original_path, artwork_thumb_path,
                perceptual_spectral_features["noisy"], perceptual_spectral_features["tonal"],
                perceptual_spectral_features["dark"], perceptual_spectral_features["bright"],
                perceptual_spectral_features["percussive"], perceptual_spectral_features["smooth"],
                perceptual_spectral_features["lufs"],
                happiness_sc, party_sc, aggressive_sc, danceability_sc,
                relaxed_sc, sad_sc, engagement_sc, approachability_sc
            )

            try:
                cursor.execute(sql_insert_query, values_tuple)
                conn.commit()
                # print(f"Successfully processed and stored {file_path_to_process}", file=sys.stderr) # Reduce noise
            except Exception as db_error:
                print(f"Database error for {file_path_to_process}: {db_error}", file=sys.stderr)
                conn.rollback()
            finally:
                conn.close()

    except Exception as e_main_process:
        print(f"FATAL error processing file {file_path_to_process}: {e_main_process}", file=sys.stderr)
        # import traceback # Keep commented unless deep debugging
        # traceback.print_exc(file=sys.stderr)

# ------------------------------------------------------------------------------
# Process All Audio Files in a Folder
# ------------------------------------------------------------------------------
def process_folder(folder_to_scan, num_concurrent_processes, max_tracks=None):
    audio_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.opus', '.wma', '.aac'}
    files_to_process_paths = []
    already_classified_count = 0

    print(f"Scanning folder: {folder_to_scan}...", file=sys.stdout)
    all_audio_file_paths_in_folder = [
        os.path.join(root, file_name)
        for root, _, files in os.walk(folder_to_scan)
        for file_name in files
        if Path(file_name).suffix.lower() in audio_formats
    ]
    
    print(f"Found {len(all_audio_file_paths_in_folder)} audio files in scan path.", file=sys.stdout)

    for fp in tqdm(all_audio_file_paths_in_folder, desc="Checking existing tracks in DB", file=sys.stdout):
        if not track_exists(fp):
            files_to_process_paths.append(fp)
        else:
            already_classified_count += 1
    
    print(f"Already classified: {already_classified_count}", file=sys.stdout)
    print(f"New/updated tracks to process: {len(files_to_process_paths)}", file=sys.stdout)

    if not files_to_process_paths:
        print("No new tracks to process.", file=sys.stdout)
        return

    if max_tracks is not None:
        files_to_process_paths = files_to_process_paths[:max_tracks]
        print(f"Limited to processing {len(files_to_process_paths)} tracks.", file=sys.stdout)

    with ProcessPoolExecutor(max_workers=num_concurrent_processes) as executor:
        future_to_path_map = {
            executor.submit(process_audio_file_path, fp_single): fp_single
            for fp_single in files_to_process_paths
        }
        for future_item in tqdm(
            as_completed(future_to_path_map),
            total=len(future_to_path_map),
            desc="Processing audio files",
            file=sys.stdout
        ):
            path_processed = future_to_path_map[future_item]
            try:
                future_item.result(timeout=300) # 5 minutes per file
            except TimeoutError:
                print(f"Timeout processing file (possible corruption or very long): {path_processed}", file=sys.stderr)
            except Exception as e_folder_proc:
                print(f"Error processing file {path_processed} in pool: {e_folder_proc}", file=sys.stderr)
    print("Folder processing completed.", file=sys.stdout)

# ------------------------------------------------------------------------------
# Faiss (Vector Similarity Search - Stub)
# ------------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None 
    FAISS_AVAILABLE = False
    print("Warning: Faiss library not installed. Vector indexing and similarity search will not be available.", file=sys.stderr)

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process and classify audio files.")
    parser.add_argument("path", help="Path to the folder or a single audio file.")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Number of concurrent processes.")
    parser.add_argument("--reanalyze-spectral", action="store_true", help="Re-analyze spectral features for tracks missing them.")
    parser.add_argument("-n", "--max-tracks", type=int, help="Limit processing to N new tracks.")
    args = parser.parse_args()

    print(f"Using database: {db_path}", file=sys.stdout)
    init_db()
    check_db_integrity()
    init_spectral_columns() # Ensures DB schema is up-to-date

    num_processes = args.threads

    if os.path.isdir(args.path):
        process_folder(args.path, num_processes, args.max_tracks)
    elif os.path.isfile(args.path):
        print(f"Processing single file: {args.path}", file=sys.stdout)
        process_audio_file_path(args.path)
        print(f"Completed processing for: {args.path}", file=sys.stdout)
    else:
        print(f"Error: Path {args.path} is not a valid file or directory.", file=_actual_stderr) # Use actual stderr
        sys.exit(1)

    if args.reanalyze_spectral:
        print("Starting re-analysis of missing spectral features...", file=sys.stdout)
        analyze_missing_spectral_features_only(num_processes)

    print("Main script execution finished.", file=sys.stdout)

# ------------------------------------------------------------------------------
# Safe Entry for Multiprocessing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()

    try:
        main()
    except Exception as unhandled_e_main:
        # This will go to sys.stderr, which might be the log file or _actual_stderr if redirection failed
        print(f"Unhandled exception in main execution: {unhandled_e_main}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        # If sys.stderr was redirected to the log file, flush and close it
        if _error_log_file_stream and sys.stderr == _error_log_file_stream:
            sys.stderr.flush()
            _error_log_file_stream.close()
        sys.stderr = _actual_stderr  # Restore original stderr
        print(f"Script finished. Error log (if any) is at: {error_log_path}", file=_actual_stderr)