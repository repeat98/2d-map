#!/usr/bin/env python
# analyzeDB.py

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
import os
from collections import Counter

# Global style for plots
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style for consistency

# --- Helper Functions ---
def ensure_dir(directory_path):
    """Ensure the directory exists, create it if not."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def load_data(db_path):
    """Load data from the SQLite database."""
    print(f"Connecting to database: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None
    try:
        conn = sqlite3.connect(db_path)
        # Load main table
        df = pd.read_sql_query("SELECT * FROM classified_tracks", conn)
        conn.close()
        print(f"Successfully loaded {len(df)} tracks.")
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def parse_json_blob(blob_series, feature_type_name):
    """
    Parses a pandas Series of JSON blobs, aggregates data, and returns a long DataFrame.
    Handles potential None values and decoding errors.
    """
    all_features_list = []
    print(f"Parsing '{feature_type_name}' JSON blobs...")
    for index, blob in blob_series.dropna().items(): # Drop NaNs to avoid errors
        try:
            # Blobs are stored as bytes, need to decode
            if isinstance(blob, bytes):
                data = json.loads(blob.decode('utf-8'))
            elif isinstance(blob, str): # If it's already a string
                data = json.loads(blob)
            else:
                print(f"Warning: Unexpected data type for blob at index {index}: {type(blob)}")
                continue

            for key, value in data.items():
                all_features_list.append({'track_id': index, feature_type_name: key, 'probability': float(value)})
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error for {feature_type_name} blob at track_id (index) {index}: {e}")
        except Exception as e:
            print(f"Warning: Could not process {feature_type_name} blob for track_id (index) {index}: {e}")

    if not all_features_list:
        print(f"No valid {feature_type_name} data found after parsing blobs.")
        return pd.DataFrame(columns=['track_id', feature_type_name, 'probability'])

    return pd.DataFrame(all_features_list)


# --- Analysis Functions ---

def analyze_numerical_features(df, output_dir):
    print("\n--- Analyzing Numerical Features ---")
    numerical_cols = [
        'bpm', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'spectral_contrast', 'spectral_flatness', 'rms', 'happiness', 'party',
        'aggressive', 'danceability', 'relaxed', 'sad', 'engagement', 'approachability'
    ]
    # Include tag and instrument probabilities if they exist
    for i in range(1, 11):
        if f'tag{i}_prob' in df.columns:
            numerical_cols.append(f'tag{i}_prob')
        if f'instrument{i}_prob' in df.columns:
            numerical_cols.append(f'instrument{i}_prob')

    # Filter out columns not present in the DataFrame
    existing_numerical_cols = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not existing_numerical_cols:
        print("No valid numerical columns found for analysis.")
        return

    desc_stats = df[existing_numerical_cols].describe().transpose()
    print("\nDescriptive Statistics for Numerical Features:")
    print(desc_stats)

    # Histograms
    print("\nGenerating histograms for numerical features...")
    for col in existing_numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hist_{col}.png'))
        plt.close()
    print(f"Histograms saved to {output_dir}")

    # Correlation Matrix
    if len(existing_numerical_cols) > 1:
        print("\nGenerating correlation matrix for numerical features...")
        corr_matrix = df[existing_numerical_cols].corr()
        plt.figure(figsize=(18, 15)) # Adjusted for more features
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
        print(f"Correlation matrix saved to {output_dir}")

def analyze_categorical_features(df, output_dir):
    print("\n--- Analyzing Categorical Features ---")

    # Key Analysis
    if 'key' in df.columns:
        print("\nAnalyzing 'key' distribution...")
        key_counts = df['key'].value_counts().nlargest(20) # Top 20 keys
        plt.figure(figsize=(12, 7))
        sns.barplot(x=key_counts.index, y=key_counts.values, palette="viridis")
        plt.title('Distribution of Musical Keys (Top 20)')
        plt.xlabel('Key')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'key_distribution.png'))
        plt.close()
        print(f"Key distribution plot saved to {output_dir}")

    # Top N Genre Tag Analysis (tag1 to tag10)
    print("\nAnalyzing Top N Genre Tags...")
    genre_tags_list = []
    for i in range(1, 11):
        col_name = f'tag{i}'
        if col_name in df.columns:
            genre_tags_list.extend(df[col_name].dropna().tolist())

    if genre_tags_list:
        genre_tag_counts = Counter(genre_tags_list).most_common(30) # Top 30 individual genre tags
        if genre_tag_counts:
            genre_df = pd.DataFrame(genre_tag_counts, columns=['Genre Tag', 'Frequency'])
            plt.figure(figsize=(15, 8)) # Adjusted size
            sns.barplot(data=genre_df, x='Genre Tag', y='Frequency', palette="mako")
            plt.title('Frequency of Top N Genre Tags (Across tag1-tag10, Top 30 shown)')
            plt.xlabel('Genre Tag')
            plt.ylabel('Total Frequency')
            plt.xticks(rotation=60, ha='right') # Increased rotation for readability
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_N_genre_tags_frequency.png'))
            plt.close()
            print(f"Top N genre tags frequency plot saved to {output_dir}")
    else:
        print("No genre tag columns (tag1-tag10) found or they are all empty.")


    # Top N Instrument Tag Analysis (instrument1 to instrument10)
    print("\nAnalyzing Top N Instrument Tags...")
    instrument_tags_list = []
    for i in range(1, 11):
        col_name = f'instrument{i}'
        if col_name in df.columns:
            instrument_tags_list.extend(df[col_name].dropna().tolist())

    if instrument_tags_list:
        instrument_tag_counts = Counter(instrument_tags_list).most_common(30) # Top 30 individual instrument tags
        if instrument_tag_counts:
            instrument_df = pd.DataFrame(instrument_tag_counts, columns=['Instrument Tag', 'Frequency'])
            plt.figure(figsize=(15, 8)) # Adjusted size
            sns.barplot(data=instrument_df, x='Instrument Tag', y='Frequency', palette="rocket")
            plt.title('Frequency of Top N Instrument Tags (Across instrument1-instrument10, Top 30 shown)')
            plt.xlabel('Instrument Tag')
            plt.ylabel('Total Frequency')
            plt.xticks(rotation=60, ha='right') # Increased rotation for readability
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_N_instrument_tags_frequency.png'))
            plt.close()
            print(f"Top N instrument tags frequency plot saved to {output_dir}")
    else:
        print("No instrument tag columns (instrument1-instrument10) found or they are all empty.")


def analyze_blob_features(df_main, output_dir):
    print("\n--- Analyzing Full Genre & Instrument Features (from Blobs) ---")

    # Genre Features from 'features' blob
    if 'features' in df_main.columns:
        df_genres_long = parse_json_blob(df_main['features'], 'genre')
        if not df_genres_long.empty:
            # Distribution of genre probabilities
            plt.figure(figsize=(10, 6))
            sns.histplot(df_genres_long['probability'], kde=True, bins=30, color='skyblue')
            plt.title('Distribution of All Genre Probabilities (from features blob)')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'all_genre_probabilities_dist.png'))
            plt.close()

            # Top 30 most frequent genres overall
            top_genres_overall = df_genres_long['genre'].value_counts().nlargest(30)
            plt.figure(figsize=(15, 8))
            sns.barplot(x=top_genres_overall.index, y=top_genres_overall.values, palette='crest')
            plt.title('Top 30 Most Frequent Genres Overall (from features blob)')
            plt.xlabel('Genre')
            plt.ylabel('Number of Tracks Present In')
            plt.xticks(rotation=60, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_30_overall_genres.png'))
            plt.close()
            print(f"Full genre feature analysis plots saved to {output_dir}")
        else:
            print("Could not analyze full genre features (blob data might be missing or unparseable).")
    else:
        print("Column 'features' (genre blob) not found.")


    # Instrument Features from 'instrument_features' blob
    if 'instrument_features' in df_main.columns:
        df_instruments_long = parse_json_blob(df_main['instrument_features'], 'instrument')
        if not df_instruments_long.empty:
            # Distribution of instrument probabilities
            plt.figure(figsize=(10, 6))
            sns.histplot(df_instruments_long['probability'], kde=True, bins=30, color='salmon')
            plt.title('Distribution of All Instrument Probabilities (from instrument_features blob)')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'all_instrument_probabilities_dist.png'))
            plt.close()

            # Top 30 most frequent instruments overall
            top_instruments_overall = df_instruments_long['instrument'].value_counts().nlargest(30)
            plt.figure(figsize=(15, 8))
            sns.barplot(x=top_instruments_overall.index, y=top_instruments_overall.values, palette='flare')
            plt.title('Top 30 Most Frequent Instruments Overall (from instrument_features blob)')
            plt.xlabel('Instrument')
            plt.ylabel('Number of Tracks Present In')
            plt.xticks(rotation=60, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_30_overall_instruments.png'))
            plt.close()
            print(f"Full instrument feature analysis plots saved to {output_dir}")
        else:
            print("Could not analyze full instrument features (blob data might be missing or unparseable).")
    else:
        print("Column 'instrument_features' (instrument blob) not found.")

def generate_text_summary(df, output_dir):
    """Generate a comprehensive text summary of the data analysis."""
    print("\nGenerating text summary...")
    summary_lines = []
    
    # Basic dataset information
    summary_lines.append("=== DATASET OVERVIEW ===")
    summary_lines.append(f"Total number of tracks: {len(df)}")
    summary_lines.append(f"Number of features: {len(df.columns)}")
    summary_lines.append("\n")
    
    # Numerical features summary
    summary_lines.append("=== NUMERICAL FEATURES SUMMARY ===")
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numerical_cols:
        desc_stats = df[numerical_cols].describe()
        summary_lines.append(desc_stats.to_string())
    summary_lines.append("\n")
    
    # Categorical features summary
    summary_lines.append("=== CATEGORICAL FEATURES SUMMARY ===")
    categorical_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            summary_lines.append(f"\nTop 10 most common values for {col}:")
            summary_lines.append(value_counts.head(10).to_string())
    summary_lines.append("\n")
    
    # Missing values summary
    summary_lines.append("=== MISSING VALUES SUMMARY ===")
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentages
    })
    summary_lines.append(missing_summary.to_string())
    
    # Write summary to file
    summary_path = os.path.join(output_dir, 'data_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Text summary saved to {summary_path}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Analyze the track database for feature insights.")
    parser.add_argument("db_path", help="Path to the SQLite database file (e.g., tracks.db)")
    args = parser.parse_args()

    output_dir = "analysis_plots"
    ensure_dir(output_dir)

    df = load_data(args.db_path)

    if df is None:
        return

    # Convert relevant probability columns to numeric, coercing errors
    for i in range(1,11):
        if f'tag{i}_prob' in df.columns:
            df[f'tag{i}_prob'] = pd.to_numeric(df[f'tag{i}_prob'], errors='coerce')
        if f'instrument{i}_prob' in df.columns:
            df[f'instrument{i}_prob'] = pd.to_numeric(df[f'instrument{i}_prob'], errors='coerce')

    analyze_numerical_features(df, output_dir)
    analyze_categorical_features(df, output_dir)
    analyze_blob_features(df, output_dir)
    generate_text_summary(df, output_dir)  # Add the new summary generation

    print("\n--- Analysis Summary ---")
    print(f"Total tracks analyzed: {len(df)}")
    # Add more summary points if desired

    # Example: Insights for clustering
    print("\nPotential Insights for Clustering:")
    if 'bpm' in df.columns:
        print(f"- BPM ranges from {df['bpm'].min():.2f} to {df['bpm'].max():.2f}, with a mean of {df['bpm'].mean():.2f}.")
    if 'happiness' in df.columns and 'aggressive' in df.columns:
        if not df[['happiness', 'aggressive']].empty and df[['happiness', 'aggressive']].corr().iloc[0,1] < -0.3:
            print("- 'happiness' and 'aggressive' seem to be somewhat negatively correlated, which is intuitive.")
    if 'features' in df.columns:
        print("- Consider using the full genre probability vectors (from 'features' blob) for more nuanced genre representation in clustering.")
    if 'instrument_features' in df.columns:
        print("- Similarly, full instrument probability vectors (from 'instrument_features' blob) can provide detailed instrument profiles.")
    print("- Review the generated histograms to identify features with multi-modal distributions, which might indicate natural clusters.")
    print("- Check the correlation matrix for highly correlated features; PCA can help, but awareness is good.")

    print(f"\nAnalysis complete. Plots saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()