import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Configuration Constants ---
# Makes the script easier to modify and understand.

# Key categories
ERROR_KEYS = {'Backspace', 'Delete'}
SPECIAL_KEYS = {'Shift', 'CapsLock', 'Enter', 'Tab'}
MODIFIER_KEYS = {'Control', 'Alt', 'Meta'} # Keys that modify others but aren't typed

# Pause detection threshold in milliseconds
PAUSE_THRESHOLD_MS = 2000  # 2 seconds

# Common n-grams for rhythm analysis
COMMON_DIGRAPHS = {'th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en', 'at', 'ed', 'to', 'or', 'ea'}
COMMON_TRIGRAPHS = {'the', 'and', 'ing', 'her', 'ere', 'ent', 'tha', 'nth', 'was', 'eth', 'for', 'dth'}


def get_stats(data: List[float], prefix: str) -> Dict[str, Any]:
    """Calculates descriptive statistics for a list of numbers."""
    if not data or not np.all(np.isfinite(data)):
        # Return a dictionary of zeros if data is empty or contains non-finite values
        return {f'{prefix}_mean': 0, f'{prefix}_std': 0, f'{prefix}_median': 0, f'{prefix}_count': 0}

    return {
        f'{prefix}_mean': np.mean(data),
        f'{prefix}_std': np.std(data),
        f'{prefix}_median': np.median(data),
        f'{prefix}_count': len(data)
    }

def calculate_session_features(session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes the key events from a single session to extract behavioral features
    using highly optimized and vectorized pandas operations.

    Args:
        session_data: The parsed JSON data for one session.

    Returns:
        A dictionary of calculated features for the session, or None if data is invalid.
    """
    try:
        key_events = session_data['behavioralData']['keyEvents']
        anxiety_score = session_data['selfReport']['anxietyScore']
        raw_text_length = session_data['behavioralData'].get('rawTextLength', 0)
    except (KeyError, TypeError):
        return None

    if not key_events:
        return None

    df = pd.DataFrame(key_events)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp', 'key', 'type'], inplace=True)
    df.sort_values('timestamp', inplace=True, ignore_index=True)
    
    if df.empty:
        return None

    # --- 1. Dwell Time (Vectorized) ---
    # Time a key is held down.
    downs = df[df['type'] == 'keydown'].copy()
    ups = df[df['type'] == 'keyup'].copy()
    
    # Create a unique press ID for each key to handle multiple presses of the same key
    downs['press_id'] = downs.groupby('key').cumcount()
    ups['press_id'] = ups.groupby('key').cumcount()
    
    # Merge to match keydown with its corresponding keyup
    merged = pd.merge(downs, ups, on=['key', 'press_id'], suffixes=('_down', '_up'))
    merged['dwell_time'] = merged['timestamp_up'] - merged['timestamp_down']
    dwell_times = merged['dwell_time'][merged['dwell_time'] > 0].tolist()

    # --- 2. Latencies (Flight Time & Inter-Key Latency) ---
    # Filter for events that represent actual typing
    typable_events = df[~df['key'].isin(MODIFIER_KEYS)].copy()
    
    # Flight Time: Time from keyup of key A to keydown of key B
    typable_events['prev_type'] = typable_events['type'].shift(1)
    typable_events['prev_timestamp'] = typable_events['timestamp'].shift(1)
    flights_df = typable_events[(typable_events['type'] == 'keydown') & (typable_events['prev_type'] == 'keyup')]
    flight_times = (flights_df['timestamp'] - flights_df['prev_timestamp']).tolist()

    # Inter-Key Latency: Time from keydown of key A to keydown of key B
    keydowns = df[df['type'] == 'keydown'].copy()
    inter_key_latencies = keydowns['timestamp'].diff().dropna().tolist()

    # --- 3. Typing Pauses ---
    # Identify long flight times as pauses
    pause_times = [ft for ft in flight_times if ft > PAUSE_THRESHOLD_MS]
    pause_count = len(pause_times)
    total_pause_duration = sum(pause_times)

    # --- 4. N-Gram Speeds (Vectorized) ---
    # Analyze rhythm of common letter combinations
    keydowns['key_lower'] = keydowns['key'].str.lower()
    
    digraph_speeds = []
    trigraph_speeds = []
    
    if len(keydowns) > 2:
        # Create sequences using vectorized string operations
        digraph_sequences = keydowns['key_lower'] + keydowns['key_lower'].shift(-1)
        trigraph_sequences = keydowns['key_lower'] + keydowns['key_lower'].shift(-1) + keydowns['key_lower'].shift(-2)
        
        # Calculate time differences
        time_diffs = keydowns['timestamp'].diff()
        
        # Filter for common n-grams and get their speeds
        digraph_speeds = time_diffs[digraph_sequences.isin(COMMON_DIGRAPHS)].tolist()
        trigraph_speeds = (time_diffs + time_diffs.shift(-1))[trigraph_sequences.isin(COMMON_TRIGRAPHS)].tolist()

    # --- 5. Typing Speed, Errors, and Special Key Usage ---
    total_duration_ms = df['timestamp'].max() - df['timestamp'].min()
    total_duration_min = total_duration_ms / (1000 * 60)
    wpm = (raw_text_length / 5) / total_duration_min if total_duration_min > 0 else 0

    total_keydowns = len(keydowns)
    error_key_count = keydowns['key'].isin(ERROR_KEYS).sum()
    error_rate = error_key_count / total_keydowns if total_keydowns > 0 else 0
    
    special_key_counts = {f'{key.lower()}_count': keydowns['key'].isin([key, key.lower()]).sum() for key in SPECIAL_KEYS}

    # --- 6. Aggregate and Return All Features ---
    features = {'anxiety_score': anxiety_score}
    features.update(get_stats(dwell_times, 'dwell_time'))
    features.update(get_stats(flight_times, 'flight_time'))
    features.update(get_stats(inter_key_latencies, 'inter_key_latency'))
    features.update(get_stats(digraph_speeds, 'digraph_speed'))
    features.update(get_stats(trigraph_speeds, 'trigraph_speed'))
    
    features['typing_speed_wpm'] = wpm
    features['error_rate'] = error_rate
    features['error_key_count'] = error_key_count
    features['pause_count'] = pause_count
    features['total_pause_duration_ms'] = total_pause_duration
    features.update(special_key_counts)
    
    return features


def main():
    """Main function to run the data processing pipeline."""
    input_csv_path = Path('data-collection-form.csv')
    output_csv_path = Path('processed_keystroke_data.csv')

    if not input_csv_path.is_file():
        print(f"Error: Input file not found at '{input_csv_path}'")
        return

    print(f"Loading raw data from '{input_csv_path}'...")
    raw_df = pd.read_csv(input_csv_path)

    processed_data = []
    print("Processing sessions with vectorized functions...")
    
    for index, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0], desc="Sessions"):
        try:
            session_json = json.loads(row.iloc[0]) # Use iloc[0] for first column
            features = calculate_session_features(session_json)
            if features:
                processed_data.append(features)
        except (json.JSONDecodeError, IndexError, TypeError) as e:
            print(f"Warning: Skipping row {index} due to parsing error: {e}")
            continue

    if not processed_data:
        print("No valid sessions were processed. Exiting.")
        return

    print("Creating final DataFrame...")
    processed_df = pd.DataFrame(processed_data).fillna(0)

    print(f"Saving processed data to '{output_csv_path}'...")
    processed_df.to_csv(output_csv_path, index=False)

    print("\n--- Processing Complete ---")
    print(f"Successfully processed {len(processed_df)} sessions.")
    print(f"Output saved to: {output_csv_path.resolve()}")
    print("\nFirst 5 rows of the processed data:")
    print(processed_df.head())
    print("\nData description:")
    print(processed_df.describe())


if __name__ == '__main__':
    main()