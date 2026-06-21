import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from collections import Counter
import soundfile as sf


def spectrogram(path,window_size,hoplength):

    data,samplerate=librosa.load(path,sr=None)
#finding the short-fourier transform
    stft_matrix = librosa.stft(data,n_fft=window_size,hop_length=hoplength)
#convert the magnitude to decibels
    spectrogram_db=librosa.amplitude_to_db(np.abs(stft_matrix),ref=np.max)
    return spectrogram_db,samplerate

def plot_spectrogram(spectrogram_db,hoplength,samplerate,song_name):
    fig=plt.figure(figsize=(12,6))
    librosa.display.specshow(spectrogram_db, 
                         sr=samplerate, 
                         hop_length=hoplength,
                         x_axis='time', 
                         y_axis='linear', 
                         cmap='magma',vmin=-80, vmax=0)

    plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
    plt.ylim(0,3500)
    plt.xlim(0,25)
    plt.title(f'Spectrogram of "{song_name}"')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    return fig



def get_peaks(path,window_size,hoplength,neighborhood_size=20,threshold_db=-55):
    spectrogram_db,samplerate=spectrogram(path,window_size,hoplength)
    local_max = maximum_filter(spectrogram_db, size=neighborhood_size)
    # a point is  peak where the point is equal to local maximum and > threshhold
    is_peak = (spectrogram_db == local_max) & (spectrogram_db > threshold_db)
    peak_freq_bins, peak_time_frames = np.where(is_peak)
    #converting the indices to physical units
    times = librosa.frames_to_time(peak_time_frames, sr=samplerate, hop_length=hoplength)
    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=window_size)[peak_freq_bins]
    return spectrogram_db,times,freqs,samplerate

def plot_constellation_of_peaks(spectrogram_db,times,freqs,hoplength,song_name,samplerate):
    fig=plt.figure(figsize=(12,6))
    librosa.display.specshow(spectrogram_db, sr=samplerate, hop_length=hoplength, 
                         x_axis='time', y_axis='linear', cmap='magma',
                         vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB', label='magnitude (dB)')
    plt.scatter(times, freqs, facecolors='none', edgecolors='cyan', s=30, alpha=0.8, linewidths=1)
    plt.ylim(0, 3500)  
    plt.xlim(0, 25)    
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.title(f'Spectrogram with Local Maxima Constellation Overlay of the song "{song_name}"')
    plt.tight_layout()
    return fig




def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            song_id TEXT,
            offset REAL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)')
    conn.commit()
    return conn


def store_song_fingerprints(conn,times, freqs, song_name):
    peaks = sorted(zip(times, freqs), key=lambda x: x[0])
    cursor=conn.cursor()
    #hashing parameters
    fan_value = 15
    min_time_delta = 0.05
    max_time_delta = 3.0
    hashes_to_insert = []
    for i in range(len(peaks)):
        anchor_time, anchor_freq = peaks[i]
        
        matches = 0
        for j in range(i + 1, len(peaks)):
            target_time, target_freq = peaks[j]
            time_delta = target_time - anchor_time
            
            if time_delta < min_time_delta:
                continue
            if time_delta > max_time_delta:
                break
            f1 = int(anchor_freq)
            f2 = int(target_freq)
            dt = round(time_delta, 3) # Round time difference to milliseconds
            
            # Create signature and hash
            signature = f"{f1}|{f2}|{dt}"
            compact_hash = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:8]
            
            # Append to our batch list (rounding anchor time for clean DB storage)
            hashes_to_insert.append((compact_hash, song_name, round(anchor_time, 3)))
            
            matches += 1
            if matches >= fan_value:
                break
        
        #inserting the batch into database
    cursor.executemany('''
        INSERT INTO fingerprints (hash, song_id, offset)
         VALUES (?, ?, ?)
    ''', hashes_to_insert)
    conn.commit()
    return len(hashes_to_insert)


def process_song_database(folder_path, db_name="song_database.db"):
    
    
    # Audio parameters
    WINDOW_SIZE = 4096
    HOP_LENGTH = WINDOW_SIZE//4
    
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: The directory '{folder_path}' does not exist.")
        return
        
    audio_extensions = ['*.mp3', '*.wav', '*.flac']
    song_files = []
    for ext in audio_extensions:
        song_files.extend(folder.glob(ext))
        
    if not song_files:
        print(f"No audio files found in '{folder_path}'.")
        return

    print(f"Found {len(song_files)} songs. Initializing database...")
    conn = setup_database(db_name)
    
    total_hashes = 0
    
    for idx, filepath in enumerate(song_files, 1):
        # Use the filename (without extension) as the unique song identifier
        song_name = filepath.stem 
        print(f"[{idx}/{len(song_files)}] Processing '{song_name}'...")
        
        try:
            _,times, freqs,_ = get_peaks(filepath, WINDOW_SIZE, HOP_LENGTH,neighborhood_size=50,threshold_db=-25)
            num_hashes = store_song_fingerprints(conn, times, freqs, song_name)
            total_hashes += num_hashes
            print(f"    -> Generated {num_hashes} hashes.")
        except Exception as e:
            print(f"    -> Error processing '{song_name}': {e}")
            
    conn.close()
    print("-" * 40)
    print(f"Processing Complete! Stored a total of {total_hashes} hashes in '{db_name}'.")




def generate_clip_hashes(times, freqs):
    
    peaks = sorted(zip(times, freqs), key=lambda x: x[0])
    
    fan_value = 15
    min_time_delta = 0.05
    max_time_delta = 3.0
    
    clip_hashes = []
    
    for i in range(len(peaks)):
        anchor_time, anchor_freq = peaks[i]
        
        matches = 0
        for j in range(i + 1, len(peaks)):
            target_time, target_freq = peaks[j]
            time_delta = target_time - anchor_time
            
            if time_delta < min_time_delta:
                continue
            if time_delta > max_time_delta:
                break
                
            f1 = int(anchor_freq)
            f2 = int(target_freq)
            dt = round(time_delta, 3) 
            
            signature = f"{f1}|{f2}|{dt}"
            compact_hash = hashlib.sha1(signature.encode('utf-8')).hexdigest()[:8]
            clip_hashes.append((compact_hash, round(anchor_time, 3)))
            
            matches += 1
            if matches >= fan_value:
                break
                
    return clip_hashes

#functions that identifies song
def match_audio_clip(clip_path, db_path="song_database.db"):
    #idenitfying songs by analyszing hash time offsets"
    print(f"Analyzing unknown clip: '{clip_path}'...")
    
    WINDOW_SIZE = 4096
    HOP_LENGTH = WINDOW_SIZE // 4
    _, times, freqs, _ = get_peaks(
        clip_path, 
        WINDOW_SIZE, 
        HOP_LENGTH, 
        neighborhood_size=50, 
        threshold_db=-25
    )
    
    # Generate hashes for the clip
    clip_hashes = generate_clip_hashes(times, freqs)
    
    if not clip_hashes:
        print("Could not extract any clear hashes from the clip.")
        return None
        
    # Group clip offsets by hash for extremely fast lookup
    clip_hash_dict = {}
    for h, offset in clip_hashes:
        if h not in clip_hash_dict:
            clip_hash_dict[h] = []
        clip_hash_dict[h].append(offset)
        
    unique_hashes = list(clip_hash_dict.keys())
    
    # Query the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # We use a Counter to build our Histogram of Relative Offsets
    alignment_histogram = Counter()
    
    #chunking our queries
    chunk_size = 900
    for i in range(0, len(unique_hashes), chunk_size):
        chunk = unique_hashes[i:i + chunk_size]
        placeholders = ','.join(['?'] * len(chunk))
        
        query = f"SELECT hash, song_id, offset FROM fingerprints WHERE hash IN ({placeholders})"
        
        for row in cursor.execute(query, chunk):
            db_hash, song_id, db_offset = row
            
            # 4. The Alignment Logic
            # Compare the database offset to every time that hash appeared in the clip
            for clip_offset in clip_hash_dict[db_hash]:
                # Calculate the relative time offset (db_offset - clip_offset)
                relative_offset = round(db_offset - clip_offset, 1)
                alignment_histogram[(song_id, relative_offset)] += 1
                
    conn.close()
    
    # 5. Determine the matched song
    if not alignment_histogram:
        print("No matches found in the database.")
        return None
        
    # the matched song is the bin with the highest number of alligned hashes
    best_match, max_score = alignment_histogram.most_common(1)[0]
    matched_song = best_match[0]
    matched_offset = best_match[1]
    
    print("-" * 40)
    print(f"MATCH FOUND: {matched_song}")
    print(f"Alignment Score: {max_score} synchronized hashes")
    print(f"Clip starts at ~{matched_offset} seconds into the song")
    print("-" * 40)
    
    return matched_song,alignment_histogram


def plot_offset_histogram(alignment_histogram, matched_song):
  
    if not alignment_histogram or not matched_song:
        return
    offsets = []
    scores = []
    
    for (song_id, offset), count in alignment_histogram.items():
        if song_id == matched_song:
            offsets.append(offset)
            scores.append(count)
            
    if not offsets:
        return

    fig=plt.figure(figsize=(12, 6))
    # Plot the offsets as a bar chart 
    plt.bar(offsets, scores, width=0.2, color='royalblue', alpha=0.8)
    # Find and highlight the winning spike
    max_score = max(scores)
    winning_offset = offsets[scores.index(max_score)]
    plt.axvline(x=winning_offset, color='red', linestyle='--', linewidth=2,
                label=f'True Alignment Offset ({winning_offset}s)')
    plt.title(f"Histogram of Relative Time Offsets: '{matched_song}'", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Time Offset (Database Time - Clip Time) in Seconds", fontsize=12)
    plt.ylabel("Number of Synchronized Hashes", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # Force the plot to show the full scale so you can see how 'flat' it really is
    # plt.ylim(0, 5) # Assuming 600 was your peak height from the match
    plt.tight_layout()
    return fig





     





