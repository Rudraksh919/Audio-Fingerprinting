import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
from  pathlib import Path
import pandas as pd
import os


#importing all the required functions from the fingerprint script
from fingerprint import (get_peaks, process_song_database, match_audio_clip, 
                         plot_offset_histogram, plot_constellation_of_peaks, 
                         plot_spectrogram, store_song_fingerprints)

from huggingface_hub import snapshot_download

if not os.path.exists("song_database.db"):
    snapshot_download(
        repo_id="astroboy619/audio-fingerprint-db",
        repo_type="dataset",
        local_dir="."
    )

#configuring UI
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")
st.title("Zapptain America: Audio Fingerprinting")
st.markdown("#SIGNALS, SYSTEMS & NETWORKS Project")
st.markdown("This app is a mini version of the famous Shazaam App . It takes a song and from the user and finds its constellations peaks and matches with a song present in out database and gives it's name")

# Create the top-level navigation tabs
tab_lib, tab_identify, tab_batch = st.tabs([" LIBRARY", " IDENTIFY", " BATCH"])

#DataBase Management
with tab_lib:
    st.markdown("### Library Management")
    # Add a new song button
    st.markdown("####  Add a New Song :-")
    st.markdown("Upload a full track to add it permanently to the database.")
    uploaded_lib_file = st.file_uploader("Upload a song (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"], key="lib_upload")
    target_folder = "SongDatabase"
    db_name_default = "song_database.db"
    if uploaded_lib_file is not None:
        if st.button("Index Uploaded Song", type="primary"):
            with st.spinner(f"Analyzing and indexing '{uploaded_lib_file.name}'..."):
                os.makedirs(target_folder, exist_ok=True)# Ensures the directory exists
                file_path = os.path.join(target_folder, uploaded_lib_file.name) # Save the file permanently to the folder
                with open(file_path, "wb") as f:
                    f.write(uploaded_lib_file.getbuffer())
                song_name = uploaded_lib_file.name.rsplit('.', 1)[0]# Extract peaks and generate hashes
                WINDOW_SIZE = 4096
                HOP_LENGTH = WINDOW_SIZE // 4
                _, times, freqs, _ = get_peaks(file_path, WINDOW_SIZE, HOP_LENGTH)
                conn = sqlite3.connect(db_name_default)# Insert directly into the database
                num_hashes = store_song_fingerprints(conn, times, freqs, song_name)
                conn.close()
                st.success(f"Successfully added '{song_name}', Generated {num_hashes} hashes.")

    st.divider()
    st.markdown("####  Current Database")#Viewing Database
    # Query the database for all unique song names
    try:
        conn = sqlite3.connect(db_name_default)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT song_id FROM fingerprints ORDER BY song_id")
        indexed_songs = [row[0] for row in cursor.fetchall()]
        conn.close()
    except sqlite3.OperationalError:
        indexed_songs = []
    if not indexed_songs:
        st.info("The database is currently empty. Upload a song above to get started.")
    else:
        st.metric(label="Total Songs Indexed", value=len(indexed_songs))
        selected_song = st.selectbox("Select a song to view its constellation peaks:", indexed_songs)# Dropdown to select and view a specific song
        if selected_song:
            # Locate the original audio file in the folder to extract its visual peaks
            folder_path = Path(target_folder)
            matching_files = list(folder_path.glob(f"{selected_song}.*"))
            if matching_files:
                target_file = matching_files[0]
                with st.spinner(f"Rendering constellation for '{selected_song}'..."):
                    WINDOW_SIZE = 4096
                    HOP_LENGTH = WINDOW_SIZE // 4
                    spectrogram_db, times, freqs, samplerate = get_peaks(str(target_file), WINDOW_SIZE, HOP_LENGTH)
                    fig = plot_constellation_of_peaks(spectrogram_db, times, freqs, HOP_LENGTH, selected_song, samplerate)
                    if fig:
                        st.pyplot(fig)
            else:
                st.warning(f"Audio file for '{selected_song}' is missing from the '{target_folder}' folder. The hashes exist in the database, but the original audio is required to draw the visual constellation.")

#Single Clip Mode
with tab_identify:
    st.markdown("### Identify a clip")
    uploaded_file = st.file_uploader("Upload a  clip", type=["wav", "mp3", "flac", "ogg", "m4a"], key="single_upload")
    # We need a default db_name for the identify and batch tabs
    db_name_default = "song_database.db"
    if uploaded_file is not None:
        temp_clip_path = f"temp_query.{uploaded_file.name.split('.')[-1]}"
        with open(temp_clip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_clip_path)
        if st.button("Identify", type="primary"):
            with st.spinner("Extracting peaks and querying database..."):
                # Run matching
                matched_song, histogram = match_audio_clip(temp_clip_path, db_name_default)
                if matched_song:
                    st.success(f"**MATCH FOUND:** {matched_song}")
                    # Visualizations
                    WINDOW_SIZE = 4096
                    HOP_LENGTH = WINDOW_SIZE // 4
                    spectrogram_db, times, freqs, samplerate = get_peaks(temp_clip_path, WINDOW_SIZE, HOP_LENGTH)
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Spectrogram", "Constellation", "Alignment Spike"])
                    with viz_tab1:
                        fig1 = plot_spectrogram(spectrogram_db, HOP_LENGTH, samplerate, "Query Clip")
                        st.pyplot(fig1)
                        
                    with viz_tab2:
                        fig2 = plot_constellation_of_peaks(spectrogram_db, times, freqs, HOP_LENGTH, "Query Clip", samplerate)
                        st.pyplot(fig2)
                        
                    with viz_tab3:
                        fig3 = plot_offset_histogram(histogram, matched_song)
                        if fig3:
                            st.pyplot(fig3)
                        else:
                            st.info("Not enough data to plot the histogram.")
                else:
                    st.error("No match found in the database.")
                    
        # Cleanup
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

#Multi Clip Mode
with tab_batch:
    st.markdown("### Identify many clips at once")
    st.markdown("Upload a set of query clips. Results are written to a standardized `results.csv`.") 
    batch_files = st.file_uploader("Upload multiple query clips", type=["wav", "mp3", "flac"], accept_multiple_files=True, key="batch_upload")
    if batch_files:
        if st.button("Run Batch", type="primary"):
            results_data = []
            progress_bar = st.progress(0)# Create progress bar
            status_text = st.empty()
            for i, file in enumerate(batch_files):
                status_text.text(f"Identifying: {file.name} ({i+1}/{len(batch_files)})")
                temp_path = f"temp_batch_{i}.{file.name.split('.')[-1]}"# Save temp file
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                matched_song, _ = match_audio_clip(temp_path, db_name_default)# Run matching
                prediction = matched_song if matched_song else "None"
                results_data.append({
                    "filename": file.name,
                    "prediction": prediction
                })
                if os.path.exists(temp_path):#deletes the temperory file
                    os.remove(temp_path)  
                progress_bar.progress((i + 1) / len(batch_files))# Update progress
            status_text.text("processing complete!")
            df = pd.DataFrame(results_data)# Generates CSV
            st.dataframe(df) # Show the table on screen
            
            # Create download button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download the results",
                data=csv_data,
                file_name='results.csv',
                mime='text/csv',
                type="primary"
            )