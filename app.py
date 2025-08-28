
import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from pydub import AudioSegment   
from src.predict import predict_genre

# Page config
st.set_page_config(page_title="Music Genre Classification", page_icon="🎵", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main {
            max-width: 950px;
            margin: auto;
        }
        .card {
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .prediction-badge {
            display: inline-block;
            padding: 12px 20px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 20px;
            margin: 20px 0;
            text-align: center;
        }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin: 12px 0 6px 0;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)
# Helper for dynamic colors
def genre_color(genre: str) -> str:
    colors = {
        "classical": "#4CAF50",   
        "disco": "#9C27B0",       
        "rock": "#F44336",        
        "jazz": "#2196F3",        
        "blues": "#00BCD4",       
        "country": "#795548",     
        "hiphop": "#FF9800",      
        "metal": "#212121",       
        "pop": "#E91E63",         
        "reggae": "#009688"      
    }
    return colors.get(genre.lower(), "#607D8B")  

st.title("🎶 Music Genre Classification")

uploaded_files = st.file_uploader(
    "Upload audio file(s)", 
    type=["au", "wav", "mp3"], 
    accept_multiple_files=True
)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]

        # Convert audio to WAV using pydub + ffmpeg 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_file_path = tmp_file.name

        try:
            audio = AudioSegment.from_file(uploaded_file, format=file_extension)
            audio.export(temp_file_path, format="wav")  # Always save as WAV
        except Exception as e:
            st.error(f"⚠️ Could not convert {uploaded_file.name}: {e}")
            continue

        with st.spinner(f"🔎 Analyzing {uploaded_file.name}..."):
            genre = predict_genre(temp_file_path)

        if genre:
            color = genre_color(genre)
            st.markdown(
                f"<div class='prediction-badge' style='background:{color}; color:white;'>"
                f"Predicted genre is: {genre}</div>", 
                unsafe_allow_html=True
            )
            results.append({"File": uploaded_file.name, "Predicted Genre": genre})
        else:
            st.error("❌ Could not predict genre.")

        # Audio Playback (WAV is guaranteed playable)
        st.markdown("### ▶️ Play Uploaded Audio")
        st.audio(temp_file_path, format="audio/wav")
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"🎵 {uploaded_file.name}")

        try:
            y, sr = librosa.load(temp_file_path, sr=None)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='section-title'>📊 Waveform</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 2))
                librosa.display.waveshow(y, sr=sr, ax=ax, color="purple")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown("<div class='section-title'>🌈 Spectrogram</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 2))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not process audio: {e}")

        os.remove(temp_file_path)
        st.markdown("</div>", unsafe_allow_html=True)  

    if len(results) > 1:
        st.markdown("## 📋 Prediction Summary")
        st.table(results)

else:
    st.info("⬆️ Upload one or more audio files to start.")
