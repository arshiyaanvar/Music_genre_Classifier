import os
import librosa
import numpy as np
import joblib

MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "music_genre_model.pkl")

data = joblib.load(MODEL_FILE)
model = data["model"]
scaler = data["scaler"]
label_encoder = data["label_encoder"]

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=30)
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    features = [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate] + mfccs_mean.tolist()
    return np.array(features)

def predict_genre(file_path):
    try:
        features = extract_features(file_path)
        features_scaled = scaler.transform([features])  # shape: (1, 24)
        pred_class = model.predict(features_scaled)
        genre = label_encoder.inverse_transform(pred_class)[0]
        return genre
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    file_path = input("Enter path to audio file: ").strip()
    genre = predict_genre(file_path)
    print(f"Predicted genre: {genre}")
    if genre:
        print(f"Predicted genre: {genre}")
    else:
        print("Prediction failed.")
