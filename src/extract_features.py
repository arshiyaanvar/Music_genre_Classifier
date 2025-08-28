import os
import librosa
import numpy as np
import pandas as pd

GENRES = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def process_dataset(output_csv="features.csv"):
    features, labels = [], []
    base_dir = os.path.join("data", "gtzan_dataset")

    for genre in GENRES:
        genre_dir = os.path.join(base_dir, genre)
        print(f"Processing genre: {genre}")
        for filename in os.listdir(genre_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_dir, filename)
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(genre)

    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")
    return df

if __name__ == "__main__":
    process_dataset()
