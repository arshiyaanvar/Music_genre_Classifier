import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder

def train_cnn(features_csv="features.csv", model_path="cnn_model.h5"):
    df = pd.read_csv(features_csv)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    X = np.expand_dims(X, axis=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(X.shape[1], 1)),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(encoder.classes_), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    model.save(model_path)
    print(f"✅ CNN model saved to {model_path}")

if __name__ == "__main__":
    train_cnn()
