import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_ml_models(features_csv="features.csv"):
    df = pd.read_csv(features_csv)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    models = {
        "SVM": SVC(kernel="linear", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.2f}")
        joblib.dump(model, f"{name}_model.pkl")
        print(f"✅ {name} model saved")

if __name__ == "__main__":
    train_ml_models()
