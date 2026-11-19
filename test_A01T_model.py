
from pathlib import Path
import numpy as np
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent

X_FILE = PROJECT_DIR / "output" / "A01T_epochs.npy"
Y_FILE = PROJECT_DIR / "output" / "A01T_labels.npy"

MODEL_FILE = (
    PROJECT_DIR
    / "model_from_C"
    / "给A_模型与接口"
    / "bci2a_csp_lda_n6_all9.joblib"
)

LABELS = {
    0: "Left",
    1: "Right",
    2: "Feet",
    3: "Tongue"
}

print("=" * 60)
print("A01T REAL EEG MODEL PREDICTION")
print("=" * 60)

# Load real EEG epochs
X = np.load(X_FILE)
y = np.load(Y_FILE)

# Load C model
model = joblib.load(MODEL_FILE)

print("EEG shape:", X.shape)
print("EEG dtype:", X.dtype)
print("Model loaded:", type(model))

# Validate
assert X.shape == (288, 22, 751)
assert X.dtype == np.float32
assert y.shape == (288,)
assert np.isfinite(X).all()

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Evaluate
accuracy = np.mean(predictions == y) * 100

print("\nPrediction completed!")
print("Prediction shape:", predictions.shape)
print("Probability shape:", probabilities.shape)

print("\nFirst 10 predictions:")

for i in range(10):
    print(
        f"Trial {i+1}: "
        f"True={LABELS[int(y[i])]}, "
        f"Predicted={LABELS[int(predictions[i])]}, "
        f"Probabilities={np.round(probabilities[i], 4)}"
    )

print("\nAccuracy:", f"{accuracy:.2f}%")

# Save results
output_file = PROJECT_DIR / "output" / "A01T_model_predictions.csv"

import csv

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([
        "trial",
        "true_label",
        "predicted_label",
        "correct",
        "left_probability",
        "right_probability",
        "feet_probability",
        "tongue_probability"
    ])

    for i in range(len(y)):
        writer.writerow([
            i + 1,
            LABELS[int(y[i])],
            LABELS[int(predictions[i])],
            predictions[i] == y[i],
            *probabilities[i]
        ])

print("\nCSV saved:", output_file)
print("\nA01T REAL EEG PREDICTION COMPLETED!")
