
"""
Realtime EEG Model Integration
Member A + Member C

Replay continuous EEG data in chunks,
detect cue-related windows and predict
using C's trained CSP + LDA model.
"""

import csv
import time
from pathlib import Path

import numpy as np
import joblib


# ==================================================
# 1. Paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_FILE = (
    PROJECT_DIR
    / "model_from_C"
    / "给A_模型与接口"
    / "bci2a_csp_lda_n6_all9.joblib"
)

CONTINUOUS_FILE = OUTPUT_DIR / "A01T_continuous_eeg.npy"
CUE_FILE = OUTPUT_DIR / "A01T_cue_samples.npy"
LABEL_FILE = OUTPUT_DIR / "A01T_cue_labels.npy"
EPOCH_FILE = OUTPUT_DIR / "A01T_epochs.npy"

CSV_FILE = OUTPUT_DIR / "A01T_realtime_integration.csv"


# ==================================================
# 2. Parameters
# ==================================================

FS = 250
N_CHANNELS = 22

CHUNK_SIZE = 25

WINDOW_START_SECONDS = 0.5
WINDOW_END_SECONDS = 3.5

WINDOW_SAMPLES = 751

# False: fast replay for initial testing
# True: replay according to actual sampling duration
REALTIME_REPLAY = False

LABEL_NAMES = {
    0: "Left",
    1: "Right",
    2: "Feet",
    3: "Tongue"
}


# ==================================================
# 3. Load resources
# ==================================================

def load_resources():

    print("=" * 60)
    print("LOADING EEG DATA AND C MODEL")
    print("=" * 60)

    required_files = [
        CONTINUOUS_FILE,
        CUE_FILE,
        LABEL_FILE,
        EPOCH_FILE,
        MODEL_FILE
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Missing file: {file}"
            )

    eeg = np.load(
        CONTINUOUS_FILE,
        allow_pickle=False
    )

    cue_samples = np.load(
        CUE_FILE,
        allow_pickle=False
    )

    labels = np.load(
        LABEL_FILE,
        allow_pickle=False
    )

    original_epochs = np.load(
        EPOCH_FILE,
        allow_pickle=False
    )

    model = joblib.load(MODEL_FILE)

    if eeg.ndim != 2 or eeg.shape[0] != N_CHANNELS:
        raise ValueError(
            f"Unexpected continuous EEG shape: {eeg.shape}"
        )

    if original_epochs.shape != (
        len(labels), N_CHANNELS, WINDOW_SAMPLES
    ):
        raise ValueError(
            f"Unexpected epoch shape: {original_epochs.shape}"
        )

    if len(cue_samples) != len(labels):
        raise ValueError("Cue and label count mismatch.")

    if not np.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf.")

    if not np.array_equal(
        np.sort(cue_samples), cue_samples
    ):
        raise ValueError("Cue samples are not chronological.")

    if not np.array_equal(
        model.classes_, np.array([0, 1, 2, 3])
    ):
        raise ValueError("Unexpected model classes.")

    print("Continuous EEG shape:", eeg.shape)
    print("Cue count:", len(cue_samples))
    print("Epoch shape:", original_epochs.shape)
    print("Model loaded:", type(model))
    print("Model classes:", model.classes_)

    print("\nAll resource checks passed!")

    return eeg, cue_samples, labels, original_epochs, model


# ==================================================
# 4. Simulate continuous data reception
# ==================================================

def run_realtime_replay(
    eeg,
    cue_samples,
    labels,
    original_epochs,
    model
):

    total_samples = eeg.shape[1]

    # Storage for samples already received
    received_buffer = np.zeros_like(eeg)

    # Calculate prediction window positions
    window_offset = int(
        WINDOW_START_SECONDS * FS
    )

    window_starts = cue_samples + window_offset

    window_ends = (
        window_starts + WINDOW_SAMPLES
    )

    if np.any(window_ends > total_samples):
        raise ValueError(
            "Some prediction windows exceed EEG recording."
        )

    results = []

    next_trial = 0
    total_chunks = 0

    total_start = time.perf_counter()

    print("\nStarting continuous EEG replay...")
    print("Chunk size:", CHUNK_SIZE)
    print("Realtime mode:", REALTIME_REPLAY)

    # Receive data chunk by chunk
    for start in range(
        0, total_samples, CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            total_samples
        )

        chunk = eeg[:, start:end]

        # Simulate receiving new EEG data
        received_buffer[:, start:end] = chunk

        total_chunks += 1

        received_until = end

        # Predict when a complete window has arrived
        while (
            next_trial < len(cue_samples)
            and window_ends[next_trial] <= received_until
        ):

            trial_start = int(
                window_starts[next_trial]
            )

            trial_end = int(
                window_ends[next_trial]
            )

            prediction_start = time.perf_counter()

            epoch = received_buffer[
                :,
                trial_start:trial_end
            ].copy()

            if epoch.shape != (
                N_CHANNELS, WINDOW_SAMPLES
            ):
                raise RuntimeError(
                    f"Invalid window shape: {epoch.shape}"
                )

            # Verify replayed window against extracted epoch
            if not np.allclose(
                epoch,
                original_epochs[next_trial],
                rtol=1e-5,
                atol=1e-9
            ):
                raise RuntimeError(
                    f"Window reconstruction mismatch at trial "
                    f"{next_trial + 1}"
                )

            model_input = epoch[np.newaxis, :, :]

            prediction = int(
                model.predict(model_input)[0]
            )

            probabilities = model.predict_proba(
                model_input
            )[0]

            prediction_ms = (
                time.perf_counter() - prediction_start
            ) * 1000

            true_label = int(labels[next_trial])

            correct = prediction == true_label

            results.append({
                "trial": next_trial + 1,
                "cue_sample": int(cue_samples[next_trial]),
                "window_start_sample": trial_start,
                "window_end_sample": trial_end,
                "true_label": true_label,
                "true_class": LABEL_NAMES[true_label],
                "predicted_label": prediction,
                "predicted_class": LABEL_NAMES[prediction],
                "correct": correct,
                "left_probability": float(probabilities[0]),
                "right_probability": float(probabilities[1]),
                "feet_probability": float(probabilities[2]),
                "tongue_probability": float(probabilities[3]),
                "prediction_ms": prediction_ms
            })

            next_trial += 1

            if next_trial % 50 == 0:
                print(
                    f"Completed {next_trial}/"
                    f"{len(cue_samples)} trials"
                )

        # Optional real-time replay
        if REALTIME_REPLAY:
            time.sleep(
                chunk.shape[1] / FS
            )

    elapsed = time.perf_counter() - total_start

    if next_trial != len(cue_samples):
        raise RuntimeError(
            f"Only {next_trial} trials predicted."
        )

    print("\nAll trials received and predicted!")

    return results, total_chunks, elapsed


# ==================================================
# 5. Save prediction results
# ==================================================

def save_results(results):

    if not results:
        raise ValueError("No results to save.")

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV saved:", CSV_FILE)


# ==================================================
# 6. Main
# ==================================================

def main():

    eeg, cue_samples, labels, epochs, model = (
        load_resources()
    )

    results, total_chunks, elapsed = (
        run_realtime_replay(
            eeg,
            cue_samples,
            labels,
            epochs,
            model
        )
    )

    save_results(results)

    correct = sum(
        row["correct"] for row in results
    )

    accuracy = correct / len(results) * 100

    avg_prediction_ms = np.mean([
        row["prediction_ms"] for row in results
    ])

    print("\n" + "=" * 60)
    print("REALTIME EEG MODEL INTEGRATION SUMMARY")
    print("=" * 60)

    print("Subject: A01")
    print("Total EEG samples:", eeg.shape[1])
    print("Total chunks received:", total_chunks)
    print("Total trials predicted:", len(results))
    print("Correct predictions:", correct)

    print(f"Accuracy: {accuracy:.2f}%")

    print(
        f"Average prediction time: "
        f"{avg_prediction_ms:.4f} ms"
    )

    print(
        f"Total replay processing time: "
        f"{elapsed:.3f} seconds"
    )

    print("\nWindow reconstruction: PASSED")
    print("CSP + LDA model integration: PASSED")
    print("All trials processed successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()
