
from pathlib import Path
import numpy as np
import mne

# ==================================================
# 1. Paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

GDF_FILE = PROJECT_DIR / "data" / "A01T.gdf"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. Parameters
# ==================================================

FS = 250
LOW_FREQ = 8.0
HIGH_FREQ = 30.0

TMIN = 0.5
TMAX = 3.5

EXPECTED_CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2",
    "EEG-3", "EEG-4", "EEG-5", "EEG-C3",
    "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4",
    "EEG-8", "EEG-9", "EEG-10", "EEG-11",
    "EEG-12", "EEG-13", "EEG-14", "EEG-Pz",
    "EEG-15", "EEG-16"
]

EVENT_LABELS = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3
}

LABEL_NAMES = {
    0: "Left",
    1: "Right",
    2: "Feet",
    3: "Tongue"
}

# ==================================================
# 3. Read GDF
# ==================================================

print("=" * 60)
print("EXTRACTING A01T EEG DATA")
print("=" * 60)

if not GDF_FILE.exists():
    raise FileNotFoundError(GDF_FILE)

raw = mne.io.read_raw_gdf(
    GDF_FILE,
    preload=True,
    verbose=False
)

print("Sampling rate:", raw.info["sfreq"])

if not np.isclose(raw.info["sfreq"], FS):
    raise ValueError("Unexpected sampling rate.")

# ==================================================
# 4. Verify channels
# ==================================================

missing = [
    ch for ch in EXPECTED_CHANNELS
    if ch not in raw.ch_names
]

if missing:
    raise ValueError(f"Missing channels: {missing}")

raw.pick_channels(
    EXPECTED_CHANNELS,
    ordered=True
)

if raw.ch_names != EXPECTED_CHANNELS:
    raise ValueError("Channel order mismatch.")

print("EEG channels:", len(raw.ch_names))
print("Channel order verified!")

# ==================================================
# 5. Filter continuous EEG
# ==================================================

raw.filter(
    l_freq=LOW_FREQ,
    h_freq=HIGH_FREQ,
    picks=EXPECTED_CHANNELS,
    method="iir",
    verbose=False
)

print("8-30 Hz filtering completed.")

# ==================================================
# 6. Save continuous EEG
# ==================================================

# Shape: (22, total_samples)
# Unit: V

continuous_eeg = raw.get_data(
    picks=EXPECTED_CHANNELS
).astype(np.float32)

continuous_file = OUTPUT_DIR / "A01T_continuous_eeg.npy"

np.save(continuous_file, continuous_eeg)

print("\nContinuous EEG shape:", continuous_eeg.shape)
print("Continuous EEG saved:", continuous_file)

# ==================================================
# 7. Read events
# ==================================================

events, event_id = mne.events_from_annotations(
    raw,
    verbose=False
)

selected_events = {
    label: event_id[label]
    for label in EVENT_LABELS
}

print("\nSelected event mapping:")
print(selected_events)

# ==================================================
# 8. Extract epochs
# ==================================================

epochs = mne.Epochs(
    raw,
    events,
    event_id=selected_events,
    tmin=TMIN,
    tmax=TMAX,
    baseline=None,
    picks=EXPECTED_CHANNELS,
    preload=True,
    reject_by_annotation=False,
    verbose=False
)

X = epochs.get_data().astype(np.float32)

event_codes = epochs.events[:, 2]

code_to_label = {
    selected_events["769"]: 0,
    selected_events["770"]: 1,
    selected_events["771"]: 2,
    selected_events["772"]: 3
}

y = np.array(
    [code_to_label[code] for code in event_codes],
    dtype=np.int64
)

# ==================================================
# 9. Save cue positions
# ==================================================

# Convert MNE absolute sample indices
# into indices relative to the beginning
# of continuous_eeg.

cue_samples = (
    epochs.events[:, 0] - raw.first_samp
).astype(np.int64)

cue_labels = y.copy()

# Check that all cue positions are valid.

if np.any(cue_samples < 0):
    raise ValueError("Invalid cue positions.")

required_end = (
    cue_samples + int(TMAX * FS) + 1
)

if np.any(required_end > continuous_eeg.shape[1]):
    raise ValueError("Some trials exceed recording boundaries.")

np.save(
    OUTPUT_DIR / "A01T_cue_samples.npy",
    cue_samples
)

np.save(
    OUTPUT_DIR / "A01T_cue_labels.npy",
    cue_labels
)

# ==================================================
# 10. Validate epochs
# ==================================================

print("\nExtracted EEG shape:", X.shape)
print("EEG dtype:", X.dtype)
print("Labels shape:", y.shape)
print("Cue samples shape:", cue_samples.shape)

for label, name in LABEL_NAMES.items():
    print(name, ":", np.sum(y == label))

if X.shape[1:] != (22, 751):
    raise ValueError(f"Unexpected epoch shape: {X.shape}")

if not np.isfinite(X).all():
    raise ValueError("EEG contains NaN or Inf.")

if not np.isfinite(continuous_eeg).all():
    raise ValueError("Continuous EEG contains NaN or Inf.")

if not np.isin(y, list(LABEL_NAMES)).all():
    raise ValueError("Invalid labels.")

if len(cue_samples) != len(X):
    raise ValueError("Cue and epoch count mismatch.")

# ==================================================
# 11. Save original epochs
# ==================================================

np.save(OUTPUT_DIR / "A01T_epochs.npy", X)
np.save(OUTPUT_DIR / "A01T_labels.npy", y)

print("\nSaved files:")
print("A01T_continuous_eeg.npy")
print("A01T_cue_samples.npy")
print("A01T_cue_labels.npy")
print("A01T_epochs.npy")
print("A01T_labels.npy")

print("\nAll validation checks passed!")
print("A01T EEG extraction completed.")
print("=" * 60)
