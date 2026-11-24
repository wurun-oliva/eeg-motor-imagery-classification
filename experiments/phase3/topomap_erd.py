"""Plot labeled motor-imagery band-power change against pre-cue fixation.

The BCI IV 2a recordings do not contain a separately labeled resting-state
condition. The reference interval here is the fixation period before each cue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multi_subject_data import load_subject

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "phase3"
FS = 250.0
TMIN, TMAX = -1.8, 3.5
CHANNEL_MAP = {
    "EEG-Fz": "Fz", "EEG-0": "FC3", "EEG-1": "FC1", "EEG-2": "FCz",
    "EEG-3": "FC2", "EEG-4": "FC4", "EEG-5": "C5", "EEG-C3": "C3",
    "EEG-6": "C1", "EEG-Cz": "Cz", "EEG-7": "C2", "EEG-C4": "C4",
    "EEG-8": "C6", "EEG-9": "CP3", "EEG-10": "CP1", "EEG-11": "CPz",
    "EEG-12": "CP2", "EEG-13": "CP4", "EEG-14": "P1", "EEG-Pz": "Pz",
    "EEG-15": "P2", "EEG-16": "POz",
}
CLASSES = {0: "Left hand", 1: "Right hand", 2: "Feet", 3: "Tongue"}
BANDS = {"Mu alpha 8–13 Hz": (8.0, 13.0), "Beta 13–30 Hz": (13.0, 30.0)}


def band_power(x: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    freq, psd = welch(x, fs=FS, nperseg=min(250, x.shape[-1]), axis=-1)
    mask = (freq >= band[0]) & (freq <= band[1])
    return trapezoid(psd[..., mask], freq[mask], axis=-1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    collected = {cls: {band: [] for band in BANDS} for cls in CLASSES}
    channel_names = None
    for subject in (f"A{i:02d}" for i in range(1, 10)):
        X, y, _, channels, _, _ = load_subject(
            DATA, subject, l_freq=4, h_freq=38, tmin=TMIN, tmax=TMAX,
        )
        channel_names = channels
        base = X[..., :int(round((-0.2 - TMIN) * FS))]
        active_start = int(round((0.5 - TMIN) * FS))
        active = X[..., active_start:]
        for cls in CLASSES:
            mask = y == cls
            for band, limits in BANDS.items():
                p0 = band_power(base[mask], limits).mean(axis=0)
                p1 = band_power(active[mask], limits).mean(axis=0)
                collected[cls][band].append(100.0 * (p1 - p0) / np.maximum(p0, 1e-30))
        print(f"Loaded {subject}: {len(y)} cue-labeled trials")

    names = [CHANNEL_MAP[name] for name in channel_names]
    info = mne.create_info(names, FS, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, on_missing="raise")
    values = {
        (cls, band): np.mean(np.stack(collected[cls][band]), axis=0)
        for cls in CLASSES for band in BANDS
    }
    absmax = max(float(np.max(np.abs(v))) for v in values.values())
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, band in enumerate(BANDS):
        for col, cls in enumerate(CLASSES):
            ax = axes[row, col]
            im, _ = mne.viz.plot_topomap(
                values[(cls, band)], info, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-absmax, absmax), contours=4,
            )
            ax.set_title(f"{CLASSES[cls]}\n{band}")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Band-power change vs pre-cue fixation (%)")
    fig.suptitle(
        "BCI IV 2a A01–A09 Motor Imagery Topomaps\n"
        "Reference is pre-cue fixation, not a separately labeled resting state",
        fontsize=13,
    )
    path = OUT / "erd_topomaps_A01-A09.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
