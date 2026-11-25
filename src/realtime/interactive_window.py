"""Manually inspect one fixed-length EEG window and request a prediction.

Browse labeled trials or move a fixed 751-sample window along the recording,
then click Predict.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.signal import sosfilt

from src.realtime.prediction_feedback import PredictionFeedback
from src.realtime.realtime_filter_spec import FS, SAMPLES_PER_EPOCH, SOS, TMIN, init_state

CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
    "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
    "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
]
LABELS = {0: "Left hand", 1: "Right hand", 2: "Feet", 3: "Tongue"}
WINDOW_SAMPLES = SAMPLES_PER_EPOCH
IMAGERY_END_SECONDS = 4.0


def load_causal_eeg(gdf_path: Path) -> np.ndarray:
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], FS):
        raise ValueError(f"需要 {FS:g} Hz 数据，实际为 {raw.info['sfreq']:g} Hz")
    missing = [name for name in CHANNELS if name not in raw.ch_names]
    if missing:
        raise ValueError(f"GDF 缺少模型通道：{missing}")
    samples = raw.get_data(picks=CHANNELS).astype(np.float64, copy=False)
    filtered, _ = sosfilt(SOS, samples, axis=-1, zi=init_state(len(CHANNELS)))
    return filtered.astype(np.float32, copy=False)


def load_labeled_trials(gdf_path: Path):
    """Return cue-aligned labels and each trial's next onset boundary."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {v: str(k).strip() for k, v in event_id.items()}
    starts = []
    cues = []
    for sample, _, code in events:
        description = id_to_code.get(int(code), "")
        sample = int(sample - raw.first_samp)
        if description == "768":
            starts.append(sample)
        elif description in {"769", "770", "771", "772"}:
            cues.append((sample, int(description) - 769))
    starts.sort()
    cues.sort()
    trials = []
    for onset, next_onset in zip(starts, starts[1:] + [raw.n_times]):
        cue = next((item for item in cues if onset <= item[0] < next_onset), None)
        if cue is not None:
            trials.append((cue[0], next_onset, cue[1]))
    return trials


def trial_start_bounds(cue_sample: int, next_onset: int) -> tuple[int, int]:
    """Return the inclusive range of starts that keep the window inside the trial."""
    first_start = cue_sample + round(TMIN * FS)
    imagery_stop = min(next_onset, cue_sample + round(IMAGERY_END_SECONDS * FS) + 1)
    return first_start, imagery_stop - WINDOW_SAMPLES


def true_label_for_window(start_sample: int, trials):
    """Only score a full window inside the cue's four-second imagery period."""
    for cue_sample, next_onset, label in trials:
        first_start, last_start = trial_start_bounds(cue_sample, next_onset)
        if first_start <= start_sample <= last_start:
            return label
    return None


def append_result(path: Path, start_sample: int, predicted: int, true_label, probabilities):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "window_start_sample", "window_end_sample_exclusive",
                "window_start_s", "window_end_s", "predicted_label", "predicted_class",
                "true_label", "true_class", "correct",
                "p_left", "p_right", "p_feet", "p_tongue",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), start_sample,
            start_sample + WINDOW_SAMPLES,
            f"{start_sample / FS:.3f}", f"{(start_sample + WINDOW_SAMPLES - 1) / FS:.3f}",
            predicted, LABELS[predicted],
            "" if true_label is None else true_label,
            "" if true_label is None else LABELS[true_label],
            "" if true_label is None else predicted == true_label,
            *[f"{p:.6f}" for p in probabilities],
        ])


def show_window(gdf_path: Path, model_path: Path, output_csv: Path):
    eeg = load_causal_eeg(gdf_path)
    trials = load_labeled_trials(gdf_path)
    max_start = eeg.shape[1] - WINDOW_SAMPLES
    if max_start < 0:
        raise ValueError("记录长度不足 751 个采样点，无法形成模型窗口")
    model = joblib.load(model_path)

    trial_windows = []
    for cue_sample, next_onset, label in trials:
        first_start, last_start = trial_start_bounds(cue_sample, next_onset)
        if 0 <= first_start <= min(last_start, max_start):
            trial_windows.append((first_start, label))

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(left=0.125, right=0.97, bottom=0.405)
    x = np.arange(WINDOW_SAMPLES) / FS
    offsets = np.arange(len(CHANNELS)) * 120.0
    lines = [
        ax.plot(x, eeg[i, :WINDOW_SAMPLES] * 1e6 + offsets[i], lw=0.8)[0]
        for i in range(len(CHANNELS))
    ]
    ax.set_yticks(offsets)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel("Time within window (seconds)")
    ax.set_title("Manual EEG Window Viewer (fixed 3 s / 751 samples)")
    ax.grid(axis="x", alpha=0.25)
    window_info = fig.text(0.125, 0.338, "", fontsize=11, color="#475569")
    feedback = PredictionFeedback(fig)

    slider_position = [0.16, 0.066, 0.44, 0.033]
    time_ax = fig.add_axes(slider_position)
    time_slider = Slider(
        time_ax, "Window start (s)", 0.0, max(1, max_start) / FS,
        valinit=0.0, valstep=1.0 / FS, valfmt="%.2f",
    )
    trial_slider = None
    mode_button = None
    if trial_windows:
        trial_ax = fig.add_axes(slider_position, label="trial slider")
        trial_slider = Slider(
            trial_ax, f"Trial (1–{len(trial_windows)})", 1, max(2, len(trial_windows)),
            valinit=1, valstep=1, valfmt="%d",
        )
        mode_button = Button(fig.add_axes([0.66, 0.05, 0.12, 0.06]), "Browse by time",
                             color="#ECF1F7", hovercolor="#DCE7F4")
    mascot_button = Button(fig.add_axes([0.79, 0.05, 0.105, 0.06]), f"Mascot: {feedback.mascot}",
                           color="#ECF1F7", hovercolor="#DCE7F4")
    mascot_button.label.set_fontsize(9)
    predict_button = Button(fig.add_axes([0.905, 0.05, 0.065, 0.06]), "Predict",
                            color="#29446C", hovercolor="#345584")
    predict_button.label.set_color("white")
    predict_button.label.set_fontweight("bold")

    trial_mode = trial_slider is not None
    start = 0
    true_label = None

    def update_window(_value=None):
        nonlocal start, true_label
        if trial_mode:
            trial_number = int(trial_slider.val)
            start, true_label = trial_windows[trial_number - 1]
            position = f"Trial {trial_number} / {len(trial_windows)}"
        else:
            start = min(round(time_slider.val * FS), max_start)
            true_label = true_label_for_window(start, trials)
            position = "Time browsing"
        end_seconds = (start + WINDOW_SAMPLES - 1) / FS
        window_info.set_text(f"{position}   ·   Recording {start / FS:.2f}–{end_seconds:.2f} s")
        segment = eeg[:, start:start + WINDOW_SAMPLES]
        for line, channel, offset in zip(lines, segment, offsets):
            line.set_ydata(channel * 1e6 + offset)
        feedback.clear(LABELS.get(true_label))
        fig.canvas.draw_idle()

    def show_active_slider():
        time_ax.set_visible(not trial_mode)
        time_slider.set_active(not trial_mode and max_start > 0)
        if trial_slider is not None:
            trial_slider.ax.set_visible(trial_mode)
            trial_slider.set_active(trial_mode and len(trial_windows) > 1)
            mode_button.label.set_text("Browse by time" if trial_mode else "Browse by trial")

    def switch_mode(_event):
        nonlocal trial_mode
        trial_mode = not trial_mode
        show_active_slider()
        if trial_mode:
            nearest_index = min(range(len(trial_windows)), key=lambda i: abs(trial_windows[i][0] - start))
            trial_slider.set_val(nearest_index + 1)
        else:
            time_slider.set_val(start / FS)

    def predict(_event):
        batch = eeg[np.newaxis, :, start:start + WINDOW_SAMPLES]
        pred = int(model.predict(batch)[0])
        probs = model.predict_proba(batch)[0]
        probabilities = "    ·    ".join(f"{name} {probs[label]:.1%}" for label, name in LABELS.items())
        feedback.show_prediction(LABELS[pred], LABELS.get(true_label), probabilities)
        append_result(output_csv, start, pred, true_label, probs)
        fig.canvas.draw_idle()

    def change_mascot(_event):
        feedback.change_mascot()
        mascot_button.label.set_text(f"Mascot: {feedback.mascot}")
        fig.canvas.draw_idle()

    time_slider.on_changed(update_window)
    if trial_slider is not None:
        trial_slider.on_changed(update_window)
        mode_button.on_clicked(switch_mode)
    predict_button.on_clicked(predict)
    mascot_button.on_clicked(change_mascot)
    show_active_slider()
    update_window()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf", type=Path, default=Path("data/A01T.gdf"))
    parser.add_argument("--model", type=Path, default=Path("bci2a_csp_lda_n6_all9.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realtime/manual_predictions_with_correctness.csv"))
    args = parser.parse_args()
    show_window(args.gdf, args.model, args.output)


if __name__ == "__main__":
    main()
