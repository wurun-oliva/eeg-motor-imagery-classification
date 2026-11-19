"""Manually inspect one fixed-length EEG window and request a prediction.

This is an interactive offline viewer. Drag the time slider to move the
751-sample window, then click Predict. It does not alter the auto-replay path.
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

from src.realtime.realtime_filter_spec import FS, SOS, init_state

CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
    "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
    "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
]
LABELS = {0: "左手", 1: "右手", 2: "双脚", 3: "舌头"}
WINDOW_SAMPLES = 751


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


def append_result(path: Path, start_sample: int, predicted: int, probabilities):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "window_start_sample", "window_end_sample_exclusive",
                "window_start_s", "window_end_s", "predicted_label", "predicted_class",
                "p_left", "p_right", "p_feet", "p_tongue",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), start_sample,
            start_sample + WINDOW_SAMPLES,
            f"{start_sample / FS:.3f}", f"{(start_sample + WINDOW_SAMPLES - 1) / FS:.3f}",
            predicted, LABELS[predicted], *[f"{p:.6f}" for p in probabilities],
        ])


def show_window(gdf_path: Path, model_path: Path, output_csv: Path):
    eeg = load_causal_eeg(gdf_path)
    model = joblib.load(model_path)
    max_start = eeg.shape[1] - WINDOW_SAMPLES
    if max_start < 0:
        raise ValueError("记录长度不足 751 个采样点，无法形成模型窗口")

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(bottom=0.22)
    x = np.arange(WINDOW_SAMPLES) / FS
    offsets = np.arange(len(CHANNELS)) * 120.0
    lines = [
        ax.plot(x, eeg[i, :WINDOW_SAMPLES] * 1e6 + offsets[i], lw=0.8)[0]
        for i in range(len(CHANNELS))
    ]
    ax.set_yticks(offsets)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel("窗口内时间（秒）")
    ax.set_title("手动 EEG 窗口查看器（固定 3 秒 / 751 点）")
    ax.grid(axis="x", alpha=0.25)
    status = fig.text(0.12, 0.12, "拖动下方滑块选择窗口，再点击“预测”", fontsize=11)

    slider_ax = fig.add_axes([0.14, 0.065, 0.66, 0.035])
    slider = Slider(
        slider_ax, "窗口起点（秒）", 0.0, max_start / FS,
        valinit=0.0, valstep=1.0 / FS, valfmt="%.2f",
    )
    button_ax = fig.add_axes([0.83, 0.045, 0.12, 0.075])
    predict_button = Button(button_ax, "预测")

    state = {"start": 0}

    def update_window(_value):
        start = int(round(slider.val * FS))
        state["start"] = start
        segment = eeg[:, start:start + WINDOW_SAMPLES]
        for i, line in enumerate(lines):
            line.set_ydata(segment[i] * 1e6 + offsets[i])
        status.set_text(
            f"当前窗口：{start / FS:.2f}–{(start + WINDOW_SAMPLES - 1) / FS:.2f} 秒；尚未预测"
        )
        fig.canvas.draw_idle()

    def predict(_event):
        start = state["start"]
        batch = eeg[np.newaxis, :, start:start + WINDOW_SAMPLES]
        pred = int(model.predict(batch)[0])
        probs = model.predict_proba(batch)[0]
        status.set_text(
            f"预测：{LABELS[pred]}  |  左手 {probs[0]:.1%}  右手 {probs[1]:.1%}  "
            f"双脚 {probs[2]:.1%}  舌头 {probs[3]:.1%}"
        )
        append_result(output_csv, start, pred, probs)
        fig.canvas.draw_idle()

    slider.on_changed(update_window)
    predict_button.on_clicked(predict)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf", type=Path, default=Path("data/A01T.gdf"))
    parser.add_argument("--model", type=Path, default=Path("bci2a_csp_lda_n6_all9.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realtime/manual_predictions.csv"))
    args = parser.parse_args()
    show_window(args.gdf, args.model, args.output)


if __name__ == "__main__":
    main()
