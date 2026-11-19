"""Causal, chunk-by-chunk replay of BCI IV 2a EEG through sliding windows.

This replays a saved GDF recording; it does not connect to a live headset.
The model artifact is loaded only when the user runs this program.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import mne
import numpy as np
from scipy.signal import sosfilt

from src.realtime.realtime_filter_spec import FS, SOS, init_state
from src.realtime.sliding_window import SlidingWindowBuffer

CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
    "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
    "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
]
EVENT_TO_LABEL = {"769": 0, "770": 1, "771": 2, "772": 3}
LABEL_NAMES = {0: "Left", 1: "Right", 2: "Feet", 3: "Tongue"}


def read_cues(raw):
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {v: str(k).strip() for k, v in event_id.items()}
    cues = []
    for sample, _, code in events:
        text = id_to_code.get(int(code), "")
        if text in EVENT_TO_LABEL:
            cues.append((int(sample - raw.first_samp), EVENT_TO_LABEL[text]))
    cues.sort()
    if not cues:
        raise ValueError("GDF contains no labeled cue events 769–772")
    return cues


def replay(gdf_path: Path, model_path: Path, output_csv: Path,
           chunk_samples=25, realtime=False):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], FS):
        raise ValueError(f"Expected {FS:g} Hz, got {raw.info['sfreq']:g} Hz")
    missing = [c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing model channels: {missing}")
    cues = read_cues(raw)
    eeg = raw.get_data(picks=CHANNELS).astype(np.float64, copy=False)  # volts
    model = joblib.load(model_path)

    # Independent state per cue lets overlapping trial windows be scheduled
    # while every state receives only samples that have arrived so far.
    states = []
    cue_cursor = 0
    zi = init_state(len(CHANNELS))
    rows = []
    received = 0

    for start in range(0, eeg.shape[1], chunk_samples):
        stop = min(start + chunk_samples, eeg.shape[1])
        block, zi = sosfilt(SOS, eeg[:, start:stop], axis=-1, zi=zi)
        block = block.astype(np.float32, copy=False)
        # Cue events become visible only when their sample has arrived.
        while cue_cursor < len(cues) and cues[cue_cursor][0] < stop:
            cue_sample, label = cues[cue_cursor]
            if cue_sample >= start:
                next_cue = cues[cue_cursor + 1][0] if cue_cursor + 1 < len(cues) else eeg.shape[1]
                states.append({
                    "cue": cue_sample, "label": label,
                    "next_cue_offset": next_cue - cue_sample,
                    "buffer": SlidingWindowBuffer(
                        sampling_rate=FS, window_seconds=3.0,
                        step_seconds=0.25, initial_offset_seconds=0.5,
                        n_channels=len(CHANNELS), subject_id=gdf_path.stem[:3], cue_sample=0,
                    ),
                })
            cue_cursor += 1

        for state in states:
            local_start = max(start, state["cue"])
            if local_start >= stop:
                continue
            chunk = block[:, local_start - start:]
            for win in state["buffer"].add_data(chunk):
                # Do not let a window cross into the next labeled trial.
                if win.end_sample > state["next_cue_offset"]:
                    continue
                model_input = win.data[np.newaxis, ...]
                if model_input.shape != (1, len(CHANNELS), 751):
                    raise ValueError(f"Unexpected model input shape {model_input.shape}")
                t0 = time.perf_counter()
                pred = int(model.predict(model_input)[0])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                rows.append({
                    "cue_sample": state["cue"],
                    "window_index": win.index,
                    "window_start_sample": state["cue"] + win.start_sample,
                    "window_end_sample_exclusive": state["cue"] + win.end_sample,
                    "window_start_s": (state["cue"] + win.start_sample) / FS,
                    "window_end_s": (state["cue"] + win.end_sample - 1) / FS,
                    "ready_at_sample": state["cue"] + win.end_sample,
                    "true_label": state["label"],
                    "predicted_label": pred,
                    "predicted_class": LABEL_NAMES[pred],
                    "correct": pred == state["label"],
                    "inference_ms": elapsed_ms,
                })
        # Retire trial schedulers after the following cue has arrived.
        states = [
            state for state in states
            if stop <= state["cue"] + state["next_cue_offset"]
        ]
        received = stop
        if realtime:
            time.sleep((stop - start) / FS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Replay complete: {len(rows)} window predictions -> {output_csv}")
    if rows:
        print(f"Window accuracy (trial labels): {np.mean([r['correct'] for r in rows]):.2%}")
        print(f"Mean model inference: {np.mean([r['inference_ms'] for r in rows]):.3f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf", type=Path, default=Path("data/A01T.gdf"))
    parser.add_argument("--model", type=Path, default=Path("bci2a_csp_lda_n6_all9.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realtime/sliding_predictions.csv"))
    parser.add_argument("--chunk-samples", type=int, default=25)
    parser.add_argument("--realtime", action="store_true", help="pace replay at the recording's sample rate")
    args = parser.parse_args()
    if args.chunk_samples < 1:
        parser.error("--chunk-samples must be positive")
    replay(args.gdf, args.model, args.output, args.chunk_samples, args.realtime)


if __name__ == "__main__":
    main()
