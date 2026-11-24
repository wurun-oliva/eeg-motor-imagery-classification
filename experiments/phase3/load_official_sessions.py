"""Load BCI IV-2a T/E GDF files with the competition's published labels.

The separate ``true_labels.zip`` is downloaded from the competition results
page.  Only the E session requires it; T labels are checked against its cue
events, so a wrong trial order fails loudly.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from multi_subject_data import _make_labeled_trial_events


def load_sessions(data_dir: Path, labels_zip: Path, *, sessions: str = "TE",
                  l_freq: float = 4.0, h_freq: float = 38.0,
                  tmin: float = 0.0, tmax: float = 4.0,
                  decim: int = 2):
    """Return X, y, subject IDs and session IDs, preserving event order."""
    if sessions not in {"T", "TE"}:
        raise ValueError("sessions must be 'T' or 'TE'")
    if not labels_zip.exists():
        raise FileNotFoundError(f"Official labels archive missing: {labels_zip}")
    xs, ys, ids, session_ids = [], [], [], []
    channel_order = None
    with zipfile.ZipFile(labels_zip) as archive:
        for sid in (f"A{i:02d}" for i in range(1, 10)):
            for session in sessions:
                path = data_dir / f"{sid}{session}.gdf"
                raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
                if not np.isclose(raw.info["sfreq"], 250):
                    raise ValueError(f"Unexpected sample rate in {path}")
                channels = [name for name in raw.ch_names if name.startswith("EEG-")]
                if len(channels) != 22:
                    raise ValueError(f"Expected 22 EEG channels in {path}")
                if channel_order is None:
                    channel_order = channels
                if channels != channel_order:
                    raise ValueError(f"Channel order differs in {path}")
                raw.pick(channels)
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq, h_freq, method="iir",
                               iir_params={"order": 8, "ftype": "butter"},
                               phase="zero", verbose="ERROR")
                official = loadmat(io.BytesIO(archive.read(f"{sid}{session}.mat")))["classlabel"].ravel().astype(int) - 1
                if len(official) != 288 or not np.array_equal(np.bincount(official, minlength=4), [72] * 4):
                    raise ValueError(f"Official labels malformed for {sid}{session}")
                if session == "T":
                    events, _, _ = _make_labeled_trial_events(raw, anchor="cue")
                    cue_labels = events[:, 2] - 1
                    if not np.array_equal(cue_labels, official):
                        raise ValueError(f"Official labels disagree with GDF cues for {sid}T")
                else:
                    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
                    cue_code = event_id.get("783")
                    if cue_code is None:
                        raise ValueError(f"Missing E-session cue 783 in {path}")
                    events = events[events[:, 2] == cue_code].copy()
                    if len(events) != len(official):
                        raise ValueError(f"Event/label count mismatch in {path}")
                    events[:, 2] = official + 1
                epochs = mne.Epochs(raw, events, event_id={str(i): i for i in range(1, 5)},
                                    tmin=tmin, tmax=tmax, baseline=None, preload=True,
                                    reject_by_annotation=True, verbose="ERROR")
                if len(epochs) != 288:
                    raise ValueError(f"Trials dropped in {path}: {len(epochs)}")
                x = epochs.get_data(copy=True)[:, :, ::decim].astype(np.float32)
                xs.append(x)
                ys.append(official.astype(np.int64))
                ids.extend([sid] * len(official))
                session_ids.extend([session] * len(official))
                print(f"Loaded {sid}{session}: {x.shape}", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.asarray(ids), np.asarray(session_ids)
