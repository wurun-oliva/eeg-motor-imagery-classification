
"""Multi-subject BCI Competition IV Dataset 2a T-session loader.

Output contract:
    X: (n_trials, 22, n_times), float32
    y: (n_trials,), int64; Left=0, Right=1, Feet=2, Tongue=3
    subject_id: (n_trials,), strings such as A01

Preprocessing convention:
- Sampling frequency: 250 Hz
- EEG channels: 22 EEG-* channels
- Bandpass filter: IIR Butterworth, 8-30 Hz
- Filter phase: zero (forward-backward, offline processing)
- Epoch: 0.5 to 3.5 seconds
- Default epoch anchor: class cue (769-772)
- Optional compatibility mode: trial onset (768)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mne
import numpy as np


EVENT_TO_LABEL = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3,
}

LABEL_NAMES = {
    0: "Left",
    1: "Right",
    2: "Feet",
    3: "Tongue",
}

TRIAL_START_CODE = "768"


def _description_to_code(description: str) -> str:
    """Extract event code from an annotation description."""
    value = str(description).strip()
    match = re.search(r"(?:^|\D)(768|769|770|771|772)(?:\D|$)", value)
    return match.group(1) if match else value


def find_training_file(
    data_dir: str | Path,
    subject_id: str
) -> Path:
    """Find one subject's T-session GDF file."""
    root = Path(data_dir).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root}"
        )

    sid = subject_id.upper()

    if not re.fullmatch(r"A0[1-9]", sid):
        raise ValueError(
            f"Invalid subject_id {subject_id!r}; expected A01...A09"
        )

    candidates = sorted(set(
        list(root.rglob(f"{sid}T*.gdf")) +
        list(root.rglob(f"{sid}t*.gdf"))
    ))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {sid}T.gdf under {root}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple files found for {sid}: {candidates}"
        )

    return candidates[0]


def _make_labeled_trial_events(
    raw: mne.io.BaseRaw,
    anchor: str = "cue"
) -> tuple[np.ndarray, dict[str, int], dict]:
    """
    Create labeled events using either cue or trial onset.

    anchor="cue":
        Use the actual sample position of event 769-772.

    anchor="onset":
        Use the sample position of event 768.
    """

    if anchor not in {"cue", "onset"}:
        raise ValueError(
            "anchor must be either 'cue' or 'onset'"
        )

    if len(raw.annotations) == 0:
        raise ValueError("No GDF annotations found.")

    events, event_id = mne.events_from_annotations(
        raw,
        verbose="ERROR"
    )

    id_to_desc = {
        value: str(key)
        for key, value in event_id.items()
    }

    parsed = [
        (
            int(sample),
            _description_to_code(id_to_desc[int(code)])
        )
        for sample, _, code in events
    ]

    parsed.sort(key=lambda item: item[0])

    trial_starts = [
        i
        for i, (_, code) in enumerate(parsed)
        if code == TRIAL_START_CODE
    ]

    labeled_events = []

    label_event_id = {
        "Left": 1,
        "Right": 2,
        "Feet": 3,
        "Tongue": 4
    }

    counts = {
        name: 0
        for name in label_event_id
    }

    for pos, idx in enumerate(trial_starts):

        start_sample = parsed[idx][0]

        if pos + 1 < len(trial_starts):
            stop_sample = parsed[trial_starts[pos + 1]][0]
        else:
            stop_sample = raw.n_times

        cue_events = [
            (sample, code)
            for sample, code in parsed[idx + 1:]
            if (
                sample < stop_sample
                and code in EVENT_TO_LABEL
            )
        ]

        if len(cue_events) != 1:
            raise ValueError(
                f"Trial at sample {start_sample}: "
                f"expected exactly one class cue (769-772), "
                f"found {cue_events}"
            )

        cue_sample, cue_code = cue_events[0]

        label = EVENT_TO_LABEL[cue_code]
        name = LABEL_NAMES[label]

        # Select the actual epoch event sample.
        if anchor == "cue":
            event_sample = cue_sample
        else:
            event_sample = start_sample

        labeled_events.append([
            event_sample,
            0,
            label_event_id[name]
        ])

        counts[name] += 1

    if not labeled_events:
        raise ValueError(
            "No 768 trial-onset markers were found."
        )

    synthetic_events = np.asarray(
        labeled_events,
        dtype=int
    )

    metadata = {
        "n_trial_start_markers": len(trial_starts),
        "class_counts": counts,
        "epoch_anchor": anchor,
        "original_event_descriptions": sorted(
            set(code for _, code in parsed)
        )
    }

    return synthetic_events, label_event_id, metadata


def load_subject(
    data_dir: str | Path,
    subject_id: str,
    *,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = 0.5,
    tmax: float = 3.5,
    expected_sfreq: float = 250.0,
    expected_eeg_channels: int = 22,
    channel_names: Optional[Sequence[str]] = None,
    anchor: str = "cue",
    verbose: str | bool = "ERROR",
):
    """
    Load one T-session file.

    Returns:
        X, y, subject_id, channels, sfreq, metadata
    """

    file_path = find_training_file(
        data_dir,
        subject_id
    )

    raw = mne.io.read_raw_gdf(
        str(file_path),
        preload=True,
        verbose=verbose
    )

    sfreq = float(raw.info["sfreq"])

    if not np.isclose(sfreq, expected_sfreq):
        raise ValueError(
            f"{subject_id}: {sfreq:g} Hz found; "
            f"expected {expected_sfreq:g} Hz"
        )

    # Select only the 22 EEG channels.
    eeg_names = [
        name
        for name in raw.ch_names
        if name.startswith("EEG-")
    ]

    if len(eeg_names) != expected_eeg_channels:
        raise ValueError(
            f"{subject_id}: expected {expected_eeg_channels} "
            f"EEG-* channels, found {len(eeg_names)}: {eeg_names}"
        )

    # Keep the reference channel order.
    if channel_names is not None:

        if set(eeg_names) != set(channel_names):
            raise ValueError(
                f"{subject_id}: EEG channel set differs "
                f"from reference. "
                f"Missing={sorted(set(channel_names)-set(eeg_names))}; "
                f"extra={sorted(set(eeg_names)-set(channel_names))}"
            )

        eeg_names = list(channel_names)

    raw.pick_channels(
        eeg_names,
        ordered=True
    )

    # 8th-order IIR Butterworth bandpass filtering.
    # phase="zero" applies forward-backward filtering for offline use.
    iir_params = {
        "order": 8,
        "ftype": "butter",
    }
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=eeg_names,
        method="iir",
        iir_params=iir_params,
        phase="zero",
        verbose=verbose
    )

    # Generate events using the selected anchor.
    synthetic_events, label_event_id, event_metadata = (
        _make_labeled_trial_events(
            raw,
            anchor=anchor
        )
    )

    epochs = mne.Epochs(
        raw,
        synthetic_events,
        event_id=label_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=eeg_names,
        preload=True,
        reject_by_annotation=True,
        event_repeated="error",
        verbose=verbose
    )

    label_from_event = {
        label_event_id[name]: label
        for label, name in LABEL_NAMES.items()
    }

    y = np.asarray(
        [
            label_from_event[int(code)]
            for code in epochs.events[:, 2]
        ],
        dtype=np.int64
    )

    X = epochs.get_data(
        copy=True
    ).astype(
        np.float32,
        copy=False
    )

    ids = np.full(
        len(y),
        subject_id.upper(),
        dtype="<U3"
    )

    metadata = {
        "file": str(file_path),
        "sfreq_hz": sfreq,
        "channels": eeg_names,
        "n_eog_channels_excluded": 3,
        "tmin_s": tmin,
        "tmax_s": tmax,
        "epoch_anchor": anchor,
        "data_unit": "V",
        "filter": {
            "method": "iir",
            "design": "Butterworth",
            "phase": "zero (forward-backward)",
            "l_freq_hz": l_freq,
            "h_freq_hz": h_freq,
            "order": 8,
            "iir_params": iir_params
        },
        "n_times": (
            int(X.shape[-1])
            if X.ndim == 3
            else None
        ),
        "event_metadata": event_metadata
    }

    if (
        X.ndim != 3
        or X.shape[0] != len(y)
        or X.shape[1] != len(eeg_names)
    ):
        raise RuntimeError(
            f"{subject_id}: invalid output shapes "
            f"X={X.shape}, y={y.shape}"
        )

    return (
        X,
        y,
        ids,
        eeg_names,
        sfreq,
        metadata
    )


def load_multiple_subjects(
    data_dir: str | Path,
    subject_ids: Iterable[str] = tuple(
        f"A{i:02d}" for i in range(1, 10)
    ),
    **preprocess_kwargs
):
    """Load and concatenate multiple subjects."""

    subject_ids = [
        sid.upper()
        for sid in subject_ids
    ]

    if not subject_ids:
        raise ValueError(
            "subject_ids cannot be empty"
        )

    xs = []
    ys = []
    ids = []

    reference_channels = None
    reference_sfreq = None

    for sid in subject_ids:

        X, y, sid_array, channels, sfreq, _ = load_subject(
            data_dir,
            sid,
            channel_names=reference_channels,
            **preprocess_kwargs
        )

        if reference_channels is None:
            reference_channels = channels
            reference_sfreq = sfreq

        elif (
            channels != reference_channels
            or not np.isclose(sfreq, reference_sfreq)
        ):
            raise ValueError(
                f"{sid}: channel order or sampling rate differs"
            )

        xs.append(X)
        ys.append(y)
        ids.append(sid_array)

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ids)
    )


def select_subjects(
    X,
    y,
    subject_id,
    include=None,
    exclude=None
):
    """Select or exclude subjects while preserving alignment."""

    if (
        len(X) != len(y)
        or len(y) != len(subject_id)
    ):
        raise ValueError(
            "X, y and subject_id must have aligned trial counts"
        )

    mask = np.ones(
        len(y),
        dtype=bool
    )

    if include is not None:
        mask &= np.isin(
            subject_id,
            [str(s).upper() for s in include]
        )

    if exclude is not None:
        mask &= ~np.isin(
            subject_id,
            [str(s).upper() for s in exclude]
        )

    return (
        X[mask],
        y[mask],
        subject_id[mask]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[
            f"A{i:02d}"
            for i in range(1, 10)
        ]
    )

    parser.add_argument(
        "--output",
        default="results/all_subjects_T.npz"
    )

    parser.add_argument(
        "--anchor",
        choices=("cue", "onset"),
        default="cue",
        help=(
            "Epoch anchor: class cue (default) "
            "or 768 trial onset."
        )
    )

    args = parser.parse_args()

    X, y, subject_id = load_multiple_subjects(
        args.data_dir,
        args.subjects,
        anchor=args.anchor
    )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        out,
        X=X,
        y=y,
        subject_id=subject_id
    )

    print(
        f"Saved {out}: "
        f"X={X.shape}, "
        f"y={y.shape}, "
        f"subject_id={subject_id.shape}"
    )


if __name__ == "__main__":
    main()