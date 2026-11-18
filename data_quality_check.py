"""Data quality checks and per-subject statistics for multi-subject EEG arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = {0: "Left", 1: "Right", 2: "Feet", 3: "Tongue"}


def check_arrays(
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    *,
    expected_channels: int = 22,
    expected_sfreq: float = 250.0,
    expected_labels=(0, 1, 2, 3),
) -> dict:
    """Validate dimensions, alignment, finite values, labels and basic amplitudes."""
    issues = []
    if X.ndim != 3:
        issues.append(f"X must be 3-D; got ndim={X.ndim}")
    if y.ndim != 1 or subject_id.ndim != 1:
        issues.append("y and subject_id must both be 1-D")
    if X.shape[0] != len(y) or len(y) != len(subject_id):
        issues.append("Trial count mismatch among X, y and subject_id")
    if X.ndim == 3 and X.shape[1] != expected_channels:
        issues.append(f"Expected {expected_channels} channels; got {X.shape[1]}")
    if len(y) and not set(np.unique(y)).issubset(set(expected_labels)):
        issues.append(f"Unexpected labels found: {sorted(set(np.unique(y)) - set(expected_labels))}")
    if len(subject_id) and any(not str(s).startswith("A") for s in np.unique(subject_id)):
        issues.append("Some subject_id values do not look like A01...A09")

    finite_fraction = float(np.isfinite(X).mean()) if X.size else 0.0
    if finite_fraction < 1.0:
        issues.append(f"X contains non-finite values; finite fraction={finite_fraction:.6f}")

    if X.size:
        # Robust, descriptive check only; does not automatically delete samples.
        abs_max = float(np.nanmax(np.abs(X)))
        global_std = float(np.nanstd(X))
        if global_std == 0:
            issues.append("X has zero global standard deviation")
    else:
        abs_max, global_std = float("nan"), float("nan")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "n_trials": int(len(y)),
        "n_channels": int(X.shape[1]) if X.ndim == 3 else None,
        "n_times": int(X.shape[2]) if X.ndim == 3 else None,
        "finite_fraction": finite_fraction,
        "absolute_max": abs_max,
        "global_std": global_std,
        "expected_sfreq_hz": expected_sfreq,
    }


def build_subject_statistics(X: np.ndarray, y: np.ndarray, subject_id: np.ndarray) -> pd.DataFrame:
    """Create one row per subject, including total and per-class trial counts."""
    if len(X) != len(y) or len(y) != len(subject_id):
        raise ValueError("X, y and subject_id must have aligned trial counts")

    rows = []
    for sid in sorted(np.unique(subject_id).astype(str)):
        mask = subject_id.astype(str) == sid
        labels, counts = np.unique(y[mask], return_counts=True)
        count_by_label = {int(label): int(count) for label, count in zip(labels, counts)}
        row = {
            "subject_id": sid,
            "n_trials": int(mask.sum()),
            "n_channels": int(X.shape[1]),
            "n_times": int(X.shape[2]),
        }
        for label, name in LABEL_NAMES.items():
            row[f"{name.lower()}_trials"] = count_by_label.get(label, 0)
        row["missing_class_count"] = sum(count_by_label.get(label, 0) == 0 for label in LABEL_NAMES)
        row["nan_or_inf_values"] = int((~np.isfinite(X[mask])).sum())
        row["mean_abs_amplitude"] = float(np.nanmean(np.abs(X[mask])))
        row["max_abs_amplitude"] = float(np.nanmax(np.abs(X[mask])))
        rows.append(row)

    return pd.DataFrame(rows)



def build_channel_statistics(X: np.ndarray, subject_id: np.ndarray) -> pd.DataFrame:
    """Per-subject/per-channel amplitude and variance diagnostics; no auto-rejection."""
    if X.ndim != 3 or len(X) != len(subject_id):
        raise ValueError("X must be 3-D and align with subject_id")
    rows = []
    for sid in sorted(np.unique(subject_id).astype(str)):
        subject_X = X[subject_id.astype(str) == sid]
        for channel_index in range(subject_X.shape[1]):
            values = subject_X[:, channel_index, :]
            rows.append({
                "subject_id": sid,
                "channel_index": channel_index,
                "std_v": float(np.std(values)),
                "mean_abs_v": float(np.mean(np.abs(values))),
                "max_abs_v": float(np.max(np.abs(values))),
                "finite_fraction": float(np.isfinite(values).mean()),
                "near_flat_std_lt_1e-12": bool(np.std(values) < 1e-12),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Check a saved multi-subject EEG NPZ file.")
    parser.add_argument("--input", required=True, help="NPZ created by multi_subject_data.py")
    parser.add_argument("--output", default="results/subject_statistics.csv")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    X, y, subject_id = data["X"], data["y"], data["subject_id"]

    report = check_arrays(X, y, subject_id)
    print("Quality check:", "PASS" if report["passed"] else "FAIL")
    for key, value in report.items():
        if key != "issues":
            print(f"{key}: {value}")
    for issue in report["issues"]:
        print("ISSUE:", issue)

    stats = build_subject_statistics(X, y, subject_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved statistics: {output}")

    channel_output = output.with_name("channel_quality.csv")
    channel_stats = build_channel_statistics(X, subject_id)
    channel_stats.to_csv(channel_output, index=False, encoding="utf-8-sig")
    print(f"Saved channel diagnostics: {channel_output}")


if __name__ == "__main__":
    main()
