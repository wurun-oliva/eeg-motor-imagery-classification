"""Compare unscaled EEG with train-only per-channel z-score under LOSO."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multi_subject_data import load_multiple_subjects

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
OUT = ROOT / "experiments" / "phase3" / "normalization_loso.csv"


def main() -> None:
    X, y, ids = load_multiple_subjects(
        ROOT / "data", subject_ids=SUBJECTS, l_freq=8, h_freq=30,
        tmin=0.5, tmax=3.5,
    )
    rows = []
    for subject in SUBJECTS:
        test = ids == subject
        train = ~test
        mean = X[train].mean(axis=(0, 2), keepdims=True)
        std = X[train].std(axis=(0, 2), keepdims=True).clip(min=1e-8)
        for method, data in (
            ("none", X),
            ("train_subject_channel_zscore", (X - mean) / std),
        ):
            model = make_pipeline(
                CSP(n_components=6, reg=None, log=True, cov_est="concat"),
                LinearDiscriminantAnalysis(),
            )
            model.fit(data[train], y[train])
            pred = model.predict(data[test])
            rows.append({
                "normalization": method,
                "test_subject": subject,
                "n_test_trials": int(test.sum()),
                "accuracy": accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro"),
                "cohen_kappa": cohen_kappa_score(y[test], pred),
                "protocol": "LOSO; T session; 22 EEG channels; CSP(n=6)+LDA; 8-30 Hz",
            })
        print(f"{subject}: {rows[-2]['accuracy']:.3f} -> {rows[-1]['accuracy']:.3f}", flush=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
