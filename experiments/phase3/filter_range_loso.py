"""LOSO comparison of common motor-imagery filter ranges on A01T-A09T."""
from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multi_subject_data import load_multiple_subjects

DATA = ROOT / "data"
OUT = ROOT / "experiments" / "phase3" / "filter_range_loso.csv"
BANDS = [(8, 30), (4, 38), (8, 13), (13, 30), (4, 8)]
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


def main() -> None:
    rows = []
    for low, high in BANDS:
        X, y, subject_ids = load_multiple_subjects(
            DATA, subject_ids=SUBJECTS, l_freq=low, h_freq=high,
            tmin=0.5, tmax=3.5,
        )
        print(f"Loaded {low}-{high} Hz: {X.shape}", flush=True)
        for held_out in SUBJECTS:
            test = subject_ids == held_out
            train = ~test
            model = Pipeline([
                ("csp", CSP(n_components=6, reg=None, log=True, cov_est="concat")),
                ("lda", LinearDiscriminantAnalysis()),
            ])
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            rows.append({
                "filter_band_hz": f"{low}-{high}",
                "low_hz": low,
                "high_hz": high,
                "test_subject": held_out,
                "n_test_trials": int(test.sum()),
                "accuracy": accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro"),
                "cohen_kappa": cohen_kappa_score(y[test], pred),
                "model": "CSP(n=6)+LDA",
                "protocol": "LOSO; T session; 22 EEG channels",
            })
            print(f"  {held_out}: {rows[-1]['accuracy']:.3f}", flush=True)
        del X, y, subject_ids
        gc.collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
