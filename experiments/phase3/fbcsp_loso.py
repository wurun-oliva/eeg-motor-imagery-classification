"""Filter-bank CSP + LDA, evaluated with leave-one-subject-out CV."""
from __future__ import annotations

import csv
import gc
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
BANDS = [(4 + 4*i, 8 + 4*i) for i in range(8)]
OUT = ROOT / "experiments" / "phase3" / "fbcsp_loso.csv"


def main() -> None:
    band_data = []
    y_ref = ids_ref = None
    for low, high in BANDS:
        X, y, ids = load_multiple_subjects(
            ROOT / "data", subject_ids=SUBJECTS, l_freq=low, h_freq=high,
            tmin=0.5, tmax=3.5,
        )
        band_data.append(X)
        if y_ref is None:
            y_ref, ids_ref = y, ids
        elif not (np.array_equal(y_ref, y) and np.array_equal(ids_ref, ids)):
            raise RuntimeError("Trial alignment differs across filter bands")
        print(f"Loaded {low}-{high} Hz {X.shape}", flush=True)

    rows = []
    for held_out in SUBJECTS:
        test = ids_ref == held_out
        train = ~test
        train_features, test_features = [], []
        for (low, high), X in zip(BANDS, band_data):
            csp = CSP(n_components=4, reg=None, log=True, cov_est="concat")
            csp.fit(X[train], y_ref[train])
            train_features.append(csp.transform(X[train]))
            test_features.append(csp.transform(X[test]))
        x_train = np.concatenate(train_features, axis=1)
        x_test = np.concatenate(test_features, axis=1)
        model = make_pipeline(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
        model.fit(x_train, y_ref[train])
        pred = model.predict(x_test)
        row = {
            "test_subject": held_out,
            "n_test_trials": int(test.sum()),
            "accuracy": accuracy_score(y_ref[test], pred),
            "macro_f1": f1_score(y_ref[test], pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(y_ref[test], pred),
            "model": "FBCSP(8 bands x 4 CSP)+shrinkage LDA",
            "bands_hz": ";".join(f"{a}-{b}" for a, b in BANDS),
            "protocol": "LOSO; T session; 22 EEG channels; cue+0.5 to +3.5 s",
        }
        rows.append(row)
        print(f"{held_out}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
