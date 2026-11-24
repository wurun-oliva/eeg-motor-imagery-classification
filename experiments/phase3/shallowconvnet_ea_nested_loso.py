"""Nested, subject-disjoint validation for ShallowConvNet + unlabeled EA.

Outer evaluation: LOSO. Inner validation: one source subject is held out
(rotated across outer folds), so validation subjects do not occur in the
inner training set. The selected epoch count is then used to retrain on all
eight outer source subjects. Euclidean alignment uses signal only, without
any labels, including for the outer held-out subject.
"""
from __future__ import annotations

import csv
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multi_subject_data import load_multiple_subjects
from shallowconvnet_loso import ShallowConvNet, euclidean_align_by_subject

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
OUT = ROOT / "experiments" / "phase3" / "shallowconvnet_ea_nested_loso.csv"
SEEDS = [11, 23, 42]
MAX_EPOCHS = 60
PATIENCE = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_fit_apply(X: np.ndarray, fit_idx: np.ndarray, apply_idx: np.ndarray):
    mean = X[fit_idx].mean(axis=(0, 2), keepdims=True)
    std = X[fit_idx].std(axis=(0, 2), keepdims=True).clip(min=1e-8)
    return ((X[fit_idx] - mean) / std).astype(np.float32), ((X[apply_idx] - mean) / std).astype(np.float32)


def to_tensor(X: np.ndarray, y: np.ndarray):
    return TensorDataset(
        torch.from_numpy(X[:, None]),
        torch.from_numpy(y).long(),
    )


def train_one(model, X_train, y_train, X_val, y_val, max_epochs, patience):
    loader = DataLoader(to_tensor(X_train, y_train), batch_size=128, shuffle=True)
    xv = torch.from_numpy(X_val[:, None])
    yv = torch.from_numpy(y_val).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state, stale = float("inf"), 1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-euclidean-alignment", action="store_true",
        help="Run the matched no-alignment control.",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    X, y, ids = load_multiple_subjects(
        ROOT / "data", subject_ids=SUBJECTS, l_freq=8, h_freq=30,
        tmin=0.5, tmax=3.5,
    )
    X = X[:, :, ::2].copy()
    if not args.no_euclidean_alignment:
        # Align each participant using only their EEG covariance; labels are not passed.
        X = euclidean_align_by_subject(X, ids)
        output_path = ROOT / "experiments" / "phase3" / "shallowconvnet_ea_nested_loso.csv"
    else:
        output_path = ROOT / "experiments" / "phase3" / "shallowconvnet_nested_loso.csv"
    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fold, test_subject in enumerate(SUBJECTS):
        test_idx = np.flatnonzero(ids == test_subject)
        source_subjects = [s for s in SUBJECTS if s != test_subject]
        val_subject = source_subjects[fold % len(source_subjects)]
        val_idx = np.flatnonzero(ids == val_subject)
        fit_idx = np.flatnonzero(np.isin(ids, [s for s in source_subjects if s != val_subject]))
        print(f"Outer={test_subject}, inner validation={val_subject}", flush=True)

        for seed in SEEDS:
            seed_all(seed + fold * 100)
            x_fit, x_val = normalize_fit_apply(X, fit_idx, val_idx)
            inner = ShallowConvNet(X.shape[1], X.shape[2])
            _, selected_epochs = train_one(
                inner, x_fit, y[fit_idx], x_val, y[val_idx],
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            del inner, x_fit, x_val

            # Refit on every outer source subject for the selected epoch count.
            x_source, x_test = normalize_fit_apply(X, np.flatnonzero(ids != test_subject), test_idx)
            seed_all(seed + fold * 100 + 1)
            final = ShallowConvNet(X.shape[1], X.shape[2])
            loader = DataLoader(to_tensor(x_source, y[ids != test_subject]), batch_size=128, shuffle=True)
            optimizer = torch.optim.Adam(final.parameters(), lr=6.25e-4, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(selected_epochs):
                final.train()
                for xb, yb in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(final(xb), yb)
                    loss.backward()
                    optimizer.step()
            final.eval()
            with torch.no_grad():
                pred = final(torch.from_numpy(x_test[:, None])).argmax(dim=1).numpy()
            truth = y[test_idx]
            row = {
                "seed": seed,
                "test_subject": test_subject,
                "inner_validation_subject": val_subject,
                "n_test_trials": len(test_idx),
                "selected_epochs": selected_epochs,
                "accuracy": accuracy_score(truth, pred),
                "macro_f1": f1_score(truth, pred, average="macro"),
                "cohen_kappa": cohen_kappa_score(truth, pred),
                "model": (
                    "ShallowConvNet + Euclidean Alignment"
                    if not args.no_euclidean_alignment else
                    "ShallowConvNet (no alignment)"
                ),
                "protocol": (
                    "Outer LOSO; subject-disjoint inner validation; 3 seeds; unlabeled target EA; 125 Hz"
                    if not args.no_euclidean_alignment else
                    "Outer LOSO; subject-disjoint inner validation; 3 seeds; no EA; 125 Hz"
                ),
            }
            rows.append(row)
            print(f"  seed={seed}: acc={row['accuracy']:.3f}, epochs={selected_epochs}", flush=True)
            with output_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del final, optimizer, x_source, x_test
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
