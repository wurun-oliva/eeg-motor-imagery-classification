"""ShallowConvNet LOSO experiment on BCI IV-2a T sessions.

This is an exploratory deep-learning comparison. Each held-out participant
is excluded from fitting and validation; channel normalization statistics
are fitted only on source participants.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multi_subject_data import load_multiple_subjects

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
OUT = ROOT / "experiments" / "phase3" / "shallowconvnet_loso.csv"
SEED = 42


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding="same", bias=False),
            nn.Conv2d(40, 40, (n_channels, 1), bias=False),
            nn.BatchNorm2d(40, eps=1e-5, momentum=0.1),
        )
        self.pool = nn.AvgPool2d((1, 60), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            x = self.pool(self.features(torch.zeros(1, 1, n_channels, n_times)).square())
            size = x.numel()
        self.classifier = nn.Linear(size, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x).square())
        x = self.dropout(torch.log(torch.clamp(x, min=1e-6)))
        return self.classifier(x.flatten(1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def euclidean_align_by_subject(X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Unsupervised per-subject EA using signal covariance only, never labels."""
    aligned = np.empty_like(X)
    for subject in np.unique(subject_ids):
        indices = np.flatnonzero(subject_ids == subject)
        trials = X[indices]
        covariance = np.mean(np.einsum("nct,ndt->ncd", trials, trials) / trials.shape[-1], axis=0)
        covariance = (covariance + covariance.T) / 2
        eigvals, eigvecs = np.linalg.eigh(covariance)
        floor = max(float(eigvals.mean()) * 1e-6, 1e-12)
        inverse_sqrt = (eigvecs * (1.0 / np.sqrt(np.maximum(eigvals, floor)))) @ eigvecs.T
        aligned[indices] = np.einsum("ij,njt->nit", inverse_sqrt, trials)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--euclidean-alignment", action="store_true",
        help="Use unlabeled per-subject covariance alignment; produces a transductive evaluation.",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_everything(SEED)
    X, y, subject_ids = load_multiple_subjects(
        ROOT / "data", subject_ids=SUBJECTS, l_freq=8, h_freq=30,
        tmin=0.5, tmax=3.5,
    )
    # After 8-30 Hz filtering, decimate by two to 125 Hz for CPU training.
    X = X[:, :, ::2].copy()
    output_path = OUT
    if args.euclidean_alignment:
        # EA is fit separately to each subject's unlabeled EEG. The held-out
        # participant's full recording contributes only signal covariance,
        # never labels; report this as unsupervised target-time adaptation.
        X = euclidean_align_by_subject(X, subject_ids)
        output_path = ROOT / "experiments" / "phase3" / "shallowconvnet_ea_loso.csv"
        print("Using unlabeled Euclidean alignment per subject.", flush=True)
    rows = []

    for held_out in SUBJECTS:
        is_test = subject_ids == held_out
        source_idx = np.flatnonzero(~is_test)
        test_idx = np.flatnonzero(is_test)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        fit_rel, val_rel = next(split.split(source_idx, y[source_idx]))
        fit_idx, val_idx = source_idx[fit_rel], source_idx[val_rel]

        # Fit normalization on the source training data only.
        mean = X[fit_idx].mean(axis=(0, 2), keepdims=True)
        std = X[fit_idx].std(axis=(0, 2), keepdims=True).clip(min=1e-8)
        Xn = ((X - mean) / std).astype(np.float32)

        def tensor(indices):
            return torch.from_numpy(Xn[indices, None]), torch.from_numpy(y[indices]).long()

        x_fit, y_fit = tensor(fit_idx)
        x_val, y_val = tensor(val_idx)
        x_test, y_test = tensor(test_idx)
        loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=128, shuffle=True)
        model = ShallowConvNet(X.shape[1], X.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-4, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        best_val, best_state, stale = float("inf"), None, 0

        for epoch in range(60):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 8:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(x_test).argmax(dim=1).cpu().numpy()
        truth = y_test.cpu().numpy()
        row = {
            "test_subject": held_out,
            "n_test_trials": int(len(test_idx)),
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(truth, pred),
            "model": "ShallowConvNet; Adam; validation early stopping",
            "epochs_run": epoch + 1,
            "normalization": "per-channel z-score; fit subjects only",
            "protocol": (
                "LOSO+unsupervised target EA; held-out labels unused; T session; 22 channels; "
                "8-30 Hz; cue+0.5 to +3.5 s; 125 Hz"
                if args.euclidean_alignment else
                "LOSO; T session; 22 channels; 8-30 Hz; cue+0.5 to +3.5 s; 125 Hz"
            ),
        }
        rows.append(row)
        print(f"{held_out}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}, epochs={epoch+1}", flush=True)
        del model, optimizer, x_fit, y_fit, x_val, y_val, x_test, y_test

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
