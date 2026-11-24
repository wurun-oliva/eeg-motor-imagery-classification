"""Small EEGNet model, evaluated with LOSO and train-only normalization."""
from __future__ import annotations

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
OUT = ROOT / "experiments" / "phase3" / "eegnet_loso.csv"
SEED = 42


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 4):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 32), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            n_features = self.temporal(torch.zeros(1, 1, n_channels, n_times)).numel()
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.classifier(self.temporal(x).flatten(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    torch.set_num_threads(2)
    set_seed(SEED)
    X, y, ids = load_multiple_subjects(
        ROOT / "data", subject_ids=SUBJECTS, l_freq=8, h_freq=30,
        tmin=0.5, tmax=3.5,
    )
    # The signal is already band-limited to 8-30 Hz; keep every second
    # sample (125 Hz) to make the CPU experiment practical.
    X = X[:, :, ::2].copy()
    rows = []
    for fold, held_out in enumerate(SUBJECTS):
        test_mask = ids == held_out
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        fit_rel, val_rel = next(split.split(train_idx, y[train_idx]))
        fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]

        # Normalize each channel using training-only statistics (no test leakage).
        mean = X[fit_idx].mean(axis=(0, 2), keepdims=True)
        std = X[fit_idx].std(axis=(0, 2), keepdims=True).clip(min=1e-8)
        Xn = ((X - mean) / std).astype(np.float32)
        def tensor(indices):
            return torch.from_numpy(Xn[indices, None]), torch.from_numpy(y[indices]).long()

        x_fit, y_fit = tensor(fit_idx)
        x_val, y_val = tensor(val_idx)
        x_test, y_test = tensor(test_idx)
        loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=256, shuffle=True)
        model = EEGNet(X.shape[1], X.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = nn.CrossEntropyLoss()
        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(15):
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
                if stale >= 4:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(x_test).argmax(dim=1).cpu().numpy()
        truth = y_test.cpu().numpy()
        row = {
            "test_subject": held_out,
            "n_test_trials": len(test_idx),
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, average="macro"),
            "cohen_kappa": cohen_kappa_score(truth, pred),
            "model": "EEGNet-8,2; Adam; early stopping",
            "epochs_run": epoch + 1,
            "normalization": "per-channel z-score; fit subjects only",
            "protocol": "LOSO; T session; 22 EEG channels; 8-30 Hz; cue+0.5 to +3.5 s",
        }
        rows.append(row)
        print(f"{held_out}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}, epochs={epoch+1}", flush=True)
        del model, optimizer, x_fit, y_fit, x_val, y_val, x_test, y_test
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
