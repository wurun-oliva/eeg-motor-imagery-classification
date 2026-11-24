"""Subject-disjoint BCI IV-2a experiments inspired by CTNet's study design.

This is an independent implementation, not a reproduction of CTNet's released
code. It supports T-only and T+E experiments. Official E labels are used only
for training *source* subjects and scoring the held-out subject. Optional EA
uses unlabeled held-out data and is therefore transductive.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_official_sessions import load_sessions
from shallowconvnet_loso import ShallowConvNet, euclidean_align_by_subject
from eegnet_loso import EEGNet

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


class DeepConvNet(nn.Module):
    """Four-block DeepConvNet (temporal+spatial first block, 3 later blocks)."""
    def __init__(self, n_channels: int, n_times: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), bias=False),
            nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10), bias=False),
            nn.BatchNorm2d(200), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_channels, n_times)).numel()
        self.classifier = nn.Linear(flat, 4)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_normalize(x: np.ndarray, fit_idx: np.ndarray):
    mean = x[fit_idx].mean(axis=(0, 2), keepdims=True)
    std = x[fit_idx].std(axis=(0, 2), keepdims=True).clip(min=1e-8)
    return ((x - mean) / std).astype(np.float32)


def trial_normalize(x: np.ndarray):
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True).clip(min=1e-8)
    return ((x - mean) / std).astype(np.float32)


def make_model(kind: str, n_channels: int, n_times: int):
    if kind == "deep":
        return DeepConvNet(n_channels, n_times)
    if kind == "eegnet":
        return EEGNet(n_channels, n_times)
    return ShallowConvNet(n_channels, n_times)


def augment_sr(x: torch.Tensor, y: torch.Tensor, pieces: int = 4):
    """Class-preserving segment recombination, confined to a training batch."""
    n, _, _, t = x.shape
    cuts = np.linspace(0, t, pieces + 1, dtype=int)
    synthetic = torch.empty_like(x)
    for label in range(4):
        positions = torch.where(y == label)[0]
        if len(positions) == 0:
            continue
        for k in range(pieces):
            donors = positions[torch.randint(len(positions), (len(positions),))]
            synthetic[positions, :, :, cuts[k]:cuts[k+1]] = x[donors, :, :, cuts[k]:cuts[k+1]]
    return torch.cat((x, synthetic)), torch.cat((y, y))


def train_epochs(model, x, y, n_epochs, *, augment, device, val=None, patience=12):
    model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None]), torch.from_numpy(y)),
                        batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_epoch, best_loss, stale = 1, float("inf"), 0
    if val is not None:
        vx, vy = val
        vx = torch.from_numpy(vx[:, None]).to(device)
        vy = torch.from_numpy(vy).to(device)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if augment:
                xb, yb = augment_sr(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                losses = []
                for start in range(0, len(vx), 256):
                    losses.append(loss_fn(model(vx[start:start+256]), vy[start:start+256]).item())
                val_loss = float(np.mean(losses))
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return best_epoch if val is not None else n_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["shallow", "deep", "eegnet"], default="shallow")
    parser.add_argument("--sessions", choices=["T", "TE"], default="TE")
    parser.add_argument("--ea", action="store_true", help="Unlabeled per-subject EA; transductive")
    parser.add_argument("--augment", action="store_true", help="S&R training augmentation")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-preprocess", action="store_true",
                        help="Use acquired 0.5-100 Hz data at 250 Hz and per-trial Z-score")
    parser.add_argument("--inner-split", choices=["subject", "random"], default="subject")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Training device: {device}", flush=True)
    x, y, ids, sessions = load_sessions(
        ROOT / "data", ROOT / "data/true_labels.zip", sessions=args.sessions,
        l_freq=None if args.paper_preprocess else 4.0,
        h_freq=None if args.paper_preprocess else 38.0,
        decim=1 if args.paper_preprocess else 2,
    )
    if args.ea:
        x = euclidean_align_by_subject(x, ids)
    if args.paper_preprocess:
        x = trial_normalize(x)
    if args.out is None:
        suffix = "_paper" if args.paper_preprocess else ""
        suffix += "_randomval" if args.inner_split == "random" else ""
        suffix += "_ea" if args.ea else ""
        suffix += "_sr" if args.augment else ""
        args.out = ROOT / "experiments/phase3" / f"{args.model}_{args.sessions}_4s{suffix}_loso.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, held_out in enumerate(SUBJECTS):
        source = np.flatnonzero(ids != held_out)
        test = np.flatnonzero(ids == held_out)
        source_subjects = [s for s in SUBJECTS if s != held_out]
        if args.inner_split == "subject":
            val_subject = source_subjects[fold % 8]
            inner_train = np.flatnonzero((ids != held_out) & (ids != val_subject))
            inner_val = np.flatnonzero(ids == val_subject)
        else:
            val_subject = "source-trial random 30%"
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                              random_state=args.seed + fold)
            train_rel, val_rel = next(splitter.split(source, y[source]))
            inner_train, inner_val = source[train_rel], source[val_rel]
        set_seed(args.seed + fold * 100)
        inner_x = x if args.paper_preprocess else source_normalize(x, inner_train)
        inner_model = make_model(args.model, x.shape[1], x.shape[2])
        best_epoch = train_epochs(inner_model, inner_x[inner_train], y[inner_train],
                                  args.max_epochs, augment=args.augment, device=device,
                                  val=(inner_x[inner_val], y[inner_val]))
        del inner_model, inner_x
        set_seed(args.seed + fold * 100 + 1)
        outer_x = x if args.paper_preprocess else source_normalize(x, source)
        model = make_model(args.model, x.shape[1], x.shape[2])
        train_epochs(model, outer_x[source], y[source], best_epoch,
                     augment=args.augment, device=device)
        model.eval()
        pred = []
        with torch.no_grad():
            for start in range(0, len(test), 256):
                pred.append(model(torch.from_numpy(outer_x[test[start:start+256], None]).to(device)).argmax(1).cpu().numpy())
        pred = np.concatenate(pred)
        for ses in ("all", *args.sessions):
            mask = np.ones(len(test), bool) if ses == "all" else sessions[test] == ses
            rows.append({
                "held_out": held_out, "test_session": ses, "n_test": int(mask.sum()),
                "validation_subject": val_subject, "selected_epochs": best_epoch,
                "accuracy": accuracy_score(y[test][mask], pred[mask]),
                "macro_f1": f1_score(y[test][mask], pred[mask], average="macro"),
                "model": args.model, "sessions": args.sessions, "ea": args.ea,
                "augment_sr": args.augment, "seed": args.seed,
                "protocol": (
                    f"LOSO; {args.inner_split} inner validation; cue+0 to +4 s; "
                    + ("acquired 0.5-100 Hz; 250 Hz; trial zscore" if args.paper_preprocess
                       else "4-38 Hz; 125 Hz; source-only channel zscore")
                    + ("; unlabeled target EA" if args.ea else "")
                ),
            })
        with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"{held_out}: {rows[-(len(args.sessions)+1)]['accuracy']:.3f} ({best_epoch} epochs)", flush=True)
    print(f"Mean LOSO accuracy: {np.mean([r['accuracy'] for r in rows if r['test_session']=='all']):.4f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
