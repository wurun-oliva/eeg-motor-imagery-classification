# -*- coding: utf-8 -*-
"""
成员4 · 任务：把 LDA 收缩（shrinkage）实验从 A01 扩展到全部 9 名被试。

目的：验证 "LDA 加适量保险丝更稳略升" 的结论在多人上是否普适，
而不是只在 A01 一个被试上成立（单被试可能是运气）。

方法：每名被试，在完全相同的预处理/CSP/同一套 5 折划分下，只换 LDA 档位。
红线：只用各被试 T 文件；E 完全不碰。
"""
import csv
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUBJECTS = [f"A0{i}" for i in range(1, 10)]   # A01..A09
DATA_DIR = "data/BCICIV_2a_gdf"
TRAIN_SUFFIX = "T.gdf"

CSP_N, CSP_LOG, CSP_REG = 4, True, None
CV = StratifiedKFold(n_splits=5, shuffle=False)

SETTINGS = [
    ("svd基线",  dict(solver="svd")),
    ("无保险丝",  dict(solver="lsqr", shrinkage=None)),
    ("auto",    dict(solver="lsqr", shrinkage="auto")),
    ("0.1",     dict(solver="lsqr", shrinkage=0.1)),
    ("0.5",     dict(solver="lsqr", shrinkage=0.5)),
    ("0.9",     dict(solver="lsqr", shrinkage=0.9)),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load_trials(path):
    """读单个 T 文件 -> 8-30Hz IIR 滤波 -> 切 4 类 -> (X, y)。映射从文件现读。"""
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.filter(8, 30, method="iir")
    events, event_id = mne.events_from_annotations(raw)
    order = ["769", "770", "771", "772"]
    motor = {s: event_id[s] for s in order if s in event_id}
    epochs = mne.Epochs(raw, events, event_id=motor,
                        tmin=0.5, tmax=3.5,
                        baseline=None, preload=True,
                        event_repeated="merge")
    X = epochs.get_data()
    class_of = {motor[s]: i for i, s in enumerate(order)}
    y = np.array([class_of[e] for e in epochs.events[:, -1]])
    return X, y


def make_pipe(lda_kwargs):
    return Pipeline([
        ("csp", CSP(n_components=CSP_N, reg=CSP_REG, log=CSP_LOG, cov_est="concat")),
        ("lda", LinearDiscriminantAnalysis(**lda_kwargs)),
    ])


def cv_one(subj):
    """一名被试：返回 (subj, [(档名, mean, std), ...])。"""
    path = f"{DATA_DIR}/{subj}{TRAIN_SUFFIX}"
    X, y = load_trials(path)
    out = []
    for name, kw in SETTINGS:
        scores = cross_val_score(make_pipe(kw), X, y, cv=CV, n_jobs=-1)
        out.append((name, float(scores.mean()), float(scores.std())))
    line = " | ".join(f"{n}={m:.1%}±{s:.1%}" for n, m, s in out)
    print(f"{subj}: {line}", flush=True)
    return subj, out


def main():
    print("=" * 78)
    print("成员4 · 9 名被试 LDA 收缩扫描（只动 LDA，其余锁死）")
    print("=" * 78)

    results = Parallel(n_jobs=-1)(delayed(cv_one)(s) for s in SUBJECTS)
    results = dict(results)  # subj -> [(name, mean, std), ...]

    names = [n for n, _ in SETTINGS]
    n_subj = len(SUBJECTS)
    mat = np.zeros((n_subj, len(names)))       # 每被试每档 mean
    mat_std = np.zeros((n_subj, len(names)))   # 每被试每档 std
    for i, subj in enumerate(SUBJECTS):
        for j, (name, mean, std) in enumerate(results[subj]):
            mat[i, j], mat_std[i, j] = mean, std

    print("-" * 78)
    for j, name in enumerate(names):
        col = mat[:, j]
        print(f"{name:6}: 9 人平均 = {col.mean():.2%} ± {col.std():.2%}"
              f"   (跨人がstd {col.std():.2%})")

    # ---- CSV：行=被试，列=档位，AVERAGE=跨被试平均与离散 ----------------
    with open("lda_shrink_all_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{n}" for n in names])
        for i, subj in enumerate(SUBJECTS):
            w.writerow([subj] + [f"{mat[i, j]:.4f}" for j in range(len(names))])
        avg = [mat[:, j].mean() for j in range(len(names))]
        spreads = [mat[:, j].std() for j in range(len(names))]
        w.writerow(["AVERAGE"] + [f"{a:.4f}" for a in avg])
        w.writerow(["ACROSS-SUBJ-STD"] + [f"{s:.4f}" for s in spreads])
    print("\n汇总表已保存: lda_shrink_all_summary.csv")

    # ---- 图：按档位的 9 人平均 + 跨被试离散(误差线) -----------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, avg, yerr=spreads, capsize=5, color=COLORS)
    ax.axhline(0.25, color="gray", ls=":", label="随机基线 25%")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("LDA 档位（只动这个变量）")
    ax.set_ylabel("9 名被试平均交叉验证准确率")
    ax.set_title("成员4 · 9 被试 LDA 收缩扫描：平均准确率（误差线=跨被试离散度）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    for xi, (a, s) in enumerate(zip(avg, spreads)):
        ax.annotate(f"{a:.1%}\n±{s:.1%}", (xi, a + 0.03), textcoords="offset points",
                    xytext=(0, 0), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("lda_shrink_all_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("对比图已保存: lda_shrink_all_plot.png")

    best_j = int(np.argmax(avg))
    print("-" * 78)
    print(f"结论：9 人平均最优档 = {names[best_j]} = {avg[best_j]:.2%}"
          f"（跨被试 ± {spreads[best_j]:.2%}）")


if __name__ == "__main__":
    main()
