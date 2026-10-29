# -*- coding: utf-8 -*-
"""
成员4 · 任务：LDA 收缩（shrinkage）单旋钮扫描（A01）。

大白话：基线的判卷老师（LDA）没加任何保险丝。我们只给 LDA 加不同强度的
保险丝（shrinkage），其余（数据/滤波/分段/CSP/划分）一概不动，用同一套
5 折交叉验证公平对比，验证“加了保险丝更稳”的假设。

数据泄漏红线：只用训练集 T；测试集 E 完全不碰。
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

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_PATH = "data/BCICIV_2a_gdf/A01T.gdf"
TMIN, TMAX = 0.5, 3.5

# 与基线完全一致的 CSP 固定挡
CSP_N = 4
CSP_LOG = True
CSP_REG = None

# 同一套 5 折划分（shuffle=False，保证各档用同一张考卷，公平可比）
CV = StratifiedKFold(n_splits=5, shuffle=False)


# 各档位的 LDA 设置（只在这里变）
LDA_SETTINGS = [
    ("0-① svd(基线)",     dict(solver="svd")),
    ("0-② lsqr+None",     dict(solver="lsqr", shrinkage=None)),
    ("1  lsqr+auto",      dict(solver="lsqr", shrinkage="auto")),
    ("2  lsqr+0.1",       dict(solver="lsqr", shrinkage=0.1)),
    ("3  lsqr+0.5",       dict(solver="lsqr", shrinkage=0.5)),
    ("4  lsqr+0.9",       dict(solver="lsqr", shrinkage=0.9)),
]


def load_trials(path):
    """读 A01T -> 8-30Hz IIR 滤波 -> 切 4 类 trial -> (X, y)。
    事件映射从文件现读（不写死数字）：769=左/770=右/771=脚/772=舌。"""
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.filter(8, 30, method="iir")
    events, event_id = mne.events_from_annotations(raw)
    order = ["769", "770", "771", "772"]
    motor = {s: event_id[s] for s in order if s in event_id}
    epochs = mne.Epochs(raw, events, event_id=motor,
                        tmin=TMIN, tmax=TMAX,
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


def main():
    X, y = load_trials(DATA_PATH)
    print(f"A01 数据加载完成: X={X.shape}, 类别数={len(np.unique(y))}")
    print("=" * 66)

    rows = []
    for name, kw in LDA_SETTINGS:
        scores = cross_val_score(make_pipe(kw), X, y, cv=CV, n_jobs=1)
        mean, std = float(scores.mean()), float(scores.std())
        rows.append((name, mean, std))
        print(f"  {name:18} -> {mean:.2%} ± {std:.2%}")

    print("=" * 66)

    # ---- 汇总 CSV ----
    with open("lda_shrink_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["档位", "LDA 设置", "cv_mean", "cv_std"])
        for i, (name, mean, std) in enumerate(rows):
            w.writerow([str(i), name, f"{mean:.4f}", f"{std:.4f}"])
    print("汇总表已保存: lda_shrink_summary.csv")

    # ---- 对比图 ----
    labels = [n.split(" ")[0] for n, _, _ in rows]
    means = [m for _, m, _ in rows]
    stds = [s for _, _, s in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"])
    ax.axhline(0.25, color="gray", ls=":", label="随机基线 25%")
    ax.axhline(0.7639, color="blue", ls="--", lw=1, label="官方基线 76.39%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("LDA 设置（只动这一个变量）")
    ax.set_ylabel("5 折交叉验证准确率")
    ax.set_title("A01 LDA 收缩扫描：保险丝强度 vs 准确率")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.1%}\n±{s:.1%}", (xi, m + 0.03),
                    textcoords="offset points", xytext=(0, 0),
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("lda_shrink_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("对比图已保存: lda_shrink_plot.png")

    best = max(rows, key=lambda r: r[1])
    print("=" * 66)
    print(f"最优档位: {best[0]} -> {best[1]:.2%} ± {best[2]:.2%}")


if __name__ == "__main__":
    main()
