# -*- coding: utf-8 -*-
"""实时端滤波/预测接口规范 —— 给成员A（确定性实现 + 系数导出）

与训练端同一组系数：IIR Butterworth, order=8, 8-30 Hz bandpass, fs=250, SOS 形式。
训练端 = 零相位（双向）；实时端 = 同一 SOS 做单向因果滤波（sosfilt），保持 zi 连续流转。
实证：因果 vs 零相位 LOSO 平均精度仅差约 -0.8pp（特征为 log 功率，对相位不敏感）。

用法：
    import realtime_filter_spec as rf
    zi = rf.init_state(22)            # 启动时初始化
    y, zi = rf.filter_blocks(x, zi)   # 每次到块 x(C,N) 后回传 zi
    n_ext = rf.window_start(step_k)   # 当前第 k 个 0.25s 步的窗口起点
"""
import json
import os

import numpy as np
from scipy.signal import butter, sosfilt

FS = 250.0
L_FREQ, H_FREQ = 8.0, 30.0
ORDER = 8                 # 单边设计阶数；训练端双向零相位 → effective 16
WINDOW_SEC = 3.0          # cue 后 0.5-3.5s
TMIN, TMAX = 0.5, 3.5     # 时窗边界（相对 cue），单位 s
STEP_SEC = 0.25
SAMPLES_PER_EPOCH = int(round(FS * WINDOW_SEC)) + 1   # 751

# 与训练端完全相同的系数（scipy；mne 显式 iir_params=order8 时与该系数一致，Δ~1e-6）
SOS = butter(ORDER, [L_FREQ, H_FREQ], "bandpass", fs=FS, output="sos")  # (8,6)


def init_state(n_channels: int) -> np.ndarray:
    """Zero state expected by scipy.sosfilt: (n_sections, n_channels, 2)."""
    return np.zeros((SOS.shape[0], n_channels, 2), dtype=np.float64)


def filter_blocks(samples: np.ndarray, zi: np.ndarray):
    """单向因果滤波一块 (C, N) 数据，返回 (滤波结果, 更新后的 zi)。"""
    return sosfilt(SOS, samples, axis=-1, zi=zi)


def step_samples(k: int) -> int:
    """第 k 步与第 k-1 步之间的窗口起点差（63/62 交替，平均 62.5=0.25s）。k 从 1 开始。"""
    return window_start(k) - window_start(k - 1)


def window_start(k: int) -> int:
    """第 k 个预测步的窗口首样本，相对 cue 样本的偏移（k 从 0 起）：
        k=0 → round(0.5*250) = 125（即 cue 后 0.5s）；
        预测窗口取 ring[csample + start(k) : csample + start(k) + 751]（751 个点，含两端）。"""
    return int(round((TMIN + k * STEP_SEC) * FS))


def main():
    # 自检：步长序列与 1 秒对齐
    print("阶数:", ORDER, "| 频带:", L_FREQ, "-", H_FREQ, "Hz | FS:", FS)
    print("SOS shape:", SOS.shape, "  每 epoch 点数:", SAMPLES_PER_EPOCH)
    steps = [step_samples(k) for k in range(1, 9)]
    print("前 8 步推进点数:", steps, "  平均:", np.mean(steps), "(应≈62.5)")
    print("前 4 步合计:", sum(steps[:4]), "点 =", sum(steps[:4]) / FS, "s")
    print("窗口起点(相对cue) k=0..3:", [window_start(k) for k in range(4)],
          "  → 每步取 ring[start : start+751]")

    out_dir = os.path.join("outputs", "phase2_memberC")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "realtime_sos_butter8_8-30_250.npy"), SOS)
    with open(os.path.join(out_dir, "realtime_sos_butter8_8-30_250.json"), "w") as f:
        json.dump({"fs": FS, "order": ORDER, "low": L_FREQ, "hi": H_FREQ,
                   "sections": sos2list(SOS)}, f, indent=1)
    print("系数已导出:", out_dir)

    chans = ["EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
             "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
             "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz",
             "EEG-15", "EEG-16"]
    with open(os.path.join(out_dir, "channels_order.json"), "w", encoding="utf-8") as f:
        json.dump({"n_channels": len(chans), "order": chans}, f, ensure_ascii=False, indent=1)
    print("channels_order.json 已写，共", len(chans), "个通道")


def sos2list(sos):
    return [[float(x) for x in row] for row in sos]


if __name__ == "__main__":
    main()
