# 实验记录 · 成员 C：跨用户（LOSO）模型训练与三项优化

> 口径：22 个头皮 EEG 通道（规范 §4.2，剔除 EOG）；2026-11-12 起全部实验按此口径。

## 0. 元信息（§8：名称 / 日期 / 负责人 / 代码版本）

| 字段 | 内容 |
|---|---|
| 实验名称 | 跨用户运动想象分类：LOSO 基线 + CSP 成分数扫描 + 标准化 + 滤波频带 |
| 负责人 | 成员 C（算法研究与改进实验） |
| 日期 | 2026-11-12 |
| 数据集 | BCI Competition IV 2a（A01–A09，共 9 人） |
| 数据文件 | `data/BCICIV_2a_gdf/A*T.gdf`（T = 训练 session） |
| 场景 | 训练 session（T）10×2 次，每受试者 288 trial（`mne` 匹配事件数实测 = 288，9 人一致） |
| Git commit ID | **暂无**（`BCI2a_repro/` 尚未 `git init`，待建立仓库后补填）。临时以 SHA-256 指纹做代码可追溯，见附录 A |
| 复现命令 | 见附录 B |

## 1. 数据集与受试者范围；各受试者 trial 数量与类别分布（§8）

- 受试者：A01–A09 全部 9 人；均来自 BCI IV 2a 训练 session（T）。
- 每受试者 T session 均 **288 trial**，类别完全平衡：左手(0)/右手(1)/双脚(2)/舌头(3) 各 **72**。
- 试次数实测量：`X.shape = (288, 22, 751)`（每受试者），`n_test_trials` 全为 288。

| 受试者 | trial 数 | 各类试次（左右脚舌） |
|---|---|---|
| A01–A09（9 人均相同） | 288 / 人 | 72 / 72 / 72 / 72（平衡） |

## 2. 预处理设置（§8：通道 / 采样率 / 滤波 / Epoch）

| 项 | 设置 | 说明 |
|---|---|---|
| 通道 | **22 个头皮 EEG，顺序显式固定**（`temp_loader.DEFAULT_EEG_CHANNELS`） | gdf 原始 25 通道，剔除 EOG 左/中/右 3 个；mne 将 EOG 读作 `eeg` 类型，无法按类型过滤，故按名单显式固定顺序 |
| 采样率 | 250 Hz（未重采样） | |
| 滤波 | IIR 带通 8–30 Hz（16 阶 Butterworth，双通零相位） | 频带实验单独改 4–38 Hz（一次只改一个变量） |
| Epoch | 事件后 0.5–3.5 s（含端点，**751 点**） | `tmin=0.5, tmax=3.5, baseline=None` |
| 标签映射 | 769→0(左手), 770→1(右手), 771→2(双脚), 772→3(舌头) | 事件号从文件现读，事件缺失也能对齐 |

**预处理变更史（§4 提醒）**：第一阶段按 gdf 原始 25 通道（含 EOG）训练基线。按规范 §4.2 明确为 22 通道后，2026-11-12 已把全部实验按 22 通道重跑；旧 25 通道结果归档 `results/_archive_25ch/`。**两份结果口径不同、不直接比绝对值**，但结论方向一致（见 §5.1）。

## 3. 模型结构与参数（§8）

Pipeline（scikit-learn）：`CSP(log=True, n_components, reg=None, cov_est=concat) → [StandardScaler?] → LinearDiscriminantAnalysis`

| 超参 | 值 |
|---|---|
| CSP 成分数 | 各实验见其章节；**新基准 = 6** |
| 正则 / log | `reg=None` / `log=True` |
| 协方差估计 | `cov_est=concat`（跨类别合并估计） |
| 标准化 | 实验 4 的对比项 `standardize True/False`，其余实验不启用 |
| 随机种子 | 42 |
| 分类器 | LDA（无 shrinkage） |

## 4. 训练 / 验证 / 测试划分（§8）

- 协议：**LOSO（留一受试者出）**：每轮完整留出 1 人当测试，其余 8 人训练；共 9 轮，每轮只评估留出者。
- 测试受试者数据**不参与**训练 / 特征选择 / 标准化拟合 / 超参选择（全流程遵守，无泄漏）。
- 标准化（若启用）随 Pipeline 仅在训练集 fit，测试集只 `transform+predict`。
- 每轮训练 8×288 = 2304 trial，测试 288 trial。

## 5. 评价指标（§6 口径）

- Accuracy（总体正确率）、Macro-F1（4 类宏平均）、Cohen's Kappa；汇总为 9 轮均值 ± 标准差。
- 随机基线 = 25%。
- 混淆矩阵：`results/loso_results_22ch_confusion.png`（总体汇总，CSP+LDA 基线）。

## 6. 实验 1：LOSO 基线（22ch，n=4，8–30 Hz）

来源：`results/loso_results_22ch.csv`（`--n-components 4 --seed 42`）

| 受试者 | Acc (%) | Macro-F1 | Kappa |
|---|---|---|---|
| A01 | 54.86 | 0.4700 | 0.3981 |
| A02 | 28.82 | 0.2588 | 0.0509 |
| A03 | 45.83 | 0.4264 | 0.2778 |
| A04 | 32.64 | 0.2129 | 0.1019 |
| A05 | 25.35 | 0.1876 | 0.0046 |
| A06 | 22.22 | 0.1663 | -0.0370 |
| A07 | 37.85 | 0.3441 | 0.1713 |
| A08 | 56.60 | 0.5642 | 0.4213 |
| A09 | 52.08 | 0.5107 | 0.3611 |
| **平均 ± 标准差** | **39.58 ± 12.42** | 0.3490 ± 0.141 | 0.1944 ± 0.166 |

结论：跨用户明显低于单受试者场景（单人多分类基线约 60%），难在脑电个体差异而非算法失灵；A06 低于随机、A02/A05 贴近随机，属困难受试者。对照 25ch 旧基线（39.62%）相差约 0.04pp → 22ch 定为新标准口径。

## 7. 实验 2：CSP 成分数扫描（22ch，8–30 Hz，LOSO）

来源：`experiments/.../csp_components_scan.py`（只改 `n_components`）；`results/csp_components_summary.csv`

| 成分数 | 平均 Acc ± std | 备注 |
|---|---|---|
| 2 | 36.50 ± 10.48 | |
| 4（基线档） | 39.58 ± 12.42 | |
| **6** | **41.98 ± 15.47** | ← **最优（较 n=4 +2.4pp）** |
| 8 | 41.13 ± 15.20 | |

n=6 逐人 Acc(%)：A01 53.47 / A02 24.65 / A03 57.29 / A04 36.46 / A05 26.39 / A06 24.31 / A07 34.03 / A08 **69.10** / A09 52.08。

结论：n=6 平均最优，但增益主要来自本已较强的 A08/A03/；困难受试者（A02/A05/A06）无改善且波动变大 → 定 **n=6 为新基准**，保留个体差异说明。

## 8. 实验 3：仅训练集拟合的标准化（22ch，n=6 锁死）

来源：`experiments/.../standardization_scan_22ch.py`；`results/standardization_summary.csv`

| 档位 | 平均 Acc ± std |
|---|---|
| 无标准化 | 41.98 ± 15.47 |
| 仅训练集标准化 | 41.98 ± 15.47 |

**9 人逐位数字完全一致，差值 +0.00%**。原因（数学性质，非 bug）：StandardScaler 是逐维可逆线性变换，LDA 决策只依赖样本间相对关系，对该类整体缩放免疫。

结论：CSP+LDA 组合下该旋钮无效，**不调**；换对尺度敏感的分类器（SVM / 神经网络 EEGNet）后必须重新评估——已计入答辩 Q&A。

## 9. 实验 4：滤波频带（22ch，n=6 锁死）

来源：`experiments/.../band_scan_22ch.py`；`results/band_summary.csv`；自检：8–30 档与实验 2 的 n=6 逐位一致

| 频带 | 平均 Acc ± std | vs 8–30 |
|---|---|---|
| **8–30 Hz（基线 / 经典）** | **41.98 ± 15.47** | — |
| 4–38 Hz（宽频带） | 40.32 ± 16.34 | **-1.66pp** |

4–38 逐人 Acc(%)：A01 46.18 / A02 25.00 / A03 61.46 / A04 36.11 / A05 24.65 / A06 25.35 / A07 **23.96** / A08 68.75 / A09 51.39。

结论：宽频带无益；多放行的 4–8 / 30–38 Hz 多为噪声，放大困难受试者的不稳定（A07 -10.1pp、A01 -7.3pp）→ **维持 8–30 Hz**。

## 10. 汇总

- 统一对比表：`results/model_comparison.csv`（**72 行 = 成分数 36 + 标准化 18 + 频带 18**，均为 22ch 口径、含逐受试者指标）。
- 本阶段主结论（22ch，CSP+LDA / LOSO）：**成分数 n=6、频带 8–30 Hz、不启用标准化 → 平均约 42%**；三处优化方向与 25ch 时代结论一致。
- 交付物清单（§6-C 对齐）：可调用接口 `make_pipe()` / `predict`（§3）；逐受试者 LOSO 结果（§6）；模型比较表；混淆矩阵图；本实验说明。

## 附录 A：代码版本指纹（git 前的临时可追溯）

| 文件 | SHA-256（前 12 位） |
|---|---|
| `src/data/temp_loader.py` | F4BAE6A5490F |
| `src/models/cross_subject_model.py` | BEF7BE3A7A8D |
| `src/evaluation/loso_evaluation.py` | 5D0F8696E2CC |
| `experiments/cross_subject_experiments/csp_components_scan.py` | 3D6B4E71A5D4 |
| `experiments/cross_subject_experiments/standardization_scan_22ch.py` | 342C33164C77 |
| `experiments/cross_subject_experiments/band_scan_22ch.py` | 84B87A29E4D5 |

## 附录 B：复现命令（cwd=`BCI2a_repro/`，Python 3.13 / mne 1.13.2 / scikit-learn 1.9.1）

```text
# 1) LOSO 基线（22ch, n=4, 8-30Hz）
python src/evaluation/loso_evaluation.py --subjects A01 A02 A03 A04 A05 A06 A07 A08 A09 --out results/loso_results_22ch.csv --n-components 4 --seed 42

# 2) CSP 成分数扫描（2/4/6/8）
python experiments/cross_subject_experiments/csp_components_scan.py

# 3) 标准化复核（n=6）
python experiments/cross_subject_experiments/standardization_scan_22ch.py

# 4) 频带复核（8-30 vs 4-38）
python experiments/cross_subject_experiments/band_scan_22ch.py
```

## 附录 C：已知缺口 / 待办

- [ ] Git commit ID 缺失 → 待建立仓库后补填；临时见附录 A 指纹。
- [ ] `temp_loader.py` 为临时接口，待成员 B 的 `multi_subject_data.py` 替换（接口签名保持 `(X, y, subject_id)`）。
- [ ] 可选后续：Riemannian Alignment / EEGNet（当前环境 `pyriemann`/`torch`/`tensorflow` 均未安装）。
