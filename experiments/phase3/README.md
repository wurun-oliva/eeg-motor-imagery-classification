# EEG 项目第三阶段实验

七项任务的完成情况和关键结果见[阶段工作总结（吴润）](./阶段工作总结_吴润.md)。

## 运行方式

### 新增：官方 T+E 标签与文献对照

将比赛[官方公布的标签](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)下载到 `data/true_labels.zip`，并确认 `data/` 内有 A01T/A01E 至 A09T/A09E 的 GDF。运行：

```bash
conda activate eeg-dl
python experiments/phase3/literature_inspired_loso.py --model shallow --sessions TE --ea --max-epochs 60
python experiments/phase3/literature_inspired_loso.py --model shallow --sessions TE --ea --augment --inner-split random --max-epochs 30
# 有 CUDA/MPS 设备时可运行论文启发的完整采样率深层模型：
python experiments/phase3/literature_inspired_loso.py --model deep --sessions TE --paper-preprocess --inner-split random --augment --max-epochs 600 --device auto
```

每次试验按整名受试者留出，输出 CSV 包含九人逐人准确率和 T/E 分开准确率。`--ea` 使用目标受试者全部无标签试次，属于测试时适配。`--inner-split random` 仅在来源受试者的试次中随机划出验证集，不使用留出受试者标签。论文对照和数据检查见[跨被试文献与复核记录](./跨被试文献与复核记录.md)。深层网络的完整九折 250 Hz 运行需较长计算时间；未完成的运行不计为成绩。

在仓库根目录打开终端，先进入已安装依赖的环境，再运行实验：

```bash
conda activate eeg-dl
python experiments/phase3/topomap_erd.py
python experiments/phase3/filter_range_loso.py
python experiments/phase3/fbcsp_loso.py
python experiments/phase3/eegnet_loso.py
python experiments/phase3/shallowconvnet_loso.py
python experiments/phase3/shallowconvnet_loso.py --euclidean-alignment
python experiments/phase3/shallowconvnet_ea_nested_loso.py
python experiments/phase3/shallowconvnet_ea_nested_loso.py --no-euclidean-alignment
python experiments/phase3/normalization_loso.py
```

## 已生成结果

| 文件 | 内容 |
|---|---|
| `erd_topomaps_A01-A09.png` | 四种运动想象类别的 Mu/Alpha、Beta 频段头皮功率变化地形图 |
| `filter_range_loso.csv` | 5 种滤波范围、9 名受试者逐人准确率、Macro-F1、Kappa |
| `fbcsp_loso.csv` | 8 个子频带的 FBCSP + 收缩 LDA 留一受试者结果 |
| `eegnet_loso.csv` | EEGNet 留一受试者逐人结果 |
| `shallowconvnet_loso.csv` | ShallowConvNet 留一受试者逐人结果 |
| `shallowconvnet_ea_loso.csv` | 加入无标签欧氏对齐的 ShallowConvNet 留一受试者结果 |
| `shallowconvnet_ea_nested_loso.csv` | 受试者隔离内层验证、三种子复核结果 |
| `shallowconvnet_nested_loso.csv` | 相同分组验证的无 EA 配对对照 |
| `normalization_loso.csv` | CSP+LDA 是否进行训练集通道 Z-score 的逐人对照 |
| `滤波范围_LOSO比较.xlsx` | 滤波范围、算法表现、归一化对照及逐人结果工作簿 |

当前汇总准确率：8–30 Hz CSP+LDA 为 42.0%；FBCSP 为 35.8%；EEGNet 为 33.3%；ShallowConvNet 为 40.0%。补做受试者隔离内层验证、三随机种子配对后，未对齐 ShallowConvNet 为 **37.6% ± 12.5%**，ShallowConvNet + EA 为 **48.8% ± 16.7%**（标准差按九名受试者三种子均值计算）；EA 提高 11.2 个百分点，且每名受试者三种子平均值均高于 CSP+LDA 的 LOSO 基线。早期试次级验证的 EA 初测 50.1% 不作为主要结论。EA 使用留出受试者无标签试次估计对齐矩阵，属于无监督测试时适配，不能与完全不接触目标数据的零样本 LOSO 混作一类结果。训练受试者通道 Z-score 与未额外标准化的 CSP+LDA 结果相同（平均准确率均为 42.0%）。

## 代码与流程自查

- 重新按 A01 同被试 5 折运行 8–30 Hz、cue 后 0.5–3.5 秒、CSP(4)+LDA，得到 **76.04% ± 6.42%**，与早期报告的 76.39% 接近，支持当前数据读取、事件标签映射和分段流程基本正确。
- A01–A09 共 2592 个试次，每名受试者 288 个，四类各 72 个；类别映射为 769=左手、770=右手、771=双脚、772=舌头。每折确实整名留出一位受试者；训练/验证没有抽取该受试者标签。
- `A01T_model_predictions.csv` 中的 61.1% 不是留出测试分数：原脚本将已训练模型直接用于 A01 的全体数据。此值不应用于报告泛化准确率。
- 已确认的实验局限：EEGNet 和 ShallowConvNet 只跑单随机种子；ShallowConvNet 是轻量自写实现而非论文复现；FBCSP 拼接了所有子带特征，尚未做折内特征筛选；EA 使用了目标受试者整段无标签试次。故这些数值可用于当前排查和比较，但还不是最终性能结论。
- 为修正早停验证中的同受试者重叠，新增 `shallowconvnet_ea_nested_loso.py`：外层整名留出；每折从八名来源受试者中再留一人作验证；用验证选出 epoch 数后，在全部八名来源受试者上重训；三个种子逐折保存到 `shallowconvnet_ea_nested_loso.csv`。此协议更贴合跨被试早停选择，结果较之前试次级验证初测低 1.3 个百分点。
- 使用同一脚本加 `--no-euclidean-alignment` 生成配对无对齐控制 `shallowconvnet_nested_loso.csv`；两版只差 EA 预处理。此配对结果显示提升主要出现在域对齐，不能归因于“只换成深度模型”。

## 实验口径和限制

- 数据为 BCI Competition IV 2a 的 A01–A09 标注 T session，22 个 EEG 通道；每人 288 个试次。
- 每轮留出 1 名受试者测试，其余 8 名训练。EEGNet 还从这 8 人中留出验证集，仅用于早停。
- 分类窗口为类别提示后 0.5–3.5 秒。该窗口是运动想象期，不包括独立标注的静息态；提示前片段属于注视/期待阶段，不应称为静息态。
- 所有模型实验使用离线零相位滤波。结果不能证明真实设备在线采集能力或实时延迟。
- 头地形图以提示前注视/期待片段作相对功率参考，不是静息态图。地形图是群体均值，不是单个受试者的诊断结论。
- `src/realtime/interactive_window.py` 是对已录制 GDF 的手动窗口查看器；滑动选择窗口后点击 Predict。仅当窗口处于有标签的运动想象试次中才显示正确/错误，否则真实标签不可用。这是离线交互演示，不是连续实时解码。
