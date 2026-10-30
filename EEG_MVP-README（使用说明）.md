# EEG 运动想象解码项目：CSP + LDA

## 1. 项目简介

本项目实现一个基于传统机器学习的离线脑电运动想象解码流程。

系统将 EEG 信号经过带通滤波和试次切分后，使用 Common Spatial Patterns（CSP）提取空间特征，并使用 Linear Discriminant Analysis（LDA）进行四分类。

四个分类目标为：

- Left hand：左手运动想象
- Right hand：右手运动想象
- Feet：双脚运动想象
- Tongue：舌头运动想象

项目包含两个程序：

- `main.py`：训练与评估模型，并保存最终模型。
- `predict.py`：加载最终模型，对独立 EEG 文件进行预测。

当前版本处理预先录制的 GDF 文件，属于离线解码原型，不是实时脑电采集系统。

## 2. 项目结构

```text
EEG_MVP/
├── main.py
├── predict.py
├── README.md
├── requirements.txt
├── data/
│   ├── A01T.gdf
│   └── A01E.gdf
└── outputs/
```

`A01T.gdf` 和 `A01E.gdf` 需要由使用者放入 `data/` 文件夹，数据文件不包含在代码中。

## 3. 环境要求

建议使用 Python 3.10 或 Python 3.11。

需要安装以下 Python 库：

- MNE
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- joblib

可以在项目目录中创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用 Windows，虚拟环境激活命令为：

```bash
.venv\Scripts\activate
```

## 4. 程序 A：训练与评估

先把带标签的训练文件 `A01T.gdf` 放到 `data/` 文件夹中。

运行：

```bash
python main.py
```

程序主要执行以下步骤：

1. 读取 GDF 文件。
2. 对 EEG 信号进行 8–30 Hz 带通滤波。
3. 按照提示事件提取 0.5–3.5 秒的试次。
4. 使用 CSP 提取空间特征。
5. 使用 LDA 完成四分类。
6. 通过五折交叉验证评估模型。
7. 输出准确率、Macro-F1、分类报告和混淆矩阵。
8. 使用全部 A01T 标签试次重新拟合最终模型。
9. 将模型保存为 `csp_lda_model.joblib`。

### 程序 A 的主要输出

- `outputs/summary.json`
- `outputs/classification_report.txt`
- `outputs/classification_report.csv`
- `outputs/confusion_matrix.csv`
- `outputs/confusion_matrix.png`
- `outputs/five_fold_accuracy.png`
- `outputs/cross_validation_predictions.csv`
- `csp_lda_model.joblib`

其中，逐试次交叉验证预测记录的是每个试次在未用该试次训练的折模型下得到的预测结果。

## 5. 程序 B：预测独立数据

运行程序 B 前，必须先成功运行程序 A，生成：

```text
csp_lda_model.joblib
```

将独立的 EEG 文件放入 `data/` 文件夹。默认情况下，程序 B 读取：

```text
data/A01E.gdf
```

运行：

```bash
python predict.py
```

也可以指定输入文件：

```bash
python predict.py --input data/A01E.gdf
```

指定输出位置：

```bash
python predict.py --input data/A01E.gdf --output outputs/my_predictions.csv
```

预测结果将保存为 CSV，每一行代表一个提取出的试次，包含：

- `trial`：试次编号
- `predicted_class_id`：预测类别编号
- `predicted_motor_imagery`：预测的运动想象类别

## 6. 方法参数

| 项目 | 设置 |
|---|---|
| 滤波频段 | 8–30 Hz |
| 试次时间窗 | 0.5–3.5 秒 |
| CSP 组件数 | 4 |
| 分类器 | LDA |
| 评估方法 | 五折交叉验证 |
| 类别数量 | 4 |

训练和预测必须使用兼容的预处理方法、采样率、EEG 通道及通道顺序。

## 7. 如何理解结果

五折交叉验证用于估计分类流程在当前数据上的表现。

程序 A 保存的最终模型则是在全部 A01T 标签试次上重新训练的模型，供程序 B 对独立数据进行预测。

不要把最终模型对训练数据的预测当作独立测试准确率。

如果 A01E 文件没有可用的真实类别标签，程序 B 只能输出预测类别，无法仅凭这些预测计算真实测试准确率。

## 8. 当前版本的局限

1. 当前程序处理的是预录制 GDF 数据，不包含实时 EEG 设备接入。
2. 当前模型针对四种运动想象类别进行分类。
3. 跨被试泛化能力尚未在本版本中验证。
4. 程序 B 默认针对标准 BCI Competition IV 2a 评估文件的事件标记配置。如果实际文件的事件标记不同，需要先核对数据集说明。
5. 模型预测反映算法的分类结果，不等于已经验证的真实运动意图。

## 9. 后续改进方向

- 使用独立且具有真实标签的数据评估模型。
- 比较 CSP + LDA 与其他传统机器学习或深度学习方法。
- 分析不同类别之间的混淆。
- 评估不同训练会话、不同被试之间的泛化能力。
- 在满足实时采集与延迟要求后，再扩展到在线解码。
