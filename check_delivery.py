
from pathlib import Path
import numpy as np
import csv

# 项目目录
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_DIR = PROJECT_DIR / "model_from_C" / "给A_模型与接口"

print("=" * 60)
print("MEMBER A DELIVERY CHECK")
print("=" * 60)

# 1. 检查模型
model_file = MODEL_DIR / "bci2a_csp_lda_n6_all9.joblib"

print("\n[1] Model file")
print("Model exists:", model_file.exists())

# 2. 检查数据文件
files = [
    "A01T_continuous_eeg.npy",
    "A01T_cue_samples.npy",
    "A01T_cue_labels.npy",
    "A01T_epochs.npy",
    "A01T_labels.npy",
]

print("\n[2] EEG data files")

for name in files:
    path = OUTPUT_DIR / name
    print(name, ":", path.exists())

# 3. 检查数据形状
print("\n[3] Data validation")

continuous = np.load(
    OUTPUT_DIR / "A01T_continuous_eeg.npy"
)

cue_samples = np.load(
    OUTPUT_DIR / "A01T_cue_samples.npy"
)

epochs = np.load(
    OUTPUT_DIR / "A01T_epochs.npy"
)

labels = np.load(
    OUTPUT_DIR / "A01T_labels.npy"
)

print("Continuous EEG:", continuous.shape)
print("Cue samples:", cue_samples.shape)
print("Epochs:", epochs.shape)
print("Labels:", labels.shape)

assert continuous.shape[0] == 22
assert len(cue_samples) == 288
assert epochs.shape == (288, 22, 751)
assert labels.shape == (288,)
assert np.isfinite(continuous).all()
assert np.isfinite(epochs).all()

print("Data validation PASSED")

# 4. 检查模型预测结果
print("\n[4] Prediction CSV")

csv_files = [
    "A01T_model_predictions.csv",
    "A01T_realtime_integration.csv",
]

for name in csv_files:
    path = OUTPUT_DIR / name

    if not path.exists():
        print(name, ": MISSING")
        continue

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(name)
    print("Prediction records:", len(rows))
    print("Columns:", reader.fieldnames)

    assert len(rows) == 288

print("\n" + "=" * 60)
print("MEMBER A DELIVERY CHECK COMPLETED")
print("=" * 60)
