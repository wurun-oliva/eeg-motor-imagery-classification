import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ===== Step 4: 加载 + 滤波 + 分段 =====
raw = mne.io.read_raw_gdf('data/A01T.gdf', preload=True)
raw.filter(l_freq=8, h_freq=30, method='iir')

events, event_id = mne.events_from_annotations(raw)

# 只保留 4 类运动想象事件
motor_events = {'769': 7, '770': 8, '771': 9, '772': 10}
epochs = mne.Epochs(raw, events, event_id=motor_events,
                    tmin=0.5, tmax=3.5,
                    baseline=None, preload=True,
                    event_repeated='merge')

print(f"Trial 数量: {len(epochs)}")

X = epochs.get_data()
y = epochs.events[:, -1]

# 标签重映射为 0-3
label_map = {7: 0, 8: 1, 9: 2, 10: 3}
y = np.array([label_map[i] for i in y])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"标签类别: {np.unique(y)}")

# ===== Step 5: CSP + LDA 分类 =====
csp = CSP(n_components=4, reg=None, log=True, cov_est='concat')
lda = LinearDiscriminantAnalysis()
pipeline = Pipeline([('csp', csp), ('lda', lda)])

scores = cross_val_score(pipeline, X, y, cv=5)

print(f"CSP+LDA 准确率: {scores.mean():.2%} (+/- {scores.std():.2%})")

# ===== Step 6: 混淆矩阵 =====
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pipeline.fit(X, y)
y_pred = pipeline.predict(X)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - CSP+LDA')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("混淆矩阵已保存为 confusion_matrix.png")
