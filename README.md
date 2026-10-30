# EEG Motor Imagery Classification

A 4-class brain-computer interface (BCI) classification pipeline using EEG signals from the BCI Competition IV-2a dataset. This project implements the classic CSP + LDA approach for motor imagery decoding.

## Dataset

- **Source**: BCI Competition IV-2a (http://www.bbci.de/competition/iv/)
- **Subjects**: 9 subjects (A01–A09)
- **Channels**: 22 EEG channels + 3 EOG channels
- **Sampling rate**: 250 Hz
- **Classes**: 4 motor imagery tasks
  - Class 0: Left hand (annotation 769)
  - Class 1: Right hand (annotation 770)
  - Class 2: Feet (annotation 771)
  - Class 3: Tongue (annotation 772)
- **Trials per subject**: 288 (Training session)

## Method

### Pipeline

1. **Bandpass filtering**: 8–30 Hz Butterworth filter (alpha + beta bands, most relevant for motor imagery)
2. **Trial extraction**: 0.5–3.5 s post-cue, 3 s windows
3. **Feature extraction**: Common Spatial Pattern (CSP), 4 components
4. **Classification**: Linear Discriminant Analysis (LDA)
5. **Evaluation**: 5-fold cross-validation

### Why CSP + LDA?

CSP maximizes the variance of one class while minimizing it for others, making it one of the most effective spatial filtering methods for motor imagery BCI. Combined with LDA, it forms a lightweight yet powerful baseline.

## Results

| Subject | Accuracy (5-fold CV) |
|---------|---------------------|
| A01     | 75.69% ± 5.57%      |
| More subjects coming... | |

- **Random baseline**: 25% (4-class)
- **Training accuracy**: 80.2%
- **Confusion matrix**: See `confusion_matrix.png`

## Installation

```bash
# Create conda environment
conda create -n bci python=3.10 -y
conda activate bci

# Install dependencies
pip install numpy scipy pandas matplotlib
pip install mne
pip install scikit-learn
```

## Usage

1. Download BCI Competition IV-2a dataset and place `.gdf` files in the `data/` folder.

2. Run the classification pipeline:

```bash
python main.py
```

3. Output:
- Cross-validation accuracy printed to console
- Confusion matrix saved as `confusion_matrix.png`

## Project Structure

```
bci_project/
├── data/
│   ├── A01T.gdf
│   ├── A01E.gdf
│   └── ...
├── main.py
├── confusion_matrix.png
└── README.md
```

## Tech Stack

- **MNE-Python**: EEG data loading, filtering, epoching
- **scikit-learn**: CSP, LDA, cross-validation, confusion matrix
- **NumPy / Matplotlib**: Data processing and visualization

## References

- BCI Competition IV: http://www.bbci.de/competition/iv/
- MNE-Python: https://mne.tools/
- CSP method: Ramoser et al., 2000, "Optimal spatial filtering of single trial EEG during imagined hand movement"
- EEGNet (for comparison): Lawhern et al., 2018, "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs"

## Future Work

- [ ] Run on all 9 subjects
- [ ] Compare with PSD + SVM baseline
- [ ] Implement EEGNet (PyTorch) for deep learning comparison
- [ ] Subject-independent model training
