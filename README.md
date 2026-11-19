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
# Causal sliding-window replay (吴润)

The original automatic sliding-window replay is preserved in `sliding_window.py`, `realtime_simulation.py`, and `realtime_model_integration.py`. A package-based causal replay entry point is also available at `src/realtime/stream_replay.py`.

It feeds a saved GDF recording in chronological chunks, carries the SOS filter state across chunks, and predicts only after each 751-sample window has arrived. Defaults are 3 s windows and 0.25 s steps after each cue. This is replay simulation from recorded data, not live headset input.

From the repository root, after placing `A01T.gdf` under `data/` and the deployment model at the root:

```bash
python -m src.realtime.stream_replay --gdf data/A01T.gdf --model bci2a_csp_lda_n6_all9.joblib
```

Add `--realtime` to pace playback at the recording sample rate. Results are written to `outputs/realtime/sliding_predictions.csv`. The deployment model includes A01 in its training data, so A01 replay label agreement is for demonstration and is not held-out performance.

## Interactive manual window (吴润)

To move a fixed 3 s window by hand, launch:

```bash
python -m src.realtime.interactive_window --gdf data/A01T.gdf --model bci2a_csp_lda_n6_all9.joblib
```

Drag the slider to choose the window start time and press **预测**. The viewer plots all 22 channels for that window and shows the predicted class and class probabilities. Each manual prediction is appended to `outputs/realtime/manual_predictions.csv`. The model requires exactly 751 samples; the slider moves the window position but does not resize it. This interactive viewer is for offline inspection and is separate from the automatic replay command above.
