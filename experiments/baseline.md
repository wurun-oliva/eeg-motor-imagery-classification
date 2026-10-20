# EEG Motor Imagery BCI - Baseline Experiment

## 1. Dataset
- Data file: A01T.gdf
- Total trials: 288
- EEG channels: 22
- EOG channels: excluded

## 2. Preprocessing
- Band-pass filter: 8–30 Hz
- Epoch time window: 0.5–3.5 s
- Baseline correction: None
- Number of time points per trial: 751

## 3. Feature Extraction
- Method: Common Spatial Patterns (CSP)
- Number of CSP components: 4
- Log transform: True

## 4. Classifier
- Model: Linear Discriminant Analysis (LDA)
- Pipeline: CSP + LDA

## 5. Evaluation
- Method: 5-fold cross-validation
- Mean accuracy: 75.69%
- Standard deviation: 5.57%

## 6. Confusion Matrix
- Class 0: 49/72 correct (68.06%)
- Class 1: 65/72 correct (90.28%)
- Class 2: 47/72 correct (65.28%)
- Class 3: 57/72 correct (79.17%)

## 7. Initial Observation
- Class 1 has the highest recognition rate.
- Class 2 has the lowest recognition rate.
- Classes 2 and 3 are frequently confused.

## 8. Notes
- This is the initial baseline experiment.
- Future experiments should change one variable at a time.
- Compare new results against this baseline.

## 9. Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.89 | 0.68 | 0.77 | 72 |
| 1 | 0.75 | 0.90 | 0.82 | 72 |
| 2 | 0.68 | 0.65 | 0.67 | 72 |
| 3 | 0.74 | 0.79 | 0.77 | 72 |

- Overall accuracy: 75.69%
- Macro F1-score: 0.76
- Weighted F1-score: 0.76

## 10. Initial Analysis

- Class 1 has the highest recall and F1-score.
- Class 2 has the lowest recall and F1-score.
- Class 2 is frequently misclassified as class 3.
- The causes of misclassification have not yet been determined.
