# Window replay integration point

Member A's `sliding_window.py`, `realtime_simulation.py`, and `window_experiment.py` belong here.

Required contract: chronological EEG samples, fixed channel order and sampling rate, windows matching the trained model input, one prediction per update, no future samples, and timestamps plus preprocessing/inference timing. This directory is intentionally a placeholder; replay decoding is not implemented yet.
