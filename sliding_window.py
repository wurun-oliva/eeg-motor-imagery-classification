
"""
Sliding Window Module
=====================

Project: EEG Motor Imagery Classification
Member: A
Version: 2.0

Purpose:
    Convert continuous multi-channel EEG data
    into overlapping fixed-length windows.

Configuration:
    Sampling rate: 250 Hz
    Window duration: 3 seconds
    First window: 0.5 seconds after class cue
    Step length: 0.25 seconds
    Samples per window: 751 (inclusive endpoints)

Input:
    numpy.ndarray
    Shape: (n_channels, n_samples)

Output:
    EEGWindow objects

Important:
    1. No EEG file loading.
    2. No filtering.
    3. No classification.
    4. No subject-specific file paths.
    5. Cue sample is provided by the caller.
"""

from dataclasses import dataclass

import numpy as np

# ============================================================
# 1. EEG Window Data Structure
# ============================================================

@dataclass
class EEGWindow:

    subject_id: str

    index: int

    start_sample: int

    # Exclusive end index for Python slicing
    end_sample: int

    start_time: float

    # Time of the last included sample
    end_time: float

    data: np.ndarray

# ============================================================
# 2. Sliding Window Buffer
# ============================================================

class SlidingWindowBuffer:

    def __init__(
        self,
        sampling_rate=250,
        window_seconds=3.0,
        step_seconds=0.25,
        initial_offset_seconds=0.5,
        n_channels=22,
        subject_id="unknown",
        cue_sample=0
    ):

        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        if window_seconds <= 0:
            raise ValueError("Window duration must be positive.")

        if step_seconds <= 0:
            raise ValueError("Step duration must be positive.")

        if initial_offset_seconds < 0:
            raise ValueError("Initial offset cannot be negative.")

        if n_channels <= 0:
            raise ValueError("Channel count must be positive.")

        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string.")

        if not isinstance(cue_sample, (int, np.integer)):
            raise TypeError("cue_sample must be an integer.")

        if cue_sample < 0:
            raise ValueError("cue_sample cannot be negative.")

        self.sampling_rate = float(sampling_rate)
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.initial_offset_seconds = float(
            initial_offset_seconds
        )

        self.n_channels = int(n_channels)
        self.subject_id = subject_id
        self.cue_sample = int(cue_sample)

        # Include both endpoints:
        # 3 seconds * 250 Hz + 1 = 751 samples
        exact_window_samples = (
            self.sampling_rate * self.window_seconds
        )

        if not np.isclose(
            exact_window_samples,
            round(exact_window_samples)
        ):
            raise ValueError(
                "Window duration must correspond to "
                "an integer number of sampling intervals."
            )

        self.window_samples = (
            int(round(exact_window_samples)) + 1
        )

        # Fractional step: 0.25 * 250 = 62.5 samples
        self.step_samples_exact = (
            self.sampling_rate * self.step_seconds
        )

        if self.step_samples_exact < 1:
            raise ValueError("Step must contain at least one sample.")

        if self.step_seconds > self.window_seconds:
            raise ValueError(
                "Step cannot exceed window duration."
            )

        self.reset()

    # --------------------------------------------------------
    # 3. Reset buffer
    # --------------------------------------------------------

    def reset(self):

        self.buffer = np.empty(
            (self.n_channels, 0),
            dtype=np.float64
        )

        self.buffer_start_sample = 0

        self.total_samples_received = 0

        self.window_index = 0

    # --------------------------------------------------------
    # 4. Calculate next window start
    # --------------------------------------------------------

    def _get_next_start_sample(self):

        # Follow Member C's specification:
        #
        # round((0.5 + k * 0.25) * 250)
        #
        # This generates 125, 188, 250, 312, 375...
        #
        # Cue sample is the absolute sample index
        # of the class cue in the continuous recording.

        relative_time = (
            self.initial_offset_seconds
            + self.window_index * self.step_seconds
        )

        relative_sample = int(
            round(relative_time * self.sampling_rate)
        )

        return self.cue_sample + relative_sample

    # --------------------------------------------------------
    # 5. Add continuous EEG data
    # --------------------------------------------------------

    def add_data(self, eeg_chunk):

        eeg_chunk = np.asarray(eeg_chunk)

        # Check dimensions
        if eeg_chunk.ndim != 2:
            raise ValueError(
                "Input must have shape "
                "(n_channels, n_samples)."
            )

        # Check channel count
        if eeg_chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"received {eeg_chunk.shape[0]}."
            )

        # Ignore empty chunks
        if eeg_chunk.shape[1] == 0:
            return []

        # Check numeric values
        if not np.issubdtype(eeg_chunk.dtype, np.number):
            raise TypeError("EEG data must be numeric.")

        # Check NaN and Inf
        if not np.all(np.isfinite(eeg_chunk)):
            raise ValueError("EEG data contains NaN or Inf.")

        # Copy input data
        eeg_chunk = eeg_chunk.astype(
            np.float64,
            copy=True
        )

        # Append new samples
        self.buffer = np.concatenate(
            [self.buffer, eeg_chunk],
            axis=1
        )

        self.total_samples_received += eeg_chunk.shape[1]

        generated_windows = []

        # Generate all complete windows
        while True:

            start_sample = self._get_next_start_sample()

            # Python end index is exclusive
            end_sample = (
                start_sample + self.window_samples
            )

            # Not enough samples yet
            if end_sample > self.total_samples_received:
                break

            local_start = (
                start_sample - self.buffer_start_sample
            )

            local_end = (
                end_sample - self.buffer_start_sample
            )

            if (
                local_start < 0
                or local_end > self.buffer.shape[1]
            ):
                raise RuntimeError(
                    "Internal buffer indexing error."
                )

            # Extract complete window
            window_data = self.buffer[
                :,
                local_start:local_end
            ].copy()

            # Time of first and last included samples
            start_time = (
                start_sample / self.sampling_rate
            )

            end_time = (
                (end_sample - 1) / self.sampling_rate
            )

            window = EEGWindow(

                subject_id=self.subject_id,

                index=self.window_index,

                start_sample=start_sample,

                end_sample=end_sample,

                start_time=start_time,

                end_time=end_time,

                data=window_data
            )

            generated_windows.append(window)

            self.window_index += 1

        # ----------------------------------------------------
        # Remove samples no longer needed
        # ----------------------------------------------------

        next_start = self._get_next_start_sample()

        discard_samples = (
            next_start - self.buffer_start_sample
        )

        if discard_samples > 0:

            discard_samples = min(
                discard_samples,
                self.buffer.shape[1]
            )

            self.buffer = self.buffer[
                :,
                discard_samples:
            ].copy()

            self.buffer_start_sample += discard_samples

        return generated_windows

    # --------------------------------------------------------
    # 6. Get buffer status
    # --------------------------------------------------------

    def get_status(self):

        return {

            "subject_id": self.subject_id,

            "sampling_rate": self.sampling_rate,

            "n_channels": self.n_channels,

            "window_seconds": self.window_seconds,

            "step_seconds": self.step_seconds,

            "initial_offset_seconds": (
                self.initial_offset_seconds
            ),

            "cue_sample": self.cue_sample,

            "window_samples": self.window_samples,

            "step_samples_exact": self.step_samples_exact,

            "total_samples_received": (
                self.total_samples_received
            ),

            "windows_generated": self.window_index,

            "buffered_samples": self.buffer.shape[1]

        }

# ============================================================
# 7. Unit Test
# ============================================================

def test_sliding_window():

    print("=" * 65)
    print("SLIDING WINDOW MODULE TEST - VERSION 2.0")
    print("=" * 65)

    # Test configuration
    fs = 250
    channels = 22
    duration = 10

    total_samples = fs * duration

    subject_id = "A01"

    # For this test, cue occurs at sample 0.
    # In real-time use, supply the actual cue sample.

    cue_sample = 0

    # Generate simulated EEG
    rng = np.random.default_rng(42)

    eeg_data = rng.normal(
        0,
        1,
        size=(channels, total_samples)
    )

    print("\nOriginal EEG shape:", eeg_data.shape)

    # Create sliding window buffer
    buffer = SlidingWindowBuffer(

        sampling_rate=fs,

        window_seconds=3.0,

        step_seconds=0.25,

        initial_offset_seconds=0.5,

        n_channels=channels,

        subject_id=subject_id,

        cue_sample=cue_sample
    )

    print("\nSampling rate:", fs, "Hz")

    print("Window samples:", buffer.window_samples)

    print("Exact step samples:", buffer.step_samples_exact)

    print("Initial offset:", 0.5, "seconds")

    print("Cue sample:", cue_sample)

    # Simulate continuous data input
    chunk_size = 110

    all_windows = []

    for start in range(
        0,
        total_samples,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_samples
        )

        eeg_chunk = eeg_data[:, start:end]

        new_windows = buffer.add_data(eeg_chunk)

        all_windows.extend(new_windows)

    # Display results
    print("\nGenerated windows:", len(all_windows))

    if not all_windows:
        raise RuntimeError("No windows were generated.")

    first = all_windows[0]
    last = all_windows[-1]

    print("\nFirst window:")

    print("Subject:", first.subject_id)

    print("Index:", first.index)

    print("Start sample:", first.start_sample)

    print("End sample (exclusive):", first.end_sample)

    print("Start time:", first.start_time)

    print("End time:", first.end_time)

    print("Data shape:", first.data.shape)

    print("\nLast window:")

    print("Subject:", last.subject_id)

    print("Index:", last.index)

    print("Start sample:", last.start_sample)

    print("End sample (exclusive):", last.end_sample)

    print("Start time:", last.start_time)

    print("End time:", last.end_time)

    print("Data shape:", last.data.shape)

    # --------------------------------------------------------
    # Automated assertions
    # --------------------------------------------------------

    assert buffer.window_samples == 751

    assert buffer.step_samples_exact == 62.5

    assert len(all_windows) == 26

    assert first.data.shape == (22, 751)

    assert last.data.shape == (22, 751)

    assert first.subject_id == "A01"

    assert last.subject_id == "A01"

    assert first.index == 0

    assert first.start_sample == 125

    assert first.end_sample == 876

    assert last.start_sample == 1688

    assert last.end_sample == 2439

    assert np.isclose(first.start_time, 0.5)

    assert np.isclose(first.end_time, 3.5)

    assert np.isclose(last.start_time, 6.752)

    assert np.isclose(last.end_time, 9.752)

    # Check C's required window start sequence
    expected_starts = [
        125,
        188,
        250,
        312,
        375,
        438
    ]

    actual_starts = [
        all_windows[i].start_sample
        for i in range(6)
    ]

    assert actual_starts == expected_starts

    # Check consecutive windows have 62/63 sample steps
    actual_steps = [
        all_windows[i + 1].start_sample
        - all_windows[i].start_sample
        for i in range(5)
    ]

    assert actual_steps == [63, 62, 62, 63, 63]

    # Check exact data extraction
    np.testing.assert_array_equal(
        first.data,
        eeg_data[:, 125:876]
    )

    # Check buffer status
    status = buffer.get_status()

    assert status["window_samples"] == 751

    assert status["windows_generated"] == 26

    print("\nAll automated tests passed!")

    print("=" * 65)

# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    test_sliding_window()