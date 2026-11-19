
"""
Real-time EEG Simulation
========================

Project: EEG Motor Imagery Classification
Member: A

Purpose:
    Simulate continuous EEG data streaming.
    Feed incoming EEG chunks into SlidingWindowBuffer.
    Generate overlapping EEG windows.
    Record window metadata and processing time.

Configuration:
    Sampling rate: 250 Hz
    EEG channels: 22
    Window length: 3 seconds
    Step length: 0.25 seconds

Input:
    numpy.ndarray
    Shape: (n_channels, n_samples)

Output:
    Continuous EEG windows
    CSV recording of window metadata

Note:
    This version tests the data pipeline using simulated EEG.
    It does not perform CSP, LDA or classification.
"""

import csv
import time

from pathlib import Path

import numpy as np

from sliding_window import SlidingWindowBuffer

# ============================================================
# 1. Real-time EEG Simulation
# ============================================================

def simulate_realtime(
    eeg_data,
    subject_id="A01",
    sampling_rate=250,
    n_channels=22,
    chunk_size=25,
    window_seconds=3.0,
    step_seconds=0.25,
    simulate_realtime=False,
    output_csv=None
):
    """
    Simulate continuous EEG data streaming.

    Parameters
    ----------
    eeg_data : numpy.ndarray
        EEG data with shape (n_channels, n_samples).

    subject_id : str
        Subject identifier, such as A01.

    sampling_rate : int
        EEG sampling rate in Hz.

    n_channels : int
        Number of EEG channels.

    chunk_size : int
        Number of samples received in each chunk.

    simulate_realtime : bool
        If True, wait for each chunk's acquisition duration.
        If False, process the data as fast as possible.

    output_csv : str or Path or None
        CSV output path. If None, no CSV is saved.

    Returns
    -------
    all_windows : list
        All generated EEGWindow objects.

    summary : dict
        Simulation statistics.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    eeg_data = np.asarray(eeg_data)

    if eeg_data.ndim != 2:
        raise ValueError(
            "EEG data must have shape "
            "(n_channels, n_samples)."
        )

    if eeg_data.shape[0] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels, "
            f"received {eeg_data.shape[0]}."
        )

    if eeg_data.shape[1] == 0:
        raise ValueError("EEG data is empty.")

    if not np.issubdtype(eeg_data.dtype, np.number):
        raise TypeError("EEG data must be numeric.")

    if not np.all(np.isfinite(eeg_data)):
        raise ValueError("EEG data contains NaN or Inf.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    total_samples = eeg_data.shape[1]

    total_duration = total_samples / sampling_rate

    # --------------------------------------------------------
    # Create sliding window buffer
    # --------------------------------------------------------

    buffer = SlidingWindowBuffer(
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        n_channels=n_channels,
        subject_id=subject_id
    )

    all_windows = []

    csv_records = []

    chunk_count = 0

    total_processing_time = 0.0

    max_processing_time = 0.0

    # --------------------------------------------------------
    # Start simulation
    # --------------------------------------------------------

    print("=" * 65)
    print("REAL-TIME EEG SIMULATION")
    print("=" * 65)

    print("\nSubject:", subject_id)
    print("Sampling rate:", sampling_rate, "Hz")
    print("Channels:", n_channels)
    print("Total samples:", total_samples)
    print("Recording duration:", round(total_duration, 3), "seconds")
    print("Chunk size:", chunk_size, "samples")
    print(
        "Simulation mode:",
        "Real-time" if simulate_realtime else "Fast"
    )

    print("\nStarting data stream...\n")

    simulation_start = time.perf_counter()

    # --------------------------------------------------------
    # Receive EEG data chunk by chunk
    # --------------------------------------------------------

    for start in range(0, total_samples, chunk_size):

        end = min(
            start + chunk_size,
            total_samples
        )

        eeg_chunk = eeg_data[:, start:end]

        # Simulate acquisition interval
        if simulate_realtime:
            time.sleep(
                eeg_chunk.shape[1] / sampling_rate
            )

        chunk_count += 1

        # Record when this chunk is processed
        chunk_receive_time = time.perf_counter()

        # Feed EEG chunk into sliding window buffer
        new_windows = buffer.add_data(eeg_chunk)

        processing_end = time.perf_counter()

        processing_time_ms = (
            processing_end - chunk_receive_time
        ) * 1000.0

        total_processing_time += processing_time_ms

        max_processing_time = max(
            max_processing_time,
            processing_time_ms
        )

        # ----------------------------------------------------
        # Record generated windows
        # ----------------------------------------------------

        for window in new_windows:

            all_windows.append(window)

            record = {

                "subject_id": window.subject_id,

                "window_index": window.index,

                "start_sample": window.start_sample,

                "end_sample": window.end_sample,

                "start_time": window.start_time,

                "end_time": window.end_time,

                "channels": window.data.shape[0],

                "window_samples": window.data.shape[1],

                "chunk_processing_ms": processing_time_ms

            }

            csv_records.append(record)

        # Display progress
        if chunk_count % 20 == 0 or end == total_samples:

            print(
                f"Received: {end}/{total_samples} samples | "
                f"Generated windows: {len(all_windows)}"
            )

    # --------------------------------------------------------
    # Finish simulation
    # --------------------------------------------------------

    simulation_end = time.perf_counter()

    simulation_elapsed = (
        simulation_end - simulation_start
    )

    status = buffer.get_status()

    # --------------------------------------------------------
    # Save CSV results
    # --------------------------------------------------------

    if output_csv is not None:

        output_path = Path(output_csv)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fieldnames = [

            "subject_id",
            "window_index",
            "start_sample",
            "end_sample",
            "start_time",
            "end_time",
            "channels",
            "window_samples",
            "chunk_processing_ms"

        ]

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(csv_records)

        print("\nCSV saved to:", output_path)

    # --------------------------------------------------------
    # Calculate summary
    # --------------------------------------------------------

    average_processing_time = (
        total_processing_time / chunk_count
        if chunk_count > 0
        else 0.0
    )

    summary = {

        "subject_id": subject_id,

        "sampling_rate": sampling_rate,

        "total_samples": total_samples,

        "total_duration": total_duration,

        "chunk_count": chunk_count,

        "window_count": len(all_windows),

        "average_chunk_processing_ms": (
            average_processing_time
        ),

        "max_chunk_processing_ms": (
            max_processing_time
        ),

        "simulation_elapsed_seconds": (
            simulation_elapsed
        ),

        "buffered_samples": (
            status["buffered_samples"]
        )

    }

    return all_windows, summary

# ============================================================
# 2. Generate Simulated EEG Data
# ============================================================

def generate_test_eeg(
    n_channels=22,
    duration_seconds=10,
    sampling_rate=250
):
    """
    Generate random EEG-like data for software testing.

    This is NOT real EEG data.
    """

    rng = np.random.default_rng(42)

    total_samples = (
        duration_seconds * sampling_rate
    )

    eeg_data = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_channels, total_samples)
    )

    return eeg_data

# ============================================================
# 3. Main Test
# ============================================================

def main():

    # Configuration
    sampling_rate = 250

    n_channels = 22

    duration_seconds = 10

    chunk_size = 25

    subject_id = "A01"

    # Generate test data
    eeg_data = generate_test_eeg(
        n_channels=n_channels,
        duration_seconds=duration_seconds,
        sampling_rate=sampling_rate
    )

    print("Original EEG shape:", eeg_data.shape)

    # Output folder:
    # D:\python代码\output

    project_root = Path(__file__).resolve().parent.parent

    output_folder = project_root / "output"

    output_csv = (
        output_folder / "realtime_simulation_test.csv"
    )

    # Run simulation
    all_windows, summary = simulate_realtime(

        eeg_data=eeg_data,

        subject_id=subject_id,

        sampling_rate=sampling_rate,

        n_channels=n_channels,

        chunk_size=chunk_size,

        window_seconds=3.0,

        step_seconds=0.25,

        simulate_realtime=False,

        output_csv=output_csv

    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("SIMULATION SUMMARY")

    print("=" * 65)

    for key, value in summary.items():

        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    # Check results

    assert len(all_windows) == 26

    assert all_windows[0].data.shape == (22, 751)

    assert all_windows[-1].data.shape == (22, 751)

    assert np.isclose(
        all_windows[0].start_time,
        0.5
    )

    assert np.isclose(
        all_windows[0].end_time,
        3.5
    )

    assert np.isclose(
        all_windows[-1].start_time,
        6.752
    )

    assert np.isclose(
        all_windows[-1].end_time,
        9.752
    )

    assert summary["window_count"] == 26

    assert summary["chunk_count"] == 100

    print("\nAll real-time simulation tests passed!")

    print("=" * 65)

# ============================================================
# 4. Program Entry
# ============================================================

if __name__ == "__main__":

    main()