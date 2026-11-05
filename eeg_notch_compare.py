
# ============================================================
# Muse EEG 50 Hz Notch Filter Analysis
#
# Channels:
#   EEG 2
#   EEG 3
#
# Analysis:
#   1. Load original EDF
#   2. Check EEG channel information
#   3. Apply 50 Hz notch filter
#   4. Compare waveforms before and after filtering
#   5. Compare Welch PSD before and after filtering
#   6. Calculate Alpha power (8-13 Hz)
#   7. Calculate Beta power (13-30 Hz)
#   8. Save figures and analysis report
#
# Original EDF will NOT be modified.
# No pandas required.
# ============================================================


# ============================================================
# 1. Import libraries
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from scipy.signal import (
    iirnotch,
    filtfilt,
    welch,
    detrend
)


# ============================================================
# 2. File paths
# ============================================================

# Get the directory containing this Python script
PROJECT_DIR = Path(__file__).resolve().parent

# Original EDF file is in the same folder as this script
EDF_PATH = PROJECT_DIR / "session_001_raw.edf"

# Output folder
OUTPUT_DIR = PROJECT_DIR / "output"

# Create output folder automatically
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Check whether EDF exists
if not EDF_PATH.exists():

    raise FileNotFoundError(
        f"\nEDF file not found:\n{EDF_PATH}\n"
        "Please put session_001_raw.edf "
        "in the same folder as this Python script."
    )


# ============================================================
# 3. Load EDF file
# ============================================================

print("\n")
print("=" * 60)
print("STEP 1: LOAD EDF")
print("=" * 60)

raw = mne.io.read_raw_edf(
    EDF_PATH,
    preload=True,
    verbose="ERROR"
)

# Sampling frequency
sfreq = float(raw.info["sfreq"])

# Recording duration
duration = float(raw.times[-1])

print("EDF file:", EDF_PATH.name)

print("Sampling rate:", sfreq, "Hz")

print("Recording duration:",
      round(duration, 3), "seconds")

print("All channels:")
print(raw.ch_names)


# ============================================================
# 4. Select EEG 2 and EEG 3
# ============================================================

print("\n")
print("=" * 60)
print("STEP 2: SELECT EEG CHANNELS")
print("=" * 60)

CHANNEL_NAMES = [
    "eeg-ch2",
    "eeg-ch3"
]

# Check channel names
for channel in CHANNEL_NAMES:

    if channel not in raw.ch_names:

        raise ValueError(
            f"Channel {channel} not found in EDF.\n"
            f"Available channels: {raw.ch_names}"
        )

# Read selected channels
# MNE internally stores EEG data in Volts
# Convert to microvolts for analysis
data = raw.get_data(
    picks=CHANNEL_NAMES
) * 1e6

times = raw.times

print("Selected channels:", CHANNEL_NAMES)

print("Number of samples:", data.shape[1])

print("Sampling frequency:", sfreq, "Hz")


# ============================================================
# 5. Original signal quality statistics
# ============================================================

print("\n")
print("=" * 60)
print("STEP 3: ORIGINAL SIGNAL STATISTICS")
print("=" * 60)

for i, channel in enumerate(CHANNEL_NAMES):

    signal = data[i]

    print("\nChannel:", channel)

    print("Minimum:",
          round(float(np.min(signal)), 4))

    print("Maximum:",
          round(float(np.max(signal)), 4))

    print("Mean:",
          round(float(np.mean(signal)), 4))

    print("Standard deviation:",
          round(float(np.std(signal)), 4))


# ============================================================
# 6. Design 50 Hz notch filter
# ============================================================

print("\n")
print("=" * 60)
print("STEP 4: DESIGN NOTCH FILTER")
print("=" * 60)

NOTCH_FREQ = 50.0

QUALITY_FACTOR = 30.0

# Check Nyquist frequency
if NOTCH_FREQ >= sfreq / 2:

    raise ValueError(
        "50 Hz is above or equal to the Nyquist frequency."
    )

# Design IIR notch filter
b, a = iirnotch(
    w0=NOTCH_FREQ,
    Q=QUALITY_FACTOR,
    fs=sfreq
)

print("Notch frequency:", NOTCH_FREQ, "Hz")

print("Quality factor:", QUALITY_FACTOR)

print("Filter type: IIR Notch")


# ============================================================
# 7. Apply zero-phase notch filtering
# ============================================================

print("\n")
print("=" * 60)
print("STEP 5: APPLY NOTCH FILTER")
print("=" * 60)

# Apply filtering along the time axis
# filtfilt performs forward-backward filtering
# This avoids phase delay.
#
# The original data array remains unchanged.

filtered_data = filtfilt(
    b,
    a,
    data,
    axis=-1
)

print("Filtering completed.")

print("Original data shape:", data.shape)

print("Filtered data shape:", filtered_data.shape)


# ============================================================
# 8. Welch PSD function
# ============================================================

def calculate_psd(signal, sampling_rate):

    # Remove linear trend
    signal = detrend(
        signal,
        type="linear"
    )

    # Welch parameters
    nperseg = min(
        4096,
        len(signal)
    )

    noverlap = nperseg // 2

    frequencies, power_spectrum = welch(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density"
    )

    return frequencies, power_spectrum


# ============================================================
# 9. Calculate frequency band power
# ============================================================

def calculate_band_power(
    frequencies,
    psd,
    low_frequency,
    high_frequency
):

    # Select frequency band
    mask = (
        (frequencies >= low_frequency) &
        (frequencies <= high_frequency)
    )

    selected_freqs = frequencies[mask]

    selected_psd = psd[mask]

    if len(selected_freqs) < 2:

        raise ValueError(
            "Not enough frequency bins in the selected band."
        )

    # Integrate PSD over frequency
    # Result unit: microvolt squared (uV^2)
    power = np.trapezoid(
        selected_psd,
        selected_freqs
    )

    return float(power)


# ============================================================
# 10. Define frequency bands
# ============================================================

FREQUENCY_BANDS = {

    "Alpha": (8.0, 13.0),

    "Beta": (13.0, 30.0),

    "50Hz": (49.0, 51.0)

}


# ============================================================
# 11. Calculate PSD and band power
# ============================================================

print("\n")
print("=" * 60)
print("STEP 6: PSD AND BAND POWER ANALYSIS")
print("=" * 60)

results = {}

psd_data = {}

for i, channel in enumerate(CHANNEL_NAMES):

    print("\n")
    print("-" * 50)
    print("Channel:", channel)
    print("-" * 50)

    # Original signal PSD
    frequencies, psd_before = calculate_psd(
        data[i],
        sfreq
    )

    # Filtered signal PSD
    _, psd_after = calculate_psd(
        filtered_data[i],
        sfreq
    )

    psd_data[channel] = {
        "frequencies": frequencies,
        "before": psd_before,
        "after": psd_after
    }

    results[channel] = {}

    # Calculate each frequency band
    for band_name, band_range in FREQUENCY_BANDS.items():

        low, high = band_range

        power_before = calculate_band_power(
            frequencies,
            psd_before,
            low,
            high
        )

        power_after = calculate_band_power(
            frequencies,
            psd_after,
            low,
            high
        )

        # Percentage change
        if power_before > 0:

            change_percent = (
                (power_after - power_before)
                / power_before
            ) * 100

        else:

            change_percent = np.nan

        # Difference in dB
        if power_before > 0 and power_after > 0:

            change_db = 10 * np.log10(
                power_after / power_before
            )

        else:

            change_db = np.nan

        results[channel][band_name] = {

            "before": power_before,

            "after": power_after,

            "change_percent": change_percent,

            "change_db": change_db

        }

        print("\nBand:", band_name)

        print(
            "Frequency range:",
            low,
            "-",
            high,
            "Hz"
        )

        print(
            "Before filtering:",
            round(power_before, 6),
            "uV^2"
        )

        print(
            "After filtering:",
            round(power_after, 6),
            "uV^2"
        )

        print(
            "Power change:",
            round(change_percent, 4),
            "%"
        )

        print(
            "Power change:",
            round(change_db, 4),
            "dB"
        )


# ============================================================
# 12. Plot waveform comparison
# ============================================================

print("\n")
print("=" * 60)
print("STEP 7: PLOT WAVEFORMS")
print("=" * 60)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True
)

# Show first 10 seconds
time_mask = times <= 10

for i, channel in enumerate(CHANNEL_NAMES):

    axes[i].plot(
        times[time_mask],
        data[i, time_mask],
        linewidth=0.7,
        label="Before 50 Hz notch"
    )

    axes[i].plot(
        times[time_mask],
        filtered_data[i, time_mask],
        linewidth=0.7,
        label="After 50 Hz notch"
    )

    axes[i].set_title(
        channel + " - Waveform Comparison"
    )

    axes[i].set_ylabel(
        "EDF scale (uV label)"
    )

    axes[i].grid(
        alpha=0.3
    )

    axes[i].legend()

axes[-1].set_xlabel(
    "Time (seconds)"
)

fig.suptitle(
    "EEG 2 and EEG 3 - Waveforms Before and After Notch"
)

plt.tight_layout()

waveform_path = (
    OUTPUT_DIR /
    "eeg_waveform_notch_comparison.png"
)

plt.savefig(
    waveform_path,
    dpi=200,
    bbox_inches="tight"
)

print("Waveform figure saved:")
print(waveform_path)

plt.close(fig)


# ============================================================
# 13. Plot Welch PSD comparison
# ============================================================

print("\n")
print("=" * 60)
print("STEP 8: PLOT PSD")
print("=" * 60)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9)
)

for i, channel in enumerate(CHANNEL_NAMES):

    frequencies = psd_data[channel]["frequencies"]

    psd_before = psd_data[channel]["before"]

    psd_after = psd_data[channel]["after"]

    # Convert PSD to dB
    psd_before_db = 10 * np.log10(
        psd_before + 1e-30
    )

    psd_after_db = 10 * np.log10(
        psd_after + 1e-30
    )

    axes[i].plot(
        frequencies,
        psd_before_db,
        label="Before notch"
    )

    axes[i].plot(
        frequencies,
        psd_after_db,
        label="After 50 Hz notch"
    )

    # Mark frequency bands
    axes[i].axvspan(
        8,
        13,
        alpha=0.10,
        label="Alpha (8-13 Hz)"
    )

    axes[i].axvspan(
        13,
        30,
        alpha=0.08,
        label="Beta (13-30 Hz)"
    )

    # Mark 50 Hz
    axes[i].axvline(
        50,
        linestyle="--",
        color="black",
        linewidth=1,
        label="50 Hz"
    )

    axes[i].set_xlim(
        1,
        80
    )

    axes[i].set_title(
        channel + " - Welch PSD Comparison"
    )

    axes[i].set_xlabel(
        "Frequency (Hz)"
    )

    axes[i].set_ylabel(
        "PSD (dB re uV^2/Hz)"
    )

    axes[i].grid(
        alpha=0.3
    )

    axes[i].legend(
        loc="best"
    )

fig.suptitle(
    "EEG 2 and EEG 3 - PSD Before and After 50 Hz Notch"
)

plt.tight_layout()

psd_path = (
    OUTPUT_DIR /
    "eeg_psd_notch_comparison.png"
)

plt.savefig(
    psd_path,
    dpi=200,
    bbox_inches="tight"
)

print("PSD figure saved:")
print(psd_path)

plt.close(fig)


# ============================================================
# 14. Plot Alpha and Beta power comparison
# ============================================================

print("\n")
print("=" * 60)
print("STEP 9: PLOT ALPHA AND BETA POWER")
print("=" * 60)

band_names = [
    "Alpha",
    "Beta"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

for i, channel in enumerate(CHANNEL_NAMES):

    for j, band_name in enumerate(band_names):

        power_before = results[channel][band_name]["before"]

        power_after = results[channel][band_name]["after"]

        axes[i, j].bar(
            ["Before", "After"],
            [power_before, power_after]
        )

        axes[i, j].set_title(
            channel + " - " + band_name + " Power"
        )

        axes[i, j].set_ylabel(
            "Power (uV^2)"
        )

        axes[i, j].grid(
            axis="y",
            alpha=0.3
        )

fig.suptitle(
    "Alpha and Beta Power Before and After Notch"
)

plt.tight_layout()

band_path = (
    OUTPUT_DIR /
    "eeg_alpha_beta_power_comparison.png"
)

plt.savefig(
    band_path,
    dpi=200,
    bbox_inches="tight"
)

print("Band power figure saved:")
print(band_path)

plt.close(fig)


# ============================================================
# 15. Save numerical report
# ============================================================

print("\n")
print("=" * 60)
print("STEP 10: SAVE ANALYSIS REPORT")
print("=" * 60)

report_path = (
    OUTPUT_DIR /
    "eeg_notch_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "Muse EEG 50 Hz Notch Analysis Report\n"
    )

    report.write(
        "=" * 55 + "\n"
    )

    report.write(
        f"EDF file: {EDF_PATH.name}\n"
    )

    report.write(
        f"Sampling rate: {sfreq} Hz\n"
    )

    report.write(
        f"Duration: {duration:.3f} seconds\n"
    )

    report.write(
        f"Notch frequency: {NOTCH_FREQ} Hz\n"
    )

    report.write(
        f"Quality factor: {QUALITY_FACTOR}\n"
    )

    report.write(
        "Channels: EEG 2, EEG 3\n\n"
    )

    for channel in CHANNEL_NAMES:

        report.write(
            "\nChannel: " + channel + "\n"
        )

        report.write(
            "-" * 45 + "\n"
        )

        for band_name in FREQUENCY_BANDS:

            item = results[channel][band_name]

            report.write(
                f"\n{band_name}:\n"
            )

            report.write(
                f"Before: {item['before']:.8f} uV^2\n"
            )

            report.write(
                f"After: {item['after']:.8f} uV^2\n"
            )

            report.write(
                f"Change: {item['change_percent']:.4f}%\n"
            )

            report.write(
                f"Change in dB: {item['change_db']:.4f} dB\n"
            )

print("Report saved:")
print(report_path)


# ============================================================
# 16. Final summary
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for channel in CHANNEL_NAMES:

    print("\nChannel:", channel)

    for band_name in FREQUENCY_BANDS:

        item = results[channel][band_name]

        print(
            band_name,
            ":",
            round(item["change_percent"], 3),
            "% change"
        )

print("\nAll analysis completed successfully.")

print("\nOriginal EDF was not modified.")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. eeg_waveform_notch_comparison.png")

print("2. eeg_psd_notch_comparison.png")

print("3. eeg_alpha_beta_power_comparison.png")

print("4. eeg_notch_analysis_report.txt")

print("\n")
print("=" * 60)
