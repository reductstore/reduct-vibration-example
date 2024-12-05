# %% Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# %% Setup save directory
directory = "../plots"
if not os.path.exists(directory):
    os.makedirs(directory)

# %% Signal in Chunks
# Generate a sample signal
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(1000)

# Define chunk size
chunk_size = 100
chunks = [signal[i : i + chunk_size] for i in range(0, len(signal), chunk_size)]

# Plot the signal with chunks
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label="Original Signal", color="#2c0549")
for i, chunk in enumerate(chunks):
    plt.axvline(
        x=i * chunk_size / 100,
        color="red",
        linestyle="--",
        label="Chunk Boundary" if i == 0 else "",
    )
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Signal Divided into 1 Second Chunks")
plt.legend()
plt.savefig(os.path.join(directory, "signal_chunks.png"), dpi=300)
plt.show()

# %% Example of an FFT
# Generate a sample signal
t = np.linspace(0, 1, 1000)
signal = (
    np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t) + np.random.randn(len(t))
)

# Perform FFT
fft_values = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(signal), t[1] - t[0])

# Plot the FFT result
plt.figure(figsize=(12, 6))
plt.plot(fft_freq, np.abs(fft_values), color="#2c0549")
plt.xlim(0, 150)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of the Signal")
plt.savefig(os.path.join(directory, "fft_example.png"), dpi=300)
plt.show()

# %% Typical Weekend Problem
# Generate a sample data retention scenario
days = np.arange(1, 11)
data_retained = [1 if (i < 6 or i > 7) else 0 for i in days]

# Plot the retention policy color in gray the first 2 days
plt.figure(figsize=(12, 6))
plt.bar(days, data_retained, color="#2c0549")

# Highlight the lost data
plt.axvspan(0, 2.5, color="gray", alpha=0.5)
plt.text(1.5, 1.01, "Data Lost", fontsize=12, color="black", ha="center")

plt.xticks(days, [f"Day {i}" for i in days])
plt.xlabel("Days")
plt.ylabel("Data Generated")
plt.title("Typical Weekend Problem")
plt.savefig(os.path.join(directory, "weekend_problem.png"), dpi=300)
plt.show()

# %% Show the aliasing effect

# Generate the base signal
f0 = 5  # Frequency of the base signal in Hz
t_continuous = np.linspace(0, 1, 1000)
signal_continuous = np.sin(2 * np.pi * f0 * t_continuous)

# Low sampling frequency (causes aliasing)
fs_low = 6  # Sampling frequency in Hz (less than 2 * f0)
t_low = np.arange(0, 1, 1 / fs_low)
signal_low = np.sin(2 * np.pi * f0 * t_low)

# High sampling frequency (samples correctly)
fs_high = 50  # Sampling frequency in Hz (greater than 2 * f0)
t_high = np.arange(0, 1, 1 / fs_high)
signal_high = np.sin(2 * np.pi * f0 * t_high)

# Create subplots
plt.figure(figsize=(12, 8))

# Plot with low sampling frequency
plt.subplot(2, 1, 1)
plt.plot(t_continuous, signal_continuous, label="Original Signal", color="#2c0549")
plt.stem(
    t_low,
    signal_low,
    linefmt="r-",
    markerfmt="ro",
    basefmt=" ",
    label=f"Sampled Signal (fs = {fs_low} Hz)",
)
plt.title("Aliasing Effect with Low Sampling Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot with high sampling frequency
plt.subplot(2, 1, 2)
plt.plot(t_continuous, signal_continuous, label="Original Signal", color="#2c0549")
plt.stem(
    t_high,
    signal_high,
    linefmt="g-",
    markerfmt="go",
    basefmt=" ",
    label=f"Sampled Signal (fs = {fs_high} Hz)",
)
plt.title("Proper Sampling with High Sampling Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(directory, "aliasing_effect.png"), dpi=300)
plt.show()

# %%
