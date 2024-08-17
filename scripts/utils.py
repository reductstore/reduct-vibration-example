"""Utility functions for generating, processing, and storing sensor data."""

import struct

import numpy as np


def generate_sensor_data(frequency: int = 1000, duration: int = 1) -> np.ndarray:
    """Generate a sine wave signal with added noise."""
    t = np.linspace(0, duration, frequency * duration)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    return signal.astype(np.float32)


def calculate_metrics(signal: np.ndarray) -> tuple:
    """Calculate the RMS, peak-to-peak, and crest factor of a signal."""
    rms = np.sqrt(np.mean(signal**2))
    peak_to_peak = np.max(signal) - np.min(signal)
    crest_factor = np.max(np.abs(signal)) / rms
    return rms, peak_to_peak, crest_factor


def pack_data(signal: np.ndarray) -> bytes:
    """Pack a signal into a byte string for storage."""
    fmt = f">{len(signal)}f"  # '>' for big-endian, 'f' for 32-bit float
    return struct.pack(fmt, *signal)
