"""Utility functions for generating, processing, and storing sensor data."""

import struct
import time
from datetime import datetime, timezone

import numpy as np


class TimeUnits:
    NANOSECOND = 1_000_000_000
    MICROSECOND = 1_000_000
    MILLISECOND = 1_000
    SECOND = 1


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


def time_to_rfc3339(timestamp: int, time_unit: int = TimeUnits.MICROSECOND) -> str:
    """Convert a timestamp to an RFC3339-formatted string."""
    return datetime.fromtimestamp(timestamp / time_unit, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


def get_current_time(time_unit: int = TimeUnits.MICROSECOND) -> int:
    """Get the current time in the specified time unit."""
    return int(time.time() * time_unit)
