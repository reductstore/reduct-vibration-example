"""Example script to demonstrate how to use the ReductStore API for storing and querying vibration data."""

import asyncio
import struct

import numpy as np
from helper_functions import (
    TimeUnits,
    calculate_metrics,
    generate_sensor_data,
    get_current_time,
    pack_data,
)
from reduct import Bucket, Client

HIGH_RMS = 1.0
HIGH_CREST_FACTOR = 3.0
HIGH_PEAK_TO_PEAK = 5.0


async def store_data(
    bucket: Bucket,
    timestamp: int,
    packed_data: bytes,
    rms: float = 0.0,
    peak_to_peak: float = 0.0,
    crest_factor: float = 0.0,
):
    """Store the sensor data in the ReductStore."""
    labels = {
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
    }
    await bucket.write("sensor_readings", packed_data, timestamp, labels=labels)


async def query_data(bucket: Bucket, start_time: int, end_time: int) -> np.ndarray:
    """Query and retrieve the stored data."""
    query_signal = np.array([], dtype=np.float32)
    async for record in bucket.query(
        "sensor_readings", start=start_time, stop=end_time
    ):
        data = await record.read_all()
        num_points = len(data) // 4
        fmt = f">{num_points}f"
        signal = struct.unpack(fmt, data)
        signal = np.array(signal, dtype=np.float32)
        query_signal = np.concatenate((query_signal, signal))

    return query_signal


async def main():
    """Example of storing and querying sensor data in ReductStore."""
    frequency = 1000
    duration = 5

    time_step = TimeUnits.MICROSECOND // frequency
    async with Client("http://localhost:8383", api_token="my-token") as client:
        bucket = await client.create_bucket("sensor_data", exist_ok=True)
        start_time = get_current_time()

        signal = generate_sensor_data(frequency=frequency, duration=duration)
        signal_chunks = np.array_split(signal, duration)

        for step, chunk in enumerate(signal_chunks):
            rms, peak_to_peak, crest_factor = calculate_metrics(chunk)
            packed_data = pack_data(chunk)
            timestamp = start_time + step * len(chunk) * time_step
            await store_data(
                bucket, timestamp, packed_data, rms, peak_to_peak, crest_factor
            )
            await asyncio.sleep(1)

        end_time = get_current_time()
        result = await query_data(bucket, start_time, end_time)
        assert np.array_equal(
            signal, result
        ), f"Data do not match with {len(signal)} signals vs {len(result)} results"


if __name__ == "__main__":
    """Run the main coroutine."""
    asyncio.run(main())
