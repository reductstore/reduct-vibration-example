"""Example script to demonstrate how to use the ReductStore API for storing and querying vibration data."""

import asyncio
import struct
import time

import numpy as np
from reduct import Bucket, Client
from utils import calculate_metrics, generate_sensor_data, pack_data


HIGH_RMS = 1.0
HIGH_CREST_FACTOR = 3.0
HIGH_PEAK_TO_PEAK = 5.0


async def setup_reductstore() -> Bucket:
    """Set up the ReductStore client and create a bucket."""
    client = Client("http://localhost:8383", api_token="my-token")
    return await client.create_bucket("sensor_data", exist_ok=True)


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
        "rms": "high" if rms > HIGH_RMS else "low",
        "peak_to_peak": "high" if peak_to_peak > HIGH_PEAK_TO_PEAK else "low",
        "crest_factor": "high" if crest_factor > HIGH_CREST_FACTOR else "low",
    }
    await bucket.write("sensor_readings", packed_data, timestamp, labels=labels)


async def query_data(bucket: Bucket, start_time: int, end_time: int):
    """Query and retrieve the stored data."""
    async for record in bucket.query(
        "sensor_readings", start=start_time, stop=end_time
    ):
        print(f"Timestamp: {record.timestamp}")
        print(f"Labels: {record.labels}")

        data = await record.read_all()
        num_points = len(data) // 4
        fmt = f">{num_points}f"
        signal = struct.unpack(fmt, data)
        signal = np.array(signal, dtype=np.float32)

        print(f"Number of data points: {num_points}")
        print(f"First few values: {signal[:5]}")
        print("---")


async def main():
    """Example of storing and querying sensor data in ReductStore."""
    bucket = await setup_reductstore()
    start_time = int(time.time() * 1_000_000)

    for _ in range(5):
        timestamp = int(time.time() * 1_000_000)
        signal = generate_sensor_data(frequency=1000, duration=1)
        rms, peak_to_peak, crest_factor = calculate_metrics(signal)
        packed_data = pack_data(signal)
        await store_data(
            bucket, timestamp, packed_data, rms, peak_to_peak, crest_factor
        )
        await asyncio.sleep(1)

    end_time = int(time.time() * 1_000_000)
    await query_data(bucket, start_time, end_time)


if __name__ == "__main__":
    """Run the main coroutine."""
    asyncio.run(main())
