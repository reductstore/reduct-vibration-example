import asyncio
import time
import struct
import numpy as np
from reduct import Client, Bucket


HIGH_RMS = 1.0
HIGH_CREST_FACTOR = 3.0
HIGH_PEAK_TO_PEAK = 5.0


# 1. Set up and connect to ReductStore
async def setup_reductstore() -> Bucket:
    client = Client("http://localhost:8383", api_token="my-token")
    return await client.create_bucket("sensor_data", exist_ok=True)


# 2. Generate simulated sensor data
def generate_sensor_data(frequency: int = 1000, duration: int = 1) -> np.ndarray:
    t = np.linspace(0, duration, frequency * duration)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    return signal.astype(np.float32)


# 3. Calculate metrics
def calculate_metrics(signal: np.ndarray) -> tuple[float, float, float]:
    rms = np.sqrt(np.mean(signal**2))
    peak_to_peak = np.max(signal) - np.min(signal)
    crest_factor = np.max(np.abs(signal)) / rms
    return rms, peak_to_peak, crest_factor


# 4. Pack data into binary format using struct
def pack_data(signal: np.ndarray) -> bytes:
    fmt = f">{len(signal)}f"  # '>' for big-endian, 'f' for 32-bit float
    return struct.pack(fmt, *signal)


# 5. Store data in ReductStore
async def store_data(
    bucket: Bucket,
    timestamp: int,
    packed_data: bytes,
    rms: float,
    peak_to_peak: float,
    crest_factor: float,
):
    labels = {
        "rms": "high" if rms > HIGH_RMS else "low",
        "peak_to_peak": "high" if peak_to_peak > HIGH_PEAK_TO_PEAK else "low",
        "crest_factor": "high" if crest_factor > HIGH_CREST_FACTOR else "low",
    }
    await bucket.write("sensor_readings", packed_data, timestamp, labels=labels)


# 6. Query and retrieve the stored data
async def query_data(bucket: Bucket, start_time: int, end_time: int):
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
    bucket = await setup_reductstore()

    # Store 5 seconds of data
    for _ in range(5):
        timestamp = int(time.time() * 1_000_000)
        signal = generate_sensor_data()
        rms, peak_to_peak, crest_factor = calculate_metrics(signal)
        packed_data = pack_data(signal)
        await store_data(
            bucket, timestamp, packed_data, rms, peak_to_peak, crest_factor
        )
        await asyncio.sleep(1)

    # Query the stored data for the last 5 seconds
    end_time = int(time.time() * 1_000_000)
    start_time = end_time - 5_000_000
    await query_data(bucket, start_time, end_time)


if __name__ == "__main__":
    asyncio.run(main())
