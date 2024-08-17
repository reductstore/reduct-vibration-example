"""Benchmark InfluxDB and ReductStore for write and read performance."""

import asyncio
import time

from influxdb_example import setup_influxdb
from influxdb_example import store_data as store_influxdb
from reduct_example import setup_reductstore
from reduct_example import store_data as store_reductstore
from utils import (
    calculate_metrics,
    generate_sensor_data,
    pack_data,
)
from plot_benchmark import (
    prepare_csv,
    read_benchmark_results,
    write_to_csv,
    plot_benchmark_results,
)

# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
TOKEN = "my-token"
ORG = "my-org"
BUCKET = "sensor_data"

# ReductStore configuration
REDUCTSTORE_URL = "http://localhost:8383"
REDUCTSTORE_TOKEN = "my-token"

# Constants for benchmarking
FREQUENCIES = [1000, 2000]  # , 5000, 10_000, 20_000, 30_000]
DURATION = 1
NUMBER_RUNS = 1

# CSV file path
CSV_FILE_PATH = "benchmark_results.csv"


async def benchmark_influxdb(frequency: int):
    """Benchmark write and read performance for InfluxDB."""
    client = await setup_influxdb()
    start_time = int(time.time() * 1_000_000)

    signal = generate_sensor_data(frequency=frequency, duration=DURATION)
    timestamp = int(time.time() * 1_000_000)

    start_write = time.time()
    await store_influxdb(client, signal, timestamp, frequency)
    end_write = time.time()

    end_time = int(time.time() * 1_000_000)
    query = f"""
    from(bucket: "{BUCKET}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement == "sensor_values")
      |> sort(columns: ["_time"])
    """

    query_api = client.query_api()
    start_read = time.time()
    result = query_api.query(query=query, org=ORG)
    end_read = time.time()

    assert len(result) == 1

    print(f"InfluxDB Frequency: {frequency} Hz")
    print(f"Write Time: {end_write - start_write:.2f} seconds")
    print(f"Read Time: {end_read - start_read:.2f} seconds")
    print("---")

    write_to_csv(
        CSV_FILE_PATH,
        "InfluxDB",
        frequency,
        end_write - start_write,
        end_read - start_read,
    )


async def benchmark_reductstore(frequency: int):
    """Benchmark write and read performance for ReductStore."""
    bucket = await setup_reductstore()
    start_time = int(time.time() * 1_000_000)

    signal = generate_sensor_data(frequency=frequency, duration=DURATION)
    rms, peak_to_peak, crest_factor = calculate_metrics(signal)
    packed_data = pack_data(signal)
    timestamp = int(time.time() * 1_000_000)

    start_write = time.time()
    await store_reductstore(bucket, timestamp, packed_data)
    end_write = time.time()

    end_time = int(time.time() * 1_000_000)
    start_read = time.time()
    async for record in bucket.query(
        "sensor_readings", start=start_time, stop=end_time
    ):
        result = await record.read_all()
    end_read = time.time()

    assert result == packed_data

    print(f"ReductStore Frequency: {frequency} Hz")
    print(f"Write Time: {end_write - start_write:.2f} seconds")
    print(f"Read Time: {end_read - start_read:.2f} seconds")
    print("---")

    write_to_csv(
        CSV_FILE_PATH,
        "ReductStore",
        frequency,
        end_write - start_write,
        end_read - start_read,
    )


async def main():
    """Run benchmarks for all frequencies."""
    prepare_csv(CSV_FILE_PATH)
    for frequency in FREQUENCIES:
        for _ in range(NUMBER_RUNS):
            await benchmark_influxdb(frequency)
            await benchmark_reductstore(frequency)
            await asyncio.sleep(1)
    data = read_benchmark_results(CSV_FILE_PATH)
    plot_benchmark_results(data)


if __name__ == "__main__":
    asyncio.run(main())
