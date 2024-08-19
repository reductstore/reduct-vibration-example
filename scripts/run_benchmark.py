"""Benchmark InfluxDB and ReductStore for write and read performance."""

import asyncio
import time

import numpy as np
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from scripts.influxdb_data_processing import store_data as store_influxdb
from scripts.influxdb_data_processing import query_data as query_influxdb
from scripts.plot_results import (
    plot_benchmark_results,
    prepare_csv,
    read_benchmark_results,
    write_to_csv,
)
from scripts.reduct_data_processing import store_data as store_reductstore
from scripts.reduct_data_processing import query_data as query_reductstore
from reduct import Client as ReductClient
from scripts.helper_functions import (
    TimeUnits,
    calculate_metrics,
    generate_sensor_data,
    get_current_time,
    pack_data,
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
FREQUENCIES = [1000, 2000, 5000, 10_000, 20_000, 30_000]
DURATION = 10
NUMBER_RUNS = 10

# CSV file path
CSV_FILE_PATH = "benchmark_results.csv"


async def benchmark_influxdb(frequency: int):
    """Benchmark write and read performance for InfluxDB."""
    time_step = TimeUnits.NANOSECOND // frequency
    async with InfluxDBClientAsync(url=INFLUXDB_URL, token=TOKEN, org=ORG) as client:
        await client.ping()
        assert await client.ping(), "InfluxDB connection failed"

        start_time = get_current_time(TimeUnits.NANOSECOND)
        signal = generate_sensor_data(frequency=frequency, duration=DURATION)

        await asyncio.sleep(DURATION)
        start_write = time.time()
        await store_influxdb(client, signal, start_time, time_step)
        end_write = time.time()

        end_time = get_current_time(TimeUnits.NANOSECOND)
        start_read = time.time()
        result = await query_influxdb(client, start_time, end_time, time_step)
        end_read = time.time()

    assert np.array_equal(
        signal, result
    ), f"Stored and queried data do not match: {len(signal)} vs {len(result)}"

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
    async with ReductClient(REDUCTSTORE_URL, api_token=REDUCTSTORE_TOKEN) as client:
        bucket = await client.create_bucket("sensor_data", exist_ok=True)
        start_time = get_current_time()

        signal = generate_sensor_data(frequency=frequency, duration=DURATION)
        signal_chunks = np.array_split(signal, DURATION)

        await asyncio.sleep(DURATION)
        start_write = time.time()
        timestamp = start_time
        for chunk in signal_chunks:
            rms, peak_to_peak, crest_factor = calculate_metrics(chunk)
            packed_data = pack_data(chunk)
            await store_reductstore(
                bucket, timestamp, packed_data, rms, peak_to_peak, crest_factor
            )
            timestamp += len(chunk) * TimeUnits.MICROSECOND // frequency
        end_write = time.time()

        end_time = get_current_time()
        start_read = time.time()
        result = await query_reductstore(bucket, start_time, end_time)
        end_read = time.time()

    assert np.array_equal(
        signal, result
    ), f"Stored and queried data do not match: {len(signal)} vs {len(result)}"

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
