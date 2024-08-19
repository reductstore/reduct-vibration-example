"""Benchmark InfluxDB and ReductStore for write and read performance."""

import asyncio
import time

import numpy as np
from helper_functions import (
    TimeUnits,
    calculate_metrics,
    generate_sensor_data,
    get_current_time,
    pack_data,
)
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_data_processing import query_data as query_influxdb
from influxdb_data_processing import store_data as store_influxdb
from plot_results import (
    plot_benchmark_results,
    prepare_csv,
    read_benchmark_results,
    write_to_csv,
)
from reduct import Client as ReductClient
from reduct_data_processing import query_data as query_reductstore
from reduct_data_processing import store_data as store_reductstore
from tqdm import tqdm

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
    ), f"Data do not match with {len(signal)} signals vs {len(result)} results"

    write_to_csv(
        CSV_FILE_PATH,
        "InfluxDB",
        frequency,
        end_write - start_write,
        end_read - start_read,
    )


async def benchmark_reductstore(frequency: int):
    """Benchmark write and read performance for ReductStore."""
    time_step = TimeUnits.MICROSECOND // frequency
    async with ReductClient(REDUCTSTORE_URL, api_token=REDUCTSTORE_TOKEN) as client:
        bucket = await client.create_bucket("sensor_data", exist_ok=True)
        start_time = get_current_time()

        signal = generate_sensor_data(frequency=frequency, duration=DURATION)
        signal_chunks = np.array_split(signal, DURATION)

        await asyncio.sleep(DURATION)

        start_write = time.time()
        for step, chunk in enumerate(signal_chunks):
            rms, peak_to_peak, crest_factor = calculate_metrics(chunk)
            packed_data = pack_data(chunk)
            timestamp = start_time + step * len(chunk) * time_step
            await store_reductstore(
                bucket, timestamp, packed_data, rms, peak_to_peak, crest_factor
            )
        end_write = time.time()

        end_time = get_current_time()

        start_read = time.time()
        result = await query_reductstore(bucket, start_time, end_time)
        end_read = time.time()

    assert np.array_equal(
        signal, result
    ), f"Data do not match with {len(signal)} signals vs {len(result)} results"

    write_to_csv(
        CSV_FILE_PATH,
        "ReductStore",
        frequency,
        end_write - start_write,
        end_read - start_read,
    )


async def run_benchmark_for_frequency(frequency: int, progress_bar: tqdm):
    for run in range(NUMBER_RUNS):
        progress_bar.set_description(
            f"Running {run + 1}/{NUMBER_RUNS} for {frequency} Hz"
        )
        await benchmark_influxdb(frequency)
        await benchmark_reductstore(frequency)
        progress_bar.update(1)


async def run_benchmarks():
    total_runs = NUMBER_RUNS * len(FREQUENCIES)

    with tqdm(total=total_runs, desc="Overall Progress", unit="run") as progress_bar:
        for frequency in FREQUENCIES:
            await run_benchmark_for_frequency(frequency, progress_bar)


async def main():
    prepare_csv(CSV_FILE_PATH)
    await run_benchmarks()
    data = read_benchmark_results(CSV_FILE_PATH)
    plot_benchmark_results(data)


if __name__ == "__main__":
    asyncio.run(main())
