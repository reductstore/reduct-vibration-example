"""Example of storing and querying sensor data in InfluxDB."""

import asyncio

import numpy as np
from influxdb_client import Point
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from scripts.helper_functions import (
    TimeUnits,
    generate_sensor_data,
    get_current_time,
    time_to_rfc3339,
)
from influxdb_client.domain.write_precision import WritePrecision

INFLUXDB_URL = "http://localhost:8086"
TOKEN = "my-token"
ORG = "my-org"
BUCKET = "sensor_data"


async def store_data(
    client: InfluxDBClientAsync, signal: np.ndarray, start_time: int, time_step: int
):
    """Store the sensor data in the InfluxDB in a batch."""
    points = []
    for step, value in enumerate(signal):
        point = (
            Point("sensor_values")
            .field("value", value)
            .time(start_time + (step * time_step), WritePrecision.NS)
        )
        points.append(point)

    await client.write_api().write(bucket=BUCKET, record=points, org=ORG)


async def query_data(
    client: InfluxDBClientAsync,
    start_time: int,
    end_time: int,
    time_step: int,
) -> np.ndarray:
    """Query and retrieve the stored data."""
    start_time_rfc3339 = time_to_rfc3339(start_time - time_step, TimeUnits.NANOSECOND)
    end_time_rfc3339 = time_to_rfc3339(end_time + time_step, TimeUnits.NANOSECOND)

    query = f"""
    from(bucket: "{BUCKET}")
      |> range(start: {start_time_rfc3339}, stop: {end_time_rfc3339})
      |> filter(fn: (r) => r._measurement == "sensor_values")
      |> sort(columns: ["_time"])
      |> keep(columns: ["_time", "_value"])
    """

    result = await client.query_api().query(query=query, org=ORG)

    if not result:
        print("No data found in query")
        return np.array([], dtype=np.float32)

    return np.array(
        [record["_value"] for record in result[0].records], dtype=np.float32
    )


async def main():
    """Example of storing and querying sensor data in InfluxDB."""
    frequency = 1000
    duration = 5

    time_step = TimeUnits.NANOSECOND // frequency
    async with InfluxDBClientAsync(url=INFLUXDB_URL, token=TOKEN, org=ORG) as client:
        await client.ping()
        assert await client.ping(), "InfluxDB connection failed"

        start_time = get_current_time(TimeUnits.NANOSECOND)
        signal = generate_sensor_data(frequency=frequency, duration=duration)

        await asyncio.sleep(duration)
        await store_data(client, signal, start_time, time_step)

        end_time = get_current_time(TimeUnits.NANOSECOND)
        result = await query_data(client, start_time, end_time, time_step)

        assert np.array_equal(
            signal, result
        ), f"Data do not match: signal {len(signal)} vs result {len(result)}"


if __name__ == "__main__":
    """Run the main coroutine."""
    asyncio.run(main())
