"""Example of storing and querying sensor data in InfluxDB."""

import asyncio
import time

import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS
from utils import generate_sensor_data


INFLUXDB_URL = "http://localhost:8086"
TOKEN = "my-token"
ORG = "my-org"
BUCKET = "sensor_data"


async def setup_influxdb():
    """Set up the InfluxDB client."""
    client = InfluxDBClient(url=INFLUXDB_URL, token=TOKEN, org=ORG)
    return client


async def store_data(client, signal: np.ndarray, timestamp: int, frequency: int):
    """Store the sensor data in the InfluxDB."""
    write_api = client.write_api(write_options=ASYNCHRONOUS)
    time_step = 1_000_000 // frequency

    for value in signal:
        point = (
            Point("sensor_values")
            .field("value", value)
            .time(timestamp, write_precision="us")
        )
        write_api.write(bucket=BUCKET, record=point, org=ORG)
        timestamp += time_step


async def query_data(client, start_time: int, end_time: int):
    """Query and retrieve the stored data."""
    query_api = client.query_api()

    query = f"""
    from(bucket: "{BUCKET}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement == "sensor_values")
      |> sort(columns: ["_time"])
    """

    result = query_api.query(query=query, org=ORG)

    for table in result:
        for record in table.records:
            print(f"Timestamp: {record.get_time()}")
            print(f"Value: {record['_value']}")
            print("---")


async def main():
    """Example of storing and querying sensor data in InfluxDB."""
    client = await setup_influxdb()
    start_time = int(time.time() * 1_000_000)

    for _ in range(5):
        timestamp = int(time.time() * 1_000_000)
        signal = generate_sensor_data(frequency=1000, duration=1)
        await store_data(client, signal, timestamp, frequency=1000)
        await asyncio.sleep(1)

    end_time = int(time.time() * 1_000_000)
    await query_data(client, start_time, end_time)


if __name__ == "__main__":
    """Run the main coroutine."""
    asyncio.run(main())
