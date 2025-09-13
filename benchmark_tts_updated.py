
import asyncio
import base64
import time
import csv
import aiohttp
import websockets
import json

SERVER = "http://35.200.228.189:8009"
WS = "ws://35.200.228.189:8009/ws"
SPEAKER_ID = "bench_speaker"
USER_ID = "bench_user"
TEXT = "Benchmarking synthesis."
REFERENCE_AUDIO_PATH = "sample.wav"
NUM_REQUESTS = 5
CONCURRENCY = 2
MODE = "ws"
results = []

async def create_profile(session):
    with open(REFERENCE_AUDIO_PATH, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "speaker_id": SPEAKER_ID,
        "reference_audio": audio_b64,
        "reference_text": "Benchmark reference.",
        "user_id": USER_ID
    }
    async with session.post(f"{SERVER}/create_profile", json=payload) as resp:
        print("Profile:", await resp.json())

async def synthesize_ws(idx):
    async with websockets.connect(WS) as ws:
        payload = {
            "action": "synthesize",
            "text": TEXT,
            "speaker_id": SPEAKER_ID,
            "language_id": "en",
            "exaggeration": 0.5,
            "user_id": USER_ID
        }
        start = time.time()
        await ws.send(json.dumps(payload))
        audio = await ws.recv()
        latency = time.time() - start
        with open(f"output_{idx}.wav", "wb") as f:
            f.write(audio)
        results.append((idx, latency, MODE, None))

async def run():
    async with aiohttp.ClientSession() as session:
        await create_profile(session)
    tasks = []
    for i in range(NUM_REQUESTS):
        tasks.append(synthesize_ws(i))
        if len(tasks) >= CONCURRENCY:
            await asyncio.gather(*tasks)
            tasks = []
    if tasks:
        await asyncio.gather(*tasks)

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Request", "Latency", "Mode", "Error"])
        writer.writerows(results)

asyncio.run(run())
