import asyncio
import base64
import time
import csv
import aiohttp
import websockets
import json
import subprocess

# Benchmark configuration
NUM_REQUESTS = 20
CONCURRENCY = 4
MODE = "ws"  # "http" or "ws"
SERVER_URL = "http://localhost:8009"
WS_URL = "ws://localhost:8009/ws"
SPEAKER_ID = "benchmark_speaker"
TEXT = "Hello, this is a benchmark test."
LANGUAGE_ID = "en"
EXAGGERATION = 0.5
USER_ID = "benchmark_user"
REFERENCE_AUDIO_PATH = "/home/ubuntu/D_Test/REPO/V1/Ref_Audios/ENGLISH/L11EN_Anika.mp3"

results = []

async def create_profile(session):
    with open(REFERENCE_AUDIO_PATH, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "speaker_id": SPEAKER_ID,
        "reference_audio": audio_b64,
        "reference_text": "Hello, this is a reference sample.",
        "user_id": USER_ID
    }
    async with session.post(f"{SERVER_URL}/create_profile", json=payload) as resp:
        print("Profile creation:", await resp.json())

async def synthesize_http(session, idx):
    payload = {
        "text": TEXT,
        "speaker_id": SPEAKER_ID,
        "language_id": LANGUAGE_ID,
        "exaggeration": EXAGGERATION,
        "user_id": USER_ID
    }
    start = time.time()
    async with session.post(f"{SERVER_URL}/synthesize", json=payload) as resp:
        data = await resp.json()
        latency = time.time() - start
        results.append((idx, latency, "http", data.get("error")))

async def synthesize_ws(idx):
    async with websockets.connect(WS_URL) as ws:
        payload = {
            "action": "synthesize",
            "text": TEXT,
            "speaker_id": SPEAKER_ID,
            "language_id": LANGUAGE_ID,
            "exaggeration": EXAGGERATION,
            "user_id": USER_ID
        }
        start = time.time()
        await ws.send(json.dumps(payload))
        response = await ws.recv()
        latency = time.time() - start
        data = json.loads(response)
        results.append((idx, latency, "ws", data.get("error")))

async def run_benchmark():
    tasks = []
    if MODE == "http":
        async with aiohttp.ClientSession() as session:
            await create_profile(session)
            for i in range(NUM_REQUESTS):
                tasks.append(synthesize_http(session, i))
                if len(tasks) >= CONCURRENCY:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
    else:
        async with aiohttp.ClientSession() as session:
            await create_profile(session)
        for i in range(NUM_REQUESTS):
            tasks.append(synthesize_ws(i))
            if len(tasks) >= CONCURRENCY:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

def log_gpu():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ])
        usage = output.decode().strip().split("\n")
        for idx, line in enumerate(usage):
            gpu_util, mem_used = line.split(',')
            print(f"GPU {idx}: Utilization={gpu_util}% Memory={mem_used}MB")
    except Exception as e:
        print("GPU logging failed:", e)

async def main():
    start_time = time.time()
    await run_benchmark()
    total_time = time.time() - start_time
    log_gpu()

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Request", "Latency", "Mode", "Error"])
        writer.writerows(results)

    latencies = [r[1] for r in results if r[3] is None]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
        throughput = len(latencies) / total_time * 60
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"P95 Latency: {p95_latency:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/min")
    else:
        print("No successful requests.")

# For environments like Jupyter
if __name__ == "__main__":
    asyncio.run(main())
