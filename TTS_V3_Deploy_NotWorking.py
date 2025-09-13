import os
import sys
from pathlib import Path
# ---------------- PATH FIX ----------------
ROOT = Path(__file__).resolve().parent
#sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
str_pth = str(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT))
import asyncio
import base64
import io
import logging
from logging.handlers import RotatingFileHandler
import pytz
import os
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
import uvicorn

from src.chatterbox_vllm.tts import ChatterboxTTS, Conditionals  # Use optimized tts_optimized.py
#from tts import ChatterboxTTS, Conditionals  # Use optimized tts_optimized.py
import multiprocessing

# === 1. IST Timezone Formatter ===
IST = pytz.timezone("Asia/Kolkata")
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        return dt.strftime("%Y-%m-%dT%H:%M:%S%z")

# === 2. Logging Configuration ===
LOG_DIR = "/home/ubuntu/D_Test/REPO/V1/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
base_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("TTSService")
logger_handler = RotatingFileHandler(f"{LOG_DIR}/llm.log", maxBytes=10_000_000, backupCount=5)
logger_handler.setFormatter(ISTFormatter(base_fmt))
logger.setLevel(logging.INFO)
logger.addHandler(logger_handler)

# Optimization flags
# QUANTIZE_T3 = os.getenv("QUANTIZE_T3", "false").lower() == "true"
# if QUANTIZE_T3:
#     logger.info("[OPTIMIZATION] Quantization enabled for T3 model (placeholder logic)")
#     # from awq import quantize_model
#     # tts_model.t3 = quantize_model(tts_model.t3, bits=4)


class TTSService:
    def __init__(self, model_path, profile_dir, batch_size=4, batch_timeout=0.5):
        self.model_path = model_path
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.tts_model = ChatterboxTTS.from_local(
            ckpt_dir=self.model_path,
            target_device="cuda",
            variant="multilingual",
            s3gen_use_fp16=True
        )
        self.speaker_profiles = {}
        self.request_queue = asyncio.Queue()
        self.load_profiles()

    def load_profiles(self):
        for profile_file in self.profile_dir.glob("*.pt"):
            try:
                speaker_id = profile_file.stem
                self.speaker_profiles[speaker_id] = Conditionals.load(profile_file).to("cpu")
                logger.info(f"Loaded speaker profile: {speaker_id}")
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")

    def encode_audio(self, wav_tensor, sr):
        buf = io.BytesIO()
        sf.write(buf, wav_tensor.numpy(), sr, format='WAV')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    async def batch_worker(self):
        while True:
            batch = []
            try:
                item = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_timeout)
                batch.append(item)
                while len(batch) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            except asyncio.TimeoutError:
                continue

            for speaker_id in set(req['speaker_id'] for req in batch):
                group = [req for req in batch if req['speaker_id'] == speaker_id]
                if speaker_id not in self.speaker_profiles:
                    for req in group:
                        self._respond_error(req, "Speaker profile not found")
                    continue

                try:
                    conds = self.speaker_profiles[speaker_id]
                    texts = [req['text'] for req in group]
                    language_id = group[0].get('language_id', 'en')
                    exaggeration = group[0].get('exaggeration', 0.5)

                    # Log GPU memory before inference
                    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
                    start_time = time.time()

                    audios = self.tts_model.generate_with_conds(
                        prompts=texts,
                        s3gen_ref=conds.gen,
                        cond_emb=conds.t3,
                        language_id=language_id,
                        exaggeration=exaggeration
                    )

                    latency = time.time() - start_time
                    mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
                    logger.info(f"[BATCH] size={len(group)} latency={latency:.2f}s "
                                f"GPU Mem Before={mem_before:.2f}MB After={mem_after:.2f}MB")

                    for req, audio in zip(group, audios):
                        audio_b64 = self.encode_audio(audio, self.tts_model.sr)
                        self._respond_success(req, {"audio": audio_b64})
                        logger.info(f"Processed user_id={req.get('user_id')} speaker_id={speaker_id}")
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    for req in group:
                        self._respond_error(req, str(e))

    def _respond_success(self, req, data):
        if 'websocket' in req:
            asyncio.create_task(req['websocket'].send_json(data))
        elif 'future' in req:
            req['future'].set_result(data)

    def _respond_error(self, req, message):
        if 'websocket' in req:
            asyncio.create_task(req['websocket'].send_json({"error": message}))
        elif 'future' in req:
            req['future'].set_result({"error": message})

    async def handle_http_synthesize(self, data):
        try:
            speaker_id = data.get("speaker_id")
            user_id = data.get("user_id", "anonymous")
            if speaker_id not in self.speaker_profiles:
                return JSONResponse(status_code=404, content={"error": "Speaker profile not found"})
            future = asyncio.get_event_loop().create_future()
            await self.request_queue.put({
                "text": data["text"],
                "speaker_id": speaker_id,
                "language_id": data.get("language_id", "en"),
                "exaggeration": data.get("exaggeration", 0.5),
                "future": future,
                "user_id": user_id
            })
            return await future
        except Exception as e:
            logger.error(f"HTTP synthesize error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def handle_http_create_profile(self, data):
        try:
            speaker_id = data["speaker_id"]
            user_id = data.get("user_id", "anonymous")
            ref_audio = data["reference_audio"]
            audio_bytes = base64.b64decode(ref_audio)
            buf = io.BytesIO(audio_bytes)
            wav, sr = sf.read(buf)
            temp_path = f"temp_{speaker_id}.wav"
            sf.write(temp_path, wav, sr)
            conds = self.tts_model.get_audio_conditionals(temp_path)
            profile = Conditionals(t3=conds[1], gen=conds[0])
            torch.save(profile, self.profile_dir / f"{speaker_id}.pt")
            self.speaker_profiles[speaker_id] = profile
            logger.info(f"Created profile speaker_id={speaker_id} user_id={user_id}")
            return {"status": "profile_created"}
        except Exception as e:
            logger.error(f"Profile creation error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def handle_ws_message(self, websocket, data):
        action = data.get("action")
        user_id = data.get("user_id", "anonymous")
        if action == "synthesize":
            speaker_id = data.get("speaker_id")
            if speaker_id not in self.speaker_profiles:
                await websocket.send_json({"error": "Speaker profile not found"})
                return
            await self.request_queue.put({
                "text": data["text"],
                "speaker_id": speaker_id,
                "language_id": data.get("language_id", "en"),
                "exaggeration": data.get("exaggeration", 0.5),
                "websocket": websocket,
                "user_id": user_id
            })
        elif action == "create_profile":
            try:
                speaker_id = data["speaker_id"]
                ref_audio = data["reference_audio"]
                audio_bytes = base64.b64decode(ref_audio)
                buf = io.BytesIO(audio_bytes)
                wav, sr = sf.read(buf)
                temp_path = f"temp_{speaker_id}.wav"
                sf.write(temp_path, wav, sr)
                conds = self.tts_model.get_audio_conditionals(temp_path)
                profile = Conditionals(t3=conds[1], gen=conds[0])
                torch.save(profile, self.profile_dir / f"{speaker_id}.pt")
                self.speaker_profiles[speaker_id] = profile
                await websocket.send_json({"status": "profile_created"})
                logger.info(f"Created profile speaker_id={speaker_id} user_id={user_id}")
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                logger.error(f"Profile creation error: {e}")


# Instantiate service
app = FastAPI()
tts_service = TTSService("/home/ubuntu/D_Test/MDL/TTS_V1", "/home/ubuntu/D_Test/MDL/TTS_V1/Speaker_Profiles")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(tts_service.batch_worker())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(request: Request):
    data = await request.json()
    return await tts_service.handle_http_synthesize(data)


@app.post("/create_profile")
async def create_profile(request: Request):
    data = await request.json()
    return await tts_service.handle_http_create_profile(data)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await tts_service.handle_ws_message(websocket, data)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# if __name__ == "__main__":
#     multiprocessing.set_start_method("spawn", force=True)
if __name__ == "__main__":
    #import multiprocessing
    #multiprocessing.set_start_method("spawn", force=True)
    tts_service = TTSService("/home/ubuntu/D_Test/MDL/TTS_V1", "/home/ubuntu/D_Test/MDL/TTS_V1/Speaker_Profiles")
    uvicorn.run("TTS_V3_Deploy:app", host="0.0.0.0", port=8009)