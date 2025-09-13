#!/usr/bin/env python3
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
from typing import List
import torchaudio as ta
from src.chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    model = ChatterboxTTS.from_pretrained(
        max_batch_size = 3,
        max_model_len = 1000,
    )

    for i, audio_prompt_path in enumerate([None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]):
        prompts = [
            "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
            "This is a separate prompt to test the batching implementation.",
            "And here is a third prompt. It's a bit longer than the first one, but not by much.",
        ]
    
        audios = model.generate(prompts, audio_prompt_path=audio_prompt_path, exaggeration=0.5)
        for audio_idx, audio in enumerate(audios):
            ta.save(f"test-{i}-{audio_idx}.mp3", audio, model.sr)

    model.shutdown()