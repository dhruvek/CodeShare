import os
import sys
from pathlib import Path

# ---------------- PATH FIX ----------------
ROOT = Path(__file__).resolve().parent
#sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
str_pth = str(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from typing import List

import torchaudio as ta

from src.chatterbox_vllm.tts import ChatterboxTTS as Vinfertts_v3

vi_tts_model = Vinfertts_v3.from_local(ckpt_dir='/home/ubuntu/D_Test/MDL/TTS_V1',target_device='cuda:0',variant='',compile=True,s3gen_use_fp16=True)

