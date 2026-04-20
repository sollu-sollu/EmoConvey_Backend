# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import base64
import logging
import os
import sys
import tempfile
import shutil

# 1. Essential heavy imports (Do these BEFORE reconfiguring stdout)
import torch
import requests
import re
import asyncio
import uuid
import time
import subprocess
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image

# 2. Safe UTF-8 Initialization (Only after Torch has loaded to prevent DLL issues)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmoConvey-Backend")

# 3. CONFIGURATION
# Using your provided Ngrok token
NGROK_AUTH_TOKEN = "36EnpN0f0tx9hre6EMpt1jRpHeb_38GYquy1KabEJGskmKeHL"
EMOTION_LLAMA_DIR       = os.getenv("EMOTION_LLAMA_DIR",       r"C:\Rajendran\EmoConvey_Backend\Emotion-LLaMA")
EMOTION_LLAMA_DEMO_CFG  = os.getenv("EMOTION_LLAMA_DEMO_CFG",  os.path.join(EMOTION_LLAMA_DIR, "eval_configs", "demo.yaml"))
EMOTION_LLAMA_GRADIO_URL = os.getenv("EMOTION_LLAMA_GRADIO_URL", "http://127.0.0.1:7889")
EMOTION_LLAMA_QUANT     = os.getenv("EMOTION_LLAMA_QUANTIZATION", "int4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global State
mock_mode = False
processing_lock = asyncio.Lock()
_gradio_process = None

# [Storage Check]
def check_disk_space(min_gb=10):
    try:
        _, _, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        if free_gb < min_gb:
            logger.error(f"❌ INSUFFICIENT STORAGE: Only {free_gb}GB free. Need at least {min_gb}GB.")
            return False
        logger.info(f"💾 Disk Space: {free_gb}GB free.")
        return True
    except: return True

# [Internal Helpers]
def _is_gradio_server_ready(url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception: return False

def _start_gradio_server() -> bool:
    global _gradio_process
    app_py = os.path.join(EMOTION_LLAMA_DIR, "app_EmotionLlamaClient.py")
    if not os.path.exists(app_py): return False

    logger.info(f"🚀 Starting Emotion-LLaMA Gradio server ({EMOTION_LLAMA_QUANT} mode)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = EMOTION_LLAMA_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    _gradio_process = subprocess.Popen(
        [sys.executable, app_py, "--cfg-path", EMOTION_LLAMA_DEMO_CFG],
        cwd=EMOTION_LLAMA_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        bufsize=1
    )

    import threading
    def _stream_output():
        for line in _gradio_process.stdout:
            line = line.rstrip()
            if line: logger.info(f"[EmoLLaMA] {line}")
    threading.Thread(target=_stream_output, daemon=True).start()
    
    logger.info("⏳ Waiting for Emotion-LLaMA to load weight file...")
    deadline = time.time() + 300
    while time.time() < deadline:
        if _gradio_process.poll() is not None:
            code = _gradio_process.returncode
            logger.error(f"❌ Emotion-LLaMA crashed (exit code: {code})")
            return False
        if _is_gradio_server_ready(EMOTION_LLAMA_GRADIO_URL): return True
        time.sleep(3)
    return False

def init_emotion_llama() -> bool:
    if _is_gradio_server_ready(EMOTION_LLAMA_GRADIO_URL): return True
    if not check_disk_space(10): return False
    return _start_gradio_server()

# [Parsing Logic - Python 3.11 Compatible Regex]
def extract_emotion_and_text(response: str):
    transcription = ""
    is_silent = False
    
    # 1. Transcription (Safe regex)
    trans_match = re.search(r'\[TRANSCRIPTION:\s*(.*?)\]', response, flags=re.DOTALL | re.IGNORECASE)
    if trans_match:
        transcription = trans_match.group(1).strip()
        response = re.sub(r'\[TRANSCRIPTION:.*?\]', '', response, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Emotion (Moved global flags to start)
    emotion = "neutral"
    emo_match = re.search(r'\[(\w+)\]', response)
    if not emo_match:
        # Python 3.11 requires global flags at the start
        emo_match = re.search(r'(?i)(detected|mood|emotion):?\s*(\w+)', response)
        if emo_match: emotion = emo_match.group(2).lower()
    else: emotion = emo_match.group(1).lower()

    # 3. Clean leaks
    text = response
    # Move (?i) flags to the absolute start of each pattern
    leak_patterns = [
        r'(?i)^(Sure!|Certainly!|Of course!|Here\'s my response:?|Based on).*?(\.|\:)\s*',
        r'^\d+\.\s*', 
        r'\[.*?\]'
    ]
    for p in leak_patterns: 
        text = re.sub(p, '', text, flags=re.DOTALL).strip()
    
    return emotion, text or "I'm here with you.", transcription, is_silent

# [Handers]
async def _handle_live(data, history):
    start = time.time()
    audio_b64 = data.get('audio', '')
    if not audio_b64: return None
    apath = os.path.join(tempfile.gettempdir(), "live.wav")
    with open(apath, "wb") as f: f.write(base64.b64decode(audio_b64))
    vpath = os.path.join(tempfile.gettempdir(), "live.mp4")
    has_img = data.get('isCamOn') and data.get('image')
    
    from moviepy.editor import ImageClip, AudioFileClip
    if has_img:
        ipath = os.path.join(tempfile.gettempdir(), "live.jpg")
        Image.open(BytesIO(base64.b64decode(data.get('image')))).convert("RGB").save(ipath, "JPEG")
        ac = AudioFileClip(apath)
        ImageClip(ipath, duration=ac.duration).set_audio(ac).write_videofile(vpath, fps=1, logger=None)
    else:
        bpath = os.path.join(tempfile.gettempdir(), "black.jpg")
        if not os.path.exists(bpath): Image.new("RGB", (224,224)).save(bpath, "JPEG")
        ac = AudioFileClip(apath)
        ImageClip(bpath, duration=ac.duration).set_audio(ac).write_videofile(vpath, fps=1, logger=None)

    raw = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.post(f"{EMOTION_LLAMA_GRADIO_URL}/api/predict/", json={"data": [vpath, "MANDATORY FORMAT: [TRANSCRIPTION: ...] [Emotion] Natural response."]}, timeout=120).json()["data"][0])
    emo, text, trans, _ = extract_emotion_and_text(raw)
    return json.dumps({"emotion":emo, "response":text, "transcription":trans, "time":round(time.time()-start,1)})

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mock_mode
    # --- Start Ngrok Tunnel ---
    try:
        from pyngrok import ngrok
        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            public_url = ngrok.connect(8000).public_url
            logger.info("="*50)
            logger.info(f"✅ NGROK TUNNEL: {public_url}")
            logger.info(f"✅ WEBHOOK URL: {public_url.replace('https','wss')}/ws/live")
            logger.info("="*50)
    except Exception as e:
        logger.error(f"❌ Ngrok Error: {e}")

    if not init_emotion_llama():
        logger.warning("⚠️ Weights failed to load. Mode: MOCK")
        mock_mode = True
    else: logger.info("✅ Weights loaded successfully!")
    yield
    if _gradio_process: _gradio_process.terminate()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

@app.websocket("/ws/live")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); history = []
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data.get('type') == 'ping': continue
            if mock_mode:
                await ws.send_text(json.dumps({"emotion":"happy","response":"Mocking... (Loading error)","time":0}))
                continue
            resp = await _handle_live(data, history)
            if resp: await ws.send_text(resp)
    except: pass

if __name__ == "__main__":
    print("\n" + "="*40 + "\n  EmoConvey Backend (Finalized Fix)\n" + "="*40)
    uvicorn.run(app, host="0.0.0.0", port=8000)
