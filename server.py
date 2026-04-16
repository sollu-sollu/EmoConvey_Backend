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
import asyncio
import subprocess
import time
import re
import requests
import torch
from io import BytesIO
from PIL import Image

# ============================================================
# Load .env (written by setup_emotion_llama.py)
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; env vars may be set externally

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmoConveyBackend")

# ============================================================
# CONFIGURATION
# ============================================================

NGROK_AUTH_TOKEN = "36EnpN0f0tx9hre6EMpt1jRpHeb_38GYquy1KabEJGskmKeHL"

# Emotion-LLaMA settings (populated by setup_emotion_llama.py via .env)
EMOTION_LLAMA_DIR       = os.getenv("EMOTION_LLAMA_DIR",       r"C:\Rajendran\EmoConvey_Backend\Emotion-LLaMA")
EMOTION_LLAMA_DEMO_CFG  = os.getenv("EMOTION_LLAMA_DEMO_CFG",  os.path.join(EMOTION_LLAMA_DIR, "eval_configs", "demo.yaml"))
EMOTION_LLAMA_GRADIO_URL = os.getenv("EMOTION_LLAMA_GRADIO_URL", "http://127.0.0.1:7889")
EMOTION_LLAMA_VRAM_GB   = float(os.getenv("EMOTION_LLAMA_VRAM_GB", "0"))
EMOTION_LLAMA_QUANT     = os.getenv("EMOTION_LLAMA_QUANTIZATION", "fp32")

# ============================================================
# Device / VRAM detection
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _detect_vram_gb() -> float:
    """Return total VRAM in GB for the primary GPU, or 0.0 for CPU."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0

def _choose_quantization(vram_gb: float) -> str:
    """Choose quantization mode from VRAM size."""
    if vram_gb <= 0:
        return "fp32"
    elif vram_gb <= 10:
        return "int4"    # 4-bit NF4  — ≤10 GB (RTX 4060, 3060 etc.)
    elif vram_gb <= 16:
        return "int8"    # 8-bit      — 10–16 GB (RTX 3090, 4080 etc.)
    else:
        return "fp16"    # half prec  — >16 GB (RTX 4090 etc.)

# Re-detect at runtime in case .env is absent
if EMOTION_LLAMA_VRAM_GB == 0:
    EMOTION_LLAMA_VRAM_GB = _detect_vram_gb()
    EMOTION_LLAMA_QUANT   = _choose_quantization(EMOTION_LLAMA_VRAM_GB)

logger.info(f"PyTorch Device: {DEVICE} | VRAM: {EMOTION_LLAMA_VRAM_GB:.1f} GB | Quant: {EMOTION_LLAMA_QUANT}")

# ============================================================
# Global State
# ============================================================

mock_mode = False
processing_lock = asyncio.Lock()
_gradio_process = None   # handle to the Emotion-LLaMA subprocess

# ============================================================
# Emotion-LLaMA Server Management
# ============================================================

def _is_gradio_server_ready(url: str, timeout: float = 3.0) -> bool:
    """Return True if the Emotion-LLaMA Gradio server is responding."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _start_gradio_server() -> bool:
    """
    Launch Emotion-LLaMA's app_EmotionLlamaClient.py as a background subprocess.
    Streams its stdout/stderr to logger so crash errors are visible.
    Returns True if the server becomes ready within 5 minutes.
    """
    global _gradio_process

    app_py = os.path.join(EMOTION_LLAMA_DIR, "app_EmotionLlamaClient.py")
    if not os.path.exists(app_py):
        logger.error(f"❌ Emotion-LLaMA app not found: {app_py}")
        logger.error("   Please run: python setup_emotion_llama.py")
        return False

    logger.info(f"🚀 Starting Emotion-LLaMA Gradio server ({EMOTION_LLAMA_QUANT} mode)...")

    env = os.environ.copy()
    env["PYTHONPATH"] = EMOTION_LLAMA_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # Ensure CUDA is visible to subprocess
    env["CUDA_VISIBLE_DEVICES"] = "0"

    _gradio_process = subprocess.Popen(
        [sys.executable, app_py,
         "--cfg-path", EMOTION_LLAMA_DEMO_CFG],
        cwd=EMOTION_LLAMA_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # ── Stream subprocess output to logger in background thread ──────────
    import threading

    def _stream_output():
        for line in _gradio_process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[EmoLLaMA] {line}")

    output_thread = threading.Thread(target=_stream_output, daemon=True)
    output_thread.start()

    logger.info("⏳ Waiting for Emotion-LLaMA to load (may take 3–5 min on first run)...")
    deadline = time.time() + 300  # 5 minute max wait
    while time.time() < deadline:
        if _gradio_process.poll() is not None:
            # Process has exited — give thread a moment to flush remaining output
            time.sleep(2)
            exit_code = _gradio_process.returncode
            logger.error(f"❌ Emotion-LLaMA process exited unexpectedly (code={exit_code}).")
            logger.error("   Check the [EmoLLaMA] log lines above for the actual error.")
            return False
        if _is_gradio_server_ready(EMOTION_LLAMA_GRADIO_URL):
            logger.info("✅ Emotion-LLaMA Gradio server is ready!")
            return True
        time.sleep(3)

    logger.error("❌ Emotion-LLaMA did not start within 5 minutes.")
    return False



def init_emotion_llama() -> bool:
    """
    Check if Emotion-LLaMA Gradio server is already running.
    If not, start it. Returns True on success.
    """
    if _is_gradio_server_ready(EMOTION_LLAMA_GRADIO_URL):
        logger.info(f"✅ Emotion-LLaMA server already running at {EMOTION_LLAMA_GRADIO_URL}")
        return True

    return _start_gradio_server()


# ============================================================
# System Prompt & Response Parsing
# ============================================================

# Emotion-specific prompt injected into every Gradio API call
EMOTION_PROMPT = (
    "You are EmoAI, a helpful and empathetic AI assistant. "
    "Analyze the emotional state of the person based on their face, voice, and words. "
    "Follow these steps for EVERY response: "
    "1. Write [TRANSCRIPTION: <exact words the user said>]. "
    "2. Detect the primary emotion and write [Emotion] (e.g., [Happy], [Sad], [Angry]). "
    "3. Respond naturally — answer questions, give advice, or continue the conversation."
)

IMAGE_ONLY_PROMPT = (
    "Analyze the person's facial expression. "
    "Start with [Emotion] matching what you see. "
    "Give a short warm observation (1–2 sentences)."
)

TEXT_ONLY_PROMPT_TEMPLATE = (
    "The user says: \"{user_text}\". "
    "You are EmoAI. Detect emotion from their words and write [Emotion]. "
    "Then respond naturally and helpfully."
)


def extract_emotion_and_text(response: str):
    """Parse [Emotion] prefix and optional [TRANSCRIPTION: ...] from model output.

    Returns: (emotion, text, transcription, is_silent)
    """
    transcription = ""
    is_silent = False

    # --- Log raw response for debugging ---
    print(f"\n{'='*50}")
    print(f"🧠 RAW AI OUTPUT: {repr(response)}")
    print(f"{'='*50}\n")

    # 1. Extract transcription if present
    trans_match = re.search(r'\[TRANSCRIPTION:\s*(.*?)\]', response, re.DOTALL | re.IGNORECASE)
    if trans_match:
        raw_trans = trans_match.group(1).strip()
        # Safety: if transcription contains brackets it's likely the AI's own response leaking in
        if '[' in raw_trans or ']' in raw_trans:
            transcription = ""   # Invalid — discard
        else:
            transcription = raw_trans
        response = re.sub(r'\[TRANSCRIPTION:.*?\]', '', response, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. Check for explicit [SILENT] marker
    if "[SILENT]" in response.upper():
        is_silent = True
        response = re.sub(r'\[SILENT\]', '', response, flags=re.IGNORECASE).strip()

    # 3. Extract emotion bracket: [Happy], [Emotion: Happy], [emotion:sad], etc.
    match = re.search(
        r'\[(?:emotion:?\s*|detected_emotion:?\s*|feelings:?\s*)?(\w+)\]\s*(.*)',
        response, re.DOTALL | re.IGNORECASE
    )
    if match:
        emotion = match.group(1).lower()
        text = match.group(2).strip()
        if not text:
            text = "I'm here with you."
        return emotion, text, transcription, is_silent

    # 4. Fallback: any single bracketed word
    bracket_match = re.search(r'\[(\w+)\]', response)
    if bracket_match:
        emotion = bracket_match.group(1).lower()
        text = re.sub(r'\[.*?\]', '', response).strip()
        if not text:
            text = "I'm here with you."
        return emotion, text, transcription, is_silent

    # 5. Keyword fallback
    emotion_keywords = {
        'happy':     ['happy', 'joy', 'cheerful', 'smile', 'grin', 'laugh', 'delighted'],
        'sad':       ['sad', 'unhappy', 'down', 'depressed', 'melancholy', 'upset'],
        'angry':     ['angry', 'anger', 'furious', 'irritated', 'annoyed', 'frustrated'],
        'anxious':   ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'concerned'],
        'calm':      ['calm', 'relaxed', 'peaceful', 'serene', 'composed'],
        'surprised': ['surprised', 'shocked', 'astonished', 'amazed'],
        'tired':     ['tired', 'exhausted', 'fatigue', 'sleepy', 'weary'],
        'focused':   ['focused', 'concentrated', 'attentive', 'engaged'],
        'neutral':   ['neutral', 'normal', 'content', 'fine', 'okay'],
    }
    lower = response.lower()
    for emo, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in lower:
                return emo, response, transcription, is_silent

    return 'neutral', response if response else "I'm here with you.", transcription, is_silent


def correct_emotion_from_text(emotion: str, response_text: str) -> str:
    """Cross-check the emotion bracket against the AI's response text.

    If the response says 'you seem sad/down' but bracket says [Relaxed],
    override the emotion to match the response sentiment.
    """
    lower = response_text.lower()
    text_emotion_cues = {
        'sad':      ['sad', 'down', 'feeling low', 'tough time', 'sorry to hear',
                     'going through', 'rough', 'difficult', 'hard time', 'upset',
                     'heartbreaking', 'loss', 'miss ', 'missing', 'lonely', 'alone'],
        'happy':    ['glad', 'great to hear', 'wonderful', 'awesome', 'fantastic',
                     "that's great", 'happy for you', 'excited', 'celebrating',
                     'congratulations', 'proud of you', 'good news'],
        'anxious':  ['worried', 'anxious', 'stressed', 'nervous', 'overwhelming',
                     'pressure', 'scared', 'afraid', 'fear', 'panic', 'uneasy'],
        'angry':    ['frustrated', 'angry', 'furious', 'annoyed', 'unfair',
                     "that's wrong", 'outrageous', 'maddening'],
        'tired':    ['exhausted', 'tired', 'worn out', 'rest', 'sleep',
                     'drained', 'burnout', 'burn out'],
        'surprised':['surprised', 'wow', 'unexpected', 'shocking', 'amazing'],
    }
    for correct_emo, cues in text_emotion_cues.items():
        if correct_emo == emotion:
            continue   # Already matches
        for cue in cues:
            if cue in lower:
                logger.info(f"  🔄 Emotion corrected: [{emotion}] -> [{correct_emo}] (response says '{cue}')")
                return correct_emo
    return emotion


# ============================================================
# Emotion-LLaMA Gradio API call helpers
# ============================================================

def _call_emotion_llama_api(video_path: str, prompt: str) -> str:
    """
    POST to the Emotion-LLaMA Gradio server and return the response text.
    Falls back to an error string on failure.
    """
    try:
        payload = {"data": [video_path, prompt]}
        resp = requests.post(
            f"{EMOTION_LLAMA_GRADIO_URL}/api/predict/",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,   # model inference can be slow on first run
        )
        resp.raise_for_status()
        result = resp.json()
        return result["data"][0] if result.get("data") else ""
    except Exception as e:
        logger.error(f"  ❌ Gradio API call failed: {e}")
        return "[Neutral] I'm having trouble processing that right now."


def _save_image_from_b64(b64_str: str, path: str) -> bool:
    """Decode a base64 image string and save it as JPEG. Returns True on success."""
    try:
        img_data = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img.save(path, "JPEG")
        return True
    except Exception as e:
        logger.warning(f"  Image decode failed: {e}")
        return False


def _build_video_from_image_audio(image_path: str, audio_path: str, out_path: str) -> bool:
    """
    Stitch a JPEG frame + WAV audio into a short MP4 for Emotion-LLaMA.
    Returns True on success.
    """
    try:
        from moviepy.editor import ImageClip, AudioFileClip
        audio_clip  = AudioFileClip(audio_path)
        duration    = audio_clip.duration
        video_clip  = ImageClip(image_path, duration=duration)
        final       = video_clip.set_audio(audio_clip)
        final.write_videofile(out_path, fps=1, logger=None, audio_codec="aac")
        return True
    except Exception as e:
        logger.warning(f"  Video build failed: {e}")
        return False


def _build_silent_video_from_image(image_path: str, out_path: str, duration: float = 2.0) -> bool:
    """Create a short silent MP4 from a single image frame."""
    try:
        from moviepy.editor import ImageClip
        clip = ImageClip(image_path, duration=duration)
        clip.write_videofile(out_path, fps=1, logger=None, audio=False)
        return True
    except Exception as e:
        logger.warning(f"  Silent video build failed: {e}")
        return False


# ============================================================
# Processing Pipeline
# ============================================================

async def process_multimodal_input(data: dict, history: list):
    """Route incoming data to the correct handler with concurrency control."""
    msg_type = data.get('type', '')

    if mock_mode:
        return _process_mock(msg_type, data)

    async with processing_lock:
        if msg_type == 'multimodal':
            return await _process_multimodal_turn(data, history)
        return await _process_local(msg_type, data.get('data', ''), history)


async def _process_local(msg_type: str, msg_data: str, history: list):
    """Process single-mode inputs (image-only, audio-only, text-only)."""
    try:
        if msg_type == 'image':
            logger.info("📸 Received image frame, processing...")
            start = time.time()

            img_path   = os.path.join(tempfile.gettempdir(), "emoconvey_frame.jpg")
            video_path = os.path.join(tempfile.gettempdir(), "emoconvey_frame_video.mp4")

            if not _save_image_from_b64(msg_data, img_path):
                return json.dumps({"emotion": "neutral", "response": "Could not decode image.", "source": "video", "silent": True, "time": 0})

            # Build a silent 2-second video from the frame
            if not _build_silent_video_from_image(img_path, video_path):
                return json.dumps({"emotion": "neutral", "response": "Video build failed.", "source": "video", "silent": True, "time": 0})

            # Run inference — loop so we offload from async thread
            raw_response = await asyncio.get_event_loop().run_in_executor(
                None, _call_emotion_llama_api, video_path, IMAGE_ONLY_PROMPT
            )

            elapsed = time.time() - start
            emotion, response_text, _, _ = extract_emotion_and_text(raw_response)

            logger.info(f"✅ Frame processed in {elapsed:.1f}s -> emotion={emotion}")

            # Image-only: ALWAYS silent (don't speak, just update emotion badge)
            return json.dumps({
                "emotion": emotion,
                "response": response_text,
                "source": "video",
                "time": round(elapsed, 1),
                "silent": True
            })

        elif msg_type == 'audio':
            logger.info("🎤 Received audio data, processing...")
            start = time.time()

            audio_path = os.path.join(tempfile.gettempdir(), "emoconvey_audio.wav")
            audio_data = base64.b64decode(msg_data)
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # Normalize to 16kHz mono
            audio_duration = 0.0
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(audio_path, sr=16000)
                audio_duration = len(y) / sr
                sf.write(audio_path, y, sr)
                logger.info(f"  Audio: {audio_duration:.1f}s @ 16kHz")
            except Exception as ne:
                logger.warning(f"  Audio normalization skipped: {ne}")

            # Build a black-frame video with audio for Emotion-LLaMA
            black_img_path = os.path.join(tempfile.gettempdir(), "emoconvey_black_frame.jpg")
            video_path     = os.path.join(tempfile.gettempdir(), "emoconvey_audio_video.mp4")

            # Create a black placeholder image if needed
            if not os.path.exists(black_img_path):
                Image.new("RGB", (224, 224), color=(0, 0, 0)).save(black_img_path, "JPEG")

            built = _build_video_from_image_audio(black_img_path, audio_path, video_path)
            if not built:
                # Fallback: just a silent video
                _build_silent_video_from_image(black_img_path, video_path)

            raw_response = await asyncio.get_event_loop().run_in_executor(
                None, _call_emotion_llama_api, video_path, EMOTION_PROMPT
            )

            elapsed = time.time() - start
            emotion, response_text, transcription, _ = extract_emotion_and_text(raw_response)
            emotion = correct_emotion_from_text(emotion, response_text)

            logger.info(f"✅ Audio processed in {elapsed:.1f}s -> emotion={emotion}")

            return json.dumps({
                "emotion": emotion,
                "response": response_text,
                "source": "audio",
                "transcription": transcription,
                "time": round(elapsed, 1),
                "silent": False
            })

        elif msg_type == 'text':
            logger.info(f"💬 Received text: '{msg_data[:50]}' Processing...")
            start = time.time()

            history.append({"role": "user", "content": msg_data})

            # Build conversation-aware prompt from history
            history_context = ""
            if len(history) > 2:
                recent = history[-6:-1]  # Last 3 exchanges (excluding current)
                history_context = " ".join(
                    f"{'User' if h['role']=='user' else 'EmoAI'}: {h['content']}"
                    for h in recent
                )

            prompt = TEXT_ONLY_PROMPT_TEMPLATE.format(user_text=msg_data)
            if history_context:
                prompt = f"Conversation so far: {history_context}. " + prompt

            # Use a black-frame silent video as placeholder for text-only
            black_img_path = os.path.join(tempfile.gettempdir(), "emoconvey_black_frame.jpg")
            video_path     = os.path.join(tempfile.gettempdir(), "emoconvey_text_video.mp4")
            if not os.path.exists(black_img_path):
                Image.new("RGB", (224, 224), color=(0, 0, 0)).save(black_img_path, "JPEG")
            _build_silent_video_from_image(black_img_path, video_path)

            raw_response = await asyncio.get_event_loop().run_in_executor(
                None, _call_emotion_llama_api, video_path, prompt
            )

            elapsed = time.time() - start
            emotion, response_text, _, _ = extract_emotion_and_text(raw_response)
            history.append({"role": "assistant", "content": response_text})

            logger.info(f"✅ Chat responded in {elapsed:.1f}s -> emotion={emotion}")

            return json.dumps({
                "emotion": emotion,
                "response": response_text,
                "source": "chat",
                "time": round(elapsed, 1)
            })

    except Exception as e:
        logger.error(f"Model Error: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "emotion": "neutral",
            "response": "Sorry, I had trouble processing that.",
            "source": msg_type,
            "time": 0
        })


async def _process_multimodal_turn(data: dict, history: list):
    """Process a combined audio + optional image turn WITH conversation memory."""
    try:
        start = time.time()

        audio_b64  = data.get('audio', '')
        image_b64  = data.get('image', '')
        is_cam_on  = data.get('isCamOn', False)

        if not audio_b64:
            return json.dumps({"emotion": "neutral", "response": "No audio received.", "source": "error", "silent": True})

        # --- Save and normalize audio ---
        audio_path = os.path.join(tempfile.gettempdir(), "emoconvey_turn_audio.wav")
        audio_data = base64.b64decode(audio_b64)
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        audio_duration = 0.0
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(y) / sr
            sf.write(audio_path, y, sr)
            logger.info(f"  Audio: {audio_duration:.1f}s @ 16kHz")
        except Exception as ne:
            logger.warning(f"  Audio normalization skipped: {ne}")

        # --- Save and build video ---
        has_image  = is_cam_on and image_b64 and len(image_b64) > 100
        video_path = os.path.join(tempfile.gettempdir(), "emoconvey_turn_video.mp4")

        if has_image:
            image_path = os.path.join(tempfile.gettempdir(), "emoconvey_turn_image.jpg")
            _save_image_from_b64(image_b64, image_path)
            built = _build_video_from_image_audio(image_path, audio_path, video_path)
        else:
            # Audio only — use a black placeholder frame
            black_img_path = os.path.join(tempfile.gettempdir(), "emoconvey_black_frame.jpg")
            if not os.path.exists(black_img_path):
                Image.new("RGB", (224, 224), color=(0, 0, 0)).save(black_img_path, "JPEG")
            built = _build_video_from_image_audio(black_img_path, audio_path, video_path)

        if not built:
            logger.warning("  Video build failed — using silent fallback")

        # --- Build prompt with anti-repetition & history ---
        last_response = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
        prompt = EMOTION_PROMPT
        if has_image:
            prompt += " Look at the person's face AND listen to their voice."
        if last_response:
            prompt += f" IMPORTANT: You already said \"{last_response[:80]}\". Say something DIFFERENT this time."

        # --- Build & log conversation context ---
        recent_history = history[-6:] if len(history) > 6 else history
        logger.info(f"🎭 Processing turn #{len(history)//2 + 1} (Audio={audio_duration:.1f}s | Image={'Yes' if has_image else 'No'} | History={len(recent_history)} msgs)...")
        logger.info(f"📝 Prompt: {prompt[:80]}...")
        if len(history) > 0:
            logger.info(f"📜 History Context: {history[-2:]}")

        # --- Call Emotion-LLaMA ---
        raw_response = await asyncio.get_event_loop().run_in_executor(
            None, _call_emotion_llama_api, video_path, prompt
        )

        elapsed = time.time() - start
        emotion, response_text, transcription, _ = extract_emotion_and_text(raw_response)

        logger.info(f"🗣️ Extracted Transcription: '{transcription}'")

        # Cross-check: if text says "you seem down" but bracket says [Relaxed], fix it
        emotion = correct_emotion_from_text(emotion, response_text)

        # --- Save to history (text only — no blobs) ---
        user_summary = transcription if transcription else "(user spoke via audio)"
        history.append({"role": "user", "content": user_summary})
        history.append({"role": "assistant", "content": response_text})

        # Trim history to prevent memory bloat (keep last 10 exchanges = 20 msgs)
        if len(history) > 20:
            history[:] = history[-20:]

        logger.info(f"✅ Turn done in {elapsed:.1f}s -> {emotion} | \"{response_text[:60]}\"")

        return json.dumps({
            "emotion": emotion,
            "response": response_text,
            "transcription": transcription,
            "source": "multimodal",
            "time": round(elapsed, 1),
            "silent": False
        })

    except Exception as e:
        logger.error(f"Multimodal Turn Error: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "emotion": "neutral",
            "response": "Sorry, I had trouble processing that.",
            "source": "error",
            "silent": False
        })


# ============================================================
# Mock Mode (UI testing without model)
# ============================================================

def _process_mock(msg_type: str, data: dict):
    """Mock responses for UI testing when the model is unavailable."""
    msg_data = data.get('data', '')
    if msg_type == 'image' or msg_type == 'multimodal':
        return json.dumps({
            "emotion": "happy",
            "response": "You look like you're in a great mood!",
            "source": "mock",
            "time": 0.5,
            "silent": msg_type == 'image'
        })
    elif msg_type == 'audio':
        return json.dumps({
            "emotion": "calm",
            "response": "You sound very relaxed.",
            "source": "mock",
            "time": 0.3
        })
    elif msg_type == 'text':
        return json.dumps({
            "emotion": "neutral",
            "response": f"I understand: {msg_data}",
            "source": "mock",
            "time": 0.1
        })
    return ""


# ============================================================
# WebSocket Manager
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# ============================================================
# FastAPI App + Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mock_mode

    logger.info("🚀 Starting EmoConvey Backend (Emotion-LLaMA)...")
    logger.info(f"   VRAM: {EMOTION_LLAMA_VRAM_GB:.1f} GB | Quantization: {EMOTION_LLAMA_QUANT}")

    if not init_emotion_llama():
        logger.warning("⚠️ Emotion-LLaMA unavailable. Running in MOCK mode.")
        logger.warning("   To fix: run `python setup_emotion_llama.py` first.")
        mock_mode = True

    # --- Start Ngrok Tunnel ---
    try:
        from pyngrok import ngrok

        if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_TOKEN_HERE":
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        public_url = ngrok.connect(8000).public_url
        ws_url = public_url.replace("https", "wss").replace("http", "ws")

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  ✅ Ngrok Tunnel Active!                             ║")
        logger.info("║                                                      ║")
        logger.info("║  PASTE THIS IN YOUR APP'S PROFILE SCREEN:            ║")
        logger.info(f"║  👉 {ws_url}/ws/live")
        logger.info("║                                                      ║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        logger.info("")
    except ImportError:
        logger.info("ℹ️ pyngrok not installed. Using localhost only.")
        logger.info("👉 Local URL: ws://10.0.2.2:8000/ws/live (Android Emulator)")
    except Exception as e:
        logger.warning(f"⚠️ Ngrok Error: {e}")
        logger.info("👉 Local URL: ws://10.0.2.2:8000/ws/live (Android Emulator)")

    logger.info(f"✅ Server Ready! Device: {DEVICE} | Mock: {mock_mode} | Model: Emotion-LLaMA")

    yield

    logger.info("Shutting down...")
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass

    # Stop the Emotion-LLaMA subprocess if we started it
    if _gradio_process and _gradio_process.poll() is None:
        logger.info("Stopping Emotion-LLaMA process...")
        _gradio_process.terminate()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Routes
# ============================================================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_history = []
    logger.info("Client Connected ✅")

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            # Ignore heartbeat pings
            if data.get('type') == 'ping':
                continue

            logger.info(f"📥 Received message type: {data.get('type', 'unknown')}")
            ai_response = await process_multimodal_input(data, session_history)
            if ai_response:
                await manager.send_personal_message(ai_response, websocket)
                logger.info(f"📤 Sent response back to client")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client Disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        import traceback
        traceback.print_exc()
        manager.disconnect(websocket)


@app.get("/")
def read_root():
    return {
        "app": "EmoConvey AI Backend",
        "model": "Emotion-LLaMA (LLaMA-2-7B)",
        "device": DEVICE,
        "vram_gb": EMOTION_LLAMA_VRAM_GB,
        "quantization": EMOTION_LLAMA_QUANT,
        "mock": mock_mode,
        "status": "Active",
        "gradio_url": EMOTION_LLAMA_GRADIO_URL,
    }


@app.get("/health")
def health_check():
    gradio_ok = _is_gradio_server_ready(EMOTION_LLAMA_GRADIO_URL)
    return {
        "status": "ok",
        "device": DEVICE,
        "gradio_server": "ready" if gradio_ok else "offline",
        "mock_mode": mock_mode,
    }


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EmoConvey AI Backend")
    print("  Powered by Emotion-LLaMA (NeurIPS 2024)")
    print("=" * 60)
    print(f"  Model     : Emotion-LLaMA / LLaMA-2-7B-chat")
    print(f"  Device    : {DEVICE}")
    print(f"  VRAM      : {EMOTION_LLAMA_VRAM_GB:.1f} GB")
    print(f"  Quant     : {EMOTION_LLAMA_QUANT}")
    print(f"  Gradio    : {EMOTION_LLAMA_GRADIO_URL}")
    print(f"  Ngrok     : {'Configured' if NGROK_AUTH_TOKEN != 'YOUR_NGROK_TOKEN_HERE' else 'Not Set'}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
