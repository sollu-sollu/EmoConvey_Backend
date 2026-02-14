import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import base64
import logging
import os
import tempfile
import asyncio
from io import BytesIO
from PIL import Image
import torch
import re
import time

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmoConveyBackend")

# ============================================================
# CONFIGURATION
# ============================================================

LOCAL_MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"

# ============================================================
# Device Setup
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"PyTorch Device: {DEVICE} (CUDA Available: {torch.cuda.is_available()})")

# Global State
model = None
processor = None
mock_mode = False
processing_lock = asyncio.Lock()

# ============================================================
# Model Initialization
# ============================================================

def init_local_model():
    """Load Qwen2.5-Omni-3B with 4-bit quantization (CUDA) or full precision (CPU)."""
    global model, processor, DEVICE
    
    logger.info(f"Loading {LOCAL_MODEL_ID} on {DEVICE}...")
    
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        if DEVICE == "cuda":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            logger.info("  Using 4-bit NF4 quantization (saves ~60% memory)...")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                LOCAL_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config
            )
        else:
            logger.info("  No CUDA detected. Loading full precision on CPU (slower)...")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                LOCAL_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        model.disable_talker()
        processor = Qwen2_5OmniProcessor.from_pretrained(LOCAL_MODEL_ID)
        
        logger.info(f"✅ {LOCAL_MODEL_ID} loaded on {DEVICE} (4-bit: {DEVICE == 'cuda'})!")
        return True
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# ============================================================
# System Prompt & Response Parsing
# ============================================================

SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {"type": "text", "text": (
            "You are EmoAI, a helpful and empathetic AI assistant. "
            "You can see and hear the user. "
            "RULES: "
            "Follow these steps for EVERY response: "
            "1. Write [TRANSCRIPTION: <exact words user said>]. "
            "2. Detect emotion and write [Emotion] (e.g., [Happy], [Sad]). "
            "3. Write your response naturally. Answer questions or chat. "
            "   Example: [TRANSCRIPTION: Hello there] [Happy] Hi! How are you?"
        )}
    ],
}

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
        # Safety check: if transcription contains brackets [], it's likely the AI's response leaked in
        if '[' in raw_trans or ']' in raw_trans:
            transcription = "" # Invalid, discard
        else:
            transcription = raw_trans
            
        response = re.sub(r'\[TRANSCRIPTION:.*?\]', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 2. Check for explicit [SILENT] marker
    if "[SILENT]" in response.upper():
        is_silent = True
        response = re.sub(r'\[SILENT\]', '', response, flags=re.IGNORECASE).strip()
    
    # 3. Extract emotion bracket: [Happy], [Emotion: Happy], [emotion:sad], etc.
    match = re.search(
        r'\[(?:emotion:?\s*|detected_emotion:?\s*|feelings:?\s*)?([\w]+)\]\s*(.*)',
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
        'happy': ['happy', 'joy', 'cheerful', 'smile', 'grin', 'laugh', 'delighted'],
        'sad': ['sad', 'unhappy', 'down', 'depressed', 'melancholy', 'upset'],
        'angry': ['angry', 'anger', 'furious', 'irritated', 'annoyed', 'frustrated'],
        'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'concerned'],
        'calm': ['calm', 'relaxed', 'peaceful', 'serene', 'composed'],
        'surprised': ['surprised', 'shocked', 'astonished', 'amazed'],
        'tired': ['tired', 'exhausted', 'fatigue', 'sleepy', 'weary'],
        'focused': ['focused', 'concentrated', 'attentive', 'engaged'],
        'neutral': ['neutral', 'normal', 'content', 'fine', 'okay'],
    }
    
    lower = response.lower()
    for emo, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in lower:
                return emo, response, transcription, is_silent
    
    return 'neutral', response if response else "I'm here with you.", transcription, is_silent


def correct_emotion_from_text(emotion: str, response_text: str) -> str:
    """Cross-check emotion bracket against the AI's own response text.
    
    If the response says 'you seem sad/down/upset' but emotion is 'relaxed',
    override the emotion to match the response sentiment.
    """
    lower = response_text.lower()
    
    # Map of emotional cues in the response text -> correct emotion
    text_emotion_cues = {
        'sad': ['sad', 'down', 'feeling low', 'tough time', 'sorry to hear', 
                'going through', 'rough', 'difficult', 'hard time', 'upset',
                'heartbreaking', 'loss', 'miss ', 'missing', 'lonely', 'alone'],
        'happy': ['glad', 'great to hear', 'wonderful', 'awesome', 'fantastic',
                  'that\'s great', 'happy for you', 'excited', 'celebrating',
                  'congratulations', 'proud of you', 'good news'],
        'anxious': ['worried', 'anxious', 'stressed', 'nervous', 'overwhelming',
                    'pressure', 'scared', 'afraid', 'fear', 'panic', 'uneasy'],
        'angry': ['frustrated', 'angry', 'furious', 'annoyed', 'unfair',
                  'that\'s wrong', 'outrageous', 'maddening'],
        'tired': ['exhausted', 'tired', 'worn out', 'rest', 'sleep',
                  'drained', 'burnout', 'burn out'],
        'surprised': ['surprised', 'wow', 'unexpected', 'shocking', 'amazing'],
    }
    
    # Check if response text strongly implies an emotion different from the bracket
    for correct_emo, cues in text_emotion_cues.items():
        if correct_emo == emotion:
            continue  # Already matches
        for cue in cues:
            if cue in lower:
                # The response text implies a different emotion than the bracket
                logger.info(f"  🔄 Emotion corrected: [{emotion}] -> [{correct_emo}] (response says '{cue}')")
                return correct_emo
    
    return emotion

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
    global model, processor
    
    try:
        from qwen_omni_utils import process_mm_info
        
        if msg_type == 'image':
            logger.info("📸 Received image frame, processing...")
            start = time.time()
            
            image_data = base64.b64decode(msg_data)
            tmp_path = os.path.join(tempfile.gettempdir(), "emoconvey_frame.jpg")
            with open(tmp_path, "wb") as f:
                f.write(image_data)
            
            conversation = [
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": "Detect the emotion on this person's face. Start with [Emotion]. Give a short warm response."}
                    ],
                },
            ]
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(model.device).to(model.dtype)
            
            text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=100)
            input_len = inputs['input_ids'].shape[1]
            generated_ids = text_ids[:, input_len:]
            result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            elapsed = time.time() - start
            raw_response = result[0].strip() if result else "Could not analyze the image."
            emotion, response_text, transcription, _ = extract_emotion_and_text(raw_response)
            
            logger.info(f"✅ Frame processed in {elapsed:.1f}s -> emotion={emotion}")
            
            # Image-only: ALWAYS silent (don't speak, just update badge)
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
            
            audio_data = base64.b64decode(msg_data)
            tmp_path = os.path.join(tempfile.gettempdir(), "emoconvey_audio.wav")
            with open(tmp_path, "wb") as f:
                f.write(audio_data)
            
            conversation = [
                SYSTEM_PROMPT,
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": tmp_path},
                        {"type": "text", "text": "Step 1: Write [TRANSCRIPTION: ...]. Step 2: Write [Emotion]. Step 3: Respond naturally."}
                    ],
                },
            ]
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(model.device).to(model.dtype)
            
            text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=150)
            input_len = inputs['input_ids'].shape[1]
            generated_ids = text_ids[:, input_len:]
            result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            elapsed = time.time() - start
            raw_response = result[0].strip() if result else "Could not analyze the audio."
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
            
            conversation = [SYSTEM_PROMPT]
            for h in history:
                conversation.append({"role": h["role"], "content": h["content"]})
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(model.device).to(model.dtype)
            
            text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=256)
            input_len = inputs['input_ids'].shape[1]
            generated_ids = text_ids[:, input_len:]
            result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            elapsed = time.time() - start
            raw_response = result[0].strip() if result else "I'm not sure how to respond to that."
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
            "response": f"Sorry, I had trouble processing that.",
            "source": msg_type,
            "time": 0
        })


async def _process_multimodal_turn(data: dict, history: list):
    """Process a combined audio + optional image turn WITH conversation memory."""
    global model, processor
    
    try:
        from qwen_omni_utils import process_mm_info
        start = time.time()
        
        audio_b64 = data.get('audio', '')
        image_b64 = data.get('image', '')
        is_cam_on = data.get('isCamOn', False)
        
        if not audio_b64:
            return json.dumps({"emotion": "neutral", "response": "No audio received.", "source": "error", "silent": True})
        
        # Save audio
        audio_data = base64.b64decode(audio_b64)
        audio_path = os.path.join(tempfile.gettempdir(), "emoconvey_turn_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        # Normalize audio to 16kHz mono
        audio_duration = 0
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(y) / sr
            sf.write(audio_path, y, sr)
            logger.info(f"  Audio: {audio_duration:.1f}s @ 16kHz")
        except Exception as ne:
            logger.warning(f"  Audio normalization skipped: {ne}")
        
        # Build multimodal content for this turn
        content = [{"type": "audio", "audio": audio_path}]
        
        has_image = is_cam_on and image_b64 and len(image_b64) > 100
        if has_image:
            image_path = os.path.join(tempfile.gettempdir(), "emoconvey_turn_image.jpg")
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            content.append({"type": "image", "image": image_path})
        
        # Build prompt — include anti-repetition if history exists
        last_response = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
        
        if has_image:
            prompt = (
                "Listen to the user and look at them. "
                "Step 1: Write [TRANSCRIPTION: ...]. "
                "Step 2: Write [Emotion] (prioritize words/tone). "
                "Step 3: Respond naturally."
            )
        else:
            prompt = (
                "Listen to the user. "
                "Step 1: Write [TRANSCRIPTION: ...]. "
                "Step 2: Write [Emotion] based on tone. "
                "Step 3: Respond naturally."
            )
        
        # Anti-repetition nudge
        if last_response:
            prompt += f" IMPORTANT: You already said \"{last_response[:80]}\". Say something DIFFERENT this time."
        
        content.append({"type": "text", "text": prompt})
        
        # ---- BUILD CONVERSATION WITH HISTORY ----
        # Include past text exchanges so the model has memory
        conversation = [SYSTEM_PROMPT]
        
        # Add recent history (keep last 6 exchanges max to avoid token overflow)
        recent_history = history[-6:] if len(history) > 6 else history
        for h in recent_history:
            conversation.append({"role": h["role"], "content": h["content"]})
        
        # Add current multimodal turn
        conversation.append({"role": "user", "content": content})
        
        logger.info(f"🎭 Processing turn #{len(history)//2 + 1} (Audio={audio_duration:.1f}s | Image={'Yes' if has_image else 'No'} | History={len(recent_history)} msgs)...")
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)
        
        # --- DEBUG: Log what we are sending ---
        logger.info(f"📝 sending {len(conversation)} msgs. Last User Prompt: {prompt[:50]}...")
        if len(history) > 0:
            logger.info(f"📜 History Context: {history[-2:]}") # Print last 2 turns
        
        text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=256)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = text_ids[:, input_len:]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        elapsed = time.time() - start
        raw_response = result[0].strip() if result else "I'm listening."
        emotion, response_text, transcription, _ = extract_emotion_and_text(raw_response)
        
        logger.info(f"🗣️ Extracted Transcription: '{transcription}'")
        
        # Cross-check: if the AI says "you seem down" but bracket says [Relaxed], fix it
        emotion = correct_emotion_from_text(emotion, response_text)
        
        # ---- SAVE TO HISTORY (text-only, no blobs) ----
        # Save a text summary of what the user said (from the model's understanding)
        user_summary = transcription if transcription else "(user spoke via audio)"
        history.append({"role": "user", "content": user_summary})
        history.append({"role": "assistant", "content": response_text})
        
        # Trim history to prevent memory bloat (keep last 10 exchanges = 20 messages)
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



def _process_mock(msg_type: str, data: dict):
    """Mock responses for UI testing."""
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
    
    logger.info("🚀 Starting EmoConvey Backend...")
    
    if not init_local_model():
        logger.warning("⚠️ Model failed to load. Running in MOCK mode.")
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
    
    logger.info(f"✅ Server Ready! Device: {DEVICE} | Mock: {mock_mode}")
    
    yield
    
    logger.info("Shutting down...")
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except:
        pass

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
    return {"app": "EmoConvey AI Backend", "model": LOCAL_MODEL_ID, "device": DEVICE, "mock": mock_mode, "status": "Active"}

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EmoConvey AI Backend")
    print("  Powered by Qwen2.5-Omni-3B")
    print("=" * 50)
    print(f"  Model  : {LOCAL_MODEL_ID}")
    print(f"  Device : {DEVICE}")
    print(f"  Ngrok  : {'Configured' if NGROK_AUTH_TOKEN != 'YOUR_NGROK_TOKEN_HERE' else 'Not Set (edit line 27)'}")
    print("=" * 50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
