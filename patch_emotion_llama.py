import os
import sys

def patch_file(filepath, replacements, create_if_missing=False):
    """Utility to perform string replacements safely."""
    if not os.path.exists(filepath):
        if create_if_missing:
            print(f"📁 Creating {filepath}...")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("")
        else:
            print(f"❌ Skipping {filepath} (Not found)")
            return False
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    changed = False
    for old_str, new_str in replacements:
        if old_str in content:
            # Only replace if new_str isn't already there (prevents double patching)
            if new_str in content and old_str != "":
                 continue
            content = content.replace(old_str, new_str)
            changed = True
        elif old_str == "" and len(content) == 0: # Handle creating new file content
            content = new_str
            changed = True
        else:
            # Silent skip if string already looks patched
            if new_str[:50] in content:
                continue
            print(f"  ⚠️ Warning: Could not find target string in {filepath}:\n    '{old_str[:50]}...'")

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully patched: {filepath}")
    else:
        print(f"⚡ No changes needed for: {filepath}")
    return changed

def run_all_patches():
    print("🚀 Starting EmoConvey Core Stabilization Patch Script (Windows/UTF-8/Multimodal)...\n")
    
    # 1. Patch modeling_llama.py (Fix transformers v4.36+ compatibility)
    # We replace the entire file with a minimal working causal model to avoid docstring crashes
    llama_path = os.path.join("Emotion-LLaMA", "minigpt4", "models", "modeling_llama.py")
    if os.path.exists(llama_path):
        with open(llama_path, 'r', encoding='utf-8') as f:
            if "LlamaForCausalLMOrig" not in f.read():
                print(f"📦 Standardizing {llama_path} for Transformers compatibility...")
                with open(llama_path, 'w', encoding='utf-8') as f:
                    f.write('from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig\n')
                    f.write('from transformers.modeling_outputs import CausalLMOutputWithPast\n\n')
                    f.write('class LlamaForCausalLM(LlamaForCausalLMOrig):\n')
                    f.write('    def forward(self, *args, **kwargs):\n')
                    f.write('        return super().forward(*args, **kwargs)\n')

    # 2. Patch Qformer.py (Missing PyTorch utils)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "Qformer.py"),
        [
            (
                "from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer",
                "def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):\n    return forward_fn(*input_tensors)\n\ndef find_pruneable_heads_and_indices(*args, **kwargs):\n    return set(), []\n\ndef prune_linear_layer(*args, **kwargs):\n    pass"
            )
        ]
    )

    # 3. Patch base_model.py (NF4 4-bit Quantization for Windows 8GB VRAM)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "base_model.py"),
        [
            (
                "from peft import (",
                "from transformers import BitsAndBytesConfig\nfrom peft import ("
            ),
            (
                "load_in_8bit=True,",
                "load_in_4bit=True,\n                bnb_4bit_use_double_quant=True,\n                bnb_4bit_quant_type=\"nf4\",\n                bnb_4bit_compute_dtype=torch.float16,"
            ),
            (
                "llama_model = prepare_model_for_int8_training(llama_model)",
                "# llama_model = prepare_model_for_int8_training(llama_model)"
            )
        ]
    )

    # 4. Patch minigpt_base.py (Fix FP32/FP16 crash and Pure-Text IndexError)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "minigpt_base.py"),
        [
            (
                "device = img_list[0].device",
                "device = img_list[0].device if len(img_list) > 0 else self.llama_model.device"
            ),
            (
                "mixed_embs = torch.cat(mixed_embs, dim=1)\n        return mixed_embs",
                "mixed_embs = torch.cat(mixed_embs, dim=1)\n        mixed_embs = mixed_embs.half()\n        return mixed_embs"
            )
        ]
    )

    # 5. Patch conversation.py (Silent Video Crash)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "conversation", "conversation.py"),
        [
            (
                "audio = video.audio",
                "audio = video.audio\n    if audio is None:\n        import numpy as np\n        return np.zeros((16000,), dtype=np.float32), 16000"
            )
        ]
    )

    # 6. Patch app_EmotionLlamaClient.py (Windows UTF-8 Support & Multimodal Bypass)
    patch_file(
        os.path.join("Emotion-LLaMA", "app_EmotionLlamaClient.py"),
        [
            (
                "import gradio as gr",
                "import gradio as gr\nimport sys\nimport io\n\nif sys.platform == \"win32\":\n    try:\n        import io\n        if hasattr(sys.stdout, 'buffer'):\n            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n        if hasattr(sys.stderr, 'buffer'):\n            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')\n    except Exception: pass"
            ),
            (
                "iface.queue().launch(server_name=\"0.0.0.0\", server_port=7889, share=False)",
                "iface.queue().launch(server_name=\"127.0.0.1\", server_port=7889, share=False)"
            ),
            (
                "def process_video_question(video_path, question):\n    if not os.path.exists(video_path):\n        return \"Error: Video file does not exist.\"",
                "def process_video_question(video_path, question):\n    if video_path != \"None\" and not os.path.exists(video_path):\n        return \"Error: Video file does not exist.\""
            ),
            (
                "chat_state.append_message(chat_state.roles[0], \"<video><VideoHere></video> <feature><FeatureHere></feature>\")\n    img_list = []\n    img_list.append(video_path)",
                "img_list = []\n    if video_path and video_path != \"None\" and os.path.exists(video_path):\n        chat_state.append_message(chat_state.roles[0], \"<video><VideoHere></video> <feature><FeatureHere></feature>\")\n        img_list.append(video_path)"
            )
        ]
    )

    # 7. Patch server.py (UTF-8 Pipeline, Prompt Strictness & Robust Sanitization)
    patch_file(
        "server.py",
        [
            (
                "import sys\nimport tempfile",
                "import sys\nimport tempfile\n\nif sys.platform == \"win32\":\n    try:\n        if hasattr(sys.stdout, 'reconfigure'):\n            sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n        if hasattr(sys.stderr, 'reconfigure'):\n            sys.stderr.reconfigure(encoding='utf-8', errors='replace')\n    except Exception: pass"
            ),
            (
                "env[\"CUDA_VISIBLE_DEVICES\"] = \"0\"",
                "env[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n    env[\"PYTHONUTF8\"] = \"1\"\n    env[\"PYTHONIOENCODING\"] = \"utf-8\""
            ),
            (
                "text=True,",
                "text=True, encoding='utf-8', errors='replace',"
            ),
            (
                "EMOTION_PROMPT = (\n    \"You are EmoAI, an empathetic assistant. \"",
                "EMOTION_PROMPT = (\n    \"You are EmoAI, a direct and empathetic assistant. \"\n    \"STRICT INSTRUCTION: DO NOT include any introductions, preambles, or conversational fillers like 'Sure!', 'I can help', or 'Based on the image'. \"\n    \"MANDATORY FORMAT (Start IMMEDIATELY with this):\\n\""
            ),
            (
                "def extract_emotion_and_text(response: str):",
                "def extract_emotion_and_text(response: str):\n    \"\"\"Parse [Emotion] prefix and optional [TRANSCRIPTION: ...] from model output.\n\n    Returns: (emotion, text, transcription, is_silent)\n    \"\"\"\n    transcription = \"\"\n    is_silent = False\n\n    # --- Log raw response for debugging ---\n    print(f\"\\n{'='*50}\")\n    print(f\"🧠 RAW AI OUTPUT: {repr(response)}\")\n    print(f\"{'='*50}\\n\")\n\n    # 1. Extract transcription if present\n    trans_match = re.search(r'\\[TRANSCRIPTION:\\s*(.*?)\\]', response, re.DOTALL | re.IGNORECASE)\n    if trans_match:\n        raw_trans = trans_match.group(1).strip()\n        if '[' in raw_trans or ']' in raw_trans:\n            transcription = \"\"   # Invalid\n        else:\n            transcription = raw_trans\n        response = re.sub(r'\\[TRANSCRIPTION:.*?\\]', '', response, flags=re.DOTALL | re.IGNORECASE).strip()\n\n    if \"[SILENT]\" in response.upper():\n        is_silent = True\n        response = re.sub(r'\\[SILENT\\]', '', response, flags=re.IGNORECASE).strip()\n\n    emotion = \"neutral\"\n    emo_match = re.search(\n        r'\\[(\\w+)\\]|(?i)detected:?\\s*(\\w+)|(?i)mood:?\\s*(\\w+)|(?i)emotion:?\\s*(\\w+)', \n        response\n    )\n    if emo_match:\n        found = emo_match.group(1) or emo_match.group(2) or emo_match.group(3) or emo_match.group(4)\n        emotion = found.lower()\n    else:\n        emotion_keywords = {\n            'happy': ['happy', 'joy', 'cheerful', 'smile', 'grin', 'laugh'],\n            'sad': ['sad', 'unhappy', 'down', 'depressed', 'upset'],\n            'angry': ['angry', 'anger', 'furious', 'annoyed', 'frustrated'],\n            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense'],\n            'calm': ['calm', 'relaxed', 'peaceful', 'serene'],\n            'surprised': ['surprised', 'shocked', 'astonished'],\n            'tired': ['tired', 'exhausted', 'fatigue', 'sleepy'],\n        }\n        lower_resp = response.lower()\n        for emo_key, cues in emotion_keywords.items():\n            if any(cue in lower_resp for cue in cues):\n                emotion = emo_key\n                break\n\n    text = response\n    for sp in [r'\\[\\w+\\]', r'(?i)^\\d+\\.\\s*(Response:?|Text:?)', r'(?i)^(Response:?|Text:?)\\s*']:\n        match = re.search(sp, text, re.MULTILINE)\n        if match:\n            text = text[match.end():].strip()\n            break\n\n    leak_patterns = [\n        r'^(Sure!|Certainly!|Of course!|I\\'d be happy to help!|Here\\'s my response:?|Here is my response:?|I can continue the conversation.*?\\.)',\n        r'(?i)^Based on (the image|the video|your|what).*?(\\.\\s*|\\:\\s*)',\n        r'(?i)^I (can|see|detect|notice|observe).*?(\\.\\s*|\\:\\s*)',\n        r'^\\d+\\.\\s*(Emotion:?|Response:?)?\\s*',\n        r'\\[.*?\\]',\n        r'(?i)detected:?\\s*\\w+',\n        r'(?i)detected emotion:?\\s*\\w+',\n        r'(?i)emotion:?\\s*\\w+',\n        r'(?i)mood:?\\s*\\w+',\n        r'^(EmoAI|Assistant|AI|System):\\s*',\n    ]\n    for _ in range(2):\n        for pattern in leak_patterns:\n            text = re.sub(r'^\\d+\\.\\s*', '', text, flags=re.MULTILINE).strip()\n            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL).strip()\n    \n    emoji_map = {':heartbreak:': '💔', ':empathy:': '🤝', ':happy:': '😊', ':sad:': '😢'}\n    for token, emoji in emoji_map.items():\n        text = text.replace(token, emoji)\n    \n    text = re.sub(r':[a-z_]+:', '', text).strip()\n    text = re.sub(r'\\s+', ' ', text).strip()\n    if not text: text = \"I'm here with you.\"\n    return emotion, text, transcription, is_silent\n\n# --- Keep original logic above, placeholder for patch matching ---"
            ),
            (
                "video_path = os.path.join(tempfile.gettempdir(), \"emoconvey_text_video.mp4\")\n            if not os.path.exists(black_img_path):\n                Image.new(\"RGB\", (224, 224), color=(0, 0, 0)).save(black_img_path, \"JPEG\")\n            _build_silent_video_from_image(black_img_path, video_path)",
                "video_path = \"None\""
            )
        ]
    )

    print("\n🎉 All stable patches applied! You can now run: python server.py")

if __name__ == "__main__":
    run_all_patches()
