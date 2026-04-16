"""
setup_emotion_llama.py
======================
One-time setup script for Emotion-LLaMA.
Works on both Windows and Ubuntu/Linux.
Run ONCE before starting the server.

    python setup_emotion_llama.py

Steps:
  1. Detect OS and GPU VRAM → choose quantization mode
  2. Clone the Emotion-LLaMA repo (if not already done)
  3. Install all required Python packages (OS-aware)
  4. Download LLaMA-2-7B-chat-hf from HuggingFace (~13 GB)
  5. Download HuBERT-large audio encoder from HuggingFace (~1.3 GB)
  6. Download Emotion-LLaMA fine-tuned checkpoint from Google Drive (~1.6 GB)
  7. Patch model configs with correct paths and quantization
  8. Write .env file for server.py
"""

import os
import sys
import platform
import subprocess
import re

# ============================================================
# OS Detection
# ============================================================
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ============================================================
# PATHS
# Change EMOTION_LLAMA_DIR if you want a different install location.
# All other paths are derived from it automatically.
# ============================================================
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
EMOTION_LLAMA_DIR  = os.path.join(SCRIPT_DIR, "Emotion-LLaMA")
CHECKPOINTS_DIR    = os.path.join(EMOTION_LLAMA_DIR, "checkpoints")
LLAMA_DIR          = os.path.join(CHECKPOINTS_DIR, "Llama-2-7b-chat-hf")
HUBERT_DIR         = os.path.join(CHECKPOINTS_DIR, "transformer", "chinese-hubert-large")
CKPT_DIR           = os.path.join(CHECKPOINTS_DIR, "save_checkpoint")
CKPT_NAME          = "Emotion_LLaMA.pth"

# Google Drive file ID for the Emotion-LLaMA demo checkpoint
GDRIVE_ID = "1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY"

# HuggingFace model repos
LLAMA_REPO  = "meta-llama/Llama-2-7b-chat-hf"
HUBERT_REPO = "TencentGameMate/chinese-hubert-large"

# ============================================================
# STEP 0 — GPU Detection & Quantization
# ============================================================

def detect_vram() -> float:
    """Return total VRAM in GB for the primary GPU, or 0.0 on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def choose_quantization(vram_gb: float) -> str:
    """
    Choose quantization mode based on VRAM:
      int4  → 4-bit NF4   for ≤ 10 GB  (e.g. RTX 4060 8GB)
      int8  → 8-bit        for ≤ 16 GB  (e.g. RTX 3090 16GB)
      fp16  → half prec    for  > 16 GB  (e.g. RTX 4090 24GB)
      fp32  → CPU fallback (no GPU)
    """
    if vram_gb <= 0:  return "fp32"
    if vram_gb <= 10: return "int4"
    if vram_gb <= 16: return "int8"
    return "fp16"


# ============================================================
# STEP 0.5 — Ensure PyTorch has CUDA support
# ============================================================

def ensure_cuda_torch():
    """
    Check if the installed PyTorch has CUDA support.
    If not (CPU-only build), reinstall with CUDA 12.4 wheels.
    This is the most common reason device shows 'cpu' instead of GPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA already working ({torch.version.cuda})")
            return
        # CUDA not available — check if it's a CPU-only install
        cuda_in_version = "cu" in torch.__version__
        if not cuda_in_version:
            print("⚠️  CPU-only PyTorch detected — reinstalling with CUDA 12.4 support...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
                check=True
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu124"],
                check=True
            )
            # Re-check
            import importlib
            import torch as torch2
            importlib.reload(torch2)
            if torch2.cuda.is_available():
                print(f"✅ PyTorch CUDA installed successfully ({torch2.version.cuda})")
            else:
                print("⚠️  CUDA still not available after reinstall.")
                print("   Check your NVIDIA driver: nvidia-smi")
        else:
            print(f"ℹ️  PyTorch has CUDA build ({torch.__version__}) but CUDA not available.")
            print("   This usually means your NVIDIA drivers need updating.")
            print("   Run: nvidia-smi  to check driver version.")
    except ImportError:
        print("📦 PyTorch not installed — installing with CUDA 12.4 support...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu124"],
            check=True
        )
        print("✅ PyTorch CUDA 12.4 installed.")


# ============================================================
# STEP 1 — Clone Emotion-LLaMA Repo
# ============================================================

def clone_emotion_llama():
    if os.path.isdir(EMOTION_LLAMA_DIR):
        print(f"✅ Emotion-LLaMA repo already at {EMOTION_LLAMA_DIR}")
        return
    print(f"📥 Cloning Emotion-LLaMA → {EMOTION_LLAMA_DIR} ...")
    subprocess.run(
        ["git", "clone", "https://github.com/ZebangCheng/Emotion-LLaMA.git",
         EMOTION_LLAMA_DIR],
        check=True
    )
    print("✅ Cloned.")


# ============================================================
# STEP 2 — Install Python Packages (OS-aware)
# ============================================================

# Package NAMES to always skip (we install correct versions ourselves)
# This matches bare names AND any specifier: torch, torch==2.x, torch>=, torch[...], etc.
_SKIP_PACKAGES = {
    # torch — installed separately with CUDA support
    "torch", "torchvision", "torchaudio",
    # nvidia CUDA libs — bundled inside PyTorch wheels
    "nvidia-cudnn-cu11", "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu11", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu11", "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu11", "nvidia-cublas-cu12",
    "nvidia-cufft-cu11", "nvidia-cufft-cu12",
    "nvidia-curand-cu11", "nvidia-curand-cu12",
    "nvidia-cusolver-cu11", "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu11", "nvidia-cusparse-cu12",
    "nvidia-nccl-cu11", "nvidia-nccl-cu12",
    "nvidia-nvtx-cu11", "nvidia-nvtx-cu12",
}

# Line-level patterns to skip (URLs etc. that can't be parsed as package names)
_SKIP_LINE_PATTERNS = [
    "https://github.com/explosion/spacy",
    "@ https://github.com/explosion/spacy",
    "en-core-web-sm",
    "en_core_web_sm",
]

_WINDOWS_EXTRA_SKIP_PACKAGES = {
    "triton",   # Linux-only OpenAI Triton GPU compiler
}


def _get_package_name(line: str) -> str:
    """Extract the bare package name from a requirements.txt line.

    Handles all formats:
      torch             → torch
      torch==2.6.0      → torch
      torch>=2.0,<3.0   → torch
      torch[cuda]       → torch
      torch @ url       → torch
    """
    import re
    # Strip extras, specifiers, and URL markers
    m = re.match(r'^([A-Za-z0-9_\-]+)', line.strip())
    return m.group(1).lower() if m else ''


def _filter_requirements(req_file: str) -> list:
    """Read and filter the Emotion-LLaMA requirements.txt for the current OS."""
    skip_pkgs = set(_SKIP_PACKAGES)
    if IS_WINDOWS:
        skip_pkgs |= _WINDOWS_EXTRA_SKIP_PACKAGES

    kept, dropped = [], []
    with open(req_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Skip lines matching URL/spacy patterns
            if any(p in stripped for p in _SKIP_LINE_PATTERNS):
                dropped.append(stripped)
                continue

            # Skip by package name
            pkg_name = _get_package_name(stripped)
            if pkg_name in skip_pkgs:
                dropped.append(stripped)
                continue

            kept.append(stripped)

    if dropped:
        print(f"  ⚠️  Skipping {len(dropped)} packages: "
              f"{[_get_package_name(d) or d[:30] for d in dropped[:5]]}"
              f"{'...' if len(dropped) > 5 else ''}")
    return kept


def install_requirements():
    """Install all packages needed to run Emotion-LLaMA + our server."""
    req_file = os.path.join(EMOTION_LLAMA_DIR, "requirements.txt")

    # Install filtered packages from their requirements.txt
    if os.path.exists(req_file):
        pkgs = _filter_requirements(req_file)
        if pkgs:
            print(f"📦 Installing {len(pkgs)} Emotion-LLaMA packages ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + pkgs,
                check=True
            )
    else:
        print("⚠️  Emotion-LLaMA requirements.txt not found — skipping.")

    # Our additional packages (work on both Windows and Linux)
    extras = [
        # Quantization — bitsandbytes 0.43+ supports Windows CUDA 12 and Linux
        "bitsandbytes>=0.43.0",
        "accelerate>=0.27.0",
        "transformers>=4.39.0",

        # Download tools
        "gdown>=5.1.0",
        "huggingface_hub>=0.22.0",
        "hf-transfer>=0.1.6",

        # Audio / video
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "numba>=0.57.0",
        "moviepy==1.0.3",
        "imageio==2.28.1",
        "imageio-ffmpeg>=0.4.7",
        "opencv-python>=4.7.0.72",

        # Web server
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "pyngrok>=7.0.0",

        # Utilities
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "sentencepiece>=0.1.99",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
    ]

    # triton is available on Linux — add it for Ubuntu
    if IS_LINUX:
        extras.append("triton>=2.1.0")   # use 2.1+ (2.0.0 had Linux issues too)

    print(f"📦 Installing {len(extras)} EmoConvey extra packages ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + extras,
        check=True
    )
    print("✅ All packages installed.")


# ============================================================
# STEP 3 — Download LLaMA-2-7B
# ============================================================

def download_llama():
    if os.path.isdir(LLAMA_DIR) and os.listdir(LLAMA_DIR):
        print(f"✅ LLaMA-2-7B already at {LLAMA_DIR}")
        return
    print(f"\n📥 Downloading LLaMA-2-7B-chat-hf (~13 GB) ...")
    print("   You must have accepted Meta's license at:")
    print("   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=LLAMA_REPO,
            local_dir=LLAMA_DIR,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin"],   # prefer safetensors
        )
        print("✅ LLaMA-2-7B downloaded.")
    except Exception as e:
        print(f"❌ LLaMA download failed: {e}")
        print(f"   Place manually at: {LLAMA_DIR}")
        sys.exit(1)


# ============================================================
# STEP 4 — Download HuBERT-large
# ============================================================

def download_hubert():
    if os.path.isdir(HUBERT_DIR) and os.listdir(HUBERT_DIR):
        print(f"✅ HuBERT-large already at {HUBERT_DIR}")
        return
    os.makedirs(HUBERT_DIR, exist_ok=True)
    print(f"\n📥 Downloading HuBERT-large (~1.3 GB) ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HUBERT_REPO,
            local_dir=HUBERT_DIR,
            local_dir_use_symlinks=False,
        )
        print("✅ HuBERT-large downloaded.")
    except Exception as e:
        print(f"❌ HuBERT download failed: {e}")
        sys.exit(1)


# ============================================================
# STEP 5 — Download Emotion-LLaMA Checkpoint
# ============================================================

def download_emotion_llama_ckpt():
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    if os.path.exists(ckpt_path):
        print(f"✅ Emotion-LLaMA checkpoint already at {ckpt_path}")
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\n📥 Downloading Emotion-LLaMA checkpoint (~1.6 GB) ...")
    try:
        import gdown
        gdown.download(id=GDRIVE_ID, output=ckpt_path, quiet=False)
        print("✅ Checkpoint downloaded.")
    except Exception as e:
        print(f"❌ Checkpoint download failed: {e}")
        print("   Manual: https://drive.google.com/file/d/1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY/view")
        print(f"   Place at: {ckpt_path}")
        sys.exit(1)


# ============================================================
# STEP 6 — Patch Model Configs
# ============================================================

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _save_yaml(path: str, data: dict):
    import yaml
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def patch_model_yaml():
    """Set the LLaMA-2 path in minigpt_v2.yaml."""
    yaml_path = os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "configs", "models", "minigpt_v2.yaml")
    if not os.path.exists(yaml_path):
        print(f"⚠️  minigpt_v2.yaml not found: {yaml_path}")
        return
    cfg = _load_yaml(yaml_path)
    cfg["llama_model"] = LLAMA_DIR.replace("\\", "/")
    _save_yaml(yaml_path, cfg)
    print(f"✅ Patched minigpt_v2.yaml → llama_model")


def patch_demo_yaml():
    """Set BOTH checkpoint paths in demo.yaml (top-level and nested model.ckpt)."""
    demo_yaml = os.path.join(EMOTION_LLAMA_DIR, "eval_configs", "demo.yaml")
    if not os.path.exists(demo_yaml):
        print(f"⚠️  demo.yaml not found: {demo_yaml}")
        return
    cfg = _load_yaml(demo_yaml)
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME).replace("\\", "/")

    # Patch top-level ckpt (used during loading)
    cfg["ckpt"] = ckpt_path

    # Patch nested model.ckpt (also used by Emotion-LLaMA's Config system)
    if "model" in cfg and isinstance(cfg["model"], dict):
        cfg["model"]["ckpt"] = ckpt_path
        print(f"✅ Patched demo.yaml → ckpt + model.ckpt → {ckpt_path}")
    else:
        print(f"✅ Patched demo.yaml → ckpt → {ckpt_path}")

    _save_yaml(demo_yaml, cfg)



def patch_hubert_path():
    """Patch the HuBERT model path in conversation.py."""
    conv_py = os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "conversation", "conversation.py")
    if not os.path.exists(conv_py):
        print(f"⚠️  conversation.py not found — skip HuBERT patch")
        return
    with open(conv_py, "r", encoding="utf-8") as f:
        src = f.read()
    old = 'model_file = "checkpoints/transformer/chinese-hubert-large"'
    new = f'model_file = r"{HUBERT_DIR}"'
    if old in src:
        src = src.replace(old, new)
        with open(conv_py, "w", encoding="utf-8") as f:
            f.write(src)
        print("✅ Patched HuBERT path in conversation.py")
    else:
        # Try to find it with a fuzzy match
        m = re.search(r'model_file\s*=\s*["\'].*chinese-hubert.*["\']', src)
        if m:
            src = src[:m.start()] + new + src[m.end():]
            with open(conv_py, "w", encoding="utf-8") as f:
                f.write(src)
            print("✅ Patched HuBERT path in conversation.py (fuzzy match)")
        else:
            print("⚠️  HuBERT path not found in conversation.py — check manually")


def patch_quantization(quantization: str):
    """Inject BitsAndBytes quantization config into the LLaMA-loading model file."""
    # The actual file is minigpt_base.py (contains LlamaForCausalLM.from_pretrained)
    # minigpt_v2.py and minigpt4.py inherit from it.
    candidates = [
        os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "models", "minigpt_base.py"),
        os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "models", "minigpt_v2.py"),
        os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "models", "minigpt4.py"),
        os.path.join(EMOTION_LLAMA_DIR, "minigpt4", "models", "mini_gpt4.py"),
    ]
    model_py = None
    for c in candidates:
        if os.path.exists(c):
            # Only patch the file that actually contains the from_pretrained call
            with open(c, "r", encoding="utf-8") as f:
                content = f.read()
            if "LlamaForCausalLM" in content and "from_pretrained" in content:
                model_py = c
                break

    if model_py is None:
        print("⚠️  Could not find LlamaForCausalLM loader — skip quantization patch")
        print("   You may need to add quantization manually to the model loading code.")
        return

    print(f"  📄 Patching: {os.path.basename(model_py)}")
    with open(model_py, "r", encoding="utf-8") as f:
        src = f.read()

    if "# EmoConvey quantization patch" in src:
        print(f"✅ Quantization already patched (mode={quantization})")
        return

    if quantization == "fp16" or quantization == "fp32":
        print(f"✅ No quantization patch needed (mode={quantization})")
        return

    if quantization == "int4":
        header = (
            "# EmoConvey quantization patch — 4-bit NF4 for ≤10 GB VRAM\n"
            "from transformers import BitsAndBytesConfig as _BnBCfg\n"
            "_quant_cfg = _BnBCfg(\n"
            "    load_in_4bit=True,\n"
            "    bnb_4bit_compute_dtype=torch.float16,\n"
            "    bnb_4bit_use_double_quant=True,\n"
            "    bnb_4bit_quant_type='nf4',\n"
            ")\n"
        )
        kwarg = ", quantization_config=_quant_cfg, device_map='auto'"
    else:  # int8
        header = (
            "# EmoConvey quantization patch — 8-bit for 10–16 GB VRAM\n"
            "from transformers import BitsAndBytesConfig as _BnBCfg\n"
            "_quant_cfg = _BnBCfg(load_in_8bit=True)\n"
        )
        kwarg = ", quantization_config=_quant_cfg, device_map='auto'"

    # Find LlamaForCausalLM.from_pretrained(...) and inject the kwarg
    new_src, n = re.subn(
        r'(LlamaForCausalLM\.from_pretrained\([^)]+)\)',
        r'\1' + kwarg + ')',
        src, count=1
    )
    if n == 0:
        print("⚠️  Could not find LlamaForCausalLM.from_pretrained — patch manually")
        return

    new_src = header + new_src
    with open(model_py, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"✅ Quantization patched in mini_gpt4.py (mode={quantization})")


# ============================================================
# STEP 7 — Write .env for server.py
# ============================================================

def write_env(quantization: str, vram_gb: float):
    env_path = os.path.join(SCRIPT_DIR, ".env")
    lines = [
        f"EMOTION_LLAMA_DIR={EMOTION_LLAMA_DIR}",
        f"EMOTION_LLAMA_DEMO_CFG={os.path.join(EMOTION_LLAMA_DIR, 'eval_configs', 'demo.yaml')}",
        f"EMOTION_LLAMA_QUANTIZATION={quantization}",
        f"EMOTION_LLAMA_VRAM_GB={vram_gb:.1f}",
        f"EMOTION_LLAMA_GRADIO_URL=http://127.0.0.1:7889",
    ]
    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Wrote .env → {env_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  EmoConvey — Emotion-LLaMA Setup")
    print(f"  OS: {platform.system()} {platform.release()}")
    print("=" * 60)

    vram_gb      = detect_vram()
    quantization = choose_quantization(vram_gb)

    print(f"\n🖥️  GPU: {f'{vram_gb:.1f} GB VRAM' if vram_gb > 0 else 'Not detected (CPU mode)'}")
    print(f"⚙️  Quantization: {quantization}")
    print(f"📁 Install path: {EMOTION_LLAMA_DIR}\n")

    clone_emotion_llama()
    ensure_cuda_torch()     # ensure CUDA torch exists before installs
    install_requirements()
    ensure_cuda_torch()     # re-run AFTER installs — catches any overwrites by sub-packages
    download_llama()
    download_hubert()
    download_emotion_llama_ckpt()
    patch_model_yaml()
    patch_demo_yaml()
    patch_hubert_path()
    patch_quantization(quantization)
    write_env(quantization, vram_gb)

    print("\n" + "=" * 60)
    print("  ✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    if IS_WINDOWS:
        print("  1. Activate venv:    .\\emo\\Scripts\\Activate.ps1")
    else:
        print("  1. Activate venv:    source emo/bin/activate")
    print("  2. Start server:     python server.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
