# -*- coding: utf-8 -*-
import os
import sys
import re

def patch_file(filepath, replacements, create_if_missing=False):
    """Utility to perform string replacements safely."""
    if not os.path.exists(filepath):
        if create_if_missing:
            print(f"📁 Creating {filepath}...")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# -*- coding: utf-8 -*-\n")
        else:
            print(f"❌ Skipping {filepath} (Not found)")
            return False
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Ensure UTF-8 header is present
    changed_header = False
    if filepath.endswith(".py") and "# -*- coding: utf-8 -*-" not in content[:100]:
        content = "# -*- coding: utf-8 -*-\n" + content
        changed_header = True

    changed = False
    for old_str, new_str in replacements:
        if new_str in content and old_str != "":
            continue
        if old_str in content:
            content = content.replace(old_str, new_str)
            changed = True

    if changed or changed_header:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully patched: {filepath}")
    else:
        print(f"⚡ No changes needed for: {filepath}")
    return changed

def run_all_patches():
    print("🚀 Restoring int4 mode and applying safe UTF-8 initialization...\n")
    
    # 1. modeling_llama.py
    llama_path = os.path.join("Emotion-LLaMA", "minigpt4", "models", "modeling_llama.py")
    if os.path.exists(llama_path):
        with open(llama_path, 'r', encoding='utf-8') as f:
            if "LlamaForCausalLMOrig" not in f.read():
                with open(llama_path, 'w', encoding='utf-8') as f:
                    f.write('# -*- coding: utf-8 -*-\nfrom transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig\nfrom transformers.modeling_outputs import CausalLMOutputWithPast\n\nclass LlamaForCausalLM(LlamaForCausalLMOrig):\n    def forward(self, *args, **kwargs):\n        return super().forward(*args, **kwargs)\n')

    # 2. base_model.py (Back to int4)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "base_model.py"),
        [
            ("from peft import (", "from transformers import BitsAndBytesConfig\nfrom peft import ("),
            (
                "load_in_8bit=True,",
                "load_in_4bit=True,\n                bnb_4bit_use_double_quant=True,\n                bnb_4bit_quant_type=\"nf4\",\n                bnb_4bit_compute_dtype=torch.float16,"
            ),
            ("llama_model = prepare_model_for_int8_training(llama_model)", "# llama_model = prepare_model_for_int8_training(llama_model)")
        ]
    )

    # 3. app_EmotionLlamaClient.py (Safe UTF-8)
    patch_file(
        os.path.join("Emotion-LLaMA", "app_EmotionLlamaClient.py"),
        [
            (
                "import gradio as gr",
                "import gradio as gr\n\ndef setup_utf8():\n    import sys\n    if sys.platform == \"win32\":\n        try:\n            if hasattr(sys.stdout, 'reconfigure'):\n                sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n            if hasattr(sys.stderr, 'reconfigure'):\n                sys.stderr.reconfigure(encoding='utf-8', errors='replace')\n        except Exception: pass\n\nsetup_utf8()"
            )
        ]
    )

    # ... other patches remain the same ...
    patch_file(os.path.join("Emotion-LLaMA", "minigpt4", "models", "Qformer.py"), [("from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer", "def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):\n    return forward_fn(*input_tensors)\n\ndef find_pruneable_heads_and_indices(*args, **kwargs):\n    return set(), []\n\ndef prune_linear_layer(*args, **kwargs):\n    pass")])
    patch_file(os.path.join("Emotion-LLaMA", "minigpt4", "models", "minigpt_base.py"), [("device = img_list[0].device", "device = img_list[0].device if len(img_list) > 0 else self.llama_model.device"), ("mixed_embs = torch.cat(mixed_embs, dim=1)\n        return mixed_embs", "mixed_embs = torch.cat(mixed_embs, dim=1)\n        mixed_embs = mixed_embs.half()\n        return mixed_embs")])
    patch_file(os.path.join("Emotion-LLaMA", "minigpt4", "conversation", "conversation.py"), [("audio = video.audio", "audio = video.audio\n    if audio is None:\n        import numpy as np\n        return np.zeros((16000,), dtype=np.float32), 16000")])

    print("\n🎉 int4 patches ready. Re-patching server.py...")

if __name__ == "__main__":
    run_all_patches()
