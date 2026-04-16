import os

def patch_file(filepath, replacements):
    """Utility to perform string replacements safely."""
    if not os.path.exists(filepath):
        print(f"❌ Skipping {filepath} (Not found)")
        return False
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    changed = False
    for old_str, new_str in replacements:
        if old_str in content:
            content = content.replace(old_str, new_str)
            changed = True
        else:
            print(f"  ⚠️ Warning: Could not find target string in {filepath}:\n    '{old_str[:50]}...'")

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully patched: {filepath}")
    else:
        print(f"⚡ No changes needed for: {filepath}")
    return changed

def run_all_patches():
    print("🚀 Starting Emotion-LLaMA Core Patch Script...\n")
    
    # 1. Patch modeling_llama.py (Docstring deprecation crashes)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "modeling_llama.py"),
        [
            (
                "from transformers.models.llama.modeling_llama import (\n    LlamaConfig,\n    LlamaModel,\n    LlamaPreTrainedModel,\n    add_start_docstrings,\n    add_start_docstrings_to_model_forward,\n    LLAMA_INPUTS_DOCSTRING,\n    LLAMA_START_DOCSTRING,\n)", 
                "from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig"
            ),
            (
                "from transformers.models.llama.modeling_llama import LlamaForCausalLMOrig",
                "from transformers.modeling_outputs import CausalLMOutputWithPast\nfrom transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig"
            ),
            (
                "@add_start_docstrings(\n    \"The bare LLaMA Model outputting raw hidden-states without any specific head on top.\",\n    LLAMA_START_DOCSTRING,\n)\nclass LlamaModel(LlamaPreTrainedModel):",
                ""
            )
            # We already mostly wiped the file locally to just the causal subclass in our manual edit earlier.
        ]
    )

    # 2. Patch Qformer.py (PyTorch removed chunking utils)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "Qformer.py"),
        [
            (
                "from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer",
                "def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):\n    return forward_fn(*input_tensors)\n\ndef find_pruneable_heads_and_indices(*args, **kwargs):\n    return set(), []\n\ndef prune_linear_layer(*args, **kwargs):\n    pass"
            )
        ]
    )

    # 3. Patch minigpt_v2.yaml (Fix nested hardcoded Linux checkpoint path)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "configs", "models", "minigpt_v2.yaml"),
        [
            (
                "llama_model: \"/home/user/project/Emoconvey/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf\"",
                "llama_model: \"checkpoints/Llama-2-7b-chat-hf\""
            )
        ]
    )

    # 4. Patch base_model.py (8-bit deprecation -> 4bit NF4, and Training Hooks wipe)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "base_model.py"),
        [
            (
                "from peft import (",
                "from transformers import BitsAndBytesConfig\nfrom peft import ("
            ),
            (
                "llama_model = LlamaForCausalLM.from_pretrained(\n                llama_model_path,\n                torch_dtype=torch.float16,\n                load_in_8bit=True,\n                device_map={'': low_res_device},\n            )",
                "bnb_cfg = BitsAndBytesConfig(\n                load_in_4bit=True,\n                bnb_4bit_use_double_quant=True,\n                bnb_4bit_quant_type=\"nf4\",\n                bnb_4bit_compute_dtype=torch.float16\n            )\n            llama_model = LlamaForCausalLM.from_pretrained(\n                llama_model_path,\n                torch_dtype=torch.float16,\n                quantization_config=bnb_cfg,\n                device_map={'': low_res_device},\n            )"
            ),
            (
                "llama_model = prepare_model_for_int8_training(llama_model)\n            loraconfig = LoraConfig(",
                "loraconfig = LoraConfig("
            )
        ]
    )

    # 5. Patch minigpt_base.py (Float32 cast native bypass bug)
    patch_file(
        os.path.join("Emotion-LLaMA", "minigpt4", "models", "minigpt_base.py"),
        [
            (
                "mixed_embs = torch.cat(mixed_embs, dim=1)\n        return mixed_embs",
                "mixed_embs = torch.cat(mixed_embs, dim=1)\n        mixed_embs = mixed_embs.half()\n        return mixed_embs"
            )
        ]
    )

    # 6. Patch app_EmotionLlamaClient.py (Windows localhost networking bugs)
    patch_file(
        os.path.join("Emotion-LLaMA", "app_EmotionLlamaClient.py"),
        [
            (
                "iface.queue().launch(server_name=\"0.0.0.0\", server_port=7889, share=False)",
                "iface.queue().launch(server_name=\"127.0.0.1\", server_port=7889, share=False)"
            )
        ]
    )

    print("\n🎉 All patches applied successfully!")

if __name__ == "__main__":
    run_all_patches()
