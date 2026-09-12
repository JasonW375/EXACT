import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 1. Placeholder path for the base model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"Loading generic architecture for: {model_id}...")

# Load the architecture only (no weights) so it is fast
# Note: the real LLaVA code wraps this in LlavaLlamaForCausalLM;
# what follows illustrates the internal structure of Llama 3.1 + LoRA
config = AutoConfig.from_pretrained(model_id)
# Build on the meta device to avoid allocating any VRAM
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

print("\n=== 1. Original Llama-3.1-8B block (one layer) ===")
print(model.model.layers[0])

# 2. LoRA configuration (mirrors the training command line)
# --lora_r 128 --lora_alpha 256
peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj"], # LLaVA typically tunes these two; all-linear is also common
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the model with LoRA
model = get_peft_model(model, peft_config)

print("\n=== 2. Structure with LoRA (note Linear becomes LoraLinear) ===")
print(model.base_model.model.model.layers[0].self_attn)

print("\n=== 3. Expected projector structure ===")
print("Given --mm_projector_type 'attn_pool+mlp2x_gelu'")
print("the structure should be: [AttentionPooling] -> [Linear] -> [GELU] -> [Linear]")
