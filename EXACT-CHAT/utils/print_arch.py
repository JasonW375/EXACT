import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 1. 模拟你的基础模型路径
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"Loading generic architecture for: {model_id}...")

# 加载基础模型结构（为了快，不加载权重，只加载结构）
# 注意：在你的实际LLaVA代码中，还会包裹一层 LlavaLlamaForCausalLM
# 这里演示 Llama 3.1 + LoRA 的内部结构
config = AutoConfig.from_pretrained(model_id)
# 使用空权重初始化以节省显存，只为了看结构
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

print("\n=== 1. 原始 Llama-3.1-8B Block 结构 (其中一层) ===")
print(model.model.layers[0])

# 2. 模拟 LoRA 配置 (参考你的命令行参数)
# --lora_r 128 --lora_alpha 256
peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj"], # LLaVA通常默认微调这两个，也可能是 all-linear
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 应用 LoRA
model = get_peft_model(model, peft_config)

print("\n=== 2. 加上 LoRA 后的结构 (注意看 Linear 层变成了 LoraLinear) ===")
print(model.base_model.model.model.layers[0].self_attn)

print("\n=== 3. 你的 Projector 结构推测 ===")
print("根据参数: --mm_projector_type 'attn_pool+mlp2x_gelu'")
print("结构应为: [AttentionPooling] -> [Linear] -> [GELU] -> [Linear]")
