import torch
import os

# Model configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "hf_token": os.environ.get("HF_AUTH_TOKEN"),
    "precision": "bfloat16",  # Options: "8bit", "bfloat16", "Q4_K_M", "Q5_K_M", 
    "device_map": "auto",
    "min_pixels": 256*28*28,
    "max_pixels": 1280*28*28,
    "max_new_tokens": 512,
    "use_flash_attention": True  # Control Flash Attention usage
}
# Quantization configurations
QUANT_CONFIGS = {
    "Q4_K_M": {
        "model_name_suffix": "-Q4_K_M",
        "torch_dtype": torch.bfloat16
    },
    "Q5_K_M": {
        "model_name_suffix": "-Q5_K_M",
        "torch_dtype": torch.bfloat16
    }
}