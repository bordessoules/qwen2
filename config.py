MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "device_map": {0: "12GiB"},
    "min_pixels": 256*28*28,
    "max_pixels": 1280*28*28,
    "max_new_tokens": 512
}
# Optional flash attention config
FLASH_ATTENTION_CONFIG = {
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "device_map": "auto"
}
