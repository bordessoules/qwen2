import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from config import MODEL_CONFIG, QUANT_CONFIGS
import argparse
import gc



def load_model(config):
    model_name = config["model_name"]
    precision = config["precision"]
    hf_token = config["hf_token"]
    device_map = config["device_map"]
    attn_impl = "flash_attention_2" if config.get("use_flash_attention") else "eager"

    if precision == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            attn_implementation=attn_impl,
            quantization_config=quantization_config
        )
    elif precision == "bfloat16":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16
        )
    elif precision in QUANT_CONFIGS:
        quantized_name = f"{model_name}{QUANT_CONFIGS[precision]['model_name_suffix']}"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_name,
            device_map=device_map,
            attn_implementation=attn_impl,
            token=hf_token,
            **QUANT_CONFIGS[precision]
        )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2VL model with configurable settings.")
    parser.add_argument('--hf_token', default=MODEL_CONFIG["hf_token"], type=str, help='Hugging Face auth token')
    parser.add_argument('--model', type=str, default=MODEL_CONFIG["model_name"], help='Model name to use')
    parser.add_argument('--precision', type=str, default=MODEL_CONFIG["precision"], help='Precision mode')
    parser.add_argument('--minpixels', type=int, default=MODEL_CONFIG["min_pixels"], help='Minimum pixels for processing')
    parser.add_argument('--maxpixels', type=int, default=MODEL_CONFIG["max_pixels"], help='Maximum pixels for processing')
    parser.add_argument('--maxtokens', type=int, default=MODEL_CONFIG["max_new_tokens"], help='Maximum number of tokens to generate')
    parser.add_argument('prompt', type=str, help='Text prompt for the model')
    parser.add_argument('images', nargs='+', help='Paths to image files')
    args = parser.parse_args()

    config = MODEL_CONFIG.copy()
    config["model_name"] = args.model
    config["precision"] = args.precision

    print(f"Loading model: {config['model_name']} with precision {config['precision']}...")
    model = load_model(config)
    processor = AutoProcessor.from_pretrained(args.model)
    print("Model loaded successfully")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": args.prompt}
        ] + [{"type": "image", "image": img} for img in args.images]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.maxtokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("Response:", response)

    del model, outputs
    torch.cuda.empty_cache()
    del processor  # Also clear the processor
    gc.collect()   # Force garbage collection

if __name__ == "__main__":
    main()