import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import MODEL_CONFIG
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Qwen2VL model with configurable settings.")
    parser.add_argument('--model', type=str, default=MODEL_CONFIG["model_name"], help='Model name to use')
    parser.add_argument('--minpixels', type=int, default=MODEL_CONFIG["min_pixels"], help='Minimum pixels for processing')
    parser.add_argument('--maxpixels', type=int, default=MODEL_CONFIG["max_pixels"], help='Maximum pixels for processing')
    parser.add_argument('--maxtokens', type=int, default=MODEL_CONFIG["max_new_tokens"], help='Maximum number of tokens to generate')
    parser.add_argument('prompt', type=str, help='Text prompt for the model')
    parser.add_argument('images', nargs='*', help='Paths to image files')
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model: {args.model}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.minpixels,
        max_pixels=args.maxpixels
    )
    print(f"Model {args.model} loaded successfully")

    # Process input
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Load images
    image_inputs, video_inputs = process_vision_info(args.images)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate response
    print("Generating response...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.maxtokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Output the generated text
    print("Response:", output_text[0])

if __name__ == "__main__":
    main()
