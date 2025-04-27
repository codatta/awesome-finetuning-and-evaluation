"""
# 0. Setup GPU Server
We suggest you should have at least 16GB GPU and 64GB CPU memory

# 1. Install Requirements
pip install torch torchvision transformers pillow requests clip-interrogator pert
pip install accelerate peft deepspeed

# 2. Prepare Annotation Data
Using CLIP or CLIP to annotate data
annotate_image('images')

# 3. Prepare the metadata.jsonl
Generate metadata.jsonl file containing image paths, captions, and style information
generate_metadata('images', 'images', "Azuki, Azuki Style, Azuki art, azuki")

# 4. Train Your LoRA Model with Bash Command
#!/bin/bash
accelerate launch \
  --mixed_precision="fp16" \
  --num_processes=1 \
  --num_machines=1 \
  --machine_rank=0 \
  --dynamo_backend="no" \
  train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="./images" \
  --resolution=768 \
  --output_dir="./images" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --num_train_epochs=10 \
  --rank=8 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --validation_epochs=5 \
  --checkpointing_steps=100
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

def generate_images(prompt, output_dir="output", num_images=1, width=720, height=720,
                    pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0",
                    lora_path="images/pytorch_lora_weights.safetensors",
                    lora_scale=0.7, seed=None):
    """
    Generate images with and without LoRA

    Args:
        prompt (str): The prompt for image generation
        output_dir (str): Output directory
        num_images (int): Number of images to generate for each type
        width (int): Image width
        height (int): Image height
        pretrained_model_path (str): Path to pretrained model
        lora_path (str): Path to LoRA weights file
        lora_scale (float): LoRA intensity
        seed (int): Random seed, set to None for random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Load base SDXL model
    print("Loading base SDXL model...")

    # Use loading method similar to train.py
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Optimize inference speed
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # First generate original images (without LoRA)
    print(f"Generating {num_images} images without LoRA...")
    for i in range(num_images):
        current_seed = torch.initial_seed() if seed is None else seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            generator=generator
        ).images[0]

        # Save image
        image_path = os.path.join(output_dir, f"no_lora_seed_{current_seed}.png")
        image.save(image_path)
        print(f"Saved image to: {image_path}")

    # Load LoRA weights
    print(f"Loading LoRA weights: {lora_path}")
    if os.path.exists(lora_path):
        try:
            # Use diffusers' LoRA loading method
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_scale)
            lora_loaded = True
        except Exception as e:
            print(f"LoRA loading error: {e}")
            print("LoRA weights are not compatible with SDXL model architecture. This usually happens when trying to use LoRA trained for SD1.5 with SDXL model.")
            print("Continuing with base model generation...")
            lora_loaded = False
    else:
        print(f"Error: LoRA file {lora_path} does not exist!")
        lora_loaded = False

    # Generate images with LoRA
    if lora_loaded:
        print(f"Generating {num_images} images with LoRA...")
        for i in range(num_images):
            current_seed = torch.initial_seed() if seed is None else seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=30,
                generator=generator
            ).images[0]

            # Save image
            image_path = os.path.join(output_dir, f"with_lora_seed_{current_seed}.png")
            image.save(image_path)
            print(f"Saved image to: {image_path}")

        # Unload LoRA weights
        pipe.unfuse_lora()
    else:
        print("Skipping LoRA image generation due to failed LoRA loading.")

    print("Image generation completed!")

# Image annotation function
def annotate_image(train_data_dir):
    # Configure paths
    output_caption_dir = f"{train_data_dir}.captions"  # Directory to store captions
    os.makedirs(output_caption_dir, exist_ok=True)

    # Choose between BLIP or CLIP Interrogator
    USE_BLIP = False  # True = use BLIP, False = use CLIP Interrogator

    # ========== Method 1: BLIP Auto Caption Generation ==========
    if USE_BLIP:
        # Load BLIP model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

        def generate_caption_blip(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to("cuda")
            caption_ids = model.generate(**inputs)
            caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
            return caption

    # ========== Method 2: CLIP Interrogator Caption Generation ==========
    else:
        ci_config = Config(device="cuda")
        ci = Interrogator(ci_config)

        def generate_caption_clip(image_path):
            image = Image.open(image_path).convert("RGB")
            caption = ci.interrogate(image)
            return caption

    # Loop through all images and generate captions
    for filename in os.listdir(train_data_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_path = os.path.join(train_data_dir, filename)
            caption = generate_caption_blip(image_path) if USE_BLIP else generate_caption_clip(image_path)

            # Generate corresponding txt file
            txt_path = os.path.join(output_caption_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            print(f"✅ Generated Caption: {filename} -> {caption}")

if __name__ == "__main__":
    # Example usage
    prompt = "sunshine boy, cigarette in mouth, sword in hand, Azuki style, high quality, vivid colors, clean background, single color background"  # You can modify this prompt as desired

    # Step 1: Annotate images
    annotate_image('images')

    # Step 2: Generate metadata
    generate_metadata('images', 'images', "Azuki, Azuki Style, Azuki art, azuki")

    # Step 3: Generate images
    generate_images(
        prompt=prompt,
        output_dir="output",
        num_images=1,
        width=720,
        height=720,
        pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0",  # Can be local path or Hugging Face model ID
        lora_path="images/pytorch_lora_weights.safetensors",
        lora_scale=0.7,
        seed=42  # Set to None for random seed
    )