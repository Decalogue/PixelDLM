"""
推理脚本：使用训练好的模型生成答案
"""

import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from model import build_jit_model
from robust_token2img import RobustToken2Img


def load_model(checkpoint_path: str, model_name: str = 'JiT-B/4', img_size: int = 64, 
               use_pixel_decoder: bool = False, pixel_decoder_depth: int = 3, device: str = 'cuda'):
    """加载训练好的模型"""
    model = build_jit_model(
        model_name=model_name,
        img_size=img_size,
        predict_clean=True,
        use_pixel_decoder=use_pixel_decoder,
        pixel_decoder_depth=pixel_decoder_depth,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤掉推理时不需要的键（如频率损失函数的权重）
    # 这些键在训练时被添加到模型中，但推理时不需要
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # 跳过频率损失函数相关的键
        if 'freq_loss_fn' in key:
            continue
        filtered_state_dict[key] = value
    
    # 加载过滤后的状态字典（允许缺少一些键）
    model.load_state_dict(filtered_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


def encode_text(text: str, token_encoder, img_size=64, device='cuda'):
    """编码文本为图像"""
    img, metadata = token_encoder.encode(text, size=(img_size, img_size))
    
    # Convert to tensor [C, H, W], normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor, metadata


def generate_text(
    model,
    num_inference_steps: int = 20,
    device: str = 'cuda',
):
    """生成文本图像（无条件生成）"""
    with torch.no_grad():
        generated_img = model.generate(
            condition=None,
            num_inference_steps=num_inference_steps,
            device=device,
        )
    
    return generated_img


def decode_text(
    generated_img: torch.Tensor,
    token_decoder,
    num_tokens: int = None,
):
    """解码图像为文本"""
    generated_np = (generated_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Decode to text
    text, token_ids = token_decoder.decode(
        generated_np,
        num_tokens=num_tokens,
        stop_on_black=True
    )
    
    return text, token_ids


def inference(
    model,
    token_decoder,
    num_inference_steps: int = 20,
    device: str = 'cuda',
    save_image: bool = False,
    output_path: str = None,
    img_size: int = 64,
):
    """完整的推理流程（无条件生成）"""
    print(f"\n生成文本中... (steps={num_inference_steps})")
    
    # Generate text image (unconditional)
    generated_img = generate_text(
        model,
        num_inference_steps=num_inference_steps,
        device=device
    )
    
    # Decode to text
    text, token_ids = decode_text(generated_img, token_decoder)
    
    print(f"\n生成的文本: {text}")
    print(f"Token 数量: {len(token_ids)}")
    
    # Save image if needed
    if save_image:
        if output_path is None:
            output_path = './generated_text.png'
        generated_np = (generated_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(output_path, generated_np)
        print(f"生成的图像已保存到: {output_path}")
    
    return text, token_ids


def main():
    parser = argparse.ArgumentParser(description='Inference with JiT-based Token2Img Diffusion Model')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='JiT-B/4', help='Model name (JiT-B/4 for 64×64, JiT-B/16 for 256×256)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--use_pixel_decoder', action='store_true', help='Use U-Net pixel decoder (DiP)')
    parser.add_argument('--pixel_decoder_depth', type=int, default=3, help='U-Net decoder depth')
    
    # Data args
    parser.add_argument('--tokenizer_path', type=str, default='/root/data/AI/pretrain/Qwen2.5-7B-Instruct', help='Tokenizer path')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt text (for future conditional generation)')
    parser.add_argument('--prompt_file', type=str, default=None, help='File with prompts (one per line)')
    
    # Generation args
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Output args
    parser.add_argument('--save_image', action='store_true', help='Save generated image')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # Load token encoder/decoder
    token_decoder = RobustToken2Img(tokenizer)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print(f"  Model: {args.model}")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print(f"  Use pixel decoder: {args.use_pixel_decoder}")
    if args.use_pixel_decoder:
        print(f"  Pixel decoder depth: {args.pixel_decoder_depth}")
    model = load_model(
        args.checkpoint, 
        args.model, 
        args.img_size,
        use_pixel_decoder=args.use_pixel_decoder,
        pixel_decoder_depth=args.pixel_decoder_depth,
        device=device
    )
    print("Model loaded successfully!")
    
    # Inference (unconditional generation)
    if args.prompt_file:
        # If prompt file provided, generate one for each line (as prompts for future conditional generation)
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        num_generations = len(prompts)
    elif args.prompt:
        # Single prompt (for future conditional generation)
        prompts = [args.prompt]
        num_generations = 1
    else:
        prompts = []  # No prompt, pure unconditional generation
        num_generations = 50
    
    # Generate
    results = []
    for i in range(num_generations):
        print(f"\n{'='*60}")
        print(f"生成 {i+1}/{num_generations}")
        if prompts:
            print(f"提示: {prompts[i]}")
        print(f"{'='*60}")
        
        output_path = os.path.join(args.output_dir, f'generated_{i}.png') if args.save_image else None
        
        text, token_ids = inference(
            model=model,
            token_decoder=token_decoder,
            num_inference_steps=args.num_inference_steps,
            device=device,
            save_image=args.save_image,
            output_path=output_path,
            img_size=args.img_size,
        )
        
        results.append({
            'prompt': prompts[i] if prompts else None,
            'generated_text': text,
            'num_tokens': len(token_ids),
        })
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {results_path}")
    
    return results


if __name__ == '__main__':
    main()
