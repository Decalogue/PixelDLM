"""
条件生成推理脚本：使用训练好的模型基于 prompt 生成 answer
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
               cond_img_size: int = 64, use_pixel_decoder: bool = False, 
               pixel_decoder_depth: int = 3, device: str = 'cuda'):
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


def encode_prompt_to_condition(prompt: str, token_encoder, cond_img_size=64, device='cuda'):
    """将 prompt 文本编码为 condition 图像"""
    cond_img, metadata = token_encoder.encode(prompt, size=(cond_img_size, cond_img_size))
    
    # Convert to tensor [C, H, W], normalize to [0, 1]
    cond_img_tensor = torch.from_numpy(cond_img).permute(2, 0, 1).float() / 255.0
    cond_img_tensor = cond_img_tensor.unsqueeze(0).to(device)
    
    return cond_img_tensor, metadata


def generate_answer(
    model,
    condition_img: torch.Tensor,
    num_inference_steps: int = 20,
    guidance_scale: float = 1.0,
    device: str = 'cuda',
):
    """基于 condition 生成 answer 图像（条件生成）"""
    # Convert condition image to patches
    condition_patches = model.image_to_patches(condition_img)  # [B, cond_patches, patch_dim]
    
    with torch.no_grad():
        generated_img = model.generate(
            condition=condition_patches,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
    
    return generated_img


def decode_answer(
    generated_img: torch.Tensor,
    token_decoder,
    num_tokens: int = None,
):
    """解码 answer 图像为文本"""
    generated_np = (generated_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Decode to text
    text, token_ids = token_decoder.decode(
        generated_np,
        num_tokens=num_tokens,
        stop_on_black=True
    )
    
    return text, token_ids


def inference_conditional(
    model,
    token_encoder,
    token_decoder,
    prompt: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 1.0,
    device: str = 'cuda',
    save_image: bool = False,
    output_path: str = None,
    cond_img_size: int = 64,
    img_size: int = 64,
):
    """完整的条件生成推理流程（prompt -> answer）"""
    print(f"\nPrompt: {prompt}")
    print(f"生成答案中... (steps={num_inference_steps}, guidance={guidance_scale})")
    
    # Encode prompt to condition image
    condition_img, cond_metadata = encode_prompt_to_condition(
        prompt, token_encoder, cond_img_size, device
    )
    cond_tokens = cond_metadata.get('num_tokens', 'N/A')
    print(f"Condition tokens: {cond_tokens}")
    
    # 调试：检查条件 patches
    condition_patches = model.image_to_patches(condition_img)
    cond_patches_count = condition_patches.shape[1]
    cond_max_patches = model.cond_max_patches if hasattr(model, 'cond_max_patches') else model._cond_max_patches if hasattr(model, '_cond_max_patches') else 'N/A'
    print(f"Condition patches: {cond_patches_count} (max: {cond_max_patches})")
    if cond_patches_count < cond_max_patches:
        padding_ratio = (cond_max_patches - cond_patches_count) / cond_max_patches
        print(f"  ⚠️  警告: 条件 patches 有 {padding_ratio*100:.1f}% 是 padding，可能影响条件信息传递")
    
    # Generate answer image (conditional generation)
    generated_img = generate_answer(
        model,
        condition_img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        device=device
    )
    
    # Decode to text
    answer, token_ids = decode_answer(generated_img, token_decoder)
    
    print(f"\n生成的答案: {answer}")
    print(f"Answer tokens: {len(token_ids)}")
    
    # Save images if needed
    if save_image:
        if output_path is None:
            output_path = './generated_answer.png'
        
        # Save answer image
        generated_np = (generated_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(output_path, generated_np)
        print(f"生成的答案图像已保存到: {output_path}")
        
        # Save condition image
        cond_output_path = output_path.replace('.png', '_condition.png')
        cond_np = (condition_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(cond_output_path, cond_np)
        print(f"条件图像已保存到: {cond_output_path}")
    
    return answer, token_ids, condition_img


def main():
    parser = argparse.ArgumentParser(description='Conditional Inference with JiT-based Token2Img Diffusion Model')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='JiT-B/4', help='Model name (JiT-B/4 for 64×64, JiT-B/16 for 256×256)')
    parser.add_argument('--img_size', type=int, default=64, help='Answer image size')
    parser.add_argument('--cond_img_size', type=int, default=64, help='Condition image size')
    parser.add_argument('--use_pixel_decoder', action='store_true', help='Use U-Net pixel decoder (DiP)')
    parser.add_argument('--pixel_decoder_depth', type=int, default=3, help='U-Net decoder depth')
    
    # Data args
    parser.add_argument('--tokenizer_path', type=str, default='/root/data/AI/pretrain/Qwen2.5-7B-Instruct', help='Tokenizer path')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt text')
    parser.add_argument('--prompt_file', type=str, default=None, help='File with prompts (one per line)')
    
    # Generation args
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Output args
    parser.add_argument('--save_image', action='store_true', help='Save generated images')
    parser.add_argument('--output_dir', type=str, default='./output_ft', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # Load token encoder/decoder
    token_encoder = RobustToken2Img(tokenizer)
    token_decoder = RobustToken2Img(tokenizer)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print(f"  Model: {args.model}")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print(f"  Condition image size: {args.cond_img_size}×{args.cond_img_size}")
    print(f"  Use pixel decoder: {args.use_pixel_decoder}")
    if args.use_pixel_decoder:
        print(f"  Pixel decoder depth: {args.pixel_decoder_depth}")
    model = load_model(
        args.checkpoint, 
        args.model, 
        args.img_size, 
        args.cond_img_size,
        use_pixel_decoder=args.use_pixel_decoder,
        pixel_decoder_depth=args.pixel_decoder_depth,
        device=device
    )
    print("Model loaded successfully!")
    
    # Get prompts
    if args.prompt_file:
        # If prompt file provided, generate one for each line
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        num_generations = len(prompts)
    elif args.prompt:
        # Single prompt
        prompts = [args.prompt]
        num_generations = 1
    else:
        # Default prompts for testing
        prompts = [
            "什么是人工智能？",
            "5+5等于多少？",
            "如何写出一本好书？",
        ]
        num_generations = len(prompts)
        print(f"未指定 prompt，使用默认测试 prompts")
    
    # Generate
    results = []
    for i in range(num_generations):
        print(f"\n{'='*60}")
        print(f"生成 {i+1}/{num_generations}")
        print(f"{'='*60}")
        
        output_path = os.path.join(args.output_dir, f'generated_{i}.png') if args.save_image else None
        
        answer, token_ids, condition_img = inference_conditional(
            model=model,
            token_encoder=token_encoder,
            token_decoder=token_decoder,
            prompt=prompts[i],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            save_image=args.save_image,
            output_path=output_path,
            cond_img_size=args.cond_img_size,
            img_size=args.img_size,
        )
        
        results.append({
            'prompt': prompts[i],
            'generated_answer': answer,
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
