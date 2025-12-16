"""
评估脚本：评估模型生成质量和 token 恢复准确率
"""

import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import json

from model import build_jit_model
from dataset import TokenImageDataset
from robust_token2img import RobustToken2Img


def load_model(checkpoint_path: str, model_name: str = 'JiT-B/4', img_size: int = 64, device: str = 'cuda'):
    """加载模型"""
    model = build_jit_model(
        model_name=model_name,
        img_size=img_size,
        predict_clean=True,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_token_recovery(
    model,
    dataloader,
    token_decoder,
    device: str = 'cuda',
    num_samples: int = 100,
):
    """评估 token 恢复准确率"""
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    total_error_rate = 0.0
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            if i >= num_samples:
                break
            
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)
            text = batch['text'][0]  # Ground truth text
            num_tokens = batch['num_tokens'].item()
            
            # Generate (unconditional for pretraining)
            generated_img = model.generate(
                condition=None,
                num_inference_steps=20,
                device=device,
            )
            
            # Convert to numpy
            generated_np = (generated_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Decode
            recovered_text, recovered_token_ids = token_decoder.decode(
                generated_np,
                num_tokens=num_tokens,
                stop_on_black=True
            )
            
            # Calculate accuracy
            # Token-level comparison
            true_tokens = token_decoder.tokenizer.encode(text, add_special_tokens=False)
            recovered_tokens = recovered_token_ids
            
            # Compare tokens
            min_len = min(len(true_tokens), len(recovered_tokens))
            if min_len > 0:
                matches = sum(1 for j in range(min_len) if true_tokens[j] == recovered_tokens[j])
                token_acc = matches / min_len
            else:
                token_acc = 0.0
            
            total_tokens += min_len
            correct_tokens += matches
            results.append({
                'original_text': text,
                'recovered_text': recovered_text,
                'token_accuracy': token_acc,
                'num_tokens': num_tokens,
            })
    
    avg_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'token_accuracy': avg_token_acc,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'results': results,
    }


def evaluate_generation_quality(
    model,
    dataloader,
    device: str = 'cuda',
    num_samples: int = 50,
):
    """评估生成质量（图像质量指标）"""
    model.eval()
    
    total_mse = 0.0
    total_l1 = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating quality')):
            if i >= num_samples:
                break
            
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)
            
            # Generate (unconditional)
            generated_img = model.generate(
                condition=None,
                num_inference_steps=20,
                device=device,
            )
            
            # Calculate metrics (only on valid pixels)
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            diff = (generated_img - clean.unsqueeze(0)) ** 2
            masked_diff = diff * mask_expanded
            mse = masked_diff.sum().item() / mask_expanded.sum().item()
            
            l1_diff = torch.abs(generated_img - clean.unsqueeze(0))
            masked_l1 = l1_diff * mask_expanded
            l1 = masked_l1.sum().item() / mask_expanded.sum().item()
            
            total_mse += mse
            total_l1 += l1
    
    avg_mse = total_mse / num_samples
    avg_l1 = total_l1 / num_samples
    
    return {
        'mse': avg_mse,
        'l1': avg_l1,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate JiT-based Token2Img Diffusion Model')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='JiT-B/4', help='Model name (JiT-B/4 for 64×64, JiT-B/16 for 256×256)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    # Data args
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--tokenizer_path', type=str, default='/root/data/AI/pretrain/Qwen2.5-7B-Instruct', help='Tokenizer path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    
    # Evaluation args
    parser.add_argument('--eval_token_recovery', action='store_true', default=True, help='Evaluate token recovery')
    parser.add_argument('--eval_quality', action='store_true', default=True, help='Evaluate generation quality')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='./evaluation_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # Load token decoder
    token_decoder = RobustToken2Img(tokenizer)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, args.img_size, device)
    print("Model loaded successfully!")
    
    # Load dataset
    dataset = TokenImageDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        img_size=args.img_size,
        use_chat_template=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    results = {}
    
    # Evaluate token recovery
    if args.eval_token_recovery:
        print("\n" + "="*60)
        print("Evaluating Token Recovery...")
        print("="*60)
        token_results = evaluate_token_recovery(
            model, dataloader, token_decoder, device, args.num_samples
        )
        results['token_recovery'] = token_results
        
        print(f"\nToken Accuracy: {token_results['token_accuracy']*100:.2f}%")
        print(f"Total Tokens: {token_results['total_tokens']}")
        print(f"Correct Tokens: {token_results['correct_tokens']}")
    
    # Evaluate generation quality
    if args.eval_quality:
        print("\n" + "="*60)
        print("Evaluating Generation Quality...")
        print("="*60)
        quality_results = evaluate_generation_quality(
            model, dataloader, device, args.num_samples
        )
        results['quality'] = quality_results
        
        print(f"\nMSE: {quality_results['mse']:.6f}")
        print(f"L1: {quality_results['l1']:.6f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {results_path}")
    
    return results


if __name__ == '__main__':
    main()
