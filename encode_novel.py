import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import time
import argparse
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
from token2img import ensure_dir, token_ids_to_img, img2text


def encode_novel_to_image(
    novel_path: str,
    output_path: str = './data/novel_1024x1024.png',
    tokenizer_path: str = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct',
    image_size: int = 1024,
    chunk_size: int = 10240
):
    """
    将长篇小说编码到指定尺寸的图像中
    
    Args:
        novel_path: 小说文件路径
        output_path: 输出图像路径
        image_size: 图像尺寸（正方形）
        tokenizer_path: Tokenizer 路径
    
    Returns:
        (实际 token 数量, 图像路径)
    """
    print(f"加载 Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"读取小说文件: {novel_path}")
    with open(novel_path, 'r', encoding='utf-8') as f:
        novel_text = f.read()
    
    print(f"小说文本长度: {len(novel_text)} 字符")
    
    print(f"分段 Tokenizing（每段 {chunk_size} 字符）...")
    
    all_token_ids = []
    total_chunks = (len(novel_text) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(novel_text), chunk_size):
        chunk_text = novel_text[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        
        # Tokenize 当前片段
        inputs = tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
        chunk_token_ids = inputs['input_ids'][0].tolist()
        all_token_ids.extend(chunk_token_ids)
        
        if chunk_num % 10 == 0 or chunk_num == total_chunks:
            print(f"  处理进度: {chunk_num}/{total_chunks} 段, 累计 tokens: {len(all_token_ids):,}")
    
    token_ids = all_token_ids
    num_tokens = len(token_ids)
    
    print(f"Token 数量: {num_tokens:,}")
    
    # 计算图像容量
    max_tokens = image_size * image_size
    print(f"图像容量: {max_tokens:,} tokens ( {image_size}×{image_size} )")
    
    # 检查是否超出容量
    if num_tokens > max_tokens:
        print(f"⚠️  警告: Token 数量 ({num_tokens:,}) 超过图像容量 ({max_tokens:,})")
        print(f"将截断到前 {max_tokens:,} 个 tokens")
        token_ids = token_ids[:max_tokens]
        num_tokens = max_tokens
    else:
        print(f"✅ Token 数量在图像容量范围内")
        utilization = num_tokens / max_tokens * 100
        print(f"图像利用率: {utilization:.2f}%")
    
    # 编码到图像
    print(f"\n编码到图像 ({image_size}×{image_size})...")
    start_time = time.time()

    print("直接使用 token_ids 编码到图像...")
    img = token_ids_to_img(
        token_ids,
        size=(image_size, image_size),
        save_path=output_path
    )
    
    encode_time = time.time() - start_time
    print(f"编码完成，耗时: {encode_time:.2f} 秒")
    
    # 验证文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"图像文件大小: {file_size:.2f} MB")
    
    return num_tokens


def decode_novel_from_image(
    image_path: str,
    num_tokens: int,
    tokenizer_path: str = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct',
    output_text_path: Optional[str] = None
):
    """
    从图像解码回小说文本
    
    Args:
        image_path: 图像路径
        num_tokens: 实际 token 数量
        tokenizer_path: Tokenizer 路径
        output_text_path: 输出文本路径（可选）
    
    Returns:
        恢复的文本
    """
    print(f"\n从图像解码...")
    print(f"图像路径: {image_path}")
    print(f"预期 token 数量: {num_tokens:,}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    start_time = time.time()
    recovered_text = img2text(image_path, num_tokens=num_tokens)
    decode_time = time.time() - start_time
    
    print(f"解码完成，耗时: {decode_time:.2f} 秒")
    print(f"恢复文本长度: {len(recovered_text):,} 字符")
    
    # 保存恢复的文本
    if output_text_path:
        os.makedirs(os.path.dirname(output_text_path) if os.path.dirname(output_text_path) else '.', exist_ok=True)
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(recovered_text)
        print(f"恢复的文本已保存到: {output_text_path}")
    
    return recovered_text


def verify_accuracy(
    original_text: str,
    recovered_text: str,
    tokenizer,
    sample_size: int = 1024
):
    """
    验证恢复准确率（采样验证）
    
    Args:
        original_text: 原始文本
        recovered_text: 恢复的文本
        tokenizer: Tokenizer
        sample_size: 采样大小（字符数）
    """
    print(f"\n验证恢复准确率...")
    
    # Tokenize 原始文本和恢复文本
    orig_inputs = tokenizer(original_text[:sample_size], return_tensors="pt", add_special_tokens=False)
    recv_inputs = tokenizer(recovered_text[:sample_size], return_tensors="pt", add_special_tokens=False)
    
    orig_tokens = orig_inputs['input_ids'][0].tolist()
    recv_tokens = recv_inputs['input_ids'][0].tolist()
    
    # 计算 token 级准确率
    min_len = min(len(orig_tokens), len(recv_tokens))
    if min_len > 0:
        matches = sum(1 for i in range(min_len) if orig_tokens[i] == recv_tokens[i])
        token_accuracy = matches / len(orig_tokens) if len(orig_tokens) > 0 else 0.0
        
        print(f"采样长度: {sample_size} 字符")
        print(f"原始 tokens: {len(orig_tokens)}")
        print(f"恢复 tokens: {len(recv_tokens)}")
        print(f"匹配 tokens: {matches}")
        print(f"Token 准确率: {token_accuracy*100:.2f}%")
        
        return token_accuracy
    else:
        print("无法计算准确率（文本为空）")
        return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将长篇小说编码到图像')
    parser.add_argument('--novel_path', type=str, 
                       default='data/金庸-倚天屠龙记.txt',
                       help='小说文件路径')
    parser.add_argument('--output_image_path', type=str,
                       default='./data/novel_1024x1024.png',
                       help='输出图像路径')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='图像尺寸（正方形）')
    parser.add_argument('--tokenizer_path', type=str,
                       default='/root/data/AI/pretrain/Qwen2.5-7B-Instruct',
                       help='Tokenizer 路径')
    parser.add_argument('--decode', action='store_true',
                       help='是否解码验证')
    parser.add_argument('--output_text_path', type=str,
                       default='./data/novel_recovered.txt',
                       help='恢复文本输出路径')
    parser.add_argument('--verify', action='store_true',
                       help='是否验证准确率')
    parser.add_argument('--chunk_size', type=int, default=10240,
                       help='分段大小')
    args = parser.parse_args()
    
    print("=" * 60)
    print("编码小说到图像")
    print("=" * 60)
    num_tokens = encode_novel_to_image(
        args.novel_path,
        args.output_image_path,
        args.tokenizer_path,
        args.image_size,
        args.chunk_size
    )
    
    # 解码验证
    if args.decode:
        print("\n" + "=" * 60)
        print("解码验证")
        print("=" * 60)
        
        # 读取原始文本（用于验证）
        with open(args.novel_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        # 如果截断了，只取对应的部分
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        chunk_text = original_text[:args.chunk_size] # 截断原始文本用于对比
        chunk_inputs = tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
        chunk_token_ids = chunk_inputs['input_ids'][0].tolist()
        chunk_num_tokens = len(chunk_token_ids)
        chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=False)
        
        recovered_text = decode_novel_from_image(
            args.output_image_path,
            chunk_num_tokens,
            args.tokenizer_path,
            args.output_text_path
        )
        
        # 验证准确率
        if args.verify:
            print("\n" + "=" * 60)
            print("准确率验证")
            print("=" * 60)
            verify_accuracy(chunk_text, recovered_text, tokenizer, sample_size=chunk_num_tokens)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"图像路径: {args.output_image_path}")
    print(f"Token 数量: {num_tokens:,}")
    print(f"图像尺寸: {args.image_size}×{args.image_size}")
    print(f"恢复文本路径: {args.output_text_path}")
