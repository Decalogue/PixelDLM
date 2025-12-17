"""
构建预训练数据集

从 /root/data/AI/creator/data/raw 目录下的 小说_1 小说_2 小说_3 小说_4 子目录
每个目录采样2500本小说，每个小说按照随机 token 长度（256-4096 tokens）进行分段
每个分段作为一个样本，构建为 dataset.py 中的格式
将1%的数据划分为验证集，放到 data/val 目录
将99%的数据作为训练集，放到 data/train 目录

注意：64*64=4096 tokens，每个样本的 token 数量随机在 256-4096 之间，模拟真实 LLM 训练数据的多样性
"""

import os
import json
import random
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer


def get_novel_files(novel_dir: str, sample_size: int = 2500) -> List[str]:
    """
    从目录中获取小说文件列表，并采样指定数量
    
    Args:
        novel_dir: 小说目录路径
        sample_size: 采样数量
    
    Returns:
        小说文件路径列表
    """
    # 获取所有 .txt 文件
    txt_files = []
    for root, dirs, files in os.walk(novel_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.TXT'):
                txt_files.append(os.path.join(root, file))
    
    # 随机采样
    if len(txt_files) > sample_size:
        txt_files = random.sample(txt_files, sample_size)
    
    return txt_files


def split_text_into_random_token_chunks(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 4096,
    min_tokens: int = 256,
    chunk_size_chars: int = 4096,
) -> List[str]:
    """
    将文本按照随机 token 长度分段（模拟真实 LLM 训练数据的多样性）
    
    优化：对于超长文本，先按字符数分段，再按 token 长度分段，避免一次性 tokenize 整个文本
    
    Args:
        text: 输入文本
        tokenizer: Tokenizer
        max_tokens: 最大 token 数（64*64=4096）
        min_tokens: 最小 token 数
        chunk_size_chars: 字符分段大小（避免超过 tokenizer 最大长度）
    
    Returns:
        文本片段列表
    """
    chunks = []
    
    # 对于超长文本，先按字符数分段（避免超过 tokenizer 最大长度 131072）
    text_chunks = []
    if len(text) > chunk_size_chars:
        for i in range(0, len(text), chunk_size_chars):
            text_chunks.append(text[i:i + chunk_size_chars])
    else:
        text_chunks = [text]
    
    # 对每个文本段进行处理
    for text_segment in text_chunks:
        # 先 tokenize 文本段
        try:
            inputs = tokenizer(text_segment, return_tensors="pt", add_special_tokens=False, truncation=False)
            token_ids = inputs['input_ids'][0].tolist()
        except Exception as e:
            # 如果还是太长，尝试截断
            try:
                inputs = tokenizer(text_segment, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=131072)
                token_ids = inputs['input_ids'][0].tolist()
            except Exception as e2:
                print(f"警告: Tokenize 失败: {e2}")
                continue
        
        # 按照随机长度分段
        i = 0
        while i < len(token_ids):
            # 随机选择 chunk 长度（在 min_tokens 和 max_tokens 之间）
            # 但不超过剩余 tokens
            remaining = len(token_ids) - i
            if remaining < min_tokens:
                # 剩余 tokens 不足，直接使用剩余部分
                chunk_size = remaining
            else:
                chunk_size = random.randint(min_tokens, min(max_tokens, remaining))
            
            if chunk_size == 0:
                break
            
            # 提取 token_ids
            chunk_token_ids = token_ids[i:i + chunk_size]
            
            # 解码回文本
            try:
                chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                if len(chunk_text.strip()) > 0:  # 跳过空片段
                    chunks.append(chunk_text)
            except Exception as e:
                print(f"警告: 解码失败: {e}")
            
            i += chunk_size
    
    return chunks


def process_novel_file(
    file_path: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 4096,
    min_tokens: int = 256,
) -> List[Dict[str, str]]:
    """
    处理单个小说文件，按照随机 token 长度分段
    
    Args:
        file_path: 小说文件路径
        tokenizer: Tokenizer
        max_tokens: 最大 token 数（64*64=4096）
        min_tokens: 最小 token 数
    
    Returns:
        样本列表，每个样本格式为 {"text": "..."}
    """
    samples = []
    
    try:
        # 读取文件（尝试不同编码）
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if text is None:
            print(f"警告: 无法读取文件 {file_path}，跳过")
            return samples
        
        # 按照随机 token 长度分段
        chunks = split_text_into_random_token_chunks(
            text, tokenizer, max_tokens, min_tokens, chunk_size_chars=4096
        )
        
        # 构建样本
        for chunk in chunks:
            samples.append({'text': chunk})
    
    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时出错: {e}")
    
    return samples


def build_pretrain_dataset(
    source_dirs: List[str],
    output_dir: str,
    tokenizer_path: str,
    samples_per_dir: int = 2500,
    max_tokens: int = 4096,
    min_tokens: int = 256,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    """
    构建预训练数据集
    
    Args:
        source_dirs: 源目录列表（小说_1, 小说_2, 小说_3, 小说_4）
        output_dir: 输出目录
        tokenizer_path: Tokenizer 路径
        samples_per_dir: 每个目录采样的小说数量
        max_tokens: 最大 token 数（64*64=4096）
        min_tokens: 最小 token 数
        val_ratio: 验证集比例（默认1%）
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 收集所有样本
    all_samples = []
    
    print("=" * 60)
    print("开始构建预训练数据集")
    print("=" * 60)
    
    # 处理每个目录
    for novel_dir in source_dirs:
        if not os.path.exists(novel_dir):
            print(f"警告: 目录不存在，跳过: {novel_dir}")
            continue
        
        print(f"\n处理目录: {novel_dir}")
        
        # 获取小说文件列表
        novel_files = get_novel_files(novel_dir, samples_per_dir)
        print(f"  找到 {len(novel_files)} 本小说")
        
        # 处理每本小说
        dir_samples = []
        for file_path in tqdm(novel_files, desc=f"  处理 {os.path.basename(novel_dir)}"):
            samples = process_novel_file(file_path, tokenizer, max_tokens, min_tokens)
            dir_samples.extend(samples)
        
        print(f"  生成 {len(dir_samples)} 个样本")
        all_samples.extend(dir_samples)
    
    print(f"\n总共生成 {len(all_samples)} 个样本")
    
    # 随机打乱
    print("\n随机打乱样本...")
    random.shuffle(all_samples)
    
    # 划分训练集和验证集
    val_size = int(len(all_samples) * val_ratio)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"\n划分数据集:")
    print(f"  训练集: {len(train_samples)} 个样本 ({len(train_samples)/len(all_samples)*100:.2f}%)")
    print(f"  验证集: {len(val_samples)} 个样本 ({len(val_samples)/len(all_samples)*100:.2f}%)")
    
    # 保存训练集（分批保存，避免单个文件过大）
    print(f"\n保存训练集到 {train_dir}...")
    batch_size = 10000  # 每个文件保存10000个样本
    for i in tqdm(range(0, len(train_samples), batch_size), desc="  保存训练集"):
        batch = train_samples[i:i + batch_size]
        batch_file = os.path.join(train_dir, f'train_{i//batch_size:04d}.json')
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
    
    # 保存验证集（分批保存）
    print(f"\n保存验证集到 {val_dir}...")
    for i in tqdm(range(0, len(val_samples), batch_size), desc="  保存验证集"):
        batch = val_samples[i:i + batch_size]
        batch_file = os.path.join(val_dir, f'val_{i//batch_size:04d}.json')
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'max_tokens': max_tokens,
        'min_tokens': min_tokens,
        'img_size': 64,  # 64*64=4096 tokens
        'samples_per_dir': samples_per_dir,
        'source_dirs': source_dirs,
    }
    
    stats_file = os.path.join(output_dir, 'stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("数据集构建完成！")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  总样本数: {len(all_samples):,}")
    print(f"  训练集: {len(train_samples):,} 个样本")
    print(f"  验证集: {len(val_samples):,} 个样本")
    print(f"  图像尺寸: 64×64 = {max_tokens} tokens")
    print(f"  Token 范围: {min_tokens}-{max_tokens} tokens/样本")
    print(f"  每个目录采样: {samples_per_dir} 本小说")
    print(f"\n输出目录:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  统计信息: {stats_file}")


def main():
    """主函数"""
    # 源目录
    base_dir = '/root/data/AI/creator/data/raw'
    source_dirs = [
        os.path.join(base_dir, '小说_1'),
        os.path.join(base_dir, '小说_2'),
        os.path.join(base_dir, '小说_3'),
        os.path.join(base_dir, '小说_4'),
    ]
    
    # 输出目录
    output_dir = '/root/data/AI/dlm/jit/data'
    
    # 参数
    tokenizer_path = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct'
    samples_per_dir = 2500  # 每个目录采样2500本小说
    max_tokens = 64 * 64  # 4096 tokens (64*64 图像容量)
    min_tokens = 256  # 最小 token 数
    val_ratio = 0.01  # 1% 验证集
    
    # 构建数据集
    build_pretrain_dataset(
        source_dirs=source_dirs,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        samples_per_dir=samples_per_dir,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        val_ratio=val_ratio,
        seed=42,
    )


if __name__ == '__main__':
    main()
