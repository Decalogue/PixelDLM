"""
构建预训练数据集

从 data/rwkv-world-v3-subsample/1m/subsample_1m.jsonl 文件读取数据
每行是一个 {"text": "xxx"} 格式的 JSON 对象
单个样本最大长度为 64*64=4096 tokens，超长则截断
将1%的数据划分为验证集，放到 data/val 目录
将99%的数据作为训练集，放到 data/train 目录
"""

import os
import json
import random
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer


def load_jsonl_file(jsonl_path: str) -> List[Dict[str, str]]:
    """
    从 JSONL 文件加载数据
    
    Args:
        jsonl_path: JSONL 文件路径，每行是一个 {"text": "xxx"} 格式的 JSON
    
    Returns:
        文本列表，每个元素是 {"text": "..."} 格式
    
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    texts = []
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"文件不存在: {jsonl_path}")
    
    print(f"读取 JSONL 文件: {jsonl_path}")
    skipped_lines = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'text' in data and isinstance(data['text'], str):
                        text = data['text'].strip()
                        if text:  # 确保文本不为空
                            texts.append({'text': text})
                        else:
                            skipped_lines += 1
                    else:
                        skipped_lines += 1
                        if line_num <= 10:  # 只显示前10个警告
                            print(f"警告: 第 {line_num} 行缺少 'text' 字段或格式不正确，跳过")
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if line_num <= 10:  # 只显示前10个错误
                        print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}，跳过")
                    continue
    except Exception as e:
        raise IOError(f"读取文件 {jsonl_path} 时出错: {e}")
    
    if skipped_lines > 0:
        print(f"跳过 {skipped_lines} 行无效数据")
    
    print(f"成功读取 {len(texts)} 条有效文本")
    
    if len(texts) == 0:
        raise ValueError(f"文件 {jsonl_path} 中没有有效的文本数据")
    
    return texts


def process_text_item(
    text_item: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_tokens: int = 4096,
    min_tokens: int = 128,
) -> List[Dict[str, str]]:
    """
    处理单个文本项，如果超过 max_tokens 则截断
    
    Args:
        text_item: 文本项，格式为 {"text": "..."}
        tokenizer: Tokenizer
        max_tokens: 最大 token 数（64*64=4096），超过则截断
        min_tokens: 最小 token 数（小于此值则跳过）
    
    Returns:
        样本列表（通常只有一个样本），每个样本格式为 {"text": "..."}
    """
    text = text_item.get('text', '').strip()
    if not text:
        return []
    
    try:
        # 使用截断 tokenize，确保不超过 max_tokens
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            add_special_tokens=False, 
            truncation=True, 
            max_length=max_tokens
        )
        token_ids = inputs['input_ids'][0].tolist()
        token_count = len(token_ids)
        
        # 检查最小长度
        if token_count < min_tokens:
            return []
        
        # 如果 token_count == max_tokens，说明可能是截断的，需要解码回文本
        # 如果 token_count < max_tokens，说明没有截断，直接使用原始文本
        if token_count == max_tokens:
            processed_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        else:
            processed_text = text
        
        if processed_text:
            return [{'text': processed_text}]
        
    except Exception as e:
        # Tokenize 失败，跳过这个文本
        pass
    
    return []


def build_pretrain_dataset(
    jsonl_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_tokens: int = 4096,
    min_tokens: int = 128,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    """
    构建预训练数据集
    
    Args:
        jsonl_path: JSONL 文件路径，每行是一个 {"text": "xxx"} 格式
        output_dir: 输出目录
        tokenizer_path: Tokenizer 路径
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
    
    # 加载 JSONL 文件
    print("=" * 60)
    print("开始构建预训练数据集")
    print("=" * 60)
    
    text_items = load_jsonl_file(jsonl_path)
    
    # 收集所有样本
    all_samples = []
    
    # 处理每个文本项
    print(f"\n处理 {len(text_items)} 条文本...")
    skipped_count = 0
    for text_item in tqdm(text_items, desc="  处理文本"):
        samples = process_text_item(text_item, tokenizer, max_tokens, min_tokens)
        if len(samples) > 0:
            all_samples.extend(samples)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 条文本（token 数 < {min_tokens} 或其他原因）")
    
    print(f"\n总共生成 {len(all_samples)} 个样本")
    
    # 检查是否有样本
    if len(all_samples) == 0:
        raise ValueError("没有生成任何样本，请检查输入数据和参数设置")
    
    # 随机打乱
    print("\n随机打乱样本...")
    random.shuffle(all_samples)
    
    # 划分训练集和验证集
    val_size = int(len(all_samples) * val_ratio)
    # 确保验证集至少有一个样本（如果总样本数很少）
    if val_size == 0 and len(all_samples) > 0:
        val_size = 1
    
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"\n划分数据集:")
    print(f"  训练集: {len(train_samples)} 个样本 ({len(train_samples)/len(all_samples)*100:.2f}%)")
    print(f"  验证集: {len(val_samples)} 个样本 ({len(val_samples)/len(all_samples)*100:.2f}%)")
    
    # 保存训练集（分批保存，避免单个文件过大）
    print(f"\n保存训练集到 {train_dir}...")
    batch_size = 10000  # 每个文件保存10000个样本
    train_file_count = 0
    for i in tqdm(range(0, len(train_samples), batch_size), desc="  保存训练集"):
        batch = train_samples[i:i + batch_size]
        batch_file = os.path.join(train_dir, f'train_{train_file_count:04d}.json')
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
            train_file_count += 1
        except Exception as e:
            print(f"警告: 保存训练集文件 {batch_file} 失败: {e}")
    
    # 保存验证集（分批保存）
    print(f"\n保存验证集到 {val_dir}...")
    val_file_count = 0
    for i in tqdm(range(0, len(val_samples), batch_size), desc="  保存验证集"):
        batch = val_samples[i:i + batch_size]
        batch_file = os.path.join(val_dir, f'val_{val_file_count:04d}.json')
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
            val_file_count += 1
        except Exception as e:
            print(f"警告: 保存验证集文件 {batch_file} 失败: {e}")
    
    # 保存统计信息
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'max_tokens': max_tokens,
        'min_tokens': min_tokens,
        'img_size': 64,  # 64*64=4096 tokens
        'source_file': jsonl_path,
        'input_texts': len(text_items),
        'train_files': train_file_count,
        'val_files': val_file_count,
        'samples_per_file': batch_size,
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
    print(f"  输入文本数: {len(text_items):,}")
    print(f"\n输出目录:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  统计信息: {stats_file}")


def main():
    """主函数"""
    # JSONL 文件路径
    jsonl_path = 'data/rwkv-world-v3-subsample/1m/subsample_1m.jsonl'
    
    # 输出目录
    output_dir = 'data'
    
    # 参数
    tokenizer_path = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct'
    max_tokens = 64 * 64  # 4096 tokens (64*64 图像容量)
    min_tokens = 128  # 最小 token 数
    val_ratio = 0.01  # 1% 验证集
    
    # 构建数据集
    build_pretrain_dataset(
        jsonl_path=jsonl_path,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        val_ratio=val_ratio,
        seed=42,
    )


if __name__ == '__main__':
    main()
