"""
构建微调数据集

从 data/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl 文件读取数据
每行是一个 {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]} 格式的 JSON 对象
转换为 {"prompt": "...", "answer": "..."} 格式
将1%的数据划分为验证集，放到 data/val_ft 目录
将99%的数据作为训练集，放到 data/train_ft 目录
"""

import os
import json
import random
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer


def load_jsonl_file(jsonl_path: str) -> List[Dict]:
    """
    从 JSONL 文件加载数据
    
    Args:
        jsonl_path: JSONL 文件路径，每行是一个 {"messages": [...]} 格式的 JSON
    
    Returns:
        对话列表，每个元素是 {"messages": [...]} 格式
    
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    conversations = []
    
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
                    if 'messages' in data and isinstance(data['messages'], list):
                        # 验证 messages 格式
                        if len(data['messages']) >= 2:
                            # 检查是否有 user 和 assistant
                            has_user = any(msg.get('role') == 'user' for msg in data['messages'])
                            has_assistant = any(msg.get('role') == 'assistant' for msg in data['messages'])
                            if has_user and has_assistant:
                                conversations.append(data)
                            else:
                                skipped_lines += 1
                                if line_num <= 10:
                                    print(f"警告: 第 {line_num} 行缺少 user 或 assistant 角色，跳过")
                        else:
                            skipped_lines += 1
                            if line_num <= 10:
                                print(f"警告: 第 {line_num} 行 messages 数量不足，跳过")
                    else:
                        skipped_lines += 1
                        if line_num <= 10:
                            print(f"警告: 第 {line_num} 行缺少 'messages' 字段或格式不正确，跳过")
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if line_num <= 10:
                        print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}，跳过")
                    continue
    except Exception as e:
        raise IOError(f"读取文件 {jsonl_path} 时出错: {e}")
    
    if skipped_lines > 0:
        print(f"跳过 {skipped_lines} 行无效数据")
    
    print(f"成功读取 {len(conversations)} 条有效对话")
    
    if len(conversations) == 0:
        raise ValueError(f"文件 {jsonl_path} 中没有有效的对话数据")
    
    return conversations


def convert_messages_to_prompt_answer(
    messages: List[Dict],
    tokenizer: AutoTokenizer,
    max_prompt_tokens: int = 4096,
    max_answer_tokens: int = 4096,
    min_answer_tokens: int = 32,
) -> Dict[str, str]:
    """
    将 messages 格式转换为 prompt/answer 格式
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        tokenizer: Tokenizer
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
    
    Returns:
        格式为 {"prompt": "...", "answer": "..."} 的字典，如果转换失败返回 None
    """
    # 提取 user 和 assistant 消息
    user_messages = []
    assistant_messages = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '').strip()
        if not content:
            continue
        
        if role == 'user':
            user_messages.append(content)
        elif role == 'assistant':
            assistant_messages.append(content)
    
    # 合并 user 消息为 prompt
    if len(user_messages) == 0:
        return None
    
    prompt = '\n'.join(user_messages)
    
    # 合并 assistant 消息为 answer
    if len(assistant_messages) == 0:
        return None
    
    answer = '\n'.join(assistant_messages)
    
    # Tokenize 检查长度
    try:
        # 检查 prompt 长度
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens
        )
        prompt_token_count = len(prompt_inputs['input_ids'][0])
        
        # 如果 prompt 被截断，需要解码回文本
        if prompt_token_count == max_prompt_tokens:
            prompt = tokenizer.decode(prompt_inputs['input_ids'][0], skip_special_tokens=True).strip()
        
        # 检查 answer 长度
        answer_inputs = tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_answer_tokens
        )
        answer_token_count = len(answer_inputs['input_ids'][0])
        
        # 检查最小长度
        if answer_token_count < min_answer_tokens:
            return None
        
        # 如果 answer 被截断，需要解码回文本
        if answer_token_count == max_answer_tokens:
            answer = tokenizer.decode(answer_inputs['input_ids'][0], skip_special_tokens=True).strip()
        
        return {
            'prompt': prompt,
            'answer': answer
        }
        
    except Exception as e:
        # Tokenize 失败，跳过
        return None


def process_conversation_item(
    conversation_item: Dict,
    tokenizer: AutoTokenizer,
    max_prompt_tokens: int = 4096,
    max_answer_tokens: int = 4096,
    min_answer_tokens: int = 32,
) -> Dict[str, str]:
    """
    处理单个对话项，转换为 prompt/answer 格式
    
    Args:
        conversation_item: 对话项，格式为 {"messages": [...]}
        tokenizer: Tokenizer
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
    
    Returns:
        格式为 {"prompt": "...", "answer": "..."} 的字典，如果处理失败返回 None
    """
    messages = conversation_item.get('messages', [])
    if not messages:
        return None
    
    return convert_messages_to_prompt_answer(
        messages,
        tokenizer,
        max_prompt_tokens,
        max_answer_tokens,
        min_answer_tokens
    )


def build_fintune_dataset(
    jsonl_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_prompt_tokens: int = 4096,
    max_answer_tokens: int = 4096,
    min_answer_tokens: int = 32,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    """
    构建微调数据集
    
    Args:
        jsonl_path: JSONL 文件路径，每行是一个 {"messages": [...]} 格式
        output_dir: 输出目录
        tokenizer_path: Tokenizer 路径
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
        val_ratio: 验证集比例（默认1%）
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train_ft')
    val_dir = os.path.join(output_dir, 'val_ft')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 加载 JSONL 文件
    print("=" * 60)
    print("开始构建微调数据集")
    print("=" * 60)
    
    conversation_items = load_jsonl_file(jsonl_path)
    
    # 收集所有样本
    all_samples = []
    
    # 处理每个对话项
    print(f"\n处理 {len(conversation_items)} 条对话...")
    skipped_count = 0
    for conversation_item in tqdm(conversation_items, desc="  处理对话"):
        sample = process_conversation_item(
            conversation_item,
            tokenizer,
            max_prompt_tokens,
            max_answer_tokens,
            min_answer_tokens
        )
        if sample is not None:
            all_samples.append(sample)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 条对话（格式不正确、token 数不足或其他原因）")
    
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
        batch_file = os.path.join(train_dir, f'train_ft_{train_file_count:04d}.json')
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
        batch_file = os.path.join(val_dir, f'val_ft_{val_file_count:04d}.json')
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
        'max_prompt_tokens': max_prompt_tokens,
        'max_answer_tokens': max_answer_tokens,
        'min_answer_tokens': min_answer_tokens,
        'source_file': jsonl_path,
        'input_conversations': len(conversation_items),
        'train_files': train_file_count,
        'val_files': val_file_count,
        'samples_per_file': batch_size,
    }
    
    stats_file = os.path.join(output_dir, 'stats_ft.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("数据集构建完成！")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  总样本数: {len(all_samples):,}")
    print(f"  训练集: {len(train_samples):,} 个样本")
    print(f"  验证集: {len(val_samples):,} 个样本")
    print(f"  Prompt 最大 tokens: {max_prompt_tokens}")
    print(f"  Answer 最大 tokens: {max_answer_tokens}")
    print(f"  Answer 最小 tokens: {min_answer_tokens}")
    print(f"  输入对话数: {len(conversation_items):,}")
    print(f"\n输出目录:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  统计信息: {stats_file}")


def main():
    """主函数"""
    # JSONL 文件路径
    jsonl_path = 'data/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl'
    
    # 输出目录
    output_dir = 'data'
    
    # 参数
    tokenizer_path = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct'
    max_prompt_tokens = 4096  # prompt 最大 token 数（64×64 图像容量）
    max_answer_tokens = 4096  # answer 最大 token 数（64×64 图像容量）
    min_answer_tokens = 32  # answer 最小 token 数
    val_ratio = 0.01  # 1% 验证集
    
    # 构建数据集
    build_fintune_dataset(
        jsonl_path=jsonl_path,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        max_prompt_tokens=max_prompt_tokens,
        max_answer_tokens=max_answer_tokens,
        min_answer_tokens=min_answer_tokens,
        val_ratio=val_ratio,
        seed=42,
    )


if __name__ == '__main__':
    main()
