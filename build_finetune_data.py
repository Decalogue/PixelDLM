"""
构建微调数据集

支持两种数据源：
1. JSONL 格式：data/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl
   格式：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
2. Parquet 格式：data/7M_core/*.parquet
   格式：{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

转换为 {"prompt": "...", "answer": "..."} 格式
将1%的数据划分为验证集，放到 data/val_ft 目录
将99%的数据作为训练集，放到 data/train_ft 目录
"""

import os
import json
import random
import glob
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoTokenizer

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: pandas 未安装，无法读取 parquet 文件。请运行: pip install pandas pyarrow")


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
    format_type: str = 'messages',  # 'messages' 或 'conversations'
) -> Optional[Dict[str, str]]:
    """
    将 messages 或 conversations 格式转换为 prompt/answer 格式
    
    Args:
        messages: 消息列表
            - messages 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            - conversations 格式: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        tokenizer: Tokenizer
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
        format_type: 数据格式类型，'messages' 或 'conversations'
    
    Returns:
        格式为 {"prompt": "...", "answer": "..."} 的字典，如果转换失败返回 None
    """
    # 提取 user/human 和 assistant/gpt 消息
    user_messages = []
    assistant_messages = []
    
    if format_type == 'messages':
        # 处理 messages 格式 (role/content)
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '').strip()
            if not content:
                continue
            
            if role == 'user':
                user_messages.append(content)
            elif role == 'assistant':
                assistant_messages.append(content)
    elif format_type == 'conversations':
        # 处理 conversations 格式 (from/value)
        for msg in messages:
            from_role = msg.get('from', '').lower()
            value = msg.get('value', '').strip()
            if not value:
                continue
            
            if from_role in ['human', 'user']:
                user_messages.append(value)
            elif from_role in ['gpt', 'assistant', 'bot']:
                assistant_messages.append(value)
    else:
        return None
    
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
    format_type: str = 'messages',
) -> Optional[Dict[str, str]]:
    """
    处理单个对话项，转换为 prompt/answer 格式
    
    Args:
        conversation_item: 对话项
            - messages 格式: {"messages": [...]}
            - conversations 格式: {"conversations": [...]}
        tokenizer: Tokenizer
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
        format_type: 数据格式类型，'messages' 或 'conversations'
    
    Returns:
        格式为 {"prompt": "...", "answer": "..."} 的字典，如果处理失败返回 None
    """
    if format_type == 'messages':
        messages = conversation_item.get('messages', [])
    elif format_type == 'conversations':
        messages = conversation_item.get('conversations', [])
        # 如果是 numpy array，转换为列表
        if HAS_PANDAS and isinstance(messages, np.ndarray):
            messages = messages.tolist()
    else:
        return None
    
    if not messages or len(messages) < 2:
        return None
    
    return convert_messages_to_prompt_answer(
        messages,
        tokenizer,
        max_prompt_tokens,
        max_answer_tokens,
        min_answer_tokens,
        format_type
    )


def load_parquet_files(parquet_dir: str) -> List[Dict]:
    """
    从 parquet 文件目录加载数据
    
    Args:
        parquet_dir: Parquet 文件目录路径
    
    Returns:
        对话列表，每个元素是 {"conversations": [...]} 格式
    """
    if not HAS_PANDAS:
        raise ImportError("pandas 未安装，无法读取 parquet 文件。请运行: pip install pandas pyarrow")
    
    conversations = []
    
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"目录不存在: {parquet_dir}")
    
    # 查找所有 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
    
    if not parquet_files:
        raise ValueError(f"目录 {parquet_dir} 中没有找到 parquet 文件")
    
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    skipped_count = 0
    
    for parquet_file in tqdm(parquet_files, desc="读取 parquet 文件"):
        try:
            df = pd.read_parquet(parquet_file)
            
            # 检查是否有 conversations 列
            if 'conversations' not in df.columns:
                print(f"警告: 文件 {parquet_file} 没有 'conversations' 列，跳过")
                continue
            
            # 处理每一行
            for idx, row in df.iterrows():
                conversations_data = row['conversations']
                
                # 如果是 numpy array，转换为列表
                if isinstance(conversations_data, np.ndarray):
                    conversations_data = conversations_data.tolist()
                
                # 验证格式
                if isinstance(conversations_data, list) and len(conversations_data) >= 2:
                    # 检查是否有 human 和 gpt
                    has_human = any(
                        msg.get('from', '').lower() in ['human', 'user'] 
                        for msg in conversations_data if isinstance(msg, dict)
                    )
                    has_gpt = any(
                        msg.get('from', '').lower() in ['gpt', 'assistant', 'bot']
                        for msg in conversations_data if isinstance(msg, dict)
                    )
                    
                    if has_human and has_gpt:
                        conversations.append({'conversations': conversations_data})
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
                    
        except Exception as e:
            print(f"警告: 读取文件 {parquet_file} 时出错: {e}")
            continue
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 条无效数据")
    
    print(f"成功读取 {len(conversations)} 条有效对话")
    
    if len(conversations) == 0:
        raise ValueError(f"从 {parquet_dir} 中没有读取到有效的对话数据")
    
    return conversations


def load_existing_dataset(train_dir: str, val_dir: str) -> tuple[List[Dict], List[Dict]]:
    """
    从已有的 train_ft 和 val_ft 目录加载数据
    
    Args:
        train_dir: 训练集目录路径
        val_dir: 验证集目录路径
    
    Returns:
        (train_samples, val_samples) 元组，每个是 [{"prompt": "...", "answer": "..."}] 格式
    """
    train_samples = []
    val_samples = []
    
    # 加载训练集
    if os.path.exists(train_dir):
        train_files = sorted(glob.glob(os.path.join(train_dir, '*.json')))
        print(f"从 {train_dir} 加载已有训练集，找到 {len(train_files)} 个文件...")
        for train_file in tqdm(train_files, desc="  加载训练集文件"):
            try:
                with open(train_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        train_samples.extend(data)
            except Exception as e:
                print(f"警告: 加载文件 {train_file} 失败: {e}")
    
    # 加载验证集
    if os.path.exists(val_dir):
        val_files = sorted(glob.glob(os.path.join(val_dir, '*.json')))
        print(f"从 {val_dir} 加载已有验证集，找到 {len(val_files)} 个文件...")
        for val_file in tqdm(val_files, desc="  加载验证集文件"):
            try:
                with open(val_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        val_samples.extend(data)
            except Exception as e:
                print(f"警告: 加载文件 {val_file} 失败: {e}")
    
    print(f"已加载训练集: {len(train_samples)} 个样本")
    print(f"已加载验证集: {len(val_samples)} 个样本")
    
    return train_samples, val_samples


def build_fintune_dataset(
    jsonl_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    existing_train_dir: Optional[str] = None,
    existing_val_dir: Optional[str] = None,
    output_dir: str = 'data',
    tokenizer_path: str = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct',
    max_prompt_tokens: int = 4096,
    max_answer_tokens: int = 4096,
    min_answer_tokens: int = 32,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    """
    构建微调数据集，支持 JSONL 和 Parquet 两种数据源，并可加载已有数据
    
    Args:
        jsonl_path: JSONL 文件路径（可选），每行是一个 {"messages": [...]} 格式
        parquet_dir: Parquet 文件目录路径（可选），包含 {"conversations": [...]} 格式
        existing_train_dir: 已有训练集目录路径（可选），如果提供则直接加载，不再处理 jsonl_path
        existing_val_dir: 已有验证集目录路径（可选），如果提供则直接加载
        output_dir: 输出目录
        tokenizer_path: Tokenizer 路径
        max_prompt_tokens: prompt 最大 token 数
        max_answer_tokens: answer 最大 token 数
        min_answer_tokens: answer 最小 token 数
        val_ratio: 验证集比例（默认1%），仅对新数据生效
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查至少有一个数据源
    if not jsonl_path and not parquet_dir and not existing_train_dir:
        raise ValueError("必须提供 jsonl_path、parquet_dir 或 existing_train_dir 至少一个数据源")
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train_ft')
    val_dir = os.path.join(output_dir, 'val_ft')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 加载已有数据
    existing_train_samples = []
    existing_val_samples = []
    if existing_train_dir or existing_val_dir:
        print("=" * 60)
        print("加载已有数据集")
        print("=" * 60)
        existing_train_samples, existing_val_samples = load_existing_dataset(
            existing_train_dir or train_dir,
            existing_val_dir or val_dir
        )
    
    # 处理新数据源
    new_samples = []
    data_sources = []
    
    if jsonl_path or parquet_dir:
        # 加载 Tokenizer（仅处理新数据时需要）
        print("\n加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        print("=" * 60)
        print("处理新数据源")
        print("=" * 60)
        
        all_conversation_items = []
        
        # 加载 JSONL 数据（如果提供了 jsonl_path 且没有使用已有数据）
        if jsonl_path and not existing_train_dir:
            print(f"\n[数据源] 加载 JSONL 文件: {jsonl_path}")
            jsonl_items = load_jsonl_file(jsonl_path)
            all_conversation_items.extend(jsonl_items)
            data_sources.append(('jsonl', jsonl_path, len(jsonl_items)))
        
        # 加载 Parquet 数据
        if parquet_dir:
            print(f"\n[数据源] 加载 Parquet 目录: {parquet_dir}")
            parquet_items = load_parquet_files(parquet_dir)
            all_conversation_items.extend(parquet_items)
            data_sources.append(('parquet', parquet_dir, len(parquet_items)))
        
        if all_conversation_items:
            print(f"\n总共加载 {len(all_conversation_items)} 条对话")
            
            # 处理每个对话项
            print(f"\n处理 {len(all_conversation_items)} 条对话...")
            skipped_count = 0
            for conversation_item in tqdm(all_conversation_items, desc="  处理对话"):
                # 判断数据格式
                if 'messages' in conversation_item:
                    format_type = 'messages'
                elif 'conversations' in conversation_item:
                    format_type = 'conversations'
                else:
                    skipped_count += 1
                    continue
                
                sample = process_conversation_item(
                    conversation_item,
                    tokenizer,
                    max_prompt_tokens,
                    max_answer_tokens,
                    min_answer_tokens,
                    format_type
                )
                if sample is not None:
                    new_samples.append(sample)
                else:
                    skipped_count += 1
            
            if skipped_count > 0:
                print(f"跳过 {skipped_count} 条对话（格式不正确、token 数不足或其他原因）")
            
            print(f"\n新数据源生成 {len(new_samples)} 个样本")
    
    # 合并已有数据和新数据
    all_samples = existing_train_samples + existing_val_samples + new_samples
    
    if len(all_samples) == 0:
        raise ValueError("没有生成任何样本，请检查输入数据和参数设置")
    
    print(f"\n总共 {len(all_samples)} 个样本（已有: {len(existing_train_samples) + len(existing_val_samples)}, 新增: {len(new_samples)}）")
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 条对话（格式不正确、token 数不足或其他原因）")
    
    print(f"\n总共生成 {len(all_samples)} 个样本")
    
    # 检查是否有样本
    if len(all_samples) == 0:
        raise ValueError("没有生成任何样本，请检查输入数据和参数设置")
    
    # 划分训练集和验证集
    if existing_train_samples or existing_val_samples:
        # 如果已有数据，保持已有划分，只对新数据划分
        print("\n合并已有数据和新数据...")
        if new_samples:
            # 对新数据划分
            random.shuffle(new_samples)
            new_val_size = int(len(new_samples) * val_ratio)
            if new_val_size == 0 and len(new_samples) > 0:
                new_val_size = 1
            
            new_train_samples = new_samples[new_val_size:]
            new_val_samples = new_samples[:new_val_size]
            
            # 合并
            train_samples = existing_train_samples + new_train_samples
            val_samples = existing_val_samples + new_val_samples
            
            print(f"已有数据: 训练集 {len(existing_train_samples)}, 验证集 {len(existing_val_samples)}")
            print(f"新数据: 训练集 {len(new_train_samples)}, 验证集 {len(new_val_samples)}")
        else:
            # 只有已有数据，直接使用
            train_samples = existing_train_samples
            val_samples = existing_val_samples
    else:
        # 没有已有数据，统一划分
        print("\n随机打乱样本...")
        random.shuffle(all_samples)
        
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
        'data_sources': data_sources,
        'existing_train_samples': len(existing_train_samples),
        'existing_val_samples': len(existing_val_samples),
        'new_samples': len(new_samples),
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
    if existing_train_samples or existing_val_samples:
        print(f"  已有数据: {len(existing_train_samples) + len(existing_val_samples):,} 个样本")
    if new_samples:
        print(f"  新数据: {len(new_samples):,} 个样本")
    if data_sources:
        print(f"\n新数据源:")
        for source_type, source_path, count in data_sources:
            print(f"  - {source_type.upper()}: {source_path} ({count:,} 条对话)")
    print(f"\n输出目录:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  统计信息: {stats_file}")


def main():
    """主函数"""
    # 数据源路径
    # jsonl_path = 'data/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl'  # 已有数据，不再处理
    parquet_dir = 'data/7M_core'  # 7M_core 数据目录（新数据）
    
    # 已有数据目录（直接利用已有构建结果）
    existing_train_dir = 'data/train_ft'
    existing_val_dir = 'data/val_ft'
    
    # 输出目录
    output_dir = 'data'
    
    # 参数
    tokenizer_path = '/root/data/AI/pretrain/Qwen2.5-7B-Instruct'
    max_prompt_tokens = 4096  # prompt 最大 token 数（64×64 图像容量）
    max_answer_tokens = 4096  # answer 最大 token 数（64×64 图像容量）
    min_answer_tokens = 32  # answer 最小 token 数
    val_ratio = 0.01  # 1% 验证集（仅对新数据生效）
    
    # 构建数据集（加载已有数据 + 处理新数据源）
    build_fintune_dataset(
        jsonl_path=None,  # 已有数据，不再处理
        parquet_dir=parquet_dir,  # 处理新的 7M_core 数据
        existing_train_dir=existing_train_dir,  # 直接利用已有训练集
        existing_val_dir=existing_val_dir,  # 直接利用已有验证集
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
