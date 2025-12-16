import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from robust_token2img import RobustToken2Img


class TokenImageDataset(Dataset):
    """
    数据集：纯文本转换为 token 编码的图像
    
    设计理念：
    1. 预训练时只需要纯文本，不需要问答对格式
    2. Base 和 Chat 版本的区别主要在于 chat_template（推理时应用）
    3. 预训练：模型学习从噪声图像恢复原始图像（自监督学习）
    4. 微调：可以通过 chat_template 格式化输入，实现条件生成
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        img_size: int = 64,
        use_chat_template: bool = False,
        max_tokens: Optional[int] = None,
        chunk_size: int = 4096,
    ):
        """
        Args:
            data_path: 数据文件路径
                - JSON格式: [{"text": "..."}, ...] 或 [{"prompt": "...", "answer": "..."}, ...]
                - 文本文件: 每行一个文本片段
                - 目录: 包含多个 JSON 或文本文件
            tokenizer: Tokenizer
            img_size: 图像尺寸
            use_chat_template: 是否使用 chat_template（用于微调，将问答对格式化为对话）
            max_tokens: 最大 token 数（None 表示使用图像容量）
            chunk_size: 分段处理的字符数（避免超过 Tokenizer 最大长度）
        """
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.use_chat_template = use_chat_template
        self.max_tokens = max_tokens or (img_size * img_size)
        self.chunk_size = chunk_size
        
        # Token encoder
        self.token_encoder = RobustToken2Img(tokenizer)
        
        # Load data
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        
        if os.path.isfile(data_path):
            if data_path.endswith('.json'):
                # JSON file
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    data = self._process_raw_data(raw_data)
            else:
                # 文本文件：每行一个文本片段
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append({'text': line})
        elif os.path.isdir(data_path):
            # Directory with JSON or text files
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                if filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                        data.extend(self._process_raw_data(raw_data))
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append({'text': line})
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        print(f"加载了 {len(data)} 个文本样本")
        return data
    
    def _process_raw_data(self, raw_data: Union[List, Dict]) -> List[Dict]:
        """处理原始数据，统一格式为 {'text': '...'}"""
        processed = []
        
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    if 'text' in item:
                        processed.append({'text': item['text']})
                    elif 'prompt' in item and 'answer' in item:
                        # 问答对：根据 use_chat_template 决定是否格式化
                        if self.use_chat_template:
                            # 使用 chat_template 格式化
                            messages = [
                                {"role": "user", "content": item['prompt']},
                                {"role": "assistant", "content": item['answer']}
                            ]
                            text = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False
                            )
                            processed.append({'text': text})
                        else:
                            # 预训练模式：直接拼接（或只使用答案）
                            # 这里我们选择拼接，让模型学习完整的对话格式
                            text = f"{item['prompt']}\n{item['answer']}"
                            processed.append({'text': text})
                    else:
                        # 其他格式，尝试转换为文本
                        text = str(item)
                        processed.append({'text': text})
                elif isinstance(item, str):
                    processed.append({'text': item})
        elif isinstance(raw_data, dict):
            # 单个字典
            if 'text' in raw_data:
                processed.append({'text': raw_data['text']})
            elif 'prompt' in raw_data and 'answer' in raw_data:
                if self.use_chat_template:
                    messages = [
                        {"role": "user", "content": raw_data['prompt']},
                        {"role": "assistant", "content": raw_data['answer']}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    processed.append({'text': text})
                else:
                    text = f"{raw_data['prompt']}\n{raw_data['answer']}"
                    processed.append({'text': text})
        
        return processed
    
    def _encode_text_to_image(self, text: str) -> np.ndarray:
        """
        将文本编码为图像
        
        注意：
        1. 在文本末尾添加 EOS token（如果存在）
        2. 剩余像素保持黑色 (0, 0, 0) 作为 padding
        3. token_id=0 也对应黑色，但这是有效的 token，不是 padding
           - padding 的判断：在 EOS token 之后的黑色像素才是 padding
           - 或者：记录实际 token 数量，在解码时使用
        """
        # 分段处理长文本（避免超过 Tokenizer 最大长度）
        if len(text) > self.chunk_size:
            # 分段 tokenize
            all_token_ids = []
            for i in range(0, len(text), self.chunk_size):
                chunk_text = text[i:i + self.chunk_size]
                inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
                chunk_token_ids = inputs['input_ids'][0].tolist()
                all_token_ids.extend(chunk_token_ids)
        else:
            # 短文本，直接 tokenize
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            all_token_ids = inputs['input_ids'][0].tolist()
        
        # 添加 EOS token（如果存在且未超过容量）
        if self.tokenizer.eos_token_id is not None:
            if len(all_token_ids) < self.max_tokens:
                all_token_ids.append(self.tokenizer.eos_token_id)
        
        # 截断到 max_tokens（保留 EOS token 的位置）
        if len(all_token_ids) > self.max_tokens:
            all_token_ids = all_token_ids[:self.max_tokens]
            # 如果被截断了，确保最后一个 token 是 EOS（如果可能）
            if self.tokenizer.eos_token_id is not None and all_token_ids[-1] != self.tokenizer.eos_token_id:
                all_token_ids[-1] = self.tokenizer.eos_token_id
        
        # 直接编码 token_ids 到图像
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for i, token_id in enumerate(all_token_ids):
            x = i % self.img_size
            y = i // self.img_size
            color = self.token_encoder.token_id_to_color(token_id)
            img[y, x] = [color[2], color[1], color[0]]  # BGR
        
        # 剩余像素保持黑色 (0, 0, 0) 作为 padding
        # 注意：token_id=0 也对应黑色，但这是有效的 token
        # padding 的判断需要记录实际 token 数量
        
        return img
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        item = self.data[idx]
        text = item['text']
        
        # Tokenize 以获取实际 token 数量
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs['input_ids'][0].tolist()
        
        # 添加 EOS token（如果存在）
        if self.tokenizer.eos_token_id is not None:
            if len(token_ids) < self.max_tokens:
                token_ids.append(self.tokenizer.eos_token_id)
        
        # 实际 token 数量（包括 EOS）
        num_tokens = min(len(token_ids), self.max_tokens)
        
        # Encode text to image
        clean_img = self._encode_text_to_image(text)
        
        # 创建 padding mask（标记哪些像素是有效的）
        # mask[i, j] = 1 表示有效像素，mask[i, j] = 0 表示 padding
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for i in range(num_tokens):
            x = i % self.img_size
            y = i // self.img_size
            mask[y, x] = 1.0
        
        # Convert to tensor
        # [H, W, C] -> [C, H, W], normalize to [0, 1]
        clean_img_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()
        
        return {
            'clean': clean_img_tensor,
            'mask': mask_tensor,  # 用于训练时 mask 掉 padding
            'num_tokens': num_tokens,  # 实际 token 数量
            'text': text,
        }


def create_dummy_dataset(output_path: str, num_samples: int = 1000, format: str = 'text'):
    """
    创建虚拟数据集用于测试
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量
        format: 数据格式 ('text' 或 'qa')
    """
    data = []
    
    if format == 'text':
        # 纯文本格式（预训练用）
        texts = [
            "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
            "Quantum computing uses quantum mechanical phenomena such as superposition and entanglement to perform computations.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy, producing oxygen as a byproduct.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            "The water cycle describes the continuous movement of water on, above, and below Earth's surface through processes like evaporation, condensation, and precipitation.",
            "Artificial intelligence has revolutionized many industries, from healthcare to transportation, by enabling machines to perform tasks that typically require human intelligence.",
            "Climate change is one of the most pressing challenges of our time, requiring global cooperation and innovative solutions.",
            "The internet has transformed how we communicate, work, and access information, creating new opportunities and challenges.",
        ]
        
        for i in range(num_samples):
            text_idx = i % len(texts)
            data.append({'text': texts[text_idx]})
    
    elif format == 'qa':
        # 问答对格式（微调用，可选）
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing.",
            "How does photosynthesis work?",
            "What is machine learning?",
            "Describe the water cycle.",
        ]
        
        answers = [
            "The capital of France is Paris.",
            "Quantum computing uses quantum mechanical phenomena to perform computations.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The water cycle describes the continuous movement of water on, above, and below Earth's surface.",
        ]
        
        for i in range(num_samples):
            q_idx = i % len(prompts)
            a_idx = i % len(answers)
            data.append({
                'prompt': prompts[q_idx],
                'answer': answers[a_idx],
            })
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Created dummy dataset with {num_samples} samples at {output_path} (format: {format})")
