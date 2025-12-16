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
        cond_img_size: Optional[int] = None,
        enable_condition: bool = False,
    ):
        """
        Args:
            data_path: 数据文件路径
                - JSON格式: [{"text": "..."}, ...] 或 [{"prompt": "...", "answer": "..."}, ...]
                - 文本文件: 每行一个文本片段
                - 目录: 包含多个 JSON 或文本文件
            tokenizer: Tokenizer
            img_size: 目标图像尺寸（默认 64×64）
            use_chat_template: 是否使用 chat_template（用于微调，将问答对格式化为对话）
            max_tokens: 最大 token 数（None 表示使用图像容量）
            chunk_size: 分段处理的字符数（避免超过 Tokenizer 最大长度）
            cond_img_size: 条件图像尺寸（默认 64×64，如果 enable_condition=True）
            enable_condition: 是否启用条件生成（如果数据有 prompt/answer，将其作为条件）
        """
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.use_chat_template = use_chat_template
        self.max_tokens = max_tokens or (img_size * img_size)
        self.chunk_size = chunk_size
        self.enable_condition = enable_condition
        self.cond_img_size = cond_img_size or (64 if enable_condition else None)
        
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
                        # 问答对：根据 enable_condition 决定处理方式
                        if self.enable_condition:
                            # 条件生成模式：保留 prompt 和 answer 分离
                            processed.append({
                                'prompt': item['prompt'],
                                'answer': item['answer']
                            })
                        elif self.use_chat_template:
                            # 使用 chat_template 格式化（无条件生成，但格式化）
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
                            # 预训练模式：直接拼接
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
                if self.enable_condition:
                    # 条件生成模式：保留 prompt 和 answer 分离
                    processed.append({
                        'prompt': raw_data['prompt'],
                        'answer': raw_data['answer']
                    })
                elif self.use_chat_template:
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
    
    def _encode_text_to_image(self, text: str, img_size: Optional[int] = None) -> np.ndarray:
        """
        将文本编码为图像
        
        Args:
            text: 输入文本
            img_size: 图像尺寸（None 表示使用 self.img_size）
        
        注意：
        1. 在文本末尾添加 EOS token（如果存在）
        2. 剩余像素使用 pad_token 的颜色（如果存在）或白色作为 padding
           - 使用 pad_token 颜色：与 tokenizer 的 padding 策略一致
           - 使用白色：白色对应的 token_id (16777215) 通常超出 vocab_size，不会与有效 token 冲突
           - 避免使用黑色 (0, 0, 0)，因为 token_id=0 可能是有效的 token
        """
        target_img_size = img_size or self.img_size
        max_tokens_for_img = target_img_size * target_img_size
        
        # 确定 padding 颜色
        # 优先使用 pad_token_id 的颜色，如果没有 pad_token_id 则使用白色
        if self.tokenizer.pad_token_id is not None:
            pad_color = self.token_encoder.token_id_to_color(self.tokenizer.pad_token_id)
            pad_color_bgr = [pad_color[2], pad_color[1], pad_color[0]]  # RGB -> BGR
        else:
            # 白色 (255, 255, 255) 对应 token_id = 16777215，通常超出 vocab_size
            pad_color_bgr = [255, 255, 255]
        
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
            if len(all_token_ids) < max_tokens_for_img:
                all_token_ids.append(self.tokenizer.eos_token_id)
        
        # 截断到 max_tokens_for_img（保留 EOS token 的位置）
        if len(all_token_ids) > max_tokens_for_img:
            all_token_ids = all_token_ids[:max_tokens_for_img]
            # 如果被截断了，确保最后一个 token 是 EOS（如果可能）
            if self.tokenizer.eos_token_id is not None and all_token_ids[-1] != self.tokenizer.eos_token_id:
                all_token_ids[-1] = self.tokenizer.eos_token_id
        
        # 初始化图像为 padding 颜色（pad_token 颜色或白色）
        img = np.full((target_img_size, target_img_size, 3), pad_color_bgr, dtype=np.uint8)
        
        # 编码 token_ids 到图像
        for i, token_id in enumerate(all_token_ids):
            x = i % target_img_size
            y = i // target_img_size
            color = self.token_encoder.token_id_to_color(token_id)
            img[y, x] = [color[2], color[1], color[0]]  # RGB -> BGR
        
        return img
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            - clean: 目标图像 [C, H, W]
            - mask: padding mask [H, W]
            - num_tokens: 实际 token 数量
            - text: 原始文本（或 answer 文本）
            - condition: 条件图像 [C, H_cond, W_cond] (可选，如果 enable_condition=True 且有 prompt)
        """
        item = self.data[idx]
        
        # 检查是否有 prompt 和 answer（条件生成模式）
        if self.enable_condition and 'prompt' in item and 'answer' in item:
            prompt_text = item['prompt']
            answer_text = item['answer']
            
            # 编码 prompt 为较小的条件图像
            cond_img = self._encode_text_to_image(prompt_text, img_size=self.cond_img_size)
            cond_img_tensor = torch.from_numpy(cond_img).permute(2, 0, 1).float() / 255.0  # [C, H_cond, W_cond]
            
            # 编码 answer 为目标图像
            target_text = answer_text
        else:
            # 无条件生成模式
            target_text = item.get('text', item.get('answer', ''))
            cond_img_tensor = None
        
        # Tokenize 目标文本以获取实际 token 数量
        inputs = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs['input_ids'][0].tolist()
        
        # 添加 EOS token（如果存在）
        if self.tokenizer.eos_token_id is not None:
            if len(token_ids) < self.max_tokens:
                token_ids.append(self.tokenizer.eos_token_id)
        
        # 实际 token 数量（包括 EOS）
        num_tokens = min(len(token_ids), self.max_tokens)
        
        # Encode target text to image
        clean_img = self._encode_text_to_image(target_text, img_size=self.img_size)
        
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
        
        result = {
            'clean': clean_img_tensor,
            'mask': mask_tensor,  # 用于训练时 mask 掉 padding
            'num_tokens': num_tokens,  # 实际 token 数量
            'text': target_text,
        }
        
        # 如果有条件，添加到返回结果
        if cond_img_tensor is not None:
            result['condition'] = cond_img_tensor
        
        return result
