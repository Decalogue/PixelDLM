"""
鲁棒的 Token 到图像编码器

核心设计：
1. 每个像素对应一个文本 token（1:1 映射）
2. 使用 256 进制分解确保不同 token 映射到不同颜色（无冲突）
3. 支持预训练：直接将文本编码到图像（无需问答对）
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer

from utils import ensure_dir


class RobustToken2Img:
    """
    鲁棒的 Token 到图像编码器
    
    设计原则：
    1. 每个像素 = 1 个 token（1:1 映射）
    2. 使用 256 进制分解：token_id → (R, G, B)
    3. 保证不同 token 映射到不同颜色（无冲突）
    4. 支持预训练：直接将文本编码到图像
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer
    ):
        """
        Args:
            tokenizer: Tokenizer 对象
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # 验证 vocab_size 是否在颜色空间范围内
        max_color_space = 256 ** 3  # 16,777,216
        if self.vocab_size > max_color_space:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) 超过颜色空间容量 ({max_color_space})。"
                f"无法保证每个 token 映射到唯一颜色。"
            )
        
        print(f"初始化 RobustToken2Img:")
        print(f"  vocab_size: {self.vocab_size:,}")
        print(f"  color_space: {max_color_space:,} (256^3)")
        print(f"  颜色利用率: {self.vocab_size / max_color_space * 100:.2f}%")
    
    def token_id_to_color(self, token_id: int) -> Tuple[int, int, int]:
        """
        将 token_id 映射为 RGB 颜色（256进制分解）
        
        这是最优的映射方式，因为：
        1. 保证了不同 token_id → 不同颜色（无冲突）
        2. 计算简单高效
        3. 反向映射也是唯一的
        
        Args:
            token_id: token ID (0 <= token_id < vocab_size)
        
        Returns:
            RGB 颜色元组 (R, G, B)，每个值在 [0, 255]
        """
        # 允许超出 vocab_size 的特殊 token（如 pad_token_id, eos_token_id）
        # 这些 token 可能在 vocab_size 范围内，也可能超出
        # 我们只检查是否为负数
        if token_id < 0:
            raise ValueError(f"token_id {token_id} 不能为负数")
        
        # 256 进制分解
        r = token_id % 256                    # 最低位（R通道）
        g = (token_id // 256) % 256          # 次低位（G通道）
        b = (token_id // (256 * 256)) % 256   # 第三位（B通道）
        
        return (r, g, b)
    
    def color_to_token_id(self, color: Tuple[int, int, int]) -> int:
        """
        将颜色转换为 token_id（反向映射）
        
        Args:
            color: 颜色 (BGR格式，OpenCV读取的图像)
        
        Returns:
            token_id
        """
        # OpenCV 读取的图像是 BGR 格式，需要转换为 RGB
        # color = (B, G, R)，需要转换为 (R, G, B)
        b, g, r = color[0], color[1], color[2]
        
        # 转换为 int 类型避免溢出
        r = int(r)
        g = int(g)
        b = int(b)
        
        # 反向映射：token_id = r + g * 256 + b * 256 * 256
        token_id = r + g * 256 + b * (256 * 256)
        
        # 验证 token_id 是否在有效范围内
        if not (0 <= token_id < self.vocab_size):
            # 可能遇到未使用的像素或噪声导致的无效颜色
            return -1  # 返回 -1 表示无效
        
        return token_id
    
    def encode(
        self,
        text: str,
        size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        将文本编码为图像
        
        Args:
            text: 输入文本
            size: 图像尺寸 (height, width)，如果为 None 则自动计算
            save_path: 保存路径（可选）
        
        Returns:
            (图像数组, 元数据字典)
            元数据包含: num_tokens, size, utilization 等
        """
        # Tokenize 文本
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs['input_ids'][0].tolist()
        num_tokens = len(token_ids)
        
        # 自动计算图像尺寸（如果未指定）
        if size is None:
            # 计算最接近的正方形尺寸
            w = int(np.ceil(np.sqrt(num_tokens)))
            h = int(np.ceil(num_tokens / w))
            size = (h, w)
        else:
            h, w = size
            # 检查容量
            max_tokens = h * w
            if num_tokens > max_tokens:
                print(f"警告: Token 数量 ({num_tokens}) 超过图像容量 ({max_tokens})，将截断")
                token_ids = token_ids[:max_tokens]
                num_tokens = max_tokens
        
        # 创建图像数组，默认像素为黑色
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 将每个 token_id 映射为像素点（从左到右，从上到下）
        for i, token_id in enumerate(token_ids):
            # 计算像素位置
            x = i % w  # 列（从左到右）
            y = i // w  # 行（从上到下）
            
            # 将 token_id 映射为颜色
            color = self.token_id_to_color(token_id)
            
            # 设置像素颜色（OpenCV 使用 BGR 格式）
            img[y, x] = [color[2], color[1], color[0]]  # BGR: (B, G, R)
        
        # 保存图像（如果指定了路径）
        if save_path:
            ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else '.')
            cv2.imwrite(save_path, img)
        
        # 计算元数据
        max_tokens = h * w
        utilization = num_tokens / max_tokens * 100 if max_tokens > 0 else 0
        
        metadata = {
            'num_tokens': num_tokens,
            'size': size,
            'utilization': utilization,
            'max_tokens': max_tokens,
        }
        
        return img, metadata
    
    def decode(
        self,
        img: np.ndarray,
        num_tokens: Optional[int] = None,
        stop_on_black: bool = True
    ) -> Tuple[str, List[int]]:
        """
        将图像解码为文本
        
        Args:
            img: 图像数组 (H, W, 3)
            num_tokens: 实际 token 数量（如果知道的话，用于截断多余的像素）
            stop_on_black: 遇到黑色像素时停止（黑色表示未使用的像素）
        
        Returns:
            (文本, token_ids 列表)
        """
        h, w, _ = img.shape
        token_ids = []
        
        # 计算需要读取的像素数量
        total_pixels = h * w
        if num_tokens is not None:
            max_pixels = num_tokens
        else:
            max_pixels = total_pixels
        
        pixel_count = 0
        
        for y in range(h):
            for x in range(w):
                if pixel_count >= max_pixels:
                    break
                
                # OpenCV 读取的是 BGR 格式
                color = tuple(img[y, x])  # (B, G, R)
                
                token_id = self.color_to_token_id(color)
                
                # 检查 EOS token
                if token_id == self.tokenizer.eos_token_id:
                    break
                
                # 检查 PAD token
                if token_id == self.tokenizer.pad_token_id:
                    continue
                
                # 检查 token_id 是否有效
                if not (0 <= token_id < self.vocab_size):
                    continue  # 跳过无效像素
                
                token_ids.append(token_id)
                pixel_count += 1
            
            if pixel_count >= max_pixels:
                break
        
        # 使用 tokenizer 解码为文本
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return text, token_ids
    
    def encode_from_file(
        self,
        text_file: str,
        size: Tuple[int, int],
        output_path: str,
        chunk_size: int = 8192
    ) -> Dict:
        """
        从文件读取文本并编码为图像（支持大文件分段处理）
        
        用于预训练：直接将各领域文本编码到图像
        
        Args:
            text_file: 文本文件路径
            size: 图像尺寸 (height, width)
            output_path: 输出图像路径
            chunk_size: 分段处理的字符数（避免超过 Tokenizer 最大长度）
        
        Returns:
            元数据字典
        """
        print(f"读取文本文件: {text_file}")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"文本长度: {len(text):,} 字符")
        
        # 分段 Tokenize（避免超过 Tokenizer 最大长度）
        print(f"分段 Tokenizing（每段 {chunk_size} 字符）...")
        
        all_token_ids = []
        total_chunks = (len(text) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            # Tokenize 当前片段
            inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False)
            chunk_token_ids = inputs['input_ids'][0].tolist()
            all_token_ids.extend(chunk_token_ids)
            
            if chunk_num % 10 == 0 or chunk_num == total_chunks:
                print(f"  处理进度: {chunk_num}/{total_chunks} 段, 累计 tokens: {len(all_token_ids):,}")
        
        token_ids = all_token_ids
        num_tokens = len(token_ids)
        
        print(f"Token 数量: {num_tokens:,}")
        
        # 检查容量
        h, w = size
        max_tokens = h * w
        print(f"图像容量: {max_tokens:,} tokens ({h}×{w})")
        
        if num_tokens > max_tokens:
            print(f"⚠️  警告: Token 数量 ({num_tokens:,}) 超过图像容量 ({max_tokens:,})")
            print(f"将截断到前 {max_tokens:,} 个 tokens")
            token_ids = token_ids[:max_tokens]
            num_tokens = max_tokens
        else:
            utilization = num_tokens / max_tokens * 100
            print(f"✅ Token 数量在图像容量范围内")
            print(f"图像利用率: {utilization:.2f}%")
        
        # 编码到图像
        print(f"\n编码到图像 ({h}×{w})...")
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, token_id in enumerate(token_ids):
            x = i % w
            y = i // w
            color = self.token_id_to_color(token_id)
            img[y, x] = [color[2], color[1], color[0]]  # BGR
        
        # 保存图像
        ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')
        cv2.imwrite(output_path, img)
        print(f"图像已保存到: {output_path}")
        
        # 返回元数据
        metadata = {
            'num_tokens': num_tokens,
            'size': size,
            'utilization': num_tokens / max_tokens * 100,
            'max_tokens': max_tokens,
        }
        
        return metadata


def test_robust_token2img():
    """测试 RobustToken2Img"""
    tokenizer = AutoTokenizer.from_pretrained('/root/data/AI/pretrain/Qwen2.5-7B-Instruct')
    encoder = RobustToken2Img(tokenizer)
    
    # 测试文本
    text = "静夜思 李白 床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    
    print("\n" + "="*60)
    print("测试编码")
    print("="*60)
    img, metadata = encoder.encode(text, size=(8, 8))
    print(f"元数据: {metadata}")
    
    print("\n" + "="*60)
    print("测试解码")
    print("="*60)
    recovered_text, recovered_token_ids = encoder.decode(img, num_tokens=metadata['num_tokens'])
    print(f"原始文本: {text}")
    print(f"恢复文本: {recovered_text}")
    print(f"是否匹配: {text == recovered_text}")
    
    # 验证 token 级别匹配
    original_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    original_token_ids = original_inputs['input_ids'][0].tolist()
    print(f"\n原始 token_ids: {original_token_ids}")
    print(f"恢复 token_ids: {recovered_token_ids}")
    print(f"Token 级别匹配: {original_token_ids == recovered_token_ids}")


if __name__ == '__main__':
    test_robust_token2img()
