import os
import cv2
import numpy as np
from utils import ensure_dir
from typing import List, Tuple, Optional
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/root/data/AI/pretrain/Qwen2.5-7B-Instruct')
id2token = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))


def token_id_to_color(token_id: int) -> Tuple[int, int, int]:
    """
    将 token_id 映射为 RGB 颜色（使用256进制分解）
    
    将 token_id 按256进制分解，确保三个通道独立且分布均匀
    token_id = r * 256^0 + g * 256^1 + b * 256^2 + ...
    
    Args:
        token_id: token ID
    
    Returns:
        RGB 颜色元组 (R, G, B)
    """
    r = token_id % 256                    # 最低位（R通道）
    g = (token_id // 256) % 256           # 次低位（G通道）
    b = (token_id // (256 * 256)) % 256   # 第三位（B通道）
    
    return (r, g, b)


def color_to_token_id(color: Tuple[int, int, int]) -> int:
    '''将颜色转换为token_id

    Args:
        color: 颜色 (BGR格式，OpenCV读取的图像)

    Returns:
        token_id: token_id
    '''
    # OpenCV 读取的图像是 BGR 格式，需要转换为 RGB
    # color = (B, G, R)，需要转换为 (R, G, B)
    b, g, r = color[0], color[1], color[2]
    
    # 转换为 int 类型避免溢出
    r = int(r)
    g = int(g)
    b = int(b)
    
    # 反向映射：token_id = r + g * 256 + b * 256 * 256
    token_id = r + g * 256 + b * (256 * 256)
    
    return token_id


def token_ids_to_img(
    token_ids: List[int],
    size: Optional[Tuple[int, int]] = None,
    save_path: str = './token2img.png',
) -> np.ndarray:
    """
    直接将 token_ids 编码为图像（避免重复 tokenize）
    
    Args:
        token_ids: Token ID 列表
        size: 图片尺寸 (height, width)，如果为 None 则自动计算
        save_path: 保存路径
    
    Returns:
        生成的图片数组 (H, W, 3)
    """
    num_tokens = len(token_ids)
    
    # 自动计算图片尺寸（如果未指定）
    if size is None:
        # 计算最接近的正方形尺寸
        w = int(np.ceil(np.sqrt(num_tokens)))
        h = int(np.ceil(num_tokens / w))
        size = (h, w)
        print(f"自动计算尺寸: {size} (H={h}, W={w})")
    else:
        h, w = size
        # 将超长文本截断
        if num_tokens > h * w:
            token_ids = token_ids[:h * w]
            num_tokens = h * w
            print(f"截断 tokens，保留前 {num_tokens} 个")
    
    # 创建图片数组，默认像素为黑色
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 将每个 token_id 映射为像素点（从左到右，从上到下）
    for i, token_id in enumerate(token_ids):
        # 计算像素位置
        x = i % w  # 列（从左到右）
        y = i // w  # 行（从上到下）
        
        # 将 token_id 映射为颜色
        color = token_id_to_color(token_id)
        
        # 设置像素颜色（OpenCV 使用 BGR 格式）
        img[y, x] = [color[2], color[1], color[0]]  # BGR: (B, G, R)
    
    # 保存图片
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else '.')
    cv2.imwrite(save_path, img)
    print(f"图片已保存到: {save_path}")
    
    return img


def token2img(
    text: str,
    tokenizer: AutoTokenizer,
    size: Optional[Tuple[int, int]] = None,
    save_path: str = './token2img.png',
    use_chat_template: bool = False
) -> np.ndarray:
    """
    将文本 tokenize 后，将每个 token_id 映射为像素点并生成图片
    
    Args:
        text: 输入文本
        tokenizer: tokenizer 对象
        size: 图片尺寸 (height, width)，如果为 None 则自动计算
        save_path: 保存路径
        use_chat_template: 是否使用 chat template（会添加特殊 token）
    
    Returns:
        生成的图片数组 (H, W, 3)
    """
    # Tokenize 文本
    if use_chat_template:
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        token_ids = inputs['input_ids'][0].tolist()
    else:
        # 直接 tokenize，不添加特殊 token
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs['input_ids'][0].tolist()
    
    print(f"token_ids: {token_ids}")
    print(f"token num: {len(token_ids)}")
    
    # 自动计算图片尺寸（如果未指定）
    if size is None:
        # 计算最接近的正方形尺寸
        num_tokens = len(token_ids)
        w = int(np.ceil(np.sqrt(num_tokens)))
        h = int(np.ceil(num_tokens / w))
        size = (h, w)
        print(f"自动计算尺寸: {size} (H={h}, W={w})")
    else:
        h, w = size
        # 将超长文本截断
        if len(token_ids) > h * w:
            token_ids = token_ids[:h * w]
            print(f"截断文本，保留前 {len(token_ids)} 个 token")
    
    # 创建图片数组，默认像素为黑色
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 将每个 token_id 映射为像素点（从左到右，从上到下）
    for i, token_id in enumerate(token_ids):
        # 计算像素位置
        x = i % w  # 列（从左到右）
        y = i // w  # 行（从上到下）
        
        # 将 token_id 映射为颜色
        color = token_id_to_color(token_id)
        
        # 设置像素颜色（OpenCV 使用 BGR 格式）
        img[y, x] = [color[2], color[1], color[0]]  # BGR: (B, G, R)
    
    # 剩余像素保持黑色（用于标识未使用的像素）
    # 这样在解码时可以知道何时停止
    
    # 保存图片
    cv2.imwrite(save_path, img)
    print(f"图片已保存到: {save_path}")
    
    return img


def token2img_batch(
    texts: list,
    tokenizer: AutoTokenizer,
    size: Optional[Tuple[int, int]] = None,
    save_dir: str = './output',
    use_chat_template: bool = False
) -> list:
    """
    批量处理多个文本
    
    Args:
        texts: 文本列表
        tokenizer: tokenizer 对象
        size: 图片尺寸
        save_dir: 保存目录
        use_chat_template: 是否使用 chat template
    
    Returns:
        图片列表
    """
    ensure_dir(save_dir)
    
    images = []
    for i, text in enumerate(texts):
        save_path = f'{save_dir}/{i}.png'
        img = token2img(text, tokenizer, size, save_path, use_chat_template)
        images.append(img)
    
    return images


def img2text(img_path: str, num_tokens: Optional[int] = None) -> str:
    '''将图片的每个像素点转换为token_id，然后转换为文本

    Args:
        img_path: 图片路径
        num_tokens: 实际 token 数量（如果知道的话，用于截断多余的像素）

    Returns:
        text: 文本
    '''
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
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
            
            # 检查是否是黑色像素（未使用的像素，RGB 都是 0）
            # 注意：OpenCV 是 BGR，所以 (0,0,0) 表示黑色
            if color == (0, 0, 0):
                # 遇到黑色像素，停止读取（假设黑色表示未使用）
                break
            
            token_id = color_to_token_id(color)
            
            # 检查 token_id 是否在有效范围内
            if 0 <= token_id < len(id2token):
                token_ids.append(token_id)
                pixel_count += 1
            else:
                # 如果超出范围，停止读取（可能是未使用的像素）
                # 注意：token_id 可能很大（如 16777215），超出 vocab_size
                break
        
        if pixel_count >= max_pixels:
            break
    
    # 使用 tokenizer 的 decode 方法转换为文本
    # skip_special_tokens=True 会跳过特殊 token，但不会跳过无效的 token_id
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    return text


if __name__ == '__main__':
    text = '静夜思 李白 床前明月光，疑是地上霜。 举头望明月，低头思故乡。'

    # 先 tokenize 获取实际 token 数量
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = inputs['input_ids'][0].tolist()
    num_tokens = len(token_ids)
    print(f"原始文本 token 数量: {num_tokens}")
    
    # 编码：文本 -> 图像
    token2img(text, tokenizer, size=(8, 8), save_path='./data/token2img.png')
    print(f"图像已保存到: ./data/token2img.png")
    print(f"图像尺寸: 8x8=64 像素")
    
    # 解码：图像 -> 文本（传入实际 token 数量）
    recovered_text = img2text('./data/token2img.png', num_tokens=num_tokens)
    print(f"\n原始文本: {text}")
    print(f"恢复文本: {recovered_text}")
    print(f"是否匹配: {text == recovered_text}")
    
    # 验证 token 级别匹配
    recovered_inputs = tokenizer(recovered_text, return_tensors="pt", add_special_tokens=False)
    recovered_token_ids = recovered_inputs['input_ids'][0].tolist()
    print(f"\n原始 token_ids: {token_ids}")
    print(f"恢复 token_ids: {recovered_token_ids}")
    print(f"Token 级别匹配: {token_ids == recovered_token_ids}")