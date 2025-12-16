# 特殊 Token 在预训练中的使用分析

## 🔍 LLM 预训练中的特殊 Token

### 1. EOS Token（End of Sequence）

**作用**：标记文档/序列的结束，用于分隔不同的文档

**使用方式**：
```python
# 预训练时，每个文档末尾添加 EOS token
text = "文档内容..."
tokens = tokenizer.encode(text, add_special_tokens=False)
tokens.append(tokenizer.eos_token_id)  # 添加 EOS token
```

**重要性**：✅ **必须使用**
- 让模型学习识别文档边界
- 在生成时，模型知道何时停止生成

### 2. BOS Token（Beginning of Sequence）

**作用**：标记序列的开始

**使用方式**：
```python
# 有些模型在开头添加 BOS token
tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
```

**重要性**：⚠️ **可选**
- 不是所有模型都使用 BOS token
- Qwen2.5 没有 BOS token（`bos_token_id = None`）

### 3. PAD Token

**作用**：用于填充短序列，使 batch 中的序列长度一致

**使用方式**：
```python
# 传统做法：使用 PAD token 填充
padded_tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
```

**重要性**：❌ **预训练时通常不使用**
- 预训练时使用 **packing** 策略：将多个短样本拼接成一个长序列
- 避免浪费计算资源在 padding 上
- 提高训练效率

---

## 📊 我们的场景分析

### 当前实现

```python
# 当前代码：不添加任何特殊 token
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
token_ids = inputs['input_ids'][0].tolist()

# 编码到图像
img = encode_token_ids_to_image(token_ids, size=(256, 256))
# 如果 token_ids 不够 256×256，剩余像素是黑色 (0, 0, 0)
```

### 问题

1. **没有 EOS token**：模型无法学习识别文档边界
2. **黑色像素作为 padding**：但模型不知道这是 padding，可能会预测这些位置
3. **没有 mask**：训练时会对 padding 位置计算 loss，浪费计算资源

---

## ✅ 解决方案

### 方案 1: 添加 EOS Token + 黑色像素作为 Padding（推荐）

```python
# 1. 在文本末尾添加 EOS token
text = "文档内容..."
tokens = tokenizer.encode(text, add_special_tokens=False)
if tokenizer.eos_token_id is not None:
    tokens.append(tokenizer.eos_token_id)

# 2. 编码到图像
img = encode_token_ids_to_image(tokens, size=(256, 256))
# 剩余像素保持黑色 (0, 0, 0) 作为 padding

# 3. 创建 mask（标记哪些像素是 padding）
mask = create_padding_mask(img)  # 黑色像素 = padding
# mask[i, j] = 0 表示 padding，mask[i, j] = 1 表示有效像素

# 4. 训练时 mask 掉 padding
loss = compute_loss(predicted_img, clean_img, mask=mask)
```

**优点**：
- ✅ 符合 LLM 预训练的做法（添加 EOS token）
- ✅ 简单直接（黑色像素 = padding）
- ✅ 训练时 mask 掉 padding，不浪费计算资源

**缺点**：
- ⚠️ 需要确保黑色 (0, 0, 0) 不会与真实 token 冲突
  - 检查：token_id=0 对应的颜色是什么？
  - 如果 token_id=0 对应黑色，需要特殊处理

### 方案 2: 使用 PAD Token（不推荐）

```python
# 使用 PAD token 填充
tokens = tokenizer.encode(text, add_special_tokens=False)
if tokenizer.eos_token_id is not None:
    tokens.append(tokenizer.eos_token_id)

# 填充到 256×256
max_tokens = 256 * 256
if len(tokens) < max_tokens:
    tokens.extend([tokenizer.pad_token_id] * (max_tokens - len(tokens)))

# 编码到图像
img = encode_token_ids_to_image(tokens, size=(256, 256))
```

**缺点**：
- ❌ 不符合 LLM 预训练的做法（预训练时通常不使用 PAD token）
- ❌ 浪费计算资源（对 padding 位置计算 loss）
- ❌ 需要额外的 mask 机制

---

## 🔧 实现建议

### 1. 添加 EOS Token

```python
def _encode_text_to_image(self, text: str) -> np.ndarray:
    """将文本编码为图像"""
    # Tokenize 文本
    inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = inputs['input_ids'][0].tolist()
    
    # 添加 EOS token（如果存在）
    if self.tokenizer.eos_token_id is not None:
        token_ids.append(self.tokenizer.eos_token_id)
    
    # 编码到图像...
```

### 2. 创建 Padding Mask

```python
def create_padding_mask(img: np.ndarray) -> np.ndarray:
    """
    创建 padding mask
    
    Args:
        img: 图像数组 (H, W, 3)
    
    Returns:
        mask: (H, W)，1 表示有效像素，0 表示 padding
    """
    # 黑色像素 (0, 0, 0) 表示 padding
    mask = (img.sum(axis=2) > 0).astype(np.float32)
    return mask
```

### 3. 训练时使用 Mask

```python
# 在训练循环中
clean_img = batch['clean']  # [B, 3, H, W]
mask = batch['mask']  # [B, H, W]

# 预测
predicted_img = model(noisy_img, timestep)

# 计算 loss（只对有效像素）
loss = mse_loss(predicted_img * mask, clean_img * mask)
loss = loss.sum() / mask.sum()  # 归一化
```

---

## 📝 检查 Token ID 0 的颜色

需要检查 `token_id=0` 对应的颜色，确保不会与黑色 padding 冲突：

```python
# 检查 token_id=0 的颜色
token_id = 0
color = token_id_to_color(token_id)  # (r, g, b)
print(f"token_id=0 对应的颜色: {color}")

# 如果是 (0, 0, 0)，需要特殊处理
if color == (0, 0, 0):
    print("⚠️ 警告: token_id=0 对应黑色，与 padding 冲突！")
    print("解决方案: 使用特殊的 padding token_id（如 pad_token_id）")
```

---

## ✅ 最终建议

1. **添加 EOS token**：在每个文本末尾添加 EOS token（如果存在）
2. **黑色像素作为 padding**：剩余像素保持黑色 (0, 0, 0)
3. **创建 padding mask**：标记哪些像素是 padding
4. **训练时 mask 掉 padding**：只对有效像素计算 loss
5. **检查 token_id=0**：确保不会与黑色 padding 冲突

---

## 🎯 总结

### LLM 预训练中的特殊 Token

- ✅ **EOS token**：必须使用（标记文档边界）
- ⚠️ **BOS token**：可选（Qwen2.5 没有）
- ❌ **PAD token**：预训练时通常不使用（使用 packing 策略）

### 我们的实现

- ✅ 添加 EOS token 到文本末尾
- ✅ 使用黑色像素作为 padding
- ✅ 创建 padding mask
- ✅ 训练时 mask 掉 padding

**结论**：应该添加 EOS token，并使用黑色像素作为 padding，同时创建 mask 在训练时忽略 padding 部分。
