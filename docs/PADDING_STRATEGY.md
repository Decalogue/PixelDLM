# Padding 策略说明

## 🎯 问题

1. **样本长度不够填满 256×256 时，需要 padding 吗？**
2. **LLM 预训练时有没有 bos_token、eos_token 和 pad_token？**

---

## 📊 LLM 预训练中的特殊 Token

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

## 🔧 我们的实现

### 1. 添加 EOS Token

```python
# 在文本末尾添加 EOS token（如果存在）
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
token_ids = inputs['input_ids'][0].tolist()

if tokenizer.eos_token_id is not None:
    if len(token_ids) < max_tokens:
        token_ids.append(tokenizer.eos_token_id)
```

### 2. 使用黑色像素作为 Padding

```python
# 编码到图像
img = np.zeros((256, 256, 3), dtype=np.uint8)  # 默认黑色

# 将 token_ids 编码到图像
for i, token_id in enumerate(token_ids):
    x = i % 256
    y = i // 256
    color = token_id_to_color(token_id)
    img[y, x] = color

# 剩余像素保持黑色 (0, 0, 0) 作为 padding
```

### 3. 创建 Padding Mask

**问题**：`token_id=0` 也对应黑色 `(0, 0, 0)`，与 padding 冲突！

**解决方案**：使用 mask 来区分有效像素和 padding

```python
# 创建 padding mask
mask = np.zeros((256, 256), dtype=np.float32)
for i in range(num_tokens):  # num_tokens 包括 EOS token
    x = i % 256
    y = i // 256
    mask[y, x] = 1.0  # 1 表示有效像素，0 表示 padding
```

### 4. 训练时使用 Mask

```python
# 在训练循环中
clean = batch['clean']  # [B, 3, H, W]
mask = batch['mask']    # [B, H, W]

# 预测
predicted_img = model(noisy_img, timestep)

# 计算 loss（只对有效像素）
mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
diff = (predicted_img - clean) ** 2
masked_diff = diff * mask_expanded
loss = masked_diff.sum() / (mask_expanded.sum() + 1e-8)  # 归一化
```

---

## ✅ 最终方案

### 1. 添加 EOS Token

- ✅ 在每个文本末尾添加 EOS token（如果存在）
- ✅ 让模型学习识别文档边界

### 2. 使用黑色像素作为 Padding

- ✅ 剩余像素保持黑色 `(0, 0, 0)`
- ⚠️ 注意：`token_id=0` 也对应黑色，但这是有效的 token

### 3. 创建 Padding Mask

- ✅ 记录实际 token 数量（`num_tokens`）
- ✅ 创建 mask：`mask[i, j] = 1` 表示有效像素，`0` 表示 padding
- ✅ 训练时只对有效像素计算 loss

---

## 📝 关键点

### 1. Token ID 0 的颜色冲突

```
token_id=0 → 颜色 (0, 0, 0) = 黑色
padding → 颜色 (0, 0, 0) = 黑色

冲突！但可以通过 mask 解决：
- 记录实际 token 数量
- 使用 mask 区分有效像素和 padding
```

### 2. 特殊 Token 的颜色

```
Qwen2.5-7B-Instruct:
- eos_token_id = 151645 → 颜色 (93, 80, 2)
- pad_token_id = 151643 → 颜色 (91, 80, 2)
- token_id=0 → 颜色 (0, 0, 0) = 黑色
```

### 3. 预训练 vs 微调

```
预训练：
- 添加 EOS token ✅
- 不使用 PAD token ❌
- 使用 mask 区分有效像素和 padding ✅

微调：
- 可以使用 chat_template 格式化输入
- 同样使用 mask 区分有效像素和 padding
```

---

## 🎯 总结

### 回答用户的问题

1. **如果某个样本长度不够填满256×256，需要 padding 吗？**
   - ✅ **需要**：使用黑色像素作为 padding
   - ✅ **但需要 mask**：使用 mask 区分有效像素和 padding
   - ✅ **训练时 mask 掉 padding**：只对有效像素计算 loss

2. **现在 LLM 预训练时有没有 bos_token、eos_token 和 pad_token？**
   - ✅ **EOS token**：必须使用（标记文档边界）
   - ⚠️ **BOS token**：可选（Qwen2.5 没有）
   - ❌ **PAD token**：预训练时通常不使用（使用 packing 策略）

### 实现要点

1. ✅ 在文本末尾添加 EOS token
2. ✅ 使用黑色像素作为 padding
3. ✅ 创建 padding mask（记录实际 token 数量）
4. ✅ 训练时只对有效像素计算 loss

---

**结论**：应该添加 EOS token，使用黑色像素作为 padding，并使用 mask 在训练时忽略 padding 部分。
