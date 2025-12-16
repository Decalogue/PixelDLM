# LLM Padding 机制详解（Qwen、Kimi 等）

## 🎯 核心结论

**训练阶段**：使用 **Right Padding**（在序列末尾添加 padding）  
**推理阶段**：使用 **Left Padding**（在序列开头添加 padding）

---

## 📊 训练阶段：Right Padding

### 为什么训练时用 Right Padding？

#### 1. 与处理方向一致

**Decoder-only 模型**（如 Qwen、Kimi）从左到右处理序列：

```
[Token_0, Token_1, Token_2, ..., Token_N, PAD, PAD, ...]
 ↑ 有效 token 连续，padding 在末尾
```

**优势**：
- ✅ 有效 token 连续，索引简单
- ✅ 与从左到右的处理方式一致
- ✅ 位置编码连续（0 到 N-1）

#### 2. 简化 Loss 计算

**训练时的 Loss 计算**：
```python
# Right padding
input_ids = [token_0, token_1, ..., token_N, PAD, PAD, ...]
labels = [token_1, token_2, ..., token_N+1, IGNORE, IGNORE, ...]
#                    ↑ 预测下一个 token        ↑ padding 位置忽略
```

**优势**：
- ✅ 有效 token 的 labels 连续
- ✅ Padding 位置用 `IGNORE_INDEX` 标记，不参与 loss 计算
- ✅ 实现简单

#### 3. Attention Mask

**Right Padding 的 Attention Mask**：
```python
attention_mask = [1, 1, ..., 1, 0, 0, ..., 0]
                 ↑ 有效 token    ↑ padding
```

**特点**：
- ✅ 有效 token 可以互相 attention
- ✅ Padding 位置被 mask 掉，不参与 attention
- ✅ 计算效率高

---

## 🔄 推理阶段：Left Padding

### 为什么推理时用 Left Padding？

#### 1. 因果注意力机制

**Decoder-only 模型使用因果注意力**：
- 每个 token 只能看到**前面的 token**
- 预测下一个 token 时，需要**最后一个 token 是有效 token**

**Right Padding 的问题**：
```
[Token_0, Token_1, ..., Token_N, PAD, PAD, ...]
                                    ↑ 最后一个 token 是 PAD
                                    ↑ 模型会基于 PAD 预测，导致错误
```

**Left Padding 的解决方案**：
```
[PAD, PAD, ..., Token_0, Token_1, ..., Token_N]
 ↑ padding    ↑ 最后一个 token 是有效 token
              ↑ 模型基于有效 token 预测，正确
```

#### 2. 批量生成

**批量生成时，不同序列长度不同**：

**Right Padding 的问题**：
```python
# Batch 中的序列
seq_1: [token_0, token_1, token_2, PAD, PAD]  # 长度 3
seq_2: [token_0, token_1, PAD, PAD, PAD]     # 长度 2

# 生成时，最后一个 token
seq_1 的最后一个 token: PAD  ❌
seq_2 的最后一个 token: PAD  ❌
# 两个序列的最后一个 token 都是 PAD，无法正确生成
```

**Left Padding 的解决方案**：
```python
# Batch 中的序列
seq_1: [PAD, PAD, token_0, token_1, token_2]  # 长度 3
seq_2: [PAD, PAD, PAD, token_0, token_1]      # 长度 2

# 生成时，最后一个 token
seq_1 的最后一个 token: token_2  ✅
seq_2 的最后一个 token: token_1  ✅
# 每个序列的最后一个 token 都是有效 token，可以正确生成
```

#### 3. 位置编码

**Left Padding 的位置编码**：
```python
# 序列：[PAD, PAD, token_0, token_1, token_2]
# 位置编码：[0, 1, 2, 3, 4]  # 但 PAD 位置被 mask 掉
# 实际有效位置：[2, 3, 4]  # token_0, token_1, token_2
```

**注意**：
- ⚠️ 位置编码从 0 开始，但有效 token 的位置不连续
- ⚠️ 需要使用 `position_ids` 来正确设置位置编码
- ⚠️ 或者使用 RoPE（Rotary Position Embedding），可以处理相对位置

---

## 🔧 Qwen 的具体实现

### 训练时配置

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 训练时：right padding
tokenizer.padding_side = "right"  # 默认值

# Tokenize
inputs = tokenizer(
    ["文本1", "文本2"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# 结果
# input_ids: [[token_0, ..., token_N, PAD, PAD], [token_0, ..., token_M, PAD, PAD]]
# attention_mask: [[1, ..., 1, 0, 0], [1, ..., 1, 0, 0]]
```

### 推理时配置

```python
# 推理时：left padding
tokenizer.padding_side = "left"

# Tokenize
inputs = tokenizer(
    ["文本1", "文本2"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# 结果
# input_ids: [[PAD, PAD, token_0, ..., token_N], [PAD, PAD, PAD, token_0, ..., token_M]]
# attention_mask: [[0, 0, 1, ..., 1], [0, 0, 0, 1, ..., 1]]
```

### 动态切换

```python
# 训练时
tokenizer.padding_side = "right"
# ... 训练代码 ...

# 推理时
tokenizer.padding_side = "left"
# ... 推理代码 ...
```

---

## 📊 对比总结

### Right Padding（训练）

| 特性 | 说明 |
|------|------|
| **位置** | 序列末尾 |
| **有效 token** | 连续（从位置 0 开始） |
| **位置编码** | 连续（0 到 N-1） |
| **Loss 计算** | 简单（有效 token 连续） |
| **适用场景** | 训练阶段 |

**序列结构**：
```
[Token_0, Token_1, ..., Token_N, PAD, PAD, ...]
 ↑ 有效 token 连续                ↑ padding
```

---

### Left Padding（推理）

| 特性 | 说明 |
|------|------|
| **位置** | 序列开头 |
| **有效 token** | 不连续（从位置 M 开始） |
| **位置编码** | 需要调整（或使用 RoPE） |
| **最后一个 token** | 总是有效 token ✅ |
| **适用场景** | 推理阶段（批量生成） |

**序列结构**：
```
[PAD, PAD, ..., Token_0, Token_1, ..., Token_N]
 ↑ padding    ↑ 有效 token（最后一个总是有效）
```

---

## ⚠️ 常见问题

### 1. 推理时使用 Right Padding 会怎样？

**问题**：
- 最后一个 token 可能是 PAD
- 模型基于 PAD 预测下一个 token，导致错误输出
- 批量生成时，不同序列的最后一个 token 都是 PAD

**解决方案**：
- 推理时切换到 left padding
- 或者使用 `generate()` 方法，它会自动处理

### 2. 位置编码如何处理？

**Right Padding**：
- 位置编码连续（0 到 N-1）
- 实现简单

**Left Padding**：
- 位置编码不连续（有效 token 从位置 M 开始）
- 需要：
  - 使用 `position_ids` 手动设置
  - 或使用 RoPE（可以处理相对位置）

### 3. Attention Mask 的区别？

**Right Padding**：
```python
attention_mask = [1, 1, ..., 1, 0, 0, ..., 0]
                 ↑ 有效 token    ↑ padding
```

**Left Padding**：
```python
attention_mask = [0, 0, ..., 0, 1, 1, ..., 1]
                 ↑ padding    ↑ 有效 token
```

---

## 🎯 最佳实践

### 训练阶段

```python
# 1. 设置 right padding
tokenizer.padding_side = "right"

# 2. Tokenize
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

# 3. 创建 labels（shift by 1）
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = IGNORE_INDEX  # 忽略 padding

# 4. 训练
loss = model(input_ids=inputs["input_ids"], labels=labels).loss
```

### 推理阶段

```python
# 1. 设置 left padding
tokenizer.padding_side = "left"

# 2. Tokenize
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

# 3. 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
```

---

## 📝 与 JiT 模型的对比

### JiT 模型的条件 Padding

**当前实现**：Right Padding（条件后面添加 padding）

```python
# 条件在序列前面，padding 在后面
cond_embed = torch.cat([cond_embed, cond_padding], dim=1)  # right padding
```

**序列结构**：
```
[Cond_real, PAD, PAD, ..., Target_0, Target_1, ..., Target_255]
 ↑ 实际条件    ↑ padding      ↑ 目标图像
```

### 与 LLM 的对比

| 特性 | LLM（训练） | LLM（推理） | JiT（条件） |
|------|------------|------------|------------|
| **Padding 位置** | Right | Left | Right |
| **原因** | 有效 token 连续 | 最后一个 token 有效 | 条件在序列前面 |
| **位置编码** | 连续 | 需要调整 | 连续 |
| **适用场景** | 训练 | 推理 | 条件生成 |

**关键区别**：
- **LLM**：训练和推理使用不同的 padding 策略
- **JiT**：条件生成时，条件在序列前面，所以用 right padding（与 LLM 训练时类似）

---

## ✅ 总结

### LLM Padding 机制

1. **训练阶段**：Right Padding
   - 有效 token 连续
   - 位置编码连续
   - Loss 计算简单

2. **推理阶段**：Left Padding
   - 最后一个 token 总是有效
   - 支持批量生成
   - 需要处理位置编码

### JiT 模型

- **条件生成**：Right Padding（条件在序列前面）
- **与 LLM 训练时类似**：有效 token 连续，位置编码连续

---

**更新日期**: 2025-12-15
