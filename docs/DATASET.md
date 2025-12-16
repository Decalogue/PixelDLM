# 数据集设计

## 🎯 设计理念

**预训练时只需要纯文本，不需要问答对格式！**

### 核心观点

1. **Base 和 Chat 版本的区别主要在于 `chat_template`**（推理时应用）
2. **预训练**：模型学习从噪声图像恢复原始图像（自监督学习）
3. **微调**：可以通过 `chat_template` 格式化输入，实现条件生成

## 📊 数据格式

### 纯文本格式（预训练用）

```json
[
  {"text": "The capital of France is Paris..."},
  {"text": "Quantum computing uses quantum mechanical..."},
  ...
]
```

### 问答对格式（微调用，可选）

```json
[
  {"prompt": "What is the capital of France?", "answer": "Paris"},
  ...
]
```

如果 `enable_condition=True`，会自动将 prompt 和 answer 分别编码为条件图像和目标图像。

### 文本文件格式

```
每行一个文本片段
The capital of France is Paris...
Quantum computing uses quantum mechanical...
...
```

## 🔧 编码流程

```python
# 1. 加载文本
text = "The capital of France is Paris..."

# 2. Tokenize（支持分段处理，避免超过 Tokenizer 最大长度）
token_ids = tokenizer.encode(text, add_special_tokens=False)

# 3. 编码到图像（64×64）
img = encode_token_ids_to_image(token_ids, size=(64, 64))

# 4. 训练：从噪声图像恢复原始图像
noisy_img = add_noise(clean_img, timestep)
predicted_img = model(noisy_img, timestep, condition=None)
loss = mse_loss(predicted_img, clean_img)
```

## 📈 使用示例

### 预训练数据准备

```python
from dataset import TokenImageDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen2.5-7B-Instruct')

# 纯文本数据
dataset = TokenImageDataset(
    data_path='./data/pretrain_texts.json',
    tokenizer=tokenizer,
    img_size=64,
    use_chat_template=False,  # 预训练不需要 chat_template
)
```

### 微调数据准备（条件生成）

```python
# 问答对数据
dataset = TokenImageDataset(
    data_path='./data/qa_pairs.json',
    tokenizer=tokenizer,
    img_size=64,
    enable_condition=True,  # 启用条件生成
    cond_img_size=64,
)
```

## ✅ 关键优势

1. **符合大语言模型训练方式**：预训练时只需要纯文本
2. **充分利用图像空间**：整个 64×64 图像用于文本（最大 4096 tokens）
3. **灵活的数据格式**：支持多种数据格式
4. **预训练和微调统一**：同一套代码，不同配置

---

**更新日期**: 2025-12-15
