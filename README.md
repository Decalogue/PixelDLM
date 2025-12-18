# PixelDLM 🎨 像素扩散语言模型 (实验中)

> **将文本直接编码为像素，用扩散模型生成文本的下一代方法**

[![Model Size](https://img.shields.io/badge/Model-0.1B-blue)]()
[![Image Size](https://img.shields.io/badge/Image-64x64-green)]()
[![Architecture](https://img.shields.io/badge/Architecture-JiT--B%2F4-orange)]()

---

## ✨ 核心亮点

- 🚀 **像素空间直接生成**：无需 VAE/潜在空间，文本直接映射到像素颜色
- 🎯 **统一架构设计**：预训练和微调使用相同的模型架构，无缝切换
- 🔥 **高效训练**：128M 参数，64×64 图像，适合快速迭代和实验
- 💡 **条件生成支持**：同时支持无条件文本生成和条件生成（prompt → answer）
- 🎨 **Token-to-Color 映射**：256 进制分解确保每个 token 映射到唯一颜色

---

## 🏗️ 模型架构

### Just image Transformer (JiT)

基于 Transformer 的扩散模型，采用**统一序列架构**设计：

```
┌─────────────────────────────────────────────────────────┐
│                    JiT-B/4 模型架构                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入: 64×64 图像 (RGB)                                  │
│   ↓                                                     │
│  Patch Embedding (4×4 patches → 256 patches)            │
│   ↓                                                     │
│  ┌─────────────────────────────────────┐                │
│  │  统一序列架构 (固定长度 512)          │                │
│  │  ┌──────────┬──────────────┐        │                │
│  │  │ Condition│   Target     │        │                │
│  │  │ (0-256)  │  (256-512)   │        │                │
│  │  └──────────┴──────────────┘        │                │
│  └─────────────────────────────────────┘                │
│   ↓                                                     │
│  Transformer Blocks × 12                                │
│  ├─ Multi-Head Attention (12 heads)                     │
│  ├─ SwiGLU MLP (MLP ratio=4)                            │
│  └─ AdaLNZero (时间条件归一化)                           │
│   ↓                                                     │
│  Output Projection                                      │
│   ↓                                                     │
│  输出: 预测的干净图像 patches                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 模型配置

| 配置项 | 值 |
|--------|-----|
| **模型名称** | JiT-B/4 |
| **参数量** | ~128M (0.128B) |
| **图像尺寸** | 64×64 |
| **Patch Size** | 4×4 |
| **Patches 数量** | 256 |
| **Embed Dim** | 768 |
| **Depth** | 12 层 |
| **Num Heads** | 12 |
| **序列长度** | 512 (条件 256 + 目标 256) |

---

## 🔄 核心流程

### 1. 文本 → 图像编码

```
文本 "Hello World"
  ↓ Tokenize
Token IDs: [1234, 5678, ...]
  ↓ Token-to-Color (256进制分解)
RGB Colors: [(R, G, B), ...]
  ↓ 填充到图像
64×64 图像 (每个像素 = 1个token)
```

**特点**：
- 每个 token 映射到唯一的 RGB 颜色（256 进制分解）
- 1:1 映射：1 个像素 = 1 个 token
- 支持最大 4096 tokens（64×64 图像容量）

### 2. 扩散训练流程

```
干净图像 (文本编码)
  ↓ 添加噪声 (随机时间步 t)
噪声图像
  ↓ Transformer 预测
预测的干净图像
  ↓ MSE Loss (mask 掉 padding)
更新模型参数
```

**训练特点**：
- 自监督学习：从噪声恢复原始图像
- 支持 padding mask：只对有效像素计算 loss
- 混合精度训练：FP16/BF16 加速

### 3. 条件生成架构

**统一序列设计**：无论是否有条件，都使用固定长度 512

```
无条件生成:
[PAD × 256 | Target × 256]  ← 前 256 个位置为 padding

条件生成:
[Condition × 256 | Target × 256]  ← 前 256 个位置为条件
```

**优势**：
- ✅ 预训练和微调架构完全一致
- ✅ 无需重新训练，直接支持条件生成
- ✅ 使用 attention mask 控制计算

### 4. 推理生成流程

```
随机噪声图像
  ↓ DDIM 采样 (20 步)
逐步去噪
  ↓ 解码颜色到 tokens
Token IDs
  ↓ Decode
生成文本
```

---

## 🎯 技术特点

### 1. Token-to-Color 映射

```python
# 256 进制分解：token_id → (R, G, B)
R = token_id % 256
G = (token_id // 256) % 256
B = (token_id // (256²)) % 256
```

- **无冲突**：不同 token 映射到不同颜色
- **可逆**：颜色可以唯一解码回 token_id
- **高效**：O(1) 编码/解码

### 2. 统一架构设计

- **固定序列长度**：512 patches（条件 256 + 目标 256）
- **Attention Mask**：控制哪些位置参与计算
- **2D 位置编码**：保留空间信息（行 + 列位置编码）

### 3. 条件生成支持

- **Prompt → Answer**：支持问答对格式数据
- **条件图像**：Prompt 编码为 64×64 图像作为条件
- **灵活切换**：同一模型支持有条件和无条件生成

---

## 📊 数据流程

### 预训练数据

```
纯文本数据
  ↓ Tokenize (最大 4096 tokens)
Token IDs
  ↓ 编码为图像 (64×64)
训练样本
  ↓ 添加噪声
训练
```

### 微调数据（条件生成）

```
问答对: {"prompt": "...", "answer": "..."}
  ↓
Prompt → 64×64 条件图像
Answer → 64×64 目标图像
  ↓
条件生成训练
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```sh
# 无条件生成（预训练）
sh ./train.sh

# 条件生成（微调）
sh ./train_ft.sh
```

### 推理生成

```python
from model import build_jit_model
from transformers import AutoTokenizer

# 加载模型
model = build_jit_model('JiT-B/4', img_size=64)
tokenizer = AutoTokenizer.from_pretrained('Qwen2.5-7B-Instruct')

# 无条件生成
text = model.generate(
    num_steps=20,
    guidance_scale=1.0,
    condition=None
)

# 条件生成
text = model.generate(
    prompt="What is AI?",
    num_steps=20,
    guidance_scale=2.0
)
```

---

## 📈 性能指标

- **模型大小**：128M 参数 (~0.5GB FP16)
- **训练速度**：~1000 tokens/s (8×H200, batch_size=512)
- **推理速度**：~20 步 DDIM 采样，<1 秒/样本
- **图像容量**：64×64 = 4096 tokens/图像

---

## 🎓 设计理念

1. **像素空间直接生成**：避免潜在空间的压缩损失
2. **统一架构**：预训练和微调使用相同模型，无缝切换
3. **简单高效**：小模型，快速训练，易于实验
4. **可扩展性**：支持更大图像和更多参数变体

---

## 📚 相关文档

### 核心文档

- [模型架构设计](./docs/ARCHITECTURE.md) - 统一架构、模型配置、Attention 机制
- [训练指南](./docs/TRAINING.md) - 训练时间估算、多卡训练、WandB 监控
- [推理指南](./docs/INFERENCE.md) - 推理流程、评估方法
- [数据集设计](./docs/DATASET.md) - 数据格式、编码流程、使用示例
- [Token 映射机制](./docs/TOKEN_MAPPING.md) - Token 到颜色映射原理
- [条件生成](./docs/CONDITIONAL_GENERATION.md) - 条件生成架构和使用

### 参考文档

- [训练问题诊断](./docs/TRAINING_ISSUES.md) - 训练问题分析与解决方案
- [Pixel DiT 改进方案](./docs/PIXEL_DIT_IMPROVEMENTS.md) - 基于最新论文的模型架构改进建议
- [统一条件生成训练方案](./docs/UNIFIED_CONDITIONAL_TRAINING.md) - 统一条件生成训练方案分析与对比
- [混合训练分析](./docs/MIXED_TRAINING.md) - 自然图像与文本编码图像混合训练
- [JiT vs ControlNet](./docs/JIT_VS_CONTROLNET.md) - 架构对比分析
- [2D RoPE Analysis](./docs/2D_ROPE_ANALYSIS.md) - 位置编码分析
- [World v3 数据集收集](./docs/WORLD_V3_DATASET_COLLECTION.md) - 数据集收集指南

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

**让文本生成更直观，让像素承载语义** ✨
