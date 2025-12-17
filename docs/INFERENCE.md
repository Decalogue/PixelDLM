# 推理指南

## 🎯 推理流程

### 无条件生成

使用预训练模型进行无条件文本生成：

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
```

### 条件生成

使用微调后的模型进行条件生成（prompt -> answer）：

```python
# 条件生成
text = model.generate(
    condition=condition_patches,  # prompt 编码后的 patches
    num_steps=20,
    guidance_scale=1.0
)
```

## 🚀 使用方法

### 无条件生成推理

#### 使用 infer.sh（推荐）

```bash
# 编辑 infer.sh 配置参数
CHECKPOINT="./run/jit_v1_test/best_model.pth"
MODEL="JiT-B/4"
IMG_SIZE=64
NUM_INFERENCE_STEPS=20

# 运行推理
./infer.sh
```

#### 使用 Python 脚本

```bash
python inference.py \
    --checkpoint ./run/jit_v1_test/best_model.pth \
    --model JiT-B/4 \
    --img_size 64 \
    --num_inference_steps 20 \
    --save_image
```

### 条件生成推理

#### 使用 infer_ft.sh（推荐）

```bash
# 编辑 infer_ft.sh 配置参数
CHECKPOINT="./run/jit_v1_ft/best_model.pth"
MODEL="JiT-B/4"
IMG_SIZE=64
COND_IMG_SIZE=64
NUM_INFERENCE_STEPS=20

# 可选：指定 prompt
# PROMPT="什么是人工智能？"
# PROMPT_FILE="./data/test_prompts.txt"

# 运行推理
./infer_ft.sh
```

#### 使用 Python 脚本

```bash
# 单个 prompt
python inference_ft.py \
    --checkpoint ./run/jit_v1_ft/best_model.pth \
    --model JiT-B/4 \
    --img_size 64 \
    --cond_img_size 64 \
    --prompt "什么是人工智能？" \
    --num_inference_steps 20 \
    --save_image

# 从文件读取 prompts
python inference_ft.py \
    --checkpoint ./run/jit_v1_ft/best_model.pth \
    --model JiT-B/4 \
    --img_size 64 \
    --cond_img_size 64 \
    --prompt_file ./data/test_prompts.txt \
    --num_inference_steps 20 \
    --save_image
```

## 📊 评估模型

```bash
python evaluate.py \
    --checkpoint ./run/jit_v1/best_model.pth \
    --data_path ./data/val \
    --num_samples 100
```

## ✅ 关键特性

1. **无条件生成**：直接从噪声生成文本图像（预训练模型）
2. **条件生成**：基于 prompt 生成 answer（微调模型）
3. **Padding Mask 支持**：只对有效像素计算指标
4. **DDIM 采样**：20 步快速采样
5. **双图像输出**：条件生成时同时保存 condition 和 answer 图像

## 📝 输出说明

### 无条件生成输出

- `generated_text.png`: 生成的文本图像
- `results.json`: 包含生成的文本和 token 数量

### 条件生成输出

- `generated_{i}.png`: 生成的 answer 图像
- `generated_{i}_condition.png`: 对应的 condition 图像
- `results.json`: 包含 prompt、生成的 answer 和 token 数量

---

**更新日期**: 2025-12-15


