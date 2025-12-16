# 推理指南

## 🎯 推理流程

### 无条件生成

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

```python
# 条件生成
text = model.generate(
    prompt="What is AI?",
    num_steps=20,
    guidance_scale=2.0
)
```

## 🚀 使用方法

### 使用 infer.sh（推荐）

```bash
# 编辑 infer.sh 配置参数
CHECKPOINT="./run/jit_v1/best_model.pth"
MODEL="JiT-B/4"
IMG_SIZE=64
NUM_INFERENCE_STEPS=20

# 运行推理
./infer.sh
```

### 使用 Python 脚本

```bash
python inference.py \
    --checkpoint ./run/jit_v1/best_model.pth \
    --model JiT-B/4 \
    --img_size 64 \
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

1. **无条件生成**：直接从噪声生成文本图像
2. **条件生成**：基于 prompt 生成 answer
3. **Padding Mask 支持**：只对有效像素计算指标
4. **DDIM 采样**：20 步快速采样

---

**更新日期**: 2025-12-15
