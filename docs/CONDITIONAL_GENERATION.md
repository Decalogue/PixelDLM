# 条件生成

## 🎯 设计理念

**统一架构**：预训练和微调使用相同的模型架构，通过条件输入实现条件生成。

## 📊 条件生成架构

### Prompt → Answer 生成

```
Prompt 文本
  ↓ 编码为图像
64×64 条件图像
  ↓ 转换为 patches
Condition Patches (256)
  ↓ 拼接
[Condition × 256 | Target × 256]
  ↓ Transformer
生成 Answer 图像
  ↓ 解码
Answer 文本
```

### 实现方式

```python
# 训练时
if enable_condition:
    prompt_img = encode_text_to_image(prompt, size=64)
    answer_img = encode_text_to_image(answer, size=64)
    
    condition = model.image_to_patches(prompt_img)
    clean_pred = model(noisy_answer, t, condition=condition)
```

## 🔄 与无条件生成的区别

| 特性 | 无条件生成 | 条件生成 |
|------|-----------|---------|
| **序列结构** | [PAD × 256 \| Target × 256] | [Condition × 256 \| Target × 256] |
| **Attention Mask** | 前 256 个位置 mask 掉 | 前 256 个位置有效 |
| **训练数据** | 纯文本 | 问答对 (prompt, answer) |
| **应用场景** | 预训练 | 微调 |

## ✅ 优势

1. **架构统一**：预训练和微调使用相同的序列长度（512）
2. **无缝切换**：无需重新训练，直接支持条件生成
3. **灵活训练**：可以混合有条件和无条件样本训练

## 📝 使用示例

### 训练条件生成模型

```bash
python train.py \
    --data_path ./data/qa_pairs \
    --enable_condition \
    --cond_img_size 64
```

### 推理条件生成

```python
# 条件生成
text = model.generate(
    prompt="What is AI?",
    num_steps=20,
    guidance_scale=2.0
)
```

---

**更新日期**: 2025-12-15
