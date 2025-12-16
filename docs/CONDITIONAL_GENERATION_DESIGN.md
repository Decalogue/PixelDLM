# 条件生成微调设计方案

## 🎯 设计目标

在预训练的基础上，支持**条件生成**：给定 prompt（问题），生成 answer（答案）。

---

## 📊 当前架构分析

### 模型已支持的条件机制

```python
def forward(self, x, t, condition=None, mask=None):
    # condition: Condition image patches [B, cond_patches, patch_dim]
    if condition is not None:
        cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
        x = torch.cat([cond_embed, x], dim=1)  # 条件放在序列前面
```

**特点**：
- ✅ 条件作为图像 patches 输入
- ✅ 条件放在序列前面（类似 prefix）
- ✅ 使用独立的 `condition_embed` 层

---

## 💡 条件生成方案设计

### 方案 1：Prompt 图像作为条件（推荐）⭐

**设计思路**：
- 将 prompt 文本编码为**较小的图像**（如 32×32 或 16×16）
- 将 answer 文本编码为**目标图像**（64×64）
- 模型学习：`prompt图像 + 噪声` → `answer图像`

**优点**：
- ✅ 利用现有架构，无需修改模型
- ✅ 条件与目标使用相同的编码方式（token → 颜色）
- ✅ 实现简单，只需修改 dataset 和训练代码

**实现**：

```python
# dataset.py 修改
def __getitem__(self, idx):
    item = self.data[idx]
    
    if self.use_chat_template and 'prompt' in item:
        # 条件生成模式
        prompt_text = item['prompt']
        answer_text = item['answer']
        
        # 编码 prompt 为较小的图像（32×32）
        prompt_img = self._encode_text_to_image(
            prompt_text, 
            size=(32, 32)  # 条件图像较小
        )
        
        # 编码 answer 为目标图像（64×64）
        answer_img = self._encode_text_to_image(
            answer_text,
            size=(64, 64)  # 目标图像
        )
        
        return {
            'clean': answer_img_tensor,  # [3, 64, 64]
            'condition': prompt_img_tensor,  # [3, 32, 32]
            'mask': answer_mask_tensor,
            'text': answer_text,
        }
```

**训练代码修改**：

```python
# train.py
if condition is not None:
    # 将 condition 图像转换为 patches
    condition_patches = model_ref.image_to_patches(condition)  # [B, cond_patches, patch_dim]
    clean_pred = model(noisy_target, t, condition=condition_patches, mask=mask)
else:
    # 无条件生成（预训练模式）
    clean_pred = model(noisy_target, t, condition=None, mask=mask)
```

---

### 方案 2：Classifier-Free Guidance（CFG）⭐

**设计思路**：
- 训练时**随机丢弃条件**（一定概率设为 None）
- 推理时使用 guidance scale 增强条件控制

**优点**：
- ✅ 更强的条件控制能力
- ✅ 可以调节条件强度（guidance scale）
- ✅ 业界标准做法（Stable Diffusion 等）

**实现**：

```python
# train.py
# 训练时随机丢弃条件（CFG）
cfg_dropout_prob = 0.1  # 10% 的概率丢弃条件
if condition is not None and random.random() > cfg_dropout_prob:
    condition_patches = model_ref.image_to_patches(condition)
else:
    condition_patches = None  # 无条件

clean_pred = model(noisy_target, t, condition=condition_patches, mask=mask)
```

**推理时使用 CFG**：

```python
# inference.py
def generate_with_cfg(
    model, condition, guidance_scale=1.5, num_inference_steps=20
):
    # 无条件预测
    pred_uncond = model(x, t, condition=None)
    
    # 有条件预测
    pred_cond = model(x, t, condition=condition)
    
    # CFG: pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    
    return pred
```

---

### 方案 3：文本嵌入直接作为条件（需要架构修改）

**设计思路**：
- 使用文本 tokenizer 的嵌入层
- 将 prompt 的 token embeddings 作为条件

**缺点**：
- ❌ 需要修改模型架构（添加文本嵌入层）
- ❌ 条件与目标使用不同的编码方式
- ❌ 实现复杂

**不推荐**，因为方案 1 更简单且有效。

---

## 🎯 推荐方案：方案 1 + 方案 2 结合

### 完整设计

1. **数据格式**：
   ```json
   {
     "prompt": "什么是机器学习？",
     "answer": "机器学习是人工智能的一个分支..."
   }
   ```

2. **条件编码**：
   - Prompt → 32×32 图像（条件）
   - Answer → 64×64 图像（目标）

3. **训练策略**：
   - 使用 CFG dropout（10-20% 概率丢弃条件）
   - 保持预训练权重，只微调条件相关部分（可选）

4. **推理**：
   - 支持无条件生成（`condition=None`）
   - 支持条件生成（提供 prompt 图像）
   - 支持 CFG（可选）

---

## 📝 实现步骤

### Step 1: 修改 dataset.py

```python
def __getitem__(self, idx):
    item = self.data[idx]
    
    if self.use_chat_template and 'prompt' in item and 'answer' in item:
        # 条件生成模式
        prompt_text = item['prompt']
        answer_text = item['answer']
        
        # 编码 prompt（条件，较小图像）
        prompt_img = self._encode_text_to_image(
            prompt_text,
            size=(32, 32),  # 条件图像：32×32
            max_tokens=32 * 32
        )
        
        # 编码 answer（目标，完整图像）
        answer_img = self._encode_text_to_image(
            answer_text,
            size=(64, 64),  # 目标图像：64×64
            max_tokens=64 * 64
        )
        
        # 转换为 tensor
        prompt_tensor = torch.from_numpy(prompt_img).permute(2, 0, 1).float() / 255.0
        answer_tensor = torch.from_numpy(answer_img).permute(2, 0, 1).float() / 255.0
        
        # 创建 mask
        answer_mask = self._create_mask(answer_text, size=64)
        
        return {
            'clean': answer_tensor,
            'condition': prompt_tensor,  # 新增
            'mask': answer_mask,
            'text': answer_text,
        }
    else:
        # 预训练模式（无条件）
        # ... 现有代码
```

### Step 2: 修改 train.py

```python
# 添加 CFG dropout 参数
cfg_dropout_prob = 0.1  # 10% 概率丢弃条件

for batch_idx, batch in enumerate(pbar):
    clean = batch['clean'].to(device)
    mask = batch['mask'].to(device)
    condition = batch.get('condition', None)  # 可能为 None（预训练模式）
    
    # CFG: 随机丢弃条件
    if condition is not None:
        condition = condition.to(device)
        if random.random() < cfg_dropout_prob:
            condition = None  # 丢弃条件
    
    # 转换为 patches
    if condition is not None:
        condition_patches = model_ref.image_to_patches(condition)
    else:
        condition_patches = None
    
    # Forward
    clean_pred = model(noisy_target, t, condition=condition_patches, mask=mask)
```

### Step 3: 修改 inference.py

```python
def generate_with_condition(
    model, prompt_text, token_encoder, num_inference_steps=20
):
    # 编码 prompt 为图像
    prompt_img = token_encoder.encode(prompt_text, size=(32, 32))
    prompt_tensor = torch.from_numpy(prompt_img).permute(2, 0, 1).float() / 255.0
    prompt_patches = model.image_to_patches(prompt_tensor.unsqueeze(0))
    
    # 条件生成
    generated_img = model.generate(
        condition=prompt_patches,
        num_inference_steps=num_inference_steps,
        device='cuda'
    )
    
    return generated_img
```

---

## 🔧 关键设计决策

### 1. 条件图像尺寸

**选项**：
- **32×32**（推荐）：1024 tokens，足够大多数 prompt
- **16×16**：256 tokens，适合短 prompt
- **64×64**：4096 tokens，与目标相同（可能浪费）

**推荐 32×32**：
- 平衡容量和效率
- 大多数 prompt 在 1024 tokens 以内
- 条件不需要太大

### 2. CFG Dropout 概率

**选项**：
- **0.1**（10%）：轻微增强无条件能力
- **0.2**（20%）：平衡条件和无条件
- **0.0**（0%）：完全条件生成（不推荐）

**推荐 0.1-0.2**：
- 保持模型的无条件生成能力
- 增强条件控制能力

### 3. 微调策略

**选项 A：全量微调**
- 微调所有参数
- 需要较小学习率（如 1e-6）

**选项 B：部分微调**
- 只微调 `condition_embed` 和最后几层
- 冻结预训练权重

**推荐选项 A**：
- 条件生成需要模型理解条件与目标的关系
- 全量微调效果更好

---

## 📊 训练数据格式

### 预训练数据（无条件）

```json
[
  {"text": "这是一段文本..."},
  {"text": "另一段文本..."}
]
```

### 微调数据（条件生成）

```json
[
  {
    "prompt": "什么是机器学习？",
    "answer": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习..."
  },
  {
    "prompt": "解释一下深度学习。",
    "answer": "深度学习是机器学习的一个子领域，使用多层神经网络..."
  }
]
```

---

## ✅ 总结

### 推荐方案

1. **条件编码**：Prompt → 32×32 图像（条件），Answer → 64×64 图像（目标）
2. **训练策略**：CFG dropout（10-20% 概率丢弃条件）
3. **微调方式**：全量微调，使用较小学习率（1e-6）
4. **推理支持**：无条件 + 条件生成

### 优势

- ✅ 利用现有架构，实现简单
- ✅ 条件与目标使用相同编码方式
- ✅ 支持 CFG，增强条件控制
- ✅ 保持无条件生成能力

---

**修改日期**: 2025-12-15
