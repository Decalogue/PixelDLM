# Loss=NaN 问题分析报告

## 🔍 问题概述

训练过程中 loss 变为 NaN，通常在训练开始后不久（5-6%）出现。

---

## 🎯 潜在问题分析

### 1. **数据编码问题** ⚠️ **高风险**

**位置**: `dataset.py` - `_encode_text_to_image`

**问题**:
- Token ID 范围是 `[0, 151643]`（vocab_size）
- 使用 256 进制分解：`r = token_id % 256`, `g = (token_id // 256) % 256`, `b = (token_id // 65536) % 256`
- **问题**：当 `token_id >= 16777216 (256^3)` 时，会溢出，但 vocab_size=151643 < 16777216，所以理论上不会溢出
- **但是**：如果 token_id 超出 vocab_size（特殊 token），可能接近或超过 256^3

**检查点**:
```python
# robust_token2img.py 中的 token_id_to_color
def token_id_to_color(self, token_id: int) -> Tuple[int, int, int]:
    if token_id < 0:
        raise ValueError(f"token_id {token_id} cannot be negative")
    r = token_id % 256
    g = (token_id // 256) % 256
    b = (token_id // (256 * 256)) % 256
    return (r, g, b)
```

**可能的问题**：
- 如果 token_id 很大（接近 256^3），b 值可能很大，导致颜色值异常
- 图像归一化到 [0, 1] 后，如果原始值异常，可能导致数值不稳定

---

### 2. **模型数值稳定性问题** ⚠️ **高风险**

#### 2.1 RMSNorm 除法问题

**位置**: `model.py` - `RMSNorm.forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
    return x / (norm + self.eps) * self.weight
```

**问题**:
- 如果 `x` 全为 0（padding），`norm` 可能为 0，虽然有 `eps=1e-6`，但可能不够
- 如果 `x` 中有 inf 或 nan，会传播

#### 2.2 AdaLNZero Scale 问题

**位置**: `model.py` - `AdaLNZero.forward`

```python
def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    shift = self.shift_mlp(c)
    scale = self.scale_mlp(c) + 1.0
    return self.norm(x) * scale.unsqueeze(1) + shift.unsqueeze(1)
```

**问题**:
- `scale_mlp` 初始化为 0，所以初始 `scale = 0 + 1.0 = 1.0`，这是正常的
- 但如果训练过程中 `scale_mlp` 输出很大（正或负），`scale` 可能很大或为负，导致数值不稳定

#### 2.3 Attention Mask 问题

**位置**: `model.py` - `MultiHeadAttention.forward`

```python
if attention_mask is not None:
    mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
    attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
```

**问题**:
- 如果 `mask_expanded == 0` 的位置很多，`attn` 中会有很多 `-inf`
- 虽然 softmax 会将 `-inf` 转换为 0，但如果所有位置都是 `-inf`，softmax 可能产生 nan

#### 2.4 Attention Scale 问题

**位置**: `model.py` - `MultiHeadAttention.__init__`

```python
self.scale = self.head_dim ** -0.5
```

**问题**:
- 如果 `head_dim` 很小，`scale` 会很大
- 对于 `embed_dim=768, num_heads=12`，`head_dim=64`，`scale=0.125`，这是正常的
- 但如果 `head_dim` 计算错误，可能导致 `scale` 异常

---

### 3. **训练过程问题** ⚠️ **高风险**

#### 3.1 噪声添加问题

**位置**: `train.py` - `add_noise_to_timestep`

```python
def add_noise_to_timestep(x: torch.Tensor, t: torch.Tensor, noise_schedule: str = 'linear') -> Tuple[torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    device = x.device
    
    noise = torch.randn_like(x)
    
    if noise_schedule == 'linear':
        alpha = 1.0 - (t.float() / 1000.0)
    else:
        alpha = 0.5 * (1 + torch.cos(torch.pi * t.float() / 1000.0))
    
    alpha = alpha.view(B, 1, 1, 1)
    noisy = alpha * x + (1 - alpha) * noise
    
    return noisy, noise
```

**问题**:
- `t` 的范围是 `[0, 999]`（`torch.randint(0, 1000, ...)`）
- `alpha = 1.0 - (t / 1000.0)`，范围是 `[0.001, 1.0]`
- 当 `t=999` 时，`alpha=0.001`，`noisy = 0.001 * x + 0.999 * noise`
- 如果 `x` 中有异常值（inf/nan），会传播到 `noisy`

#### 3.2 Loss 计算问题

**位置**: `train.py` - `train_epoch_optimized`

```python
# 计算 masked loss
diff = (clean_pred_img - clean) ** 2
masked_diff = diff * mask_expanded
loss = masked_diff.sum() / (mask_expanded.sum() + 1e-8)  # 归一化
loss = loss / gradient_accumulation_steps
```

**问题**:
- 如果 `clean_pred_img` 或 `clean` 中有 inf/nan，`diff` 会包含 inf/nan
- 如果 `mask_expanded.sum() == 0`（所有像素都是 padding），虽然有 `1e-8`，但如果 `masked_diff.sum()` 也是 0，`loss = 0 / 1e-8 = 0`，这是正常的
- 但如果 `masked_diff.sum()` 是 inf，`loss = inf / 1e-8 = inf`

#### 3.3 学习率问题

**当前设置**: `LR=1e-4`（train_test.sh）

**问题**:
- 对于扩散模型，`1e-4` 可能偏大
- 如果梯度很大，`lr * grad` 可能导致参数更新过大，产生 inf/nan

#### 3.4 梯度裁剪问题

**当前设置**: `MAX_GRAD_NORM=1.0`

**问题**:
- 梯度裁剪在 AMP 模式下需要先 `scaler.unscale_()`，代码中已经做了
- 但如果梯度本身包含 inf/nan，裁剪可能无效

---

### 4. **混合精度训练问题** ⚠️ **中等风险**

**位置**: `train.py` - AMP context

**问题**:
- 混合精度训练可能导致数值精度损失
- 如果某些操作在 fp16 下不稳定，可能产生 inf/nan
- 特别是 RMSNorm 中的除法操作

---

## 🔧 修复建议

### 优先级 1: 添加数值稳定性检查

1. **在 loss 计算前检查输入**:
```python
# 检查 clean_pred_img 和 clean 是否包含 nan/inf
if torch.isnan(clean_pred_img).any() or torch.isinf(clean_pred_img).any():
    print(f"Warning: clean_pred_img contains nan/inf at batch {batch_idx}")
    continue

if torch.isnan(clean).any() or torch.isinf(clean).any():
    print(f"Warning: clean contains nan/inf at batch {batch_idx}")
    continue
```

2. **在模型输出后检查**:
```python
clean_pred = model(noisy_target, t, condition=None, mask=mask)
if torch.isnan(clean_pred).any() or torch.isinf(clean_pred).any():
    print(f"Warning: model output contains nan/inf at batch {batch_idx}")
    continue
```

### 优先级 2: 修复 RMSNorm

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 添加数值稳定性检查
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("Warning: RMSNorm input contains nan/inf")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    
    norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
    # 确保 norm 不会太小
    norm = torch.clamp(norm, min=self.eps)
    return x / norm * self.weight
```

### 优先级 3: 降低学习率

```bash
# train_test.sh
LR=1e-5  # 从 1e-4 降低到 1e-5
```

### 优先级 4: 增强梯度裁剪

```python
# 在梯度裁剪前检查
if use_amp:
    scaler.unscale_(optimizer)
    # 检查梯度是否包含 nan/inf
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Warning: gradient contains nan/inf in {name}")
                param.grad.zero_()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

### 优先级 5: 修复 Attention Mask

```python
# 确保不会所有位置都是 -inf
if attention_mask is not None:
    mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
    
    # 检查是否所有位置都是 -inf
    if (attn == float('-inf')).all():
        # 如果所有位置都是 -inf，设置为均匀分布
        attn = torch.zeros_like(attn)
    else:
        attn = attn.softmax(dim=-1)
```

---

## 🎯 最可能的原因

基于分析，**最可能的原因是**：

1. **模型输出包含 inf/nan**（可能是 RMSNorm 或 AdaLNZero 的数值不稳定）
2. **学习率过大**（1e-4 对于扩散模型可能偏大）
3. **数据编码问题**（虽然理论上不会溢出，但需要验证）

---

## 📝 调试步骤

1. **添加数值检查**：在关键位置添加 nan/inf 检查
2. **降低学习率**：从 1e-4 降低到 1e-5 或 5e-6
3. **检查数据**：验证 token_id 范围和颜色编码
4. **禁用混合精度**：测试是否与 AMP 相关
5. **简化模型**：使用更小的模型测试

---

**修改日期**: 2024-12-15
