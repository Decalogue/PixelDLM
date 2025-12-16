# Padding Mask 支持说明

## 🎯 概述

`model_jit.py` 现在完全支持 padding mask，可以在训练和推理时正确处理 padding 区域。

---

## 🔧 实现细节

### 1. 图像 Mask 到 Patch Mask 转换

```python
def image_mask_to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
    """
    将图像 mask 转换为 patch mask
    
    Args:
        mask: Image mask [B, H, W] - 1 表示有效像素，0 表示 padding
    
    Returns:
        patch_mask: Patch mask [B, num_patches] - 1 表示有效 patch，0 表示 padding patch
    """
    # 将 mask 重塑为 patches，对每个 patch 取平均值
    # 如果 patch 中大部分像素是有效的，则认为 patch 是有效的
    mask_patches = mask.reshape(B, n, p, n, p)
    mask_patches = mask_patches.mean(dim=(2, 4))
    patch_mask = (mask_patches > 0.5).float()
    
    return patch_mask
```

**原理**：
- 将图像 mask `[B, H, W]` 转换为 patch mask `[B, num_patches]`
- 对每个 patch 内的像素取平均值
- 如果平均值 > 0.5，则认为 patch 是有效的

### 2. Attention Mask 支持

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # ...
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            # Mask out padding positions: set to large negative value before softmax
            attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        # ...
```

**原理**：
- 在 attention 计算中，将 padding patches 的 attention 权重设为 `-inf`
- 经过 softmax 后，padding patches 的 attention 权重为 0
- 模型不会关注 padding 区域

### 3. Forward 方法支持 Mask

```python
def forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Convert image mask to patch mask
    if mask is not None:
        patch_mask = self.image_mask_to_patch_mask(mask)
        # 如果有 condition，需要扩展 patch_mask
        if condition is not None:
            cond_mask = torch.ones(B, condition.shape[1], device=patch_mask.device)
            patch_mask = torch.cat([cond_mask, patch_mask], dim=1)
    
    # Transformer blocks (with attention mask)
    for block in self.blocks:
        x = block(x, t_embed, attention_mask=patch_mask)
    
    # 将 padding patches 的输出置零
    if patch_mask is not None:
        patch_mask_expanded = patch_mask.unsqueeze(-1)
        x = x * patch_mask_expanded
    
    return x
```

**关键点**：
1. 将图像 mask 转换为 patch mask
2. 在 attention 中使用 patch mask
3. 在输出时将 padding patches 置零

---

## 📊 使用示例

### 训练时

```python
# 在 train.py 中
for batch in dataloader:
    clean = batch['clean'].to(device)  # [B, 3, H, W]
    mask = batch['mask'].to(device)    # [B, H, W]
    
    # 添加噪声
    noisy_target, _ = add_noise_to_timestep(clean, t)
    
    # Forward pass with mask
    clean_pred = model(noisy_target, t, condition=None, mask=mask)
    
    # Convert to image
    clean_pred_img = model.patches_to_image(clean_pred)
    
    # Loss with mask (只对有效像素计算)
    mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
    diff = (clean_pred_img - clean) ** 2
    masked_diff = diff * mask_expanded
    loss = masked_diff.sum() / (mask_expanded.sum() + 1e-8)
```

### 推理时（可选）

```python
# 在生成时也可以使用 mask
# 例如：只生成前 N 个 tokens，后面的保持为 padding

mask = torch.ones(B, H, W)
mask[:, num_tokens//H:, :] = 0  # 后面的像素是 padding

# 生成时传入 mask
generated = model.generate(condition=condition, mask=mask)
```

---

## ✅ 优势

### 1. 训练效率

- ✅ **不浪费计算资源**：模型不会关注 padding 区域
- ✅ **更准确的梯度**：只对有效像素计算 loss
- ✅ **更快的收敛**：模型专注于学习有效内容

### 2. 模型性能

- ✅ **更好的表示**：模型学习区分有效内容和 padding
- ✅ **更准确的生成**：生成时知道哪些区域是 padding
- ✅ **更稳定的训练**：避免 padding 区域的噪声影响

### 3. 灵活性

- ✅ **支持变长序列**：不同样本可以有不同的有效长度
- ✅ **支持条件生成**：可以与 condition 一起使用
- ✅ **支持批量训练**：batch 中的样本可以有不同的长度

---

## 🔍 技术细节

### 1. Patch Mask 转换

```
图像 mask: [B, 256, 256]
    ↓
重塑为 patches: [B, 16, 16, 16, 16]  (patch_size=16)
    ↓
对每个 patch 取平均值: [B, 16, 16]
    ↓
阈值化 (>0.5): [B, 16, 16]
    ↓
展平: [B, 256]
```

### 2. Attention Mask 应用

```
Attention scores: [B, num_heads, N, N]
    ↓
Apply mask: mask_expanded = [B, 1, 1, N]
    ↓
Masked fill: attn.masked_fill(mask == 0, -inf)
    ↓
Softmax: padding positions → 0
    ↓
Output: padding patches 不参与 attention
```

### 3. 输出 Mask 应用

```
Output patches: [B, num_patches, patch_dim]
    ↓
Patch mask: [B, num_patches]
    ↓
Expand: [B, num_patches, 1]
    ↓
Multiply: output * mask
    ↓
Result: padding patches → 0
```

---

## 📝 注意事项

### 1. Mask 格式

- **图像 mask**: `[B, H, W]`，值为 1（有效）或 0（padding）
- **Patch mask**: `[B, num_patches]`，值为 1（有效）或 0（padding）

### 2. Condition 处理

- 如果有 condition，patch mask 需要扩展以包含 condition patches
- Condition patches 总是有效的（mask = 1）

### 3. 阈值选择

- 当前使用阈值 0.5 来判断 patch 是否有效
- 如果 patch 中 >50% 的像素是有效的，则认为 patch 是有效的
- 可以根据需要调整阈值

---

## 🎯 总结

### 支持的场景

1. ✅ **训练时**：使用 mask 在 loss 中忽略 padding
2. ✅ **Attention 中**：使用 mask 避免关注 padding 区域
3. ✅ **输出时**：使用 mask 将 padding patches 置零
4. ✅ **生成时**：可以使用 mask 控制生成区域

### 关键改进

1. ✅ 添加 `image_mask_to_patch_mask` 方法
2. ✅ `MultiHeadAttention` 支持 `attention_mask` 参数
3. ✅ `TransformerBlock` 传递 `attention_mask`
4. ✅ `forward` 方法支持 `mask` 参数
5. ✅ 输出时将 padding patches 置零

---

**结论**：`model_jit.py` 现在完全支持 padding mask，可以在训练和推理时正确处理 padding 区域，提高训练效率和模型性能！
