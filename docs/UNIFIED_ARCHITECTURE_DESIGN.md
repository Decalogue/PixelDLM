# 统一架构设计：固定序列长度

## 🎯 问题

**当前问题**：
- 无条件生成：序列长度 = 256
- 条件生成：序列长度 = 512（256 + 256）
- **架构不一致**：预训练和微调使用不同的序列长度

---

## ✅ 解决方案：固定最大序列长度

### 设计思路

**统一架构**：无论是否有条件，都使用**固定的最大序列长度**（512）

**实现方式**：
1. **无条件时**：前 256 个位置使用 padding（或特殊 token），后 256 个是目标图像
2. **有条件时**：前 256 个是条件，后 256 个是目标图像
3. **使用 attention mask**：控制哪些位置参与计算

---

## 📊 架构设计

### 固定序列结构

```
总序列长度：512 patches（固定）

无条件生成：
[PAD, PAD, ..., PAD, | Target_0, Target_1, ..., Target_255]
  ↑ 256 个 padding    ↑ 256 个目标 patches

条件生成：
[Cond_0, Cond_1, ..., Cond_255, | Target_0, Target_1, ..., Target_255]
  ↑ 256 个条件 patches          ↑ 256 个目标 patches
```

### Attention Mask

```python
# 无条件：mask 掉前 256 个位置
mask_uncond = [0, 0, ..., 0, | 1, 1, ..., 1]
               ↑ 256 个 0    ↑ 256 个 1

# 条件：前 256 个位置有效
mask_cond = [1, 1, ..., 1, | 1, 1, ..., 1]
            ↑ 256 个 1     ↑ 256 个 1
```

---

## 🔧 实现方案

### 方案 1：Padding + Attention Mask（推荐）⭐

**设计**：
- 无条件时：前 256 个位置使用零 padding
- 使用 attention mask 控制计算
- 架构完全统一

**优点**：
- ✅ 架构完全一致
- ✅ 预训练和微调使用相同架构
- ✅ 实现简单

**缺点**：
- ⚠️ 无条件时前 256 个位置仍会计算（但会被 mask 掉，实际不参与注意力）

---

### 方案 2：特殊 Token（更优雅）

**设计**：
- 无条件时：前 256 个位置使用特殊的"无条件" token
- 模型学习区分"无条件"和"真实条件"

**优点**：
- ✅ 架构完全一致
- ✅ 模型可以学习"无条件"的特殊表示

**缺点**：
- ⚠️ 需要额外的学习过程

---

## 🎯 推荐实现：方案 1

### 架构修改

```python
class JiT(nn.Module):
    def __init__(self, ...):
        # 固定最大序列长度
        self.max_seq_len = 512  # 256 (条件) + 256 (目标)
        
        # 目标图像 patches
        self.num_patches = 256
        
        # 2D 位置编码（扩展到最大序列长度）
        # 前 256 个：条件位置（16×16）
        # 后 256 个：目标位置（16×16）
        max_grid_size = 16  # 16×16 = 256
        self.pos_embed_h = nn.Parameter(torch.zeros(1, max_grid_size, embed_dim))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, max_grid_size, embed_dim))
        
        # 条件位置编码（前 256 个位置）
        self.cond_pos_embed_h = nn.Parameter(torch.zeros(1, max_grid_size, embed_dim))
        self.cond_pos_embed_w = nn.Parameter(torch.zeros(1, max_grid_size, embed_dim))
    
    def forward(self, x, t, condition=None, mask=None):
        B = x.shape[0]
        
        # 目标图像嵌入和位置编码
        x_target = self.patch_embed(x_patches)  # [B, 256, embed_dim]
        x_target = x_target + self._get_2d_pos_embed(256, self.pos_embed_h, self.pos_embed_w)
        
        # 条件处理
        if condition is not None:
            # 有条件：使用真实条件
            cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
            cond_embed = cond_embed + self._get_2d_pos_embed(cond_patches, self.cond_pos_embed_h, self.cond_pos_embed_w)
            # 如果条件 patches < 256，需要 padding
            if cond_embed.shape[1] < 256:
                padding = 256 - cond_embed.shape[1]
                cond_padding = torch.zeros(B, padding, self.embed_dim, device=cond_embed.device)
                cond_embed = torch.cat([cond_embed, cond_padding], dim=1)
        else:
            # 无条件：使用零 padding
            cond_embed = torch.zeros(B, 256, self.embed_dim, device=x_target.device)
        
        # 拼接：总长度固定为 512
        x = torch.cat([cond_embed, x_target], dim=1)  # [B, 512, embed_dim]
        
        # 创建 attention mask
        if condition is not None:
            # 有条件：前 256 个位置有效（根据实际条件长度）
            cond_mask = torch.ones(B, condition.shape[1], device=x.device)
            if condition.shape[1] < 256:
                cond_padding_mask = torch.zeros(B, 256 - condition.shape[1], device=x.device)
                cond_mask = torch.cat([cond_mask, cond_padding_mask], dim=1)
            target_mask = torch.ones(B, 256, device=x.device)
            attention_mask = torch.cat([cond_mask, target_mask], dim=1)  # [B, 512]
        else:
            # 无条件：前 256 个位置无效（padding）
            cond_mask = torch.zeros(B, 256, device=x.device)
            target_mask = torch.ones(B, 256, device=x.device)
            attention_mask = torch.cat([cond_mask, target_mask], dim=1)  # [B, 512]
        
        # Transformer blocks（使用统一的 attention mask）
        for block in self.blocks:
            x = block(x, t_embed, attention_mask=attention_mask)
        
        # 提取目标部分（后 256 个位置）
        x = x[:, 256:, :]  # [B, 256, embed_dim]
        
        return x
```

---

## 📊 架构对比

### 修改前

```
无条件：
序列长度: 256
架构: [Target_0, ..., Target_255]

条件：
序列长度: 320-512（可变）
架构: [Cond_0, ..., Cond_N, Target_0, ..., Target_255]
```

### 修改后

```
无条件：
序列长度: 512（固定）
架构: [PAD_0, ..., PAD_255, Target_0, ..., Target_255]
Mask:  [0, ..., 0, 1, ..., 1]

条件：
序列长度: 512（固定）
架构: [Cond_0, ..., Cond_255, Target_0, ..., Target_255]
Mask:  [1, ..., 1, 1, ..., 1]
```

---

## ✅ 优势

### 1. 架构完全统一

- ✅ 预训练和微调使用相同的序列长度（512）
- ✅ 位置编码统一
- ✅ Transformer blocks 处理相同的输入维度

### 2. 训练一致性

- ✅ 模型从一开始就适应 512 长度的序列
- ✅ 位置编码学习条件+目标的组合模式
- ✅ 无需在微调时适应新的序列长度

### 3. 实现简单

- ✅ 只需修改 forward 方法
- ✅ 使用 attention mask 控制计算
- ✅ 无需修改 Transformer blocks

---

## ⚠️ 注意事项

### 1. 计算开销

- 无条件时前 256 个位置仍会计算（但会被 mask 掉）
- 可以通过优化 attention 实现减少计算（如 Flash Attention）

### 2. 位置编码

- 前 256 个位置：条件位置编码
- 后 256 个位置：目标位置编码
- 需要确保位置编码正确

### 3. 预训练策略

- 预训练时也使用固定 512 长度
- 无条件时前 256 个是 padding
- 让模型从一开始就适应这种架构

---

## 📝 实现步骤

### Step 1: 修改模型初始化

```python
def __init__(self, ...):
    # 固定最大序列长度
    self.max_seq_len = 512
    self.num_patches = 256
    self.cond_max_patches = 256
```

### Step 2: 修改 forward 方法

- 无条件时：前 256 个位置使用零 padding
- 有条件时：前 256 个位置使用真实条件（可能 padding）
- 总长度固定为 512

### Step 3: 统一 attention mask

- 无条件：前 256 个位置 mask 掉
- 有条件：根据实际条件长度设置 mask

---

## ✅ 总结

### 问题回答

**能否统一架构？**

**可以！** 通过固定最大序列长度为 512，使用 padding + attention mask。

### 推荐方案

**方案 1：Padding + Attention Mask**
- 架构完全统一
- 实现简单
- 预训练和微调使用相同架构

### 关键点

1. ✅ 固定序列长度：512（256 条件 + 256 目标）
2. ✅ 无条件时：前 256 个位置是 padding
3. ✅ 使用 attention mask 控制计算
4. ✅ 预训练时也使用这种架构

---

**修改日期**: 2025-12-15




