# 条件位置编码修复方案

## 🚨 问题分析

### 当前代码的问题

```python
# 位置编码（只针对目标图像）
x = self.patch_embed(x_patches)  # [B, num_patches, embed_dim]
x = x + self.pos_embed  # pos_embed: [1, num_patches, embed_dim]

# 条件嵌入
if condition is not None:
    cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
    x = torch.cat([cond_embed, x], dim=1)  # [B, cond_patches + num_patches, embed_dim]
```

**问题**：
1. ❌ `pos_embed` 只针对 `num_patches`，条件部分没有位置编码
2. ❌ 拼接后序列长度变化，但位置编码不匹配
3. ❌ 条件部分的位置信息丢失

---

## ✅ 修复方案

### 方案 1：为条件添加独立的位置编码（推荐）⭐

**设计**：
- 条件使用独立的位置编码（可学习或固定）
- 目标图像使用原有的位置编码

**实现**：

```python
class JiT(nn.Module):
    def __init__(self, ...):
        # 原有位置编码（目标图像）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 条件位置编码（可学习）
        # 假设条件图像是 32×32，patch_size=4，则 cond_patches = (32//4)^2 = 64
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))  # 可配置
        
        # 或者使用固定位置编码
        # self.cond_pos_embed = self._create_cond_pos_embed(max_cond_patches=64)
    
    def forward(self, x, t, condition=None, mask=None):
        # 目标图像嵌入和位置编码
        x = self.patch_embed(x_patches)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed  # [B, num_patches, embed_dim]
        
        # 条件嵌入和位置编码
        if condition is not None:
            cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
            # 为条件添加位置编码
            cond_pos = self.cond_pos_embed[:, :cond_embed.shape[1], :]  # 截取或填充
            cond_embed = cond_embed + cond_pos
            # 拼接
            x = torch.cat([cond_embed, x], dim=1)  # [B, cond_patches + num_patches, embed_dim]
```

**优点**：
- ✅ 条件有独立的位置信息
- ✅ 实现简单
- ✅ 可学习的位置编码更灵活

---

### 方案 2：使用相对位置编码（更灵活）

**设计**：
- 使用相对位置编码，自动适应序列长度
- 条件和目标使用不同的位置编码空间

**实现**：

```python
class JiT(nn.Module):
    def __init__(self, ...):
        # 使用可学习的相对位置编码
        self.max_seq_len = 1024  # 足够大的序列长度
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        
    def forward(self, x, t, condition=None, mask=None):
        # 目标图像
        x = self.patch_embed(x_patches)
        x = x + self.pos_embed[:, :x.shape[1], :]  # 动态截取
        
        # 条件
        if condition is not None:
            cond_embed = self.condition_embed(condition)
            # 条件使用不同的位置编码范围
            cond_start_idx = 0
            cond_embed = cond_embed + self.pos_embed[:, cond_start_idx:cond_start_idx+cond_embed.shape[1], :]
            x = torch.cat([cond_embed, x], dim=1)
```

**优点**：
- ✅ 更灵活，适应不同长度的条件
- ✅ 条件与目标使用不同的位置空间

---

### 方案 3：条件不使用位置编码（简单但不推荐）

**设计**：
- 条件作为"全局上下文"，不需要位置信息
- 只对目标图像使用位置编码

**实现**：

```python
if condition is not None:
    cond_embed = self.condition_embed(condition)
    # 条件不加位置编码（作为全局上下文）
    x = self.patch_embed(x_patches)
    x = x + self.pos_embed  # 只对目标图像加位置编码
    x = torch.cat([cond_embed, x], dim=1)
```

**缺点**：
- ❌ 条件失去了位置信息
- ❌ 如果条件较长，可能影响效果

---

## 🎯 推荐实现（方案 1）

### 完整修复代码

```python
class JiT(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        cond_img_size: int = 32,  # 新增：条件图像尺寸
        ...
    ):
        # 原有代码
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 新增：条件位置编码
        self.cond_num_patches = (cond_img_size // patch_size) ** 2
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, self.cond_num_patches, embed_dim))
        
        # 初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cond_pos_embed, std=0.02)  # 条件位置编码
    
    def forward(self, x, t, condition=None, mask=None):
        # 目标图像嵌入和位置编码
        x = self.patch_embed(x_patches)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed  # [B, num_patches, embed_dim]
        
        # 条件嵌入和位置编码
        if condition is not None:
            cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
            
            # 为条件添加位置编码
            # 如果 cond_patches 小于 cond_num_patches，截取
            # 如果 cond_patches 大于 cond_num_patches，需要处理（通常不会）
            cond_pos = self.cond_pos_embed[:, :cond_embed.shape[1], :]
            cond_embed = cond_embed + cond_pos
            
            # 拼接
            x = torch.cat([cond_embed, x], dim=1)  # [B, cond_patches + num_patches, embed_dim]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_embed, attention_mask=patch_mask)
        
        # 提取目标部分（移除条件）
        if condition is not None:
            x = x[:, condition.shape[1]:, :]
        
        return x
```

---

## 📊 维度变化分析

### 无条件生成

```
输入: x_patches [B, 256, 48]  # 64×64, patch_size=4 → 256 patches
  ↓ patch_embed
x: [B, 256, 768]
  ↓ + pos_embed [1, 256, 768]
x: [B, 256, 768]
  ↓ Transformer
x: [B, 256, 768]
  ↓ output_proj
x: [B, 256, 48]
```

### 条件生成（修复后）

```
条件: condition [B, 64, 48]  # 32×32, patch_size=4 → 64 patches
  ↓ condition_embed
cond_embed: [B, 64, 768]
  ↓ + cond_pos_embed [1, 64, 768]
cond_embed: [B, 64, 768]

目标: x_patches [B, 256, 48]
  ↓ patch_embed
x: [B, 256, 768]
  ↓ + pos_embed [1, 256, 768]
x: [B, 256, 768]

  ↓ concat
x: [B, 320, 768]  # 64 + 256 = 320
  ↓ Transformer (处理 320 个 tokens)
x: [B, 320, 768]
  ↓ 提取目标部分
x: [B, 256, 768]  # 移除前 64 个（条件）
  ↓ output_proj
x: [B, 256, 48]
```

---

## ✅ 总结

### 问题确认

1. ✅ **维度确实会变大**：`cond_patches + num_patches`
2. ✅ **位置编码不匹配**：条件部分没有位置编码
3. ✅ **需要修复**：为条件添加独立的位置编码

### 推荐方案

**方案 1：独立的条件位置编码**
- 实现简单
- 条件有独立的位置信息
- 推荐使用

### 修复要点

1. 添加 `cond_pos_embed` 参数
2. 在条件嵌入后添加位置编码
3. 确保维度匹配（截取或填充）

---

**修改日期**: 2025-12-15
