# Token 到颜色映射

## 🎯 核心机制

### 256 进制分解

```python
def token_id_to_color(token_id: int) -> Tuple[int, int, int]:
    r = token_id % 256                    # 最低位（R通道）
    g = (token_id // 256) % 256          # 次低位（G通道）
    b = (token_id // (256 * 256)) % 256   # 第三位（B通道）
    return (r, g, b)
```

### 特点

- **确定性映射**：每个 token_id 对应唯一的 RGB 颜色
- **无冲突**：不同 token 映射到不同颜色
- **可逆**：颜色可以唯一解码回 token_id
- **高效**：O(1) 编码/解码

## 📊 映射示例

```
token_id=0     → (0, 0, 0)     黑色
token_id=1     → (1, 0, 0)     深红色
token_id=256   → (0, 1, 0)     深绿色
token_id=257   → (1, 1, 0)     黄绿色
token_id=65536 → (0, 0, 1)     深蓝色
```

## ✅ 为什么看起来像噪声？

### 数学映射 vs 视觉结构

- **自然图像**：有空间相关性、边缘、纹理等视觉结构
- **Token 颜色映射**：是数学上的 256 进制分解，相邻 token 的颜色可能差异很大

**这是正常的！** 模型不需要理解视觉结构，只需要学习从噪声恢复确定的 token 颜色。

## 🔄 编码/解码流程

### 编码：文本 → 图像

```
文本 "Hello World"
  ↓ Tokenize
Token IDs: [1234, 5678, ...]
  ↓ Token-to-Color (256进制分解)
RGB Colors: [(R, G, B), ...]
  ↓ 填充到图像
64×64 图像 (每个像素 = 1个token)
```

### 解码：图像 → 文本

```
64×64 图像
  ↓ 提取像素颜色
RGB Colors: [(R, G, B), ...]
  ↓ Color-to-Token (256进制组合)
Token IDs: [1234, 5678, ...]
  ↓ Decode
文本 "Hello World"
```

## 💡 设计优势

1. **1:1 映射**：1 个像素 = 1 个 token
2. **无信息损失**：编码和解码完全可逆
3. **支持最大 4096 tokens**（64×64 图像容量）
4. **Padding 处理**：使用 pad_token 颜色或白色作为 padding

---

**更新日期**: 2025-12-15
