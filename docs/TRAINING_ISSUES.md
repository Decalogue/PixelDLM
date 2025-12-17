# 训练问题诊断与解决方案

## 🔍 问题现象

从训练监控图表观察到：

1. **Loss 快速收敛**：前 2000 步就降到 <0.2
2. **Learning rate 持续线性增长**：整个训练过程都在 warmup 阶段
3. **Grad norm 不稳定**：
   - 前 2000 步：从 1,500,000 快速下降到 50k-150k
   - 2k-7k 步：相对稳定在 50k-150k
   - 7k 步后：出现巨大波动和尖峰（>1,500,000）

## ⚠️ 问题诊断

### 问题 1: Warmup 时间过长

**原因**：
- `warmup_steps = effective_batches_per_epoch * warmup_epochs`
- 如果数据集很大，`effective_batches_per_epoch` 可能达到数千
- 导致 `warmup_steps` 非常大（可能 > 10,000）
- 整个训练过程都在 warmup，学习率持续增长

**影响**：
- 学习率持续增长，后期学习率过大
- 引起梯度爆炸（grad_norm 尖峰）
- 训练不稳定

### 问题 2: 模型过早收敛

**现象**：
- Loss 在前 2000 步就快速下降到很低
- 但后续训练不稳定（grad_norm 波动大）

**可能原因**：
- 学习率过大导致陷入 sharp minima
- 或模型容量小，快速过拟合

### 问题 3: 梯度裁剪不够严格

**现象**：
- Grad norm 尖峰达到 1,500,000+
- 即使有梯度裁剪（MAX_GRAD_NORM=0.5），也可能不够

## ✅ 解决方案

### 1. 限制 Warmup 时间（已修复）

**修改**：
```python
# 限制 warmup_steps 最大值为 1000
warmup_steps = min(effective_batches_per_epoch * args.warmup_epochs, 1000)
```

**效果**：
- Warmup 最多 1000 步
- 之后进入 cosine decay 阶段
- 学习率不再持续增长

### 2. 增强梯度裁剪（已修复）

**修改**：
```bash
# train.sh
MAX_GRAD_NORM=0.1  # 从 0.5 降到 0.1
```

**效果**：
- 更严格的梯度裁剪
- 防止梯度爆炸
- 训练更稳定

### 3. 调整学习率（可选）

**建议**：
- 如果 loss 收敛太快，可以降低学习率
- 当前：`LR=5e-6`
- 建议：`LR=1e-5` 到 `5e-5`（根据实际情况调整）

### 4. 监控训练指标

**关键指标**：
- `train/loss`：应该平稳下降，不应该过早收敛
- `train/learning_rate`：warmup 后应该开始 decay
- `train/grad_norm`：应该稳定在合理范围（< 100,000）

## 📊 预期效果

修复后应该看到：

1. **Learning rate**：
   - 前 1000 步：线性增长（warmup）
   - 之后：开始 cosine decay

2. **Grad norm**：
   - 稳定在合理范围（< 100,000）
   - 不再出现巨大尖峰

3. **Loss**：
   - 平稳下降，不会过早收敛
   - 或收敛后保持稳定

## 🔧 验证方法

重新训练后，检查 WandB 图表：

1. **Learning rate 曲线**：
   - ✅ 前 1000 步线性增长
   - ✅ 之后开始下降（cosine decay）

2. **Grad norm 曲线**：
   - ✅ 稳定在 < 100,000
   - ✅ 不再出现 > 1,000,000 的尖峰

3. **Loss 曲线**：
   - ✅ 平稳下降或稳定在低值

---

**更新日期**: 2025-12-15


