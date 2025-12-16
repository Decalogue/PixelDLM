# WandB 训练监控指南

## 🎯 概述

训练脚本已集成 WandB (Weights & Biases) 用于实时监控训练过程，包括 loss、learning rate、gradient norm 等关键指标。

---

## ✅ 功能特性

### 1. 自动记录指标

训练过程中自动记录以下指标：

- **`train/loss`**: 每个优化步骤的训练损失
- **`train/learning_rate`**: 当前学习率
- **`train/grad_norm`**: 梯度范数（用于监控梯度爆炸/消失）
- **`train/epoch_loss`**: 每个 epoch 的平均损失
- **`train/epoch_time`**: 每个 epoch 的训练时间
- **`train/samples_per_second`**: 训练吞吐量
- **`train/best_loss`**: 最佳损失值

### 2. 配置记录

自动记录训练配置：
- 模型名称和图像尺寸
- 批次大小和梯度累积步数
- 学习率和优化器设置
- 混合精度训练设置
- 其他超参数

---

## 🚀 使用方法

### 方法 1: 使用 train.sh（推荐）

```bash
# 编辑 train.sh，设置 WandB 参数
USE_WANDB=true
WANDB_PROJECT="jit-diffusion"
WANDB_NAME="jit-v1"
WANDB_ENTITY="your-username"  # 可选，留空则使用 wandb login 的默认设置

# 运行训练
./train.sh
```

**设置用户名的方法**：
1. **在 train.sh 中设置**（推荐）：
   ```bash
   WANDB_ENTITY="your-username"
   ```

2. **使用环境变量**：
   ```bash
   export WANDB_ENTITY="your-username"
   ./train.sh
   ```

3. **使用 wandb login**（如果不设置 WANDB_ENTITY，会使用登录时的默认设置）：
   ```bash
   wandb login
   ```

### 方法 2: 直接使用 Python

```bash
python train.py \
    --data_path ./data/train \
    --use_wandb \
    --wandb_project "jit-diffusion" \
    --wandb_name "jit-v1" \
    --wandb_entity "your-username" \  # 可选
    --batch_size 32 \
    --epochs 100
```

---

## 📊 查看监控结果

### 1. 在线查看（推荐）

训练开始后，WandB 会自动生成一个 URL，例如：
```
https://wandb.ai/your-username/jit-diffusion/runs/xxx
```

在终端中会显示类似信息：
```
Wandb initialized: project=jit-diffusion, name=jit-v1
View run at: https://wandb.ai/...
```

### 2. 本地查看

如果需要在本地查看（需要安装 wandb）：

```bash
# 安装 wandb
pip install wandb

# 登录（首次使用需要）
wandb login

# 训练时会自动同步到云端
```

---

## 🔧 配置选项

### train.sh 中的配置

```bash
# 启用/禁用 WandB
USE_WANDB=true  # 或 false

# 项目名称
WANDB_PROJECT="jit-diffusion"

# 运行名称（可选，默认自动生成）
WANDB_NAME="jit-v1"

# WandB 用户名/实体（可选，留空则使用 wandb login 的默认设置）
WANDB_ENTITY="your-username"  # 或留空使用默认
```

### Python 参数

```bash
--use_wandb              # 启用 WandB
--wandb_project NAME     # 项目名称
--wandb_name NAME        # 运行名称（可选）
--wandb_entity NAME      # WandB 用户名/实体（可选，留空则使用默认）
```

---

## 📈 监控指标说明

### 1. train/loss

- **含义**: 每个优化步骤的训练损失（MSE loss）
- **频率**: 每个梯度累积步骤后记录
- **用途**: 监控训练是否正常收敛

**正常情况**:
- 损失应该逐渐下降
- 不应该出现突然的跳跃或 NaN

### 2. train/learning_rate

- **含义**: 当前学习率（考虑 warmup 和 cosine decay）
- **频率**: 每个优化步骤后记录
- **用途**: 确认学习率调度器正常工作

**正常情况**:
- Warmup 阶段：从 0 逐渐增加到目标学习率
- 训练阶段：按 cosine 衰减

### 3. train/grad_norm

- **含义**: 所有参数的梯度 L2 范数
- **频率**: 每个优化步骤后记录
- **用途**: 监控梯度爆炸/消失

**正常情况**:
- 梯度范数应该在合理范围内（例如 0.1 - 10）
- 如果梯度范数过大（>100），可能存在梯度爆炸
- 如果梯度范数过小（<0.001），可能存在梯度消失

### 4. train/epoch_loss

- **含义**: 每个 epoch 的平均损失
- **频率**: 每个 epoch 结束后记录
- **用途**: 观察整体训练趋势

### 5. train/samples_per_second

- **含义**: 训练吞吐量（样本/秒）
- **频率**: 每个 epoch 结束后记录
- **用途**: 评估训练效率

---

## 🛠️ 故障排除

### 问题 1: WandB 未安装

**错误信息**:
```
Warning: wandb not available. Install with: pip install wandb
```

**解决方法**:
```bash
pip install wandb
wandb login  # 首次使用需要登录
```

### 问题 2: 无法连接到 WandB 服务器

**解决方法**:
- 检查网络连接
- 使用 `wandb offline` 模式（数据会保存在本地，稍后同步）

```bash
export WANDB_MODE=offline
./train.sh
```

### 问题 3: 不想使用 WandB

**解决方法**:
- 在 `train.sh` 中设置 `USE_WANDB=false`
- 或运行 Python 时不添加 `--use_wandb` 参数

---

## 📝 最佳实践

1. **使用有意义的运行名称**
   ```bash
   WANDB_NAME="jit-v1-bs32-lr1e4"
   ```

2. **为不同实验使用不同项目**
   ```bash
   WANDB_PROJECT="jit-diffusion-64x64"
   WANDB_PROJECT="jit-diffusion-256x256"
   ```

3. **定期检查监控指标**
   - 训练开始后立即检查 loss 是否正常下降
   - 监控 gradient norm 是否在合理范围
   - 确认 learning rate 调度正常

4. **保存重要运行**
   - 在 WandB 界面中标记重要运行
   - 添加标签和注释便于后续查找

---

## 🎯 示例

### 完整训练命令

```bash
# 使用 train.sh（推荐）
./train.sh

# 或直接使用 Python
python train.py \
    --data_path ./data/train \
    --model JiT-B/4 \
    --img_size 64 \
    --batch_size 32 \
    --epochs 100 \
    --use_wandb \
    --wandb_project "jit-diffusion" \
    --wandb_name "jit-v1-bs32"
```

### 查看结果

训练开始后，在终端中会看到：
```
Wandb initialized: project=jit-diffusion, name=jit-v1-bs32
View run at: https://wandb.ai/your-username/jit-diffusion/runs/xxx
```

打开该 URL 即可实时查看训练进度和指标。

---

**修改日期**: 2024-12-15
**版本**: v1.0




