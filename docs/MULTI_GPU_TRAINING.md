# 多卡训练指南

## 🎯 概述

训练脚本已支持分布式训练（DDP - Distributed Data Parallel），可以在多张 GPU 上并行训练，加速训练过程。

---

## ✅ 支持情况

### 1. 代码层面

✅ **已支持分布式训练**：
- 使用 PyTorch DDP (DistributedDataParallel)
- 自动检测环境变量（RANK, WORLD_SIZE, LOCAL_RANK）
- 使用 DistributedSampler 分配数据
- 支持梯度同步和模型同步

### 2. 启动方式

- **单卡训练**: 使用 `train.sh`
- **多卡训练**: 使用 `train_multi_gpu.sh` 或 `torchrun`

---

## 🚀 使用方法

### 方法 1: 使用 train_multi_gpu.sh（推荐）

```bash
# 使用 2 张 GPU
./train_multi_gpu.sh 2

# 使用 4 张 GPU
./train_multi_gpu.sh 4
```

### 方法 2: 直接使用 torchrun

```bash
# 使用 2 张 GPU
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --data_path ./data/train \
    --batch_size 64 \
    --epochs 100 \
    --use_amp \
    --use_wandb
```

### 方法 3: 使用 torch.distributed.launch（旧方式）

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --data_path ./data/train \
    --batch_size 64 \
    --epochs 100
```

---

## 📊 批次大小说明

### 重要概念

- **`--batch_size`**: 每张 GPU 的批次大小
- **有效批次大小** = `batch_size × GPU数量 × gradient_accumulation_steps`

### 示例

假设：
- `batch_size = 64`（每张 GPU）
- `num_gpus = 2`
- `gradient_accumulation_steps = 1`

则：
- **有效批次大小** = 64 × 2 × 1 = **128**

### 调整建议

如果使用多卡训练，通常需要：
1. **保持有效批次大小不变**：减少每张 GPU 的 `batch_size`
2. **或增加有效批次大小**：保持每张 GPU 的 `batch_size` 不变

**示例**：
- 单卡：`batch_size=128`，有效批次 = 128
- 双卡：`batch_size=64`，有效批次 = 128（保持不变）
- 双卡：`batch_size=128`，有效批次 = 256（增加）

---

## 🔧 配置说明

### train_multi_gpu.sh 配置

```bash
# GPU 数量（通过命令行参数传入）
NUM_GPUS=${1:-2}  # 默认 2 张 GPU

# 每张 GPU 的批次大小
BATCH_SIZE=64

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS=1

# 计算总有效批次大小
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
```

### 环境变量

`torchrun` 会自动设置以下环境变量：
- `RANK`: 全局进程排名（0 到 world_size-1）
- `LOCAL_RANK`: 本地 GPU 排名（0 到 nproc_per_node-1）
- `WORLD_SIZE`: 总进程数（等于 GPU 数量）
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 主节点端口

---

## 📈 性能优化

### 1. 批次大小优化

**4090 24GB 单卡建议**：
- `batch_size = 64-128`（根据显存使用情况调整）

**4090 24GB 双卡建议**：
- 每张 GPU: `batch_size = 64`
- 总有效批次: `128`

### 2. 学习率调整

多卡训练时，通常需要**线性缩放学习率**：

```python
# 单卡学习率
lr_single = 5e-5

# 双卡学习率（线性缩放）
lr_multi = lr_single * num_gpus  # 5e-5 * 2 = 1e-4
```

**注意**：当前脚本使用固定学习率，多卡训练时可能需要手动调整。

### 3. 数据加载优化

多卡训练时，每个进程会：
- 使用 `DistributedSampler` 自动分配数据
- 每个 epoch 的数据顺序会不同（通过 `sampler.set_epoch(epoch)` 设置）

---

## 🛠️ 故障排除

### 问题 1: NCCL 初始化失败

**错误信息**:
```
NCCL error: unhandled system error
```

**解决方法**:
1. 检查 GPU 是否可见：
   ```bash
   nvidia-smi
   ```

2. 设置 NCCL 调试：
   ```bash
   export NCCL_DEBUG=INFO
   ```

3. 尝试不同的端口：
   ```bash
   torchrun --master_port=29501 ...
   ```

### 问题 2: 显存不足

**解决方法**:
1. 减少每张 GPU 的 `batch_size`
2. 增加 `gradient_accumulation_steps`
3. 确保 `USE_AMP=true`（混合精度训练）

### 问题 3: 训练速度没有提升

**可能原因**:
1. 数据加载成为瓶颈（增加 `num_workers`）
2. 模型太小，通信开销大于计算收益
3. 批次大小太小

**解决方法**:
1. 增加 `num_workers`（在 `train.py` 中）
2. 使用更大的 `batch_size`
3. 确保数据加载足够快

---

## 📝 示例

### 双卡训练示例

```bash
# 使用 train_multi_gpu.sh
./train_multi_gpu.sh 2

# 或直接使用 torchrun
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --data_path ./data/train \
    --model JiT-B/4 \
    --img_size 64 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --use_amp \
    --use_wandb \
    --wandb_project "jit-diffusion" \
    --wandb_name "jit-v1-2gpu"
```

### 四卡训练示例

```bash
./train_multi_gpu.sh 4
```

---

## ✅ 检查清单

- [x] 代码支持 DDP
- [x] 使用 DistributedSampler
- [x] 支持 torchrun 启动
- [x] 提供多卡训练脚本
- [x] 支持 WandB 监控（只在 rank 0 记录）

---

## 🎯 总结

✅ **训练脚本已完全支持多卡训练**

✅ **使用 `train_multi_gpu.sh` 可以轻松启动多卡训练**

✅ **自动处理数据分配和梯度同步**

---

**修改日期**: 2024-12-15
**版本**: v1.0




