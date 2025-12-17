#!/bin/bash
# 训练测试脚本（只运行 2 个 epoch 用于测试）
# 使用: ./train_test.sh

set -e

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seeme

# 训练参数（测试用，减少 epoch 和 batch size）
export CUDA_VISIBLE_DEVICES=1

MODEL="JiT-B/4"
IMG_SIZE=64
COND_IMG_SIZE=64
ENABLE_CONDITION=false
DATA_PATH="./data/train"
TOKENIZER_PATH="/root/data/AI/pretrain/Qwen2.5-7B-Instruct"
BATCH_SIZE=32
EPOCHS=1 # 只训练 1 个 epoch 用于测试
LR=5e-6
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=1
OUTPUT_DIR="./run/jit_v1_test"
USE_AMP=true
GRADIENT_ACCUMULATION_STEPS=2
MAX_GRAD_NORM=0.1  # 降低梯度裁剪阈值，防止梯度爆炸
SAVE_INTERVAL=1
LOG_INTERVAL=1
USE_WANDB=true 
WANDB_PROJECT="jit-diffusion-test"
WANDB_NAME="test-run"
WANDB_ENTITY="decalogue"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 构建训练命令
TRAIN_CMD="python train.py \
    --model ${MODEL} \
    --img_size ${IMG_SIZE} \
    --data_path ${DATA_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_NAME}"

# 如果启用混合精度，添加 --use_amp 参数
if [ "$USE_AMP" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_amp"
fi

# 如果启用 wandb，添加 --use_wandb 参数
if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_wandb"
    # 如果指定了 wandb entity，添加参数
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_CMD="${TRAIN_CMD} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

# 运行训练
echo "=========================================="
echo "开始训练测试（2 个 epoch）"
echo "=========================================="
echo "模型: ${MODEL}"
echo "图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
echo "数据路径: ${DATA_PATH}"
echo "批次大小: ${BATCH_SIZE}"
echo "梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "有效批次大小: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "混合精度: ${USE_AMP}"
echo "Wandb 监控: ${USE_WANDB}"
if [ "$USE_WANDB" = true ]; then
    echo "Wandb 项目: ${WANDB_PROJECT}"
    echo "Wandb 运行名: ${WANDB_NAME}"
    if [ -n "$WANDB_ENTITY" ]; then
        echo "Wandb 用户名: ${WANDB_ENTITY}"
    else
        echo "Wandb 用户名: 使用默认（wandb login 设置）"
    fi
fi
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo

eval ${TRAIN_CMD}

echo ""
echo "=========================================="
echo "训练测试完成！"
echo "=========================================="
if [ "$USE_WANDB" = true ]; then
    echo "查看 WandB 监控结果:"
    echo "  https://wandb.ai/decalogue/${WANDB_PROJECT}"
fi
echo "=========================================="
