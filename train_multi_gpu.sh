#!/bin/bash
# 多卡训练脚本
# 使用: ./train_multi_gpu.sh [num_gpus]
# 例如: ./train_multi_gpu.sh 2  # 使用 2 张 GPU

set -e

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seeme

# 获取 GPU 数量（默认为 2）
NUM_GPUS=${1:-2}

# 训练参数
MODEL="JiT-B/4"
IMG_SIZE=64
DATA_PATH="./data/train"
TOKENIZER_PATH="/root/data/AI/pretrain/Qwen2.5-7B-Instruct"
BATCH_SIZE=32  # 每张 GPU 的 batch size
EPOCHS=100
LR=5e-6
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=5
OUTPUT_DIR="./run/jit_v1"
USE_AMP=true
GRADIENT_ACCUMULATION_STEPS=1
MAX_GRAD_NORM=0.5
SAVE_INTERVAL=10
LOG_INTERVAL=10
USE_WANDB=true
WANDB_PROJECT="jit-diffusion"
WANDB_NAME="jit-v1-multi-gpu"
WANDB_ENTITY="decalogue"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 计算有效批次大小
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))

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
echo "开始多卡训练"
echo "=========================================="
echo "GPU 数量: ${NUM_GPUS}"
echo "模型: ${MODEL}"
echo "图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
echo "数据路径: ${DATA_PATH}"
echo "每张 GPU 批次大小: ${BATCH_SIZE}"
echo "梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "总有效批次大小: ${EFFECTIVE_BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "混合精度: ${USE_AMP}"
echo "Wandb 监控: ${USE_WANDB}"
if [ "$USE_WANDB" = true ]; then
    echo "Wandb 项目: ${WANDB_PROJECT}"
    echo "Wandb 运行名: ${WANDB_NAME}"
    if [ -n "$WANDB_ENTITY" ]; then
        echo "Wandb 用户名: ${WANDB_ENTITY}"
    fi
fi
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo

# 使用 torchrun 启动多卡训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train.py \
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
    --wandb_name ${WANDB_NAME} \
    --wandb_entity ${WANDB_ENTITY} \
    $([ "$USE_AMP" = true ] && echo "--use_amp") \
    $([ "$USE_WANDB" = true ] && echo "--use_wandb")

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
