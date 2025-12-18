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
LR=1e-5
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=1
OUTPUT_DIR="./run/jit_v1_test"
USE_AMP=true
GRADIENT_ACCUMULATION_STEPS=2
MAX_GRAD_NORM=1  # 降低梯度裁剪阈值，防止梯度爆炸
SAVE_INTERVAL=1
LOG_INTERVAL=1
USE_WANDB=true 
WANDB_PROJECT="jit-diffusion-test"
WANDB_NAME="test-run"
WANDB_ENTITY="decalogue"

# 像素解码器和频率损失参数（新架构改进）
# 建议：先测试频率感知损失（计算开销小），再测试 U-Net 解码器（效果显著）
USE_PIXEL_DECODER=true  # 是否使用 U-Net 像素解码器（效果显著，但计算开销 +30%）
PIXEL_DECODER_DEPTH=3  # U-Net 深度
USE_FREQ_LOSS=true  # 是否使用频率感知损失（推荐先启用，计算开销小 +5%，通用性强）
FREQ_LOSS_QUALITY=75  # JPEG 质量 (1-100)，仅影响损失计算权重，不影响 token-color 映射
FREQ_LOSS_WEIGHT=0.5  # 频率损失权重
MSE_LOSS_WEIGHT=0.5  # MSE 损失权重

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

# 如果启用像素解码器，添加参数
if [ "$USE_PIXEL_DECODER" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_pixel_decoder"
    if [ -n "$PIXEL_DECODER_DEPTH" ]; then
        TRAIN_CMD="${TRAIN_CMD} --pixel_decoder_depth ${PIXEL_DECODER_DEPTH}"
    fi
fi

# 如果启用频率感知损失，添加参数
if [ "$USE_FREQ_LOSS" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_freq_loss"
    if [ -n "$FREQ_LOSS_QUALITY" ]; then
        TRAIN_CMD="${TRAIN_CMD} --freq_loss_quality ${FREQ_LOSS_QUALITY}"
    fi
    if [ -n "$FREQ_LOSS_WEIGHT" ]; then
        TRAIN_CMD="${TRAIN_CMD} --freq_loss_weight ${FREQ_LOSS_WEIGHT}"
    fi
    if [ -n "$MSE_LOSS_WEIGHT" ]; then
        TRAIN_CMD="${TRAIN_CMD} --mse_loss_weight ${MSE_LOSS_WEIGHT}"
    fi
fi

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
echo "像素解码器: ${USE_PIXEL_DECODER}"
if [ "$USE_PIXEL_DECODER" = true ]; then
    echo "  U-Net 深度: ${PIXEL_DECODER_DEPTH}"
fi
echo "频率感知损失: ${USE_FREQ_LOSS}"
if [ "$USE_FREQ_LOSS" = true ]; then
    echo "  JPEG 质量: ${FREQ_LOSS_QUALITY}"
    echo "  频率损失权重: ${FREQ_LOSS_WEIGHT}"
    echo "  MSE 损失权重: ${MSE_LOSS_WEIGHT}"
fi
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
    echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
fi
echo "=========================================="
