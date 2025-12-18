#!/bin/bash
# 条件生成推理脚本
# 使用: ./infer_ft.sh

set -e

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seeme

# 推理参数
export CUDA_VISIBLE_DEVICES=1

CHECKPOINT="./run/jit_v1_ft/best_model.pth"  # 微调后的模型
MODEL="JiT-B/4"
IMG_SIZE=64
COND_IMG_SIZE=64
TOKENIZER_PATH="/root/data/AI/pretrain/Qwen2.5-7B-Instruct"
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=1.0
OUTPUT_DIR="./output_ft"
SAVE_IMAGE=true

# 像素解码器参数（需要与训练时一致）
USE_PIXEL_DECODER=true  # 是否使用 U-Net 像素解码器
PIXEL_DECODER_DEPTH=3  # U-Net 深度

# 可选：指定 prompt 或 prompt 文件
# PROMPT="什么是人工智能？"
# PROMPT_FILE="./data/test_prompts.txt"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 构建推理命令
INFER_CMD="python inference_ft.py \
    --checkpoint ${CHECKPOINT} \
    --model ${MODEL} \
    --img_size ${IMG_SIZE} \
    --cond_img_size ${COND_IMG_SIZE} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --output_dir ${OUTPUT_DIR}"

# 如果启用像素解码器，添加参数
if [ "$USE_PIXEL_DECODER" = true ]; then
    INFER_CMD="${INFER_CMD} --use_pixel_decoder --pixel_decoder_depth ${PIXEL_DECODER_DEPTH}"
fi

# 如果指定了 prompt，添加参数
if [ -n "${PROMPT}" ]; then
    INFER_CMD="${INFER_CMD} --prompt \"${PROMPT}\""
fi

# 如果指定了 prompt 文件，添加参数
if [ -n "${PROMPT_FILE}" ]; then
    INFER_CMD="${INFER_CMD} --prompt_file ${PROMPT_FILE}"
fi

# 如果启用保存图像，添加参数
if [ "$SAVE_IMAGE" = true ]; then
    INFER_CMD="${INFER_CMD} --save_image"
fi

# 打印配置
echo "=========================================="
echo "条件生成推理配置"
echo "=========================================="
echo "检查点: ${CHECKPOINT}"
echo "模型: ${MODEL}"
echo "答案图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
echo "条件图像尺寸: ${COND_IMG_SIZE}×${COND_IMG_SIZE}"
echo "推理步数: ${NUM_INFERENCE_STEPS}"
echo "Guidance Scale: ${GUIDANCE_SCALE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "保存图像: ${SAVE_IMAGE}"
echo "使用像素解码器: ${USE_PIXEL_DECODER}"
if [ "$USE_PIXEL_DECODER" = true ]; then
    echo "像素解码器深度: ${PIXEL_DECODER_DEPTH}"
fi
if [ -n "${PROMPT}" ]; then
    echo "Prompt: ${PROMPT}"
elif [ -n "${PROMPT_FILE}" ]; then
    echo "Prompt 文件: ${PROMPT_FILE}"
else
    echo "Prompt: 使用默认测试 prompts"
fi
echo "=========================================="
echo

# 检查模型是否存在
if [ ! -f "${CHECKPOINT}" ]; then
    echo "警告: 模型检查点不存在: ${CHECKPOINT}"
    echo "请先运行微调训练或检查模型路径"
    echo "是否继续使用默认 prompts 进行测试？(y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# 运行推理
eval ${INFER_CMD}

echo ""
echo "=========================================="
echo "条件生成推理完成！"
echo "结果已保存到: ${OUTPUT_DIR}"
echo "=========================================="
