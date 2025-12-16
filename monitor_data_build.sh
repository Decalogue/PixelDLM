#!/bin/bash
# 监控数据构建进程的脚本

echo "=========================================="
echo "数据构建进程监控"
echo "=========================================="
echo

# 检查进程状态
echo "1. 进程状态:"
if ps aux | grep -v grep | grep "build_pretrain_data.py" > /dev/null; then
    PID=$(ps aux | grep -v grep | grep "build_pretrain_data.py" | awk '{print $2}')
    CPU=$(ps aux | grep -v grep | grep "build_pretrain_data.py" | awk '{print $3}')
    MEM=$(ps aux | grep -v grep | grep "build_pretrain_data.py" | awk '{print $4}')
    echo "  ✅ 进程运行中 (PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%)"
else
    echo "  ❌ 进程未运行"
fi
echo

# 查看最新日志
echo "2. 最新日志 (最后20行):"
echo "----------------------------------------"
tail -20 build_pretrain_data.log 2>/dev/null || echo "  日志文件不存在"
echo

# 检查数据目录
echo "3. 数据目录状态:"
if [ -d "data/train" ]; then
    TRAIN_COUNT=$(find data/train -name "*.json" 2>/dev/null | wc -l)
    TRAIN_SIZE=$(du -sh data/train 2>/dev/null | awk '{print $1}')
    echo "  训练集: $TRAIN_COUNT 个文件, 总大小: $TRAIN_SIZE"
else
    echo "  训练集: 目录不存在"
fi

if [ -d "data/val" ]; then
    VAL_COUNT=$(find data/val -name "*.json" 2>/dev/null | wc -l)
    VAL_SIZE=$(du -sh data/val 2>/dev/null | awk '{print $1}')
    echo "  验证集: $VAL_COUNT 个文件, 总大小: $VAL_SIZE"
else
    echo "  验证集: 目录不存在"
fi
echo

# 检查统计信息
if [ -f "data/stats.json" ]; then
    echo "4. 统计信息:"
    echo "----------------------------------------"
    cat data/stats.json | python -m json.tool 2>/dev/null || cat data/stats.json
    echo
else
    echo "4. 统计信息: 尚未生成"
    echo
fi

echo "=========================================="
echo "提示:"
echo "  - 实时查看日志: tail -f build_pretrain_data.log"
echo "  - 终止进程: pkill -f build_pretrain_data.py"
echo "=========================================="
