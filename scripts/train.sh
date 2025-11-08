#!/bin/bash

# Transformer 对话摘要训练脚本
# 使用方法:
#   bash scripts/train.sh              # 自动检测设备 (MPS/CUDA/CPU)
#   bash scripts/train.sh mps          # 强制使用 MPS
#   bash scripts/train.sh cuda         # 强制使用 CUDA
#   bash scripts/train.sh cpu          # 强制使用 CPU
#   bash scripts/train.sh --quick      # 快速测试 (10 epochs)

echo "=========================================="
echo "Transformer 对话摘要训练"
echo "=========================================="
echo ""

# 设置项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 找不到Python"
    exit 1
fi

echo "Python版本: $(python --version)"
echo ""

# 解析参数
DEVICE_ARG=""
EPOCHS_ARG=""

for arg in "$@"; do
    case $arg in
        mps|cuda|cpu)
            DEVICE_ARG="--device $arg"
            echo "设备: $arg (手动指定)"
            ;;
        --quick)
            EPOCHS_ARG="--num_epochs 10"
            echo "模式: 快速测试 (10 epochs)"
            ;;
    esac
done

if [ -z "$DEVICE_ARG" ]; then
    echo "设备: 自动检测"
fi

echo "配置文件: configs/summarization_config.yaml"
echo ""
echo "开始训练..."
echo ""

# 执行训练
python src/train.py \
    --config configs/summarization_config.yaml \
    --seed 42 \
    $DEVICE_ARG \
    $EPOCHS_ARG

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果保存在: results/summarization/"
echo "  - best_model.pt           最佳模型checkpoint"
echo "  - training_history.json   训练历史数据"
echo "  - training_curve.png      训练曲线图"
echo "  - summary_samples.txt     展示样本(10个)"
echo "  - rouge_scores.json       ROUGE评估结果(完整测试集819样本)"
echo "  - config_used.yaml        训练配置备份"
echo "  - training_summary.txt    训练总结报告"
echo ""
