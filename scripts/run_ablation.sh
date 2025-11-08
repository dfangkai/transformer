#!/bin/bash

# 消融实验运行脚本
# 依次运行所有消融实验，验证各组件的重要性
#
# 使用方法:
#   bash scripts/run_ablation.sh              # 运行所有实验
#   bash scripts/run_ablation.sh baseline     # 只运行指定实验
#   bash scripts/run_ablation.sh --device mps # 指定设备

echo "=========================================="
echo "Transformer 消融实验"
echo "=========================================="
echo ""
echo "本脚本将运行以下消融实验："
echo "  1. baseline                  - 完整模型（对照组）"
echo "  2. no_positional             - 去除位置编码"
echo "  3. no_gradient_accumulation  - 去除梯度累积"
echo ""

# 设置项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# 消融实验列表
EXPERIMENTS=(
    "baseline"
    "no_positional"
    "no_gradient_accumulation"
)

# 解析参数
DEVICE_ARG=""
SELECTED_EXP=""

for arg in "$@"; do
    case $arg in
        --device)
            shift
            DEVICE_ARG="--device $1"
            shift
            ;;
        mps|cuda|cpu)
            DEVICE_ARG="--device $arg"
            ;;
        baseline|no_positional|no_gradient_accumulation)
            SELECTED_EXP="$arg"
            ;;
    esac
done

# 如果指定了单个实验，只运行该实验
if [ -n "$SELECTED_EXP" ]; then
    EXPERIMENTS=("$SELECTED_EXP")
    echo "只运行实验: $SELECTED_EXP"
    echo ""
fi

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行实验
for exp in "${EXPERIMENTS[@]}"; do
    echo "=========================================="
    echo "开始实验: $exp"
    echo "=========================================="
    echo "配置文件: configs/ablation/${exp}.yaml"
    echo ""

    # 运行训练
    python src/train.py \
        --config "configs/ablation/${exp}.yaml" \
        --seed 42 \
        $DEVICE_ARG

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 实验 $exp 完成"
        echo ""
    else
        echo ""
        echo "❌ 实验 $exp 失败"
        echo ""
        exit 1
    fi
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo "所有消融实验完成！"
echo "=========================================="
echo "总耗时: ${HOURS}小时${MINUTES}分钟"
echo ""
echo "结果保存在: results/ablation/"
echo ""
echo "查看结果对比："
echo "  python scripts/compare_ablation.py"
echo ""
