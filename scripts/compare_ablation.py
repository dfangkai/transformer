"""
消融实验结果对比工具

读取所有消融实验的结果，生成对比表格和可视化图表。
"""

import os
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_experiment_results(exp_name: str):
    """
    加载单个实验的结果
    """
    model_path = f"results/ablation/{exp_name}/best_model.pt"
    rouge_path = f"results/ablation/{exp_name}/rouge_scores.json"

    if not os.path.exists(model_path):
        print(f"  未找到实验结果: {exp_name}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        result = {
            'experiment': exp_name,
            'train_loss': checkpoint.get('train_loss', float('nan')),
            'val_loss': checkpoint.get('val_loss', float('nan')),
            'epoch': checkpoint.get('epoch', 0),
            'num_params': sum(p.numel() for p in checkpoint['model_state_dict'].values() if isinstance(p, torch.Tensor))
        }

        # 加载ROUGE分数（如果存在）
        if os.path.exists(rouge_path):
            with open(rouge_path, 'r') as f:
                rouge_scores = json.load(f)
                result['rouge1'] = rouge_scores['rouge1']['mean']
                result['rouge2'] = rouge_scores['rouge2']['mean']
                result['rougeL'] = rouge_scores['rougeL']['mean']
        else:
            result['rouge1'] = float('nan')
            result['rouge2'] = float('nan')
            result['rougeL'] = float('nan')

        return result
    except Exception as e:
        print(f" 加载 {exp_name} 失败: {e}")
        return None


def compare_experiments():
    """
    对比所有消融实验的结果
    """
    print("=" * 80)
    print("消融实验结果对比")
    print("=" * 80)
    print()

    # 实验列表
    experiments = [
        ("baseline", "完整模型（对照组）"),
        ("no_positional", "去除位置编码"),
        ("no_gradient_accumulation", "去除梯度累积")
    ]

    # 收集结果
    results = []
    for exp_name, exp_desc in experiments:
        result = load_experiment_results(exp_name)
        if result:
            result['description'] = exp_desc
            results.append(result)

    if not results:
        print(" 没有找到任何实验结果")
        print("\n请先运行消融实验:")
        print("  bash scripts/run_ablation.sh")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 重新排列列顺序
    columns_order = ['experiment', 'description', 'val_loss', 'train_loss', 'rouge1', 'rouge2', 'rougeL', 'epoch', 'num_params']
    df = df[columns_order]

    # 按验证loss排序
    df = df.sort_values('val_loss')

    # 打印表格
    print("\n 实验结果对比表")
    print("-" * 120)
    print(f"{'实验名称':<30} {'验证Loss':>10} {'训练Loss':>10} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9} {'Epoch':>7} {'参数量':>12}")
    print("-" * 120)

    for _, row in df.iterrows():
        rouge1_str = f"{row['rouge1']:.4f}" if not pd.isna(row['rouge1']) else "N/A"
        rouge2_str = f"{row['rouge2']:.4f}" if not pd.isna(row['rouge2']) else "N/A"
        rougeL_str = f"{row['rougeL']:.4f}" if not pd.isna(row['rougeL']) else "N/A"

        print(f"{row['description']:<30} {row['val_loss']:>10.4f} {row['train_loss']:>10.4f} "
              f"{rouge1_str:>9} {rouge2_str:>9} {rougeL_str:>9} "
              f"{row['epoch']:>7} {row['num_params']:>12,}")

    print("-" * 120)

    # 找出最佳配置
    best_idx = df['val_loss'].idxmin()
    best_exp = df.loc[best_idx]

    print(f"\n 最佳配置: {best_exp['description']}")
    print(f"   验证Loss: {best_exp['val_loss']:.4f}")
    print(f"   训练Loss: {best_exp['train_loss']:.4f}")
    if not pd.isna(best_exp['rouge1']):
        print(f"   ROUGE-1:  {best_exp['rouge1']:.4f}")
        print(f"   ROUGE-2:  {best_exp['rouge2']:.4f}")
        print(f"   ROUGE-L:  {best_exp['rougeL']:.4f}")

    # 计算各配置相对baseline的性能变化
    if 'baseline' in df['experiment'].values:
        baseline = df[df['experiment'] == 'baseline'].iloc[0]
        baseline_loss = baseline['val_loss']

        print(f"\n 相对Baseline的性能变化:")
        print("-" * 120)
        print(f"{'实验名称':<30} {'验证Loss变化':>15} {'ROUGE-1变化':>15} {'ROUGE-2变化':>15} {'ROUGE-L变化':>15}")
        print("-" * 120)

        for _, row in df.iterrows():
            if row['experiment'] != 'baseline':
                # Loss变化（越小越好）
                loss_delta = row['val_loss'] - baseline_loss
                loss_delta_pct = (loss_delta / baseline_loss) * 100
                loss_symbol = "📉" if loss_delta < 0 else "📈"
                loss_str = f"{loss_delta:+.4f} ({loss_delta_pct:+.2f}%)"

                # ROUGE变化（越大越好）
                rouge1_str = rouge2_str = rougeL_str = "N/A"
                if not pd.isna(row['rouge1']) and not pd.isna(baseline['rouge1']):
                    r1_delta = row['rouge1'] - baseline['rouge1']
                    r1_delta_pct = (r1_delta / baseline['rouge1']) * 100
                    rouge1_str = f"{r1_delta:+.4f} ({r1_delta_pct:+.2f}%)"

                    r2_delta = row['rouge2'] - baseline['rouge2']
                    r2_delta_pct = (r2_delta / baseline['rouge2']) * 100
                    rouge2_str = f"{r2_delta:+.4f} ({r2_delta_pct:+.2f}%)"

                    rL_delta = row['rougeL'] - baseline['rougeL']
                    rL_delta_pct = (rL_delta / baseline['rougeL']) * 100
                    rougeL_str = f"{rL_delta:+.4f} ({rL_delta_pct:+.2f}%)"

                print(f"{loss_symbol} {row['description']:<28} {loss_str:>15} {rouge1_str:>15} {rouge2_str:>15} {rougeL_str:>15}")

    # 生成可视化图表
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)

    # 创建保存目录
    os.makedirs('results/ablation', exist_ok=True)

    # 1. 验证Loss对比柱状图
    plt.figure(figsize=(12, 6))

    colors = ['#2ecc71' if exp == 'baseline' else '#3498db' for exp in df['experiment']]
    bars = plt.bar(range(len(df)), df['val_loss'], color=colors, alpha=0.8)

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, df['val_loss'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Convert Chinese descriptions to English for x-axis labels
    label_mapping = {
        '完整模型（对照组）': 'Baseline\n(Full Model)',
        '去除位置编码': 'No Positional\nEncoding',
        '去除梯度累积': 'No Gradient\nAccumulation'
    }
    x_labels = [label_mapping.get(desc, desc) for desc in df['description']]

    plt.xticks(range(len(df)), x_labels, rotation=0, ha='center')
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Ablation Study - Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = 'results/ablation/comparison_val_loss.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" 保存验证Loss对比图: {save_path}")
    plt.close()

    # 2. 训练Loss vs 验证Loss散点图
    plt.figure(figsize=(10, 8))

    # Convert Chinese descriptions to English for legend
    for _, row in df.iterrows():
        color = '#2ecc71' if row['experiment'] == 'baseline' else '#3498db'
        marker = 'o' if row['experiment'] == 'baseline' else '^'
        size = 200 if row['experiment'] == 'baseline' else 150

        # Use English label
        eng_label = label_mapping.get(row['description'], row['description'])
        plt.scatter(row['train_loss'], row['val_loss'],
                   color=color, marker=marker, s=size, alpha=0.7,
                   label=eng_label.replace('\n', ' '))

    # 添加对角线
    min_loss = min(df['train_loss'].min(), df['val_loss'].min())
    max_loss = max(df['train_loss'].max(), df['val_loss'].max())
    plt.plot([min_loss, max_loss], [min_loss, max_loss], 'k--', alpha=0.3, label='y=x')

    plt.xlabel('Training Loss', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Ablation Study - Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = 'results/ablation/comparison_train_vs_val.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" 保存训练vs验证Loss图: {save_path}")
    plt.close()

    # 3. 保存CSV结果
    csv_path = 'results/ablation/comparison_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存结果CSV: {csv_path}")

    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)


if __name__ == "__main__":
    compare_experiments()
