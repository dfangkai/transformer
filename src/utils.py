"""
工具函数模块

包含可视化、辅助函数等
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Optional


def visualize_positional_encoding(
    d_model: int,
    max_len: int = 100,
    save_path: str = "results/figures/positional_encoding.png"
) -> None:
    """
    可视化位置编码矩阵

    生成位置编码的热力图，展示不同位置和维度的编码值。
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建位置编码矩阵
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 创建热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pe.numpy(),
        cmap='RdBu',
        center=0,
        xticklabels=range(0, d_model, d_model // 10) if d_model > 10 else range(d_model),
        yticklabels=range(0, max_len, max_len // 10) if max_len > 10 else range(max_len),
        cbar_kws={'label': 'Encoding Value'}
    )

    plt.title(f'Positional Encoding Heatmap (d_model={d_model}, max_len={max_len})')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"位置编码可视化已保存到: {save_path}")
    plt.close()


def visualize_attention(
    attention_weights: torch.Tensor,
    src_tokens: Optional[List[str]] = None,
    tgt_tokens: Optional[List[str]] = None,
    save_path: str = "results/figures/attention_weights.png",
    head_idx: int = 0
) -> None:
    """
    可视化注意力权重

    生成注意力权重的热力图，展示源序列和目标序列之间的注意力分布。
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 选择特定batch和head的注意力权重
    if attention_weights.dim() == 4:
        # [batch, num_heads, seq_len_q, seq_len_k]
        attn = attention_weights[0, head_idx].detach().cpu().numpy()
    elif attention_weights.dim() == 3:
        # [num_heads, seq_len_q, seq_len_k]
        attn = attention_weights[head_idx].detach().cpu().numpy()
    elif attention_weights.dim() == 2:
        # [seq_len_q, seq_len_k]
        attn = attention_weights.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected attention_weights shape: {attention_weights.shape}")

    seq_len_q, seq_len_k = attn.shape

    # 创建热力图
    plt.figure(figsize=(10, 8))

    # 如果提供了tokens，使用它们作为标签
    if tgt_tokens and len(tgt_tokens) == seq_len_q:
        yticklabels = tgt_tokens
    else:
        yticklabels = [f"Q{i}" for i in range(seq_len_q)]

    if src_tokens and len(src_tokens) == seq_len_k:
        xticklabels = src_tokens
    else:
        xticklabels = [f"K{i}" for i in range(seq_len_k)]

    sns.heatmap(
        attn,
        cmap='viridis',
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={'label': 'Attention Weight'},
        annot=seq_len_q <= 10 and seq_len_k <= 10,  # 只在小矩阵时显示数值
        fmt='.3f'
    )

    plt.title(f'Attention Weights (Head {head_idx})')
    plt.xlabel('Key (Source)')
    plt.ylabel('Query (Target)')
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"注意力权重可视化已保存到: {save_path}")
    plt.close()


# ========== Mask生成工具函数 ==========

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建padding mask
    """
    # seq != pad_idx 会产生 [batch, seq_len] 的布尔张量
    # unsqueeze两次得到 [batch, 1, 1, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_future_mask(size: int) -> torch.Tensor:
    """
    创建future mask（下三角矩阵，用于防止看到未来信息）
    """
    # torch.triu创建上三角矩阵，diagonal=1表示对角线上方
    # 我们需要下三角，所以取反
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # ~mask 得到下三角矩阵（包括对角线）
    # unsqueeze两次得到 [1, 1, size, size]
    return (~mask).unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建目标序列的mask（同时考虑padding和future）

    在Decoder中，目标序列需要两种mask：
    1. Padding mask：屏蔽padding位置
    2. Future mask：防止看到未来的token
    """
    tgt_len = tgt.size(1)

    # 1. 创建padding mask用于key positions (列): [batch, 1, 1, tgt_len]
    tgt_pad_mask_col = create_padding_mask(tgt, pad_idx)

    # 2. 创建padding mask用于query positions (行): [batch, 1, tgt_len, 1]
    tgt_pad_mask_row = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)

    # 3. 创建future mask: [1, 1, tgt_len, tgt_len]
    # 确保在与tgt相同的设备上
    tgt_future_mask = create_future_mask(tgt_len).to(tgt.device)

    # 4. 组合三个mask（逻辑与）
    # padding列mask广播到 [batch, 1, tgt_len, tgt_len]
    # padding行mask广播到 [batch, 1, tgt_len, tgt_len]
    # future mask广播到 [batch, 1, tgt_len, tgt_len]
    return tgt_pad_mask_col & tgt_pad_mask_row & tgt_future_mask


# ========== 模型统计工具函数 ==========

def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型的可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module) -> None:
    """
    打印模型的详细信息
    """
    total_params = count_parameters(model)
    print("=" * 80)
    print(f"模型信息")
    print("=" * 80)
    print(f"总参数数量: {total_params:,}")
    print(f"参数大小 (MB): {total_params * 4 / (1024 ** 2):.2f}")  # 假设float32
    print("\n各层参数数量:")
    print("-" * 80)

    # 统计各个子模块的参数
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:30s} | {num_params:>15,} 参数")

    print("=" * 80)

    # 打印模型结构摘要
    print("\n模型结构摘要:")
    print("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60s} | shape: {str(tuple(param.shape)):20s} | {param.numel():>10,}")
    print("=" * 80)


# ========== 训练辅助工具函数 ==========

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: str = "results/figures/training_curve.png"
) -> None:
    """
    绘制训练曲线
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 设置x轴为整数
    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()
