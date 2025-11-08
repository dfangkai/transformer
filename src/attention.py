"""
注意力机制模块

实现缩放点积注意力（Scaled Dot-Product Attention）和多头注意力（Multi-Head Attention）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算缩放点积注意力
    """
    # 获取d_k维度（Q的最后一个维度）
    d_k = Q.size(-1)

    # 计算注意力分数: QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用mask（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 计算注意力权重（沿着最后一个维度进行softmax）
    attention_weights = F.softmax(scores, dim=-1)

    # 计算输出: attention_weights @ V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力层
        """
        super(MultiHeadAttention, self).__init__()

        # 确保d_model可以被num_heads整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义Q, K, V的线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 定义输出的线性变换层
        self.W_O = nn.Linear(d_model, d_model)

        # Dropout层
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入分割成多个头
        """
        batch_size, seq_len, d_model = x.size()

        # 重塑: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 转置: [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个头合并回原始维度
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # 转置: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)

        # 重塑: [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多头注意力的前向传播
        """
        batch_size = Q.size(0)

        # 1. 通过线性层进行变换
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 分割成多个头
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算缩放点积注意力
        # output: [batch, num_heads, seq_len_q, d_k]
        # attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 合并多个头
        # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, d_model]
        attn_output = self.combine_heads(attn_output)

        # 5. 通过输出线性层
        # [batch, seq_len_q, d_model] -> [batch, seq_len_q, d_model]
        output = self.W_O(attn_output)

        # 6. 应用dropout
        output = self.dropout(output)

        return output, attention_weights
