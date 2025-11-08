"""
Transformer基础层模块

包含前馈神经网络和残差连接等基础层。
"""

import torch
import torch.nn as nn
from typing import Callable


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (位置前馈神经网络)

    对序列中的每个位置独立地应用相同的两层全连接网络。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化Position-wise Feed-Forward Network
        """
        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # 第一个全连接层：扩展维度
        self.fc1 = nn.Linear(d_model, d_ff)

        # 第二个全连接层：恢复到原始维度
        self.fc2 = nn.Linear(d_ff, d_model)

        # Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 第一层全连接 + ReLU激活
        # [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        hidden = self.relu(self.fc1(x))

        # Dropout
        hidden = self.dropout(hidden)

        # 第二层全连接
        # [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        output = self.fc2(hidden)

        return output


class SublayerConnection(nn.Module):
    """
    残差连接 + Layer Normalization

    实现Transformer中的子层连接，包括：
    1. Layer Normalization
    2. 子层（如注意力或FFN）
    3. Dropout
    4. 残差连接
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True
    ):
        """
        初始化残差连接层
        """
        super(SublayerConnection, self).__init__()

        self.d_model = d_model
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # Layer Normalization（如果不使用则用Identity替代）
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        """
        应用残差连接到子层
        """
        # Pre-LN架构: x + Dropout(Sublayer(LayerNorm(x)))
        # 1. 对输入进行Layer Normalization（如果启用）
        normalized = self.norm(x)

        # 2. 通过子层
        sublayer_output = sublayer(normalized)

        # 3. 应用Dropout
        sublayer_output = self.dropout(sublayer_output)

        # 4. 残差连接（如果启用）
        if self.use_residual:
            return x + sublayer_output
        else:
            return sublayer_output
