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

    公式: FFN(x) = max(0, xW1 + b1)W2 + b2

    参数:
        d_model (int): 模型的嵌入维度（输入和输出维度）
        d_ff (int): 前馈网络的隐藏层维度
        dropout (float): dropout概率，默认0.1

    属性:
        fc1 (nn.Linear): 第一个全连接层，d_model -> d_ff
        fc2 (nn.Linear): 第二个全连接层，d_ff -> d_model
        dropout (nn.Dropout): Dropout层
        relu (nn.ReLU): ReLU激活函数

    示例:
        >>> ffn = PositionWiseFeedForward(d_model=256, d_ff=1024)
        >>> x = torch.randn(32, 10, 256)  # [batch, seq_len, d_model]
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([32, 10, 256])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化Position-wise Feed-Forward Network

        Args:
            d_model: 模型的嵌入维度
            d_ff: 前馈网络的隐藏层维度（通常是d_model的4倍）
            dropout: dropout概率
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

        Args:
            x: 输入张量，shape: [batch, seq_len, d_model]

        Returns:
            输出张量，shape: [batch, seq_len, d_model]
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

    公式: x + Dropout(Sublayer(LayerNorm(x)))

    注意：这里使用的是Pre-LN的架构（在子层前进行normalization），
    与原始论文中的Post-LN（在子层后normalization）不同。
    Pre-LN通常训练更稳定。

    参数:
        d_model (int): 模型的嵌入维度
        dropout (float): dropout概率，默认0.1
        use_residual (bool): 是否使用残差连接，默认True
        use_layernorm (bool): 是否使用LayerNorm，默认True

    属性:
        norm (nn.LayerNorm or nn.Identity): Layer Normalization层
        dropout (nn.Dropout): Dropout层
        use_residual (bool): 是否使用残差连接
        use_layernorm (bool): 是否使用LayerNorm

    示例:
        >>> sublayer_conn = SublayerConnection(d_model=256)
        >>> x = torch.randn(32, 10, 256)
        >>> # 定义一个简单的子层函数
        >>> def sublayer(x):
        ...     return x * 2
        >>> output = sublayer_conn(x, sublayer)
        >>> output.shape
        torch.Size([32, 10, 256])
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

        Args:
            d_model: 模型的嵌入维度
            dropout: dropout概率
            use_residual: 是否使用残差连接（用于消融实验）
            use_layernorm: 是否使用LayerNorm（用于消融实验）
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

        Args:
            x: 输入张量，shape: [batch, seq_len, d_model]
            sublayer: 子层函数，接受归一化后的x作为输入

        Returns:
            输出张量，shape: [batch, seq_len, d_model]
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
