"""
位置编码模块

实现Transformer的位置编码（Positional Encoding），用于为序列中的每个位置添加位置信息。
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层

    使用正弦和余弦函数生成位置编码，使模型能够学习序列中的位置信息。

    公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中:
        - pos: 位置索引 (0 to max_len-1)
        - i: 维度索引 (0 to d_model/2-1)
        - d_model: 模型的嵌入维度

    参数:
        d_model (int): 模型的嵌入维度
        max_len (int): 序列的最大长度，默认5000
        dropout (float): dropout概率，默认0.1
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码层

        Args:
            d_model: 模型的嵌入维度
            max_len: 序列的最大长度
            dropout: dropout概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 创建位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项 10000^(2i/d_model)
        # div_term shape: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 应用正弦到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)

        # 应用余弦到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 将位置编码注册为buffer（不作为模型参数，但会被保存和加载）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码添加到输入嵌入

        Args:
            x: 输入张量，shape: [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量，shape: [batch_size, seq_len, d_model]
        """
        # 添加位置编码到输入
        # self.pe[:, :x.size(1)] 选择与输入序列长度相同的位置编码
        x = x + self.pe[:, :x.size(1)]

        # 应用dropout
        return self.dropout(x)
