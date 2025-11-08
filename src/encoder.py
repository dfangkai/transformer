"""
Transformer Encoder模块

包含EncoderLayer和完整的Encoder实现。
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from src.attention import MultiHeadAttention
from src.layers import PositionWiseFeedForward, SublayerConnection
from src.positional import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Transformer Encoder层

    每个Encoder层包含两个子层：
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    每个子层后都有残差连接和Layer Normalization。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True
    ):
        """
        初始化Encoder层
        """
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # 多头自注意力层
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # 两个残差连接层（一个用于attention，一个用于FFN）
        # 传递use_residual和use_layernorm参数以支持消融实验
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(
                d_model=d_model,
                dropout=dropout,
                use_residual=use_residual,
                use_layernorm=use_layernorm
            )
            for _ in range(2)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder层的前向传播
        """
        # 第一个子层：Multi-Head Self-Attention + 残差连接 + LayerNorm
        # SublayerConnection会根据配置自动处理是否使用残差和LayerNorm
        x = self.sublayer_connections[0](
            x,
            lambda x: self.self_attn(x, x, x, mask)[0]
        )

        # 第二个子层：Feed-Forward Network + 残差连接 + LayerNorm
        x = self.sublayer_connections[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    完整的Transformer Encoder

    包含：
    1. Token Embedding层
    2. Positional Encoding
    3. N个EncoderLayer
    4. 最终的Layer Normalization
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_residual: bool = True,
        use_layernorm: bool = True
    ):
        """
        初始化Encoder
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_positional_encoding = use_positional_encoding
        self.use_layernorm = use_layernorm

        # Token嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码层
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # N个Encoder层
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_residual=use_residual,
                use_layernorm=use_layernorm
            )
            for _ in range(num_layers)
        ])

        # 最终的Layer Normalization
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # 嵌入缩放因子（原始论文中使用sqrt(d_model)来缩放嵌入）
        self.scale = math.sqrt(d_model)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        # 使用Xavier初始化嵌入层
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder的前向传播
        """
        # 1. Token嵌入
        # [batch, src_len] -> [batch, src_len, d_model]
        x = self.embedding(src)

        # 2. 缩放嵌入（论文中提到使用sqrt(d_model)来缩放）
        x = x * self.scale

        # 3. 添加位置编码（如果启用）
        # [batch, src_len, d_model] -> [batch, src_len, d_model]
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # 4. 通过N个Encoder层
        for layer in self.layers:
            x = layer(x, src_mask)

        # 5. 最终的Layer Normalization
        x = self.norm(x)

        return x

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        为源序列创建mask
        """
        # 创建padding mask（1表示有效位置，0表示padding位置）
        # [batch, src_len] -> [batch, 1, 1, src_len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
