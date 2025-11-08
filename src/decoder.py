"""
Transformer Decoder模块

包含DecoderLayer和完整的Decoder实现。
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from src.attention import MultiHeadAttention
from src.layers import PositionWiseFeedForward, SublayerConnection
from src.positional import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Transformer Decoder层

    每个Decoder层包含三个子层：
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
    3. Position-wise Feed-Forward Network

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
        初始化Decoder层
        """
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # Masked多头自注意力层
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # 多头交叉注意力层（Encoder-Decoder Attention）
        self.cross_attn = MultiHeadAttention(
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

        # 三个残差连接层（self-attn, cross-attn, ffn）
        # 传递use_residual和use_layernorm参数以支持消融实验
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(
                d_model=d_model,
                dropout=dropout,
                use_residual=use_residual,
                use_layernorm=use_layernorm
            )
            for _ in range(3)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder层的前向传播
        """
        # 第一个子层：Masked Multi-Head Self-Attention + 残差连接 + LayerNorm
        # SublayerConnection会根据配置自动处理是否使用残差和LayerNorm
        x = self.sublayer_connections[0](
            x,
            lambda x: self.self_attn(x, x, x, tgt_mask)[0]
        )

        # 第二个子层：Multi-Head Cross-Attention + 残差连接 + LayerNorm
        x = self.sublayer_connections[1](
            x,
            lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)[0]
        )

        # 第三个子层：Feed-Forward Network + 残差连接 + LayerNorm
        x = self.sublayer_connections[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    完整的Transformer Decoder

    包含：
    1. Token Embedding层
    2. Positional Encoding
    3. N个DecoderLayer
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
        初始化Decoder
        """
        super(Decoder, self).__init__()

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

        # N个Decoder层
        self.layers = nn.ModuleList([
            DecoderLayer(
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

        # 嵌入缩放因子
        self.scale = math.sqrt(d_model)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        # 使用Xavier初始化嵌入层
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder的前向传播 
        """
        # 1. Token嵌入
        # [batch, tgt_len] -> [batch, tgt_len, d_model]
        x = self.embedding(tgt)

        # 2. 缩放嵌入
        x = x * self.scale

        # 3. 添加位置编码（如果启用）
        # [batch, tgt_len, d_model] -> [batch, tgt_len, d_model]
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # 4. 通过N个Decoder层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # 5. 最终的Layer Normalization
        x = self.norm(x)

        return x

    def make_tgt_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        为目标序列创建mask（同时包含padding mask和future mask）
        """
        from src.utils import create_target_mask
        return create_target_mask(tgt, pad_idx)
