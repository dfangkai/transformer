"""
完整的Transformer模型

实现了经典的Encoder-Decoder Transformer架构。
"""

import torch
import torch.nn as nn
from typing import Optional

from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    """
    完整的Transformer模型（Encoder-Decoder架构）

    这是"Attention Is All You Need"论文中描述的经典Transformer模型。
    包含Encoder、Decoder和最终的投影层。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_positional_encoding: bool = True,
        use_residual: bool = True,
        use_layernorm: bool = True
    ):
       
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_len=max_len,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_len=max_len,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )

        # 投影层：将decoder输出映射到目标词汇表
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        # 初始化投影层参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        # 使用Xavier初始化投影层
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    
        # 自动创建masks（如果没有提供）
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        # Encoder前向传播
        encoder_output = self.encoder(src, src_mask)

        # Decoder前向传播
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # 投影到词汇表
        output = self.projection(decoder_output)

        return output

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        仅运行Encoder
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        仅运行Decoder
        """
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.projection(decoder_output)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        为源序列创建padding mask
        """
        return self.encoder.make_src_mask(src, self.pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        为目标序列创建mask（padding + future）
        """
        return self.decoder.make_tgt_mask(tgt, self.pad_idx)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        start_symbol: int = 1,
        end_symbol: int = 2,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        贪婪解码生成目标序列

        使用贪婪搜索策略，每一步选择概率最高的token。
        """
        self.eval()

        batch_size = src.size(0)
        device = src.device

        # 1. Encode源序列
        src_mask = self.make_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # 2. 初始化目标序列（只包含start_symbol）
        # shape: [batch, 1]
        tgt = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)

        # 3. 逐个生成token
        for _ in range(max_len - 1):
            # 创建目标mask
            tgt_mask = self.make_tgt_mask(tgt)

            # Decoder前向传播
            # output shape: [batch, current_len, tgt_vocab_size]
            output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

            # 获取最后一个位置的logits
            # next_token_logits shape: [batch, tgt_vocab_size]
            next_token_logits = output[:, -1, :]

            # 应用temperature
            next_token_logits = next_token_logits / temperature

            # 贪婪选择概率最高的token
            # next_token shape: [batch, 1]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # 将新token添加到序列末尾
            # tgt shape: [batch, current_len + 1]
            tgt = torch.cat([tgt, next_token], dim=1)

            # 检查是否所有序列都生成了end_symbol
            # 如果所有序列都包含end_symbol，提前终止
            if (next_token == end_symbol).all():
                break

        return tgt

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 50,
        start_symbol: int = 1,
        end_symbol: int = 2,
        length_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Beam Search解码

        使用beam search策略生成更高质量的输出序列。
        """
        self.eval()

        if src.size(0) != 1:
            raise ValueError("Beam search currently only supports batch_size=1")

        device = src.device

        # 1. Encode源序列
        src_mask = self.make_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # 2. 初始化beam
        # 每个beam是一个序列和对应的分数
        beams = [(torch.full((1, 1), start_symbol, dtype=torch.long, device=device), 0.0)]

        # 3. 逐步扩展beam
        for _ in range(max_len - 1):
            all_candidates = []

            for seq, score in beams:
                # 如果序列已经结束，直接添加到候选
                if seq[0, -1].item() == end_symbol:
                    all_candidates.append((seq, score))
                    continue

                # 创建mask并解码
                tgt_mask = self.make_tgt_mask(seq)
                output = self.decode(seq, encoder_output, src_mask, tgt_mask)

                # 获取最后一个位置的log概率
                next_token_logits = output[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # 获取top-k个候选token
                top_log_probs, top_indices = log_probs.topk(beam_size)

                # 为每个候选创建新的beam
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    all_candidates.append((new_seq, new_score))

            # 按分数排序，选择top-k个beam
            # 应用长度惩罚
            ordered = sorted(
                all_candidates,
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True
            )
            beams = ordered[:beam_size]

            # 如果所有beam都结束，提前终止
            if all(seq[0, -1].item() == end_symbol for seq, _ in beams):
                break

        # 返回得分最高的序列
        best_seq, _ = beams[0]
        return best_seq
