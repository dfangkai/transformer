"""
数据加载模块 - 对话摘要任务

负责加载SAMSum数据集、训练/加载自定义BPE分词器、创建DataLoader等。
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import Tuple
import os


def load_samsum(split: str = 'train'):
    """
    加载SAMSum对话摘要数据集

    Args:
        split: 数据集分割，'train'、'validation'或'test'

    Returns:
        Dataset对象，每个样本格式 {'dialogue': str, 'summary': str}
    """
    dataset = load_dataset("knkarthick/samsum", split=split)
    return dataset


def train_bpe_tokenizer(
    dataset,
    vocab_size: int = 8000,
    save_path: str = "tokenizer.json"
):
    """
    在SAMSum数据集上训练BPE分词器

    Args:
        dataset: SAMSum数据集
        vocab_size: 词表大小
        save_path: 保存路径

    Returns:
        训练好的Tokenizer对象
    """
    print(f"\n{'='*80}")
    print("训练BPE分词器")
    print(f"{'='*80}")
    print(f"词表大小: {vocab_size}")
    print(f"训练样本数: {len(dataset):,}")

    # 初始化BPE分词器
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # 设置预分词器（按空格分词）
    tokenizer.pre_tokenizer = Whitespace()

    # 配置训练器
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        show_progress=True,
        min_frequency=2
    )

    # 准备训练数据（对话 + 摘要）
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            texts = batch['dialogue'] + batch['summary']
            yield texts

    # 训练分词器
    print("\n开始训练分词器...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 配置后处理（添加特殊token）
    tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[
            ("<sos>", tokenizer.token_to_id("<sos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # 保存分词器
    tokenizer.save(save_path)
    print(f"✅ 分词器已保存到: {save_path}")
    print(f"实际词表大小: {tokenizer.get_vocab_size()}")

    return tokenizer


def load_tokenizer(tokenizer_path: str = "tokenizer.json"):
    """
    加载自训练的BPE分词器

    Args:
        tokenizer_path: 分词器文件路径

    Returns:
        Tokenizer对象
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"分词器文件不存在: {tokenizer_path}\n"
            f"请先运行训练脚本以生成分词器"
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✅ 加载分词器: {tokenizer_path}")
    print(f"词表大小: {tokenizer.get_vocab_size()}")

    return tokenizer


class SummarizationDataset(Dataset):
    """
    对话摘要数据集

    将原始数据集转换为PyTorch Dataset，每个样本包含：
    - src: 对话token id序列
    - tgt: 摘要token id序列

    Args:
        dataset: 原始数据集
        tokenizer: BPE tokenizer
        max_src_len: 源序列最大长度
        max_tgt_len: 目标序列最大长度
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 128
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # 获取特殊token的ID
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.sos_id = tokenizer.token_to_id("<sos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (src_ids, tgt_ids)，都是LongTensor
        """
        example = self.dataset[idx]

        # Tokenize对话（源序列）- 不添加特殊token，手动添加
        src_encoding = self.tokenizer.encode(example['dialogue'], add_special_tokens=False)
        src_tokens = src_encoding.ids[:self.max_src_len - 2]  # 留空间给<sos>和<eos>

        # Tokenize摘要（目标序列）
        tgt_encoding = self.tokenizer.encode(example['summary'], add_special_tokens=False)
        tgt_tokens = tgt_encoding.ids[:self.max_tgt_len - 2]

        # 添加<sos>和<eos>
        src_ids = [self.sos_id] + src_tokens + [self.eos_id]
        tgt_ids = [self.sos_id] + tgt_tokens + [self.eos_id]

        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)


def collate_fn(batch, pad_idx: int = 0):
    """
    批处理函数，将不同长度的序列padding到相同长度

    Args:
        batch: 批次数据，[(src1, tgt1), (src2, tgt2), ...]
        pad_idx: padding token的索引

    Returns:
        (src_batch, tgt_batch)，都padding到批次内的最大长度
    """
    src_batch, tgt_batch = zip(*batch)

    # Padding到批次内的最大长度
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_batch, tgt_batch


def create_dataloaders(
    dataset,
    tokenizer,
    batch_size: int = 32,
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """
    创建DataLoader

    Args:
        dataset: 原始数据集
        tokenizer: BPE tokenizer
        batch_size: 批次大小
        max_src_len: 源序列最大长度
        max_tgt_len: 目标序列最大长度
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据

    Returns:
        DataLoader对象
    """
    summarization_dataset = SummarizationDataset(
        dataset,
        tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )

    pad_idx = tokenizer.token_to_id("<pad>")

    return DataLoader(
        summarization_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True
    )
