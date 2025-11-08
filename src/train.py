"""
Transformer模型训练脚本 - 对话摘要任务

提供完整的训练流程，包括数据加载、模型训练、评估和推理。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time

from src.data_loader import (
    load_samsum,
    train_bpe_tokenizer,
    load_tokenizer,
    create_dataloaders
)
from src.model import Transformer
from src.utils import (
    count_parameters,
    print_model_info,
    plot_training_curves,
    create_padding_mask,
    create_target_mask
)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    Warmup + Cosine学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总训练步数

    Returns:
        学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup阶段：线性增长
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """
    早停机制

    当验证loss在patience个epoch内没有改善时停止训练
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 当前验证loss

        Returns:
            是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"  初始化最佳验证Loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  触发早停！验证Loss在{self.patience}个epoch内没有改善")
        else:
            if self.verbose:
                print(f"  验证Loss改善: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    log_interval: int = 100,
    gradient_accumulation_steps: int = 1
) -> float:
    """
    训练一个epoch（支持梯度累积）

    Args:
        model: Transformer模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch数
        log_interval: 日志打印间隔
        gradient_accumulation_steps: 梯度累积步数

    Returns:
        平均训练loss
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    # 梯度累积优化器零梯度只在累积步骤完成时调用
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for i, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)

        # 准备decoder输入（去掉最后一个token）和目标（去掉第一个token）
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 创建masks
        src_mask = create_padding_mask(src, pad_idx=0)
        tgt_mask = create_target_mask(tgt_input, pad_idx=0)

        # 前向传播
        output = model(src, tgt_input, src_mask, tgt_mask)

        # 计算loss（CrossEntropyLoss会自动忽略padding）
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        # 梯度累积：将loss除以累积步数
        loss = loss / gradient_accumulation_steps

        # 反向传播
        loss.backward()

        # 只在累积步骤完成时更新参数
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == num_batches:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps  # 恢复真实的loss

        # 更新进度条
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })

    return total_loss / num_batches


def evaluate(
    model,
    dataloader,
    criterion,
    device
) -> float:
    """
    评估模型

    Args:
        model: Transformer模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        平均验证loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Evaluating'):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src, pad_idx=0)
            tgt_mask = create_target_mask(tgt_input, pad_idx=0)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


def summarize_dialogue(
    model,
    dialogue: str,
    tokenizer,
    device,
    max_len: int = 128,
    debug: bool = False
) -> str:
    """
    生成对话摘要

    Args:
        model: Transformer模型
        dialogue: 对话文本
        tokenizer: BPE tokenizer
        device: 设备
        max_len: 最大生成长度
        debug: 是否打印调试信息

    Returns:
        生成的摘要
    """
    model.eval()

    # Tokenize对话
    encoding = tokenizer.encode(dialogue, add_special_tokens=False)
    src_ids = [tokenizer.token_to_id("<sos>")] + encoding.ids[:510] + [tokenizer.token_to_id("<eos>")]
    src = torch.LongTensor(src_ids).unsqueeze(0).to(device)

    if debug:
        print(f"\n[DEBUG] 对话token数: {len(src_ids)}")
        print(f"[DEBUG] 词表大小: {tokenizer.get_vocab_size()}")

    # 使用generate方法进行贪婪解码
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")

    with torch.no_grad():
        output = model.generate(
            src,
            max_len=max_len,
            start_symbol=sos_id,
            end_symbol=eos_id
        )

    if debug:
        print(f"[DEBUG] 生成的token IDs: {output[0].tolist()[:20]}...")
        print(f"[DEBUG] 生成token数: {output[0].size(0)}")

    # Decode到文本
    summary = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    return summary


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train Transformer model for dialogue summarization')
    parser.add_argument('--config', type=str, default='configs/summarization_config.yaml', help='配置文件路径')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数（覆盖配置文件）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（覆盖配置文件）')
    parser.add_argument('--device', type=str, default=None, help='设备（覆盖配置文件）')
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置文件
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device

    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # 设置device
    if config['device'] == 'auto':
        # 自动检测：优先级 MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"使用设备: {device} (自动检测)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用设备: {device} (自动检测)")
        else:
            device = torch.device('cpu')
            print(f"使用设备: {device} (自动检测)")
    elif config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"使用设备: {device}")
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: {device}")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # ========== 数据加载 ==========
    print("\n" + "=" * 80)
    print("数据加载")
    print("=" * 80)

    print("加载SAMSum对话摘要数据集...")
    train_data = load_samsum('train')
    val_data = load_samsum('validation')
    test_data = load_samsum('test')

    print(f"训练集大小: {len(train_data):,}")
    print(f"验证集大小: {len(val_data):,}")
    print(f"测试集大小: {len(test_data):,}")

    # 训练或加载tokenizer
    tokenizer_path = config.get('tokenizer_path', 'tokenizer.json')
    vocab_size = config.get('vocab_size', 8000)

    if not os.path.exists(tokenizer_path):
        print(f"\n分词器文件不存在，开始训练BPE分词器...")
        tokenizer = train_bpe_tokenizer(
            train_data,
            vocab_size=vocab_size,
            save_path=tokenizer_path
        )
    else:
        print(f"\n加载已有分词器: {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)

    # 创建DataLoader
    print("\n创建DataLoader...")
    train_loader = create_dataloaders(
        train_data,
        tokenizer,
        batch_size=config['batch_size'],
        max_src_len=config['max_src_len'],
        max_tgt_len=config['max_tgt_len'],
        num_workers=0,
        shuffle=True
    )
    val_loader = create_dataloaders(
        val_data,
        tokenizer,
        batch_size=config['batch_size'],
        max_src_len=config['max_src_len'],
        max_tgt_len=config['max_tgt_len'],
        num_workers=0,
        shuffle=False
    )

    print(f"训练批次数: {len(train_loader):,}")
    print(f"验证批次数: {len(val_loader):,}")

    # ========== 模型创建 ==========
    print("\n" + "=" * 80)
    print("模型创建")
    print("=" * 80)

    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("<pad>")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        pad_idx=pad_idx,
        use_positional_encoding=config.get('use_positional_encoding', True)
    ).to(device)

    num_params = count_parameters(model)
    print(f"模型参数数量: {num_params:,}")
    print(f"参数大小: {num_params * 4 / (1024 ** 2):.2f} MB")

    # ========== 优化器和调度器 ==========
    print("\n" + "=" * 80)
    print("优化器设置")
    print("=" * 80)

    # 梯度累积
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        print(f"梯度累积步数: {gradient_accumulation_steps}")
        print(f"有效batch size: {config['batch_size'] * gradient_accumulation_steps}")

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = len(train_loader) * config['num_epochs'] // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        config['warmup_steps'],
        num_training_steps
    )

    print(f"学习率: {config['learning_rate']}")
    print(f"Warmup步数: {config['warmup_steps']}")
    print(f"总训练步数: {num_training_steps:,}")

    # Loss函数（忽略padding + Label Smoothing）
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=config.get('label_smoothing', 0.1)
    )

    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        min_delta=config.get('early_stopping_min_delta', 0.0),
        verbose=True
    )

    # ========== 训练 ==========
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*80}")

        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            config['log_interval'],
            gradient_accumulation_steps
        )

        # 评估
        val_loss = evaluate(model, val_loader, criterion, device)

        # 更新学习率（每个epoch更新一次）
        for _ in range(len(train_loader) // gradient_accumulation_steps):
            scheduler.step()

        # 记录loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 打印统计
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch} 完成:")
        print(f"  训练Loss: {train_loss:.4f}")
        print(f"  验证Loss: {val_loss:.4f}")
        print(f"  耗时: {elapsed_time/60:.1f}分钟")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'tokenizer_path': tokenizer_path,
                'vocab_size': vocab_size
            }
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"  ✅ 保存最佳模型 (验证Loss: {val_loss:.4f})")

        # 检查早停
        if early_stopping(val_loss):
            print(f"\n提前停止训练（Epoch {epoch}）")
            break

    # ========== 训练完成 ==========
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"最佳验证Loss: {best_val_loss:.4f}")
    print(f"实际训练轮数: {len(train_losses)}")

    # 保存训练历史
    print("\n保存训练历史...")
    import json
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'total_time_minutes': total_time / 60,
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
    }
    with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 训练历史已保存")

    # 绘制训练曲线（保存到实验专属目录）
    print("\n绘制训练曲线...")
    plot_training_curves(
        train_losses,
        val_losses,
        os.path.join(config['save_dir'], 'training_curve.png')
    )

    # ========== 测试摘要生成 ==========
    print("\n" + "=" * 80)
    print("测试摘要生成")
    print("=" * 80)

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 1. 生成展示样本（10个代表性样本，用于可读性展示）
    print("\n生成展示样本 (10个代表性样本)...")
    display_indices = [0, 10, 20, 50, 100, 200, 300, 400, 500, 700]
    display_samples = [(test_data[i]['dialogue'], test_data[i]['summary'])
                       for i in display_indices if i < len(test_data)]

    with open(os.path.join(config['save_dir'], 'summary_samples.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("测试摘要生成结果（展示样本）\n")
        f.write("=" * 80 + "\n\n")

        for i, (dialogue, ref_summary) in enumerate(display_samples, 1):
            # 第一个样本启用调试模式
            debug_mode = (i == 1)
            generated_summary = summarize_dialogue(
                model, dialogue, tokenizer, device,
                max_len=config['max_tgt_len'],
                debug=debug_mode
            )

            output = f"样本 {i}:\n"
            output += f"  对话: {dialogue[:300]}{'...' if len(dialogue) > 300 else ''}\n"
            output += f"  参考摘要: {ref_summary}\n"
            output += f"  生成摘要: {generated_summary}\n"
            output += "-" * 80 + "\n\n"

            if i <= 3:  # 只在终端显示前3个样本
                print(output)
            f.write(output)

    print(f"  ✅ 已生成并保存 {len(display_samples)} 个展示样本")

    # 2. 在完整测试集上进行ROUGE评估（819个样本）
    print(f"\n在完整测试集上进行ROUGE评估 (共 {len(test_data)} 个样本)...")
    print("  这可能需要几分钟，请耐心等待...")

    generated_summaries = []
    reference_summaries = []

    # 批量生成摘要用于评估
    from tqdm import tqdm
    for item in tqdm(test_data, desc="  生成摘要"):
        dialogue = item['dialogue']
        ref_summary = item['summary']

        # 生成摘要（不启用调试模式以加快速度）
        generated_summary = summarize_dialogue(
            model, dialogue, tokenizer, device,
            max_len=config['max_tgt_len'],
            debug=False
        )

        generated_summaries.append(generated_summary)
        reference_summaries.append(ref_summary)

    print(f"  ✅ 已生成 {len(generated_summaries)} 个摘要用于评估")

    # 计算ROUGE分数
    print("\n计算ROUGE评估指标...")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for ref, gen in zip(reference_summaries, generated_summaries):
        scores = scorer.score(ref, gen)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # 计算平均分数
    avg_rouge = {
        'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']),
        'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']),
        'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    }

    print(f"  ROUGE-1: {avg_rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_rouge['rougeL']:.4f}")
    print(f"  ✅ ROUGE评估完成")

    # 保存ROUGE评估结果
    rouge_results = {
        'rouge1': {
            'mean': avg_rouge['rouge1'],
            'std': sum((s - avg_rouge['rouge1'])**2 for s in rouge_scores['rouge1']) / len(rouge_scores['rouge1']) ** 0.5,
            'scores': rouge_scores['rouge1']
        },
        'rouge2': {
            'mean': avg_rouge['rouge2'],
            'std': sum((s - avg_rouge['rouge2'])**2 for s in rouge_scores['rouge2']) / len(rouge_scores['rouge2']) ** 0.5,
            'scores': rouge_scores['rouge2']
        },
        'rougeL': {
            'mean': avg_rouge['rougeL'],
            'std': sum((s - avg_rouge['rougeL'])**2 for s in rouge_scores['rougeL']) / len(rouge_scores['rougeL']) ** 0.5,
            'scores': rouge_scores['rougeL']
        },
        'num_samples': len(generated_summaries),
        'note': 'Evaluated on complete test set'
    }
    with open(os.path.join(config['save_dir'], 'rouge_scores.json'), 'w') as f:
        json.dump(rouge_results, f, indent=2)
    print(f"  ✅ ROUGE分数已保存")

    # 保存配置文件副本
    print("\n保存配置文件副本...")
    import shutil
    config_backup_path = os.path.join(config['save_dir'], 'config_used.yaml')
    if os.path.exists(args.config):
        shutil.copy(args.config, config_backup_path)
        print(f"  ✅ 配置文件已备份")

    # 生成训练总结报告
    print("\n生成训练总结报告...")
    with open(os.path.join(config['save_dir'], 'training_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("训练总结报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("实验配置:\n")
        f.write(f"  实验名称: {config.get('experiment_name', 'default')}\n")
        f.write(f"  配置文件: {args.config}\n")
        f.write(f"  设备: {device}\n")
        f.write(f"  随机种子: {config['seed']}\n\n")

        f.write("模型配置:\n")
        f.write(f"  d_model: {config['d_model']}\n")
        f.write(f"  num_heads: {config['num_heads']}\n")
        f.write(f"  num_encoder_layers: {config['num_encoder_layers']}\n")
        f.write(f"  num_decoder_layers: {config['num_decoder_layers']}\n")
        f.write(f"  vocab_size: {vocab_size}\n")
        f.write(f"  总参数量: {count_parameters(model):,}\n\n")

        f.write("训练配置:\n")
        f.write(f"  batch_size: {config['batch_size']}\n")
        f.write(f"  gradient_accumulation_steps: {config.get('gradient_accumulation_steps', 1)}\n")
        f.write(f"  有效batch_size: {config['batch_size'] * config.get('gradient_accumulation_steps', 1)}\n")
        f.write(f"  learning_rate: {config['learning_rate']}\n")
        f.write(f"  max_epochs: {config['num_epochs']}\n\n")

        f.write("训练结果:\n")
        f.write(f"  实际训练轮数: {len(train_losses)}\n")
        f.write(f"  最佳验证Loss: {best_val_loss:.4f}\n")
        f.write(f"  最终训练Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  最终验证Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  总训练时间: {total_time/60:.1f}分钟\n\n")

        f.write("评估指标 (ROUGE):\n")
        f.write(f"  ROUGE-1 F1: {avg_rouge['rouge1']:.4f}\n")
        f.write(f"  ROUGE-2 F1: {avg_rouge['rouge2']:.4f}\n")
        f.write(f"  ROUGE-L F1: {avg_rouge['rougeL']:.4f}\n")
        f.write(f"  评估样本数: {len(generated_summaries)} (完整测试集)\n\n")

        f.write("测试样本:\n")
        f.write(f"  展示样本数: {len(display_samples)} (保存在 summary_samples.txt)\n")
        f.write(f"  ROUGE评估样本数: {len(generated_summaries)} (完整测试集)\n\n")

        f.write("保存的文件:\n")
        f.write(f"  - best_model.pt (最佳模型)\n")
        f.write(f"  - training_history.json (训练历史)\n")
        f.write(f"  - training_curve.png (训练曲线)\n")
        f.write(f"  - summary_samples.txt ({len(display_samples)}个展示样本)\n")
        f.write(f"  - rouge_scores.json (ROUGE评估结果)\n")
        f.write(f"  - config_used.yaml (配置备份)\n")
        f.write(f"  - training_summary.txt (本文件)\n")

    print(f"  ✅ 训练总结报告已生成")

    print("\n" + "=" * 80)
    print(f"所有结果已保存到: {config['save_dir']}")
    print("=" * 80)
    print(f"  - best_model.pt (最佳模型)")
    print(f"  - training_history.json (训练历史数据)")
    print(f"  - training_curve.png (训练曲线图)")
    print(f"  - summary_samples.txt ({len(display_samples)}个展示样本)")
    print(f"  - rouge_scores.json (ROUGE评估结果)")
    print(f"  - config_used.yaml (配置文件备份)")
    print(f"  - training_summary.txt (训练总结报告)")
    print("=" * 80)
    print(f"\nROUGE评估结果:")
    print(f"  ROUGE-1: {avg_rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_rouge['rougeL']:.4f}")


if __name__ == '__main__':
    main()
