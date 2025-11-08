# Transformer from Scratch - 对话摘要

完整的 Transformer 实现（不使用 `nn.Transformer`），用于对话摘要任务。

## 特点

- **从零实现** Encoder-Decoder Transformer 架构
- **完整组件**：Multi-Head Attention、Position-wise FFN、Positional Encoding、Residual & LayerNorm
- **自训练BPE分词器**：词表8k（相比BERT 30k减少73%），更适配对话文本
- **梯度累积**：小显存模拟大batch size，提升训练稳定性
- **训练优化**：AdamW、学习率调度、梯度裁剪、早停机制
- **评估指标**：ROUGE-1/2/L 自动评估摘要质量
- **数据集**：SAMSum对话摘要（14.7k训练样本）
- **硬件加速**：支持 MPS/CUDA/CPU

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 开始训练

```bash
# 默认训练（自动检测设备）
bash scripts/train.sh

# 指定设备
bash scripts/train.sh mps     # MacBook M系列
bash scripts/train.sh cuda    # NVIDIA GPU
bash scripts/train.sh cpu     # CPU

# 快速测试（10 epochs）
bash scripts/train.sh --quick
```

**首次运行**会自动训练BPE分词器（约1-2分钟），保存为 `tokenizer.json`，后续运行直接加载。

### 训练时间参考

| 设备 | 完整训练 | 快速测试 (10 epochs) |
|-----|---------|---------------------|
| MacBook M5 (MPS) | 30-40分钟 | 10-15分钟 |
| NVIDIA GPU (CUDA) | 20-30分钟 | 7-10分钟 |
| CPU | 2-3小时 | 40-60分钟 |

*实际时间取决于早停机制触发时间*

## 项目结构

```
Transformer from Scratch/
├── src/                          # 源代码
│   ├── attention.py              # Multi-Head Attention
│   ├── positional.py             # Positional Encoding
│   ├── layers.py                 # FFN, Residual, LayerNorm
│   ├── encoder.py                # Encoder
│   ├── decoder.py                # Decoder
│   ├── model.py                  # 完整 Transformer 模型
│   ├── data_loader.py            # 数据加载 + BPE分词器训练
│   ├── train.py                  # 训练脚本（含早停、梯度累积）
│   └── utils.py                  # 工具函数
├── scripts/
│   ├── train.sh                  # 训练脚本
│   ├── run_ablation.sh           # 消融实验运行脚本
│   └── compare_ablation.py       # 消融实验结果对比
├── configs/
│   ├── summarization_config.yaml # 默认配置
│   └── ablation/                 # 消融实验配置
│       ├── baseline.yaml
│       ├── no_positional.yaml
│       └── no_gradient_accumulation.yaml
├── data/                         # 数据集文件
│   └── samsum.zip                # SAMSum数据集压缩包
├── results/                      # 训练结果
│   ├── summarization/            # 默认训练结果
│   └── ablation/                 # 消融实验结果
├── requirements.txt
└── README.md
```

## 配置说明

### 模型配置 (configs/summarization_config.yaml)

```yaml
# 模型参数
d_model: 256              # 模型维度
num_heads: 8              # 注意力头数
num_encoder_layers: 4     # Encoder层数
num_decoder_layers: 4     # Decoder层数
d_ff: 1024               # 前馈网络维度
dropout: 0.1             # Dropout比率

# 训练参数
batch_size: 16           # 批次大小
learning_rate: 3e-4      # 学习率
num_epochs: 30           # 最大训练轮数
gradient_accumulation_steps: 2  # 有效batch_size = 32

# 数据参数
vocab_size: 8000         # BPE分词器词表大小
max_src_len: 512         # 对话最大长度
max_tgt_len: 128         # 摘要最大长度

# 设备配置
device: auto                # 设备选择: auto(自动检测), mps, cuda, cpu
seed: 42                    # 随机种子

# 早停配置
early_stopping_patience: 5  # 容忍轮数
```

**设备配置说明**：
- `auto` - 自动检测最佳设备（优先级: MPS > CUDA > CPU）
- `mps` - 使用 Apple Silicon 的 MPS 加速
- `cuda` - 使用 NVIDIA GPU 的 CUDA 加速
- `cpu` - 使用 CPU（较慢）

### 关键改进

#### 1. 自训练BPE分词器

**优点**：
- 词表从30k优化到8k，减少73%
- 模型参数从~24M降至~20M，减少17%
- 更适配对话文本特点（口语化表达、缩写等）
- 无需依赖大型预训练模型

**实现**：首次运行时自动在SAMSum训练集上训练BPE分词器，保存为 `tokenizer.json`

#### 2. 梯度累积

**优点**：
- 在小显存下模拟大batch size效果
- 默认配置：batch_size=16 × accumulation_steps=2 = 有效batch_size=32
- 提升训练稳定性，更好的泛化能力

**调整**：根据显存情况调整配置
```yaml
# 显存充足 (8GB+)
batch_size: 32
gradient_accumulation_steps: 1

# 显存适中 (4-8GB)
batch_size: 16
gradient_accumulation_steps: 2

# 显存紧张 (<4GB)
batch_size: 8
gradient_accumulation_steps: 4
```

## 使用训练好的模型

```python
import torch
from tokenizers import Tokenizer
from src.model import Transformer

# 加载模型
checkpoint = torch.load('results/summarization/best_model.pt')
config = checkpoint['config']

# 加载BPE分词器
tokenizer = Tokenizer.from_file(checkpoint['tokenizer_path'])

# 创建模型
model = Transformer(
    src_vocab_size=checkpoint['vocab_size'],
    tgt_vocab_size=checkpoint['vocab_size'],
    d_model=config['d_model'],
    num_heads=config['num_heads'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    d_ff=config['d_ff'],
    max_len=config['max_len'],
    dropout=config['dropout'],
    pad_idx=tokenizer.token_to_id("<pad>")
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 生成摘要
dialogue = """
Alice: Hey, are you free tomorrow?
Bob: Yeah, what's up?
Alice: Want to grab coffee at 3pm?
Bob: Sure, sounds good!
"""

# 编码
encoding = tokenizer.encode(dialogue.strip(), add_special_tokens=False)
src_ids = [tokenizer.token_to_id("<sos>")] + encoding.ids[:510] + [tokenizer.token_to_id("<eos>")]
src = torch.LongTensor(src_ids).unsqueeze(0)

# 生成
with torch.no_grad():
    output = model.generate(
        src,
        max_len=128,
        start_symbol=tokenizer.token_to_id("<sos>"),
        end_symbol=tokenizer.token_to_id("<eos>")
    )
    summary = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

print(f"摘要: {summary}")
```

## 数据集

**SAMSum** - 对话摘要数据集
- 训练集：14,732 对话-摘要对
- 验证集：818 对话-摘要对
- 测试集：819 对话-摘要对
- **数据集来源**: https://huggingface.co/datasets/knkarthick/samsum
- **本地存储**: 数据集压缩包已存放在 `data/samsum.zip`，便于离线使用
- 首次运行时会自动加载数据集

**示例**：
```
对话:
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Amanda: Sorry, can't find it.
Hannah: I think I'll just go to her place :)

摘要:
Hannah needs Betty's number but Amanda doesn't have it.
```

## 故障排除

### MPS 不可用
```bash
bash scripts/train.sh cpu
```

### 内存不足
编辑 `configs/summarization_config.yaml`，减小 batch_size：
```yaml
batch_size: 8
gradient_accumulation_steps: 4  # 保持有效batch_size=32
```

### 下载数据集失败
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/train.sh
```

## 消融实验

通过移除关键组件，验证其重要性。

### 实验配置

| 实验名称 | 说明 | 修改内容 |
|---------|------|---------|
| **baseline** | 完整模型（对照组） | - |
| **no_positional** | 去除位置编码 | `use_positional_encoding: false` |
| **no_gradient_accumulation** | 去除梯度累积 | `gradient_accumulation_steps: 2 → 1` |

### 运行消融实验

```bash
# 运行所有实验
bash scripts/run_ablation.sh

# 运行单个实验
bash scripts/run_ablation.sh baseline

# 指定设备
bash scripts/run_ablation.sh --device mps

# 对比结果
python scripts/compare_ablation.py
```

### 结果文件结构

**主实验结果** (`results/summarization/`):
```
results/summarization/
├── best_model.pt              # 最佳模型checkpoint
├── training_history.json      # 训练历史数据（loss等）
├── training_curve.png         # 训练曲线图
├── summary_samples.txt        # 展示样本生成结果（10个样本）
├── rouge_scores.json          # ROUGE评估结果（完整测试集819样本）
├── config_used.yaml          # 训练时使用的配置备份
└── training_summary.txt      # 训练总结报告
```

**消融实验结果** (`results/ablation/`):
```
results/ablation/
├── baseline/                    # 完整模型（每个实验独立保存）
│   ├── best_model.pt
│   ├── training_history.json
│   ├── training_curve.png
│   ├── summary_samples.txt         # 10个展示样本
│   ├── rouge_scores.json           # 完整测试集评估
│   ├── config_used.yaml
│   └── training_summary.txt
├── no_positional/               # 去除位置编码
│   └── （同上结构）
├── no_gradient_accumulation/    # 去除梯度累积
│   └── （同上结构）
├── comparison_val_loss.png      # 验证Loss对比图
├── comparison_train_vs_val.png  # 训练vs验证Loss图
└── comparison_results.csv       # 结果对比表
```

### 结果文件说明

1. **best_model.pt** - 最佳模型checkpoint，包含模型权重、优化器状态等
2. **training_history.json** - 训练历史数据，可用于绘图和分析
   ```python
   import json
   with open('results/summarization/training_history.json') as f:
       history = json.load(f)
   print(f"训练轮数: {history['total_epochs']}")
   print(f"最佳验证Loss: {history['best_val_loss']}")
   ```
3. **training_curve.png** - 训练和验证loss曲线图
4. **summary_samples.txt** - 10个展示样本的生成结果，用于查看生成质量
5. **rouge_scores.json** - ROUGE评估结果（完整测试集819样本）
   ```python
   import json
   with open('results/summarization/rouge_scores.json') as f:
       rouge = json.load(f)
   print(f"ROUGE-1: {rouge['rouge1']['mean']:.4f}")
   print(f"ROUGE-2: {rouge['rouge2']['mean']:.4f}")
   print(f"ROUGE-L: {rouge['rougeL']['mean']:.4f}")
   ```
6. **config_used.yaml** - 训练时使用的完整配置，确保可重现
7. **training_summary.txt** - 训练总结报告，包含所有关键信息（含ROUGE分数）

### 验证目标

- **位置编码** - Transformer核心特性，验证其对序列建模的重要性
- **梯度累积** - 本项目优化，验证其对训练稳定性的影响

### 评估指标

训练结束后自动在**完整测试集**（819个样本）上计算ROUGE指标：

- **ROUGE-1**: 衡量生成摘要与参考摘要的单词重叠度
- **ROUGE-2**: 衡量生成摘要与参考摘要的双词组（bigram）重叠度
- **ROUGE-L**: 基于最长公共子序列（LCS）的相似度

**ROUGE分数解读**：
- 0.3-0.4：基本可用的摘要质量
- 0.4-0.5：良好的摘要质量
- 0.5+：优秀的摘要质量

**评估说明**：
- ROUGE评估在完整测试集（819个样本）上进行，确保评估结果的统计显著性
- 展示样本（10个）保存在 `summary_samples.txt` 中，用于查看生成质量
- 所有ROUGE分数（平均值、标准差、详细分数）保存在 `rouge_scores.json` 中

## 核心实现

### Multi-Head Attention (`src/attention.py`)
```python
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

### Positional Encoding (`src/positional.py`)
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Early Stopping (`src/train.py`)
- 监控验证loss，连续5个epoch无改善时停止
- 避免过拟合，节省训练时间

---

**作者**: dongfangkai | **更新**: 2025-11-08
