# 消融实验配置

本目录包含3个消融实验的配置文件，用于验证Transformer关键组件的重要性。

**训练配置**：所有实验设置为15轮（主实验在15轮触发早停），早停容忍度为5轮。

## 实验列表

### 1. baseline.yaml - 完整模型（对照组）
包含所有优化组件的完整配置：
- ✅ 位置编码
- ✅ 8个注意力头
- ✅ 4层Encoder/Decoder
- ✅ 梯度累积（有效batch_size=32）

### 2. no_positional.yaml - 去除位置编码
验证位置编码的重要性：
- ❌ **不使用位置编码**
- ✅ 其他组件保持不变

**验证目标**: Transformer的核心特性，位置编码是否对序列建模至关重要

**预期**: 性能显著下降，因为模型无法区分序列位置信息

### 3. no_gradient_accumulation.yaml - 去除梯度累积
验证梯度累积对训练稳定性的影响：
- ❌ **梯度累积步数：2 → 1**
- ⬇️ 有效batch_size：32 → 16
- ✅ 其他组件保持不变

**验证目标**: 本项目的优化改进，梯度累积是否提升训练稳定性

**预期**: 训练更不稳定，可能性能略有下降

## 快速开始

### 运行所有实验
```bash
# 默认自动检测设备（MPS > CUDA > CPU）
bash scripts/run_ablation.sh
```

### 运行单个实验
```bash
bash scripts/run_ablation.sh baseline
bash scripts/run_ablation.sh no_positional
bash scripts/run_ablation.sh no_gradient_accumulation
```

### 指定设备（可选）
```bash
# 强制使用特定设备
bash scripts/run_ablation.sh --device mps   # Apple Silicon
bash scripts/run_ablation.sh --device cuda  # NVIDIA GPU
bash scripts/run_ablation.sh --device cpu   # CPU
```

**注意**：配置文件中已设置 `device: auto`，会自动检测最佳设备，通常不需要手动指定

### 对比结果
```bash
python scripts/compare_ablation.py
```

## 结果解读

运行完所有实验后，会生成：

1. **对比表格** (终端输出)
   - 各实验的验证Loss、训练Loss、ROUGE分数
   - 相对baseline的性能变化

2. **可视化图表** (`results/ablation/`)
   - `comparison_val_loss.png` - 验证Loss对比柱状图
   - `comparison_train_vs_val.png` - 训练vs验证Loss散点图
   - `comparison_results.csv` - 完整结果CSV

3. **实验结果文件** (`results/ablation/[experiment_name]/`)
   - `best_model.pt` - 最佳模型checkpoint
   - `training_history.json` - 训练历史数据（每个epoch的loss）
   - `training_curve.png` - 训练曲线图
   - `summary_samples.txt` - 10个展示样本的生成结果
   - `rouge_scores.json` - ROUGE评估结果（完整测试集819样本）
   - `config_used.yaml` - 训练配置备份
   - `training_summary.txt` - 训练总结报告（含ROUGE分数）

## 实验时间

每个实验预计耗时（MacBook M5/MPS，15 epochs）：
- 单个实验：10-20分钟（可能更早触发早停）
- 所有实验（3个）：30-60分钟

**说明**：主实验在15轮触发早停，消融实验设为相同轮数以保持可比性。实际训练时间取决于早停机制触发时间。

## 自定义实验

复制现有配置文件并修改参数：

```bash
cp configs/ablation/baseline.yaml configs/ablation/my_experiment.yaml
# 编辑 my_experiment.yaml
bash scripts/run_ablation.sh my_experiment
```

## 注意事项

1. **设备自动检测**：配置文件中 `device: auto`，会自动选择最佳设备（MPS > CUDA > CPU）
2. **首次运行会训练BPE分词器**（约1-2分钟），后续实验共用
3. **所有实验使用相同的随机种子**（seed=42），确保可重现
4. **早停机制可能导致不同实验训练轮数不同**

---

详细文档请查看主 [README.md](../../README.md)
