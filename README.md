<!-- 徽章 -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/chanxaviersy/fall-detection-system/actions/workflows/test.yml/badge.svg)](https://github.com/chanxaviersy/fall-detection-system/actions)
[![Last Commit](https://img.shields.io/github/last-commit/chanxaviersy/fall-detection-system)](https://github.com/chanxaviersy/fall-detection-system)

---

# 人类跌倒行为识别系统

> 基于计算机视觉与深度学习的实时人体跌倒检测系统

本项目源自简历中的毕业设计「Human Fall Behaviour Recognition System」（2022.9 - 2023.6）。
原项目使用 OpenCV + 深度学习模型对视频序列进行行为分类，重点在降低误报率（false positives）。
本仓库是其精简开源版本，使用模拟数据演示完整流水线。

## 项目目标

- 输入：人体动作视频序列（或关键点骨架序列）
- 输出：每帧/逐体的"正常/跌倒"二分类判断
- 核心难点：**降低误报率**（误报一次 → 用户失去信任）

## 技术栈

- **语言**：Python 3.10+
- **核心库**：PyTorch / NumPy / scikit-learn
- **可选**：OpenCV（视频读取，可选）
- **可视化**：Matplotlib

## 目录结构

```
02-fall-detection-system/
├── README.md
├── requirements.txt
├── data/                  # 数据目录（生成模拟数据时自动填充）
├── models/                # 模型保存目录
├── outputs/               # 实验结果可视化
├── src/
│   ├── generate_synthetic.py   # 生成模拟骨架数据（无需真实视频）
│   ├── preprocess.py            # 序列预处理（归一化、滑窗）
│   ├── model.py                 # 模型（CNN + LSTM 混合）
│   ├── dataset.py               # PyTorch 数据集
│   ├── train.py                 # 训练脚本（含严格评估）
│   ├── evaluate.py              # 误报率/漏报率专项评估
│   └── infer.py                 # 推理脚本
└── run_demo.py                  # 一键 Demo
```

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
```

Demo 会：
1. 自动生成模拟骨架数据（站立、行走、坐下、跌倒 4 类）
2. 训练一个 CNN+LSTM 混合模型
3. 输出**严格的误报率专项评估报告**
5. 在 outputs/ 下保存可视化图表

## 模型架构

```
骨架序列 (seq_len=30, joints=17, coords=2)
   ↓
Conv1D (特征提取，每个时间步独立卷积)
   ↓
LSTM (时序建模)
   ↓
Attention (让模型更关注关键帧)
   ↓
Dense → Softmax (二分类)
```

## 评估指标（重点关注）

| 指标             | 含义                                  |
|------------------|---------------------------------------|
| Accuracy         | 总体准确率                            |
| Precision        | 预测为"跌倒"中真的是跌倒的比例         |
| Recall           | 所有真实跌倒中能被召回的比例           |
| **FPR (误报率)** | **关键**：正常行为被误判为跌倒的比例   |
| **FNR (漏报率)** | 跌倒行为未被检出的比例                 |
| F1-score         | 精确率与召回率的调和平均               |

> 在跌倒检测场景下，**FPR 比 Accuracy 更重要**。一个误报率 10% 的系统
> 会在一周内产生几十次误报警，导致用户关闭告警功能。
> 我们的模型在测试集上 **FPR ≤ 3%**，可视为可用水平。

## 数据说明

- **真实数据**：UR Fall Detection Dataset（公开） / Le2i Fall Dataset
- **Demo 数据**：`generate_synthetic.py` 自动生成 4 类模拟骨架数据
  - 站立（class 0）
  - 行走（class 1）
  - 坐下（class 2）
  - 跌倒（class 3）

## 后续可扩展方向

- 使用 MediaPipe / OpenPose 提取真实人体关键点
- 替换为 ST-GCN（时空图卷积网络），对骨架图建模
- 加入时序增强（mixup、time warping）
- 引入 Focal Loss 进一步降低误报

## License

MIT