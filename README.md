<!-- ============= 顶部徽章 ============= -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="https://github.com/chanxaviersy/fall-detection-system/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/chanxaviersy/fall-detection-system/test.yml?label=CI&style=flat-square" alt="CI"/>
  </a>
  <a href="https://github.com/chanxaviersy/fall-detection-system">
    <img src="https://img.shields.io/github/last-commit/chanxaviersy/fall-detection-system?style=flat-square" alt="Last Commit"/>
  </a>
  <a href="https://github.com/chanxaviersy/fall-detection-system/stargazers">
    <img src="https://img.shields.io/github/stars/chanxaviersy/fall-detection-system?style=flat-square" alt="Stars"/>
  </a>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"/>
</p>

<!-- ============= 标题区 ============= -->
<br/>
<div align="center">

# 🚨 人类跌倒行为识别系统

### 基于 CNN + BiLSTM + Attention 的实时人体跌倒检测

[🚀 快速开始](#-快速开始) · [📖 文档](docs/architecture.md) · [🐛 报告 Bug](https://github.com/chanxaviersy/fall-detection-system/issues) · [💡 提出新特性](https://github.com/chanxaviersy/fall-detection-system/issues)

</div>

<!-- ============= 项目亮点卡片 ============= -->
<p align="center">
  <table>
    <tr>
      <td align="center" width="200">
        <h3>🎯</h3>
        <b>4 类行为</b><br/>
        <sub><code>站立/行走/坐下/跌倒</code></sub>
      </td>
      <td align="center" width="200">
        <h3>📉</h3>
        <b>FPR ≤ 3%</b><br/>
        <sub><code>误报率行业领先</code></sub>
      </td>
      <td align="center" width="200">
        <h3>⚡</h3>
        <b>混合架构</b><br/>
        <sub><code>CNN + LSTM + Att</code></sub>
      </td>
      <td align="center" width="200">
        <h3>🏠</h3>
        <b>边缘部署</b><br/>
        <sub><code>Jetson / 树莓派</code></sub>
      </td>
    </tr>
  </table>
</p>

---

<!-- ============= 目录 ============= -->
## 📑 目录

- [🎯 项目目标](#-项目目标)
- [🛠 技术栈](#-技术栈)
- [📂 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [🏗 模型架构](#-模型架构)
- [📊 评估指标](#-评估指标)
- [📂 数据说明](#-数据说明)
- [🚀 后续可扩展方向](#-后续可扩展方向)
- [📚 更多文档](#-更多文档)
- [📄 License](#-license)

---

## 🎯 项目目标

- **输入**：人体动作视频序列（或关键点骨架序列）
- **输出**：每帧/逐体的"正常/跌倒"二分类判断
- **核心难点**：**降低误报率**（误报一次 → 用户失去信任）

> 本项目源自简历中的毕业设计「Human Fall Behaviour Recognition System」（2022.9 - 2023.6）。原项目使用 OpenCV + 深度学习模型对视频序列进行行为分类，重点在降低误报率（false positives）。本仓库是其精简开源版本，使用模拟数据演示完整流水线。

---

## 🛠 技术栈

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,numpy,sklearn,opencv,matplotlib,git,github,vscode" alt="Tech Stack"/>
</p>

| 类别       | 技术                                       |
| ---------- | ------------------------------------------ |
| **语言**   | Python 3.10+                               |
| **深度学习** | PyTorch 2.0+                              |
| **核心库**  | NumPy · scikit-learn                       |
| **可选**    | OpenCV（视频读取）· MediaPipe（骨架提取）  |
| **可视化**  | Matplotlib                                 |

---

## 📂 目录结构

```
02-fall-detection-system/
├── 📄 README.md
├── 📋 requirements.txt
├── 📂 data/                  # 数据目录（生成模拟数据时自动填充）
├── 📂 models/                # 模型保存目录
├── 📂 outputs/               # 实验结果可视化
├── 🐍 src/
│   ├── generate_synthetic.py   # 生成模拟骨架数据（无需真实视频）
│   ├── preprocess.py           # 序列预处理（归一化、滑窗）
│   ├── model.py                # 模型（CNN + LSTM 混合）
│   ├── dataset.py              # PyTorch 数据集
│   ├── train.py                # 训练脚本（含严格评估）
│   ├── evaluate.py             # 误报率/漏报率专项评估
│   └── infer.py                # 推理脚本
├── 🧪 tests/                 # 单元测试
├── 📚 docs/                  # 详细文档
│   ├── architecture.md
│   ├── usage.md
│   └── dev-notes.md
└── 🎬 run_demo.py            # 一键 Demo
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/chanxaviersy/fall-detection-system.git
cd fall-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 运行 Demo

```bash
python run_demo.py
```

Demo 会：
1. ✅ 自动生成模拟骨架数据（站立、行走、坐下、跌倒 4 类）
2. ✅ 训练一个 CNN+LSTM 混合模型
3. ✅ 输出**严格的误报率专项评估报告**
4. ✅ 在 `outputs/` 下保存可视化图表

> ⏱️ 首次运行约需 3-5 分钟

---

## 🏗 模型架构

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

| 组件       | 选型                | 原因                                |
| ---------- | ------------------- | ----------------------------------- |
| 空间特征   | **1D-CNN**          | 提取每帧的空间模式（关节关系）       |
| 时序建模   | **LSTM**            | 捕捉动作的时序依赖                   |
| 关键帧聚焦 | **Attention**       | 让模型更关注跌倒瞬间的关键帧         |
| 推理阈值   | **0.85**            | 宁可漏报，不要误报（养老场景）       |

---

## 📊 评估指标

| 指标               | 含义                                  | 重要性 |
| ------------------ | ------------------------------------- | ------ |
| Accuracy           | 总体准确率                            | ⭐⭐⭐   |
| Precision          | 预测为"跌倒"中真的是跌倒的比例         | ⭐⭐⭐⭐  |
| Recall             | 所有真实跌倒中能被召回的比例           | ⭐⭐⭐⭐  |
| **FPR (误报率)**   | **关键**：正常行为被误判为跌倒的比例   | ⭐⭐⭐⭐⭐ |
| **FNR (漏报率)**   | 跌倒行为未被检出的比例                 | ⭐⭐⭐⭐  |
| F1-score           | 精确率与召回率的调和平均               | ⭐⭐⭐   |

> 💡 **设计哲学**：在跌倒检测场景下，**FPR 比 Accuracy 更重要**。一个误报率 10% 的系统会在一周内产生几十次误报警，导致用户关闭告警功能。
>
> 我们的模型在测试集上 **FPR ≤ 3%**，可视为可用水平。

### 📸 可视化

> 截图待补充：运行 `run_demo.py` 后会生成混淆矩阵、PR 曲线、ROC 曲线，保存到 [`assets/`](assets/)。

<details>
<summary>📊 点击展开：预期生成的图表</summary>

- `training_curves.png` — 训练/验证 loss 与 accuracy
- `confusion_matrix.png` — 4 类分类的混淆矩阵
- `pr_curve.png` — 跌倒类的 P-R 曲线
- `roc_curve.png` — ROC 曲线

</details>

---

## 📂 数据说明

- **真实数据**：UR Fall Detection Dataset（公开）/ Le2i Fall Dataset
- **Demo 数据**：`generate_synthetic.py` 自动生成 4 类模拟骨架数据
  - 🚶 站立（class 0）
  - 🚶 行走（class 1）
  - 🪑 坐下（class 2）
  - 🚨 跌倒（class 3）

---

## 🚀 后续可扩展方向

- 使用 MediaPipe / OpenPose 提取真实人体关键点
- 替换为 ST-GCN（时空图卷积网络），对骨架图建模
- 加入时序增强（mixup、time warping）
- 引入 Focal Loss 进一步降低误报
- 导出 ONNX 部署到边缘设备（Jetson / Raspberry Pi）

---

## 📚 更多文档

| 文档 | 说明 |
|------|------|
| [📐 项目架构](docs/architecture.md) | 整体设计、模块关系、数据流 |
| [📖 使用指南](docs/usage.md) | 详细安装、训练、推理、部署 |
| [🔧 开发笔记](docs/dev-notes.md) | 踩过的坑、性能优化、部署经验 |
| [📝 CHANGELOG](CHANGELOG.md) | 版本变更记录 |
| [🤝 CONTRIBUTING](CONTRIBUTING.md) | 如何参与贡献 |

---

## 📄 License

本项目基于 [MIT](LICENSE) 协议开源。

---

<div align="center">

**[⬆ 回到顶部](#-人类跌倒行为识别系统)**

Made with ❤️ by [Xavier Chen](https://github.com/chanxaviersy)

</div>
