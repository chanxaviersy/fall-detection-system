# 详细使用指南

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 推荐 GPU（推理可 CPU）

## 安装

```bash
git clone https://github.com/chanxaviersy/fall-detection-system.git
cd fall-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 配置

```bash
cp .env.example .env
# 编辑模型参数、阈值等
```

## 运行

### 生成合成数据并训练

```bash
python run_demo.py
```

流程：生成合成骨架数据 → 训练 → 评估 → 输出报告

### 训练自定义数据

```python
import numpy as np
from src.dataset import SkeletonDataset
from src.model import FallDetectionModel
from torch.utils.data import DataLoader

# 准备数据 (N, T, J, C) 和标签 (N,)
X = np.load("my_skeletons.npy")
y = np.load("my_labels.npy")

ds = SkeletonDataset(X, y)
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = FallDetectionModel(input_dim=34, num_classes=4)
# ... 训练循环
```

### 推理单样本

```python
import torch
from src.model import FallDetectionModel
from src.infer import predict_single

model = FallDetectionModel(input_dim=34, num_classes=4)
model.load_state_dict(torch.load("models/best.pt"))
model.eval()

# 1 个 30 帧的骨架序列
skeleton = np.random.randn(30, 17, 2)
result = predict_single(model, skeleton, confidence_threshold=0.85)
print(result)  # {'class': 3, 'confidence': 0.92, 'is_fall': True}
```

## 常见问题

**Q: MediaPipe 装不上？**
A: 用 `pip install mediapipe` 或换 `mediapipe-silicon`（Apple Silicon）。

**Q: 推理延迟太高？**
A: 用 `torch.jit.script` 转 TorchScript，或导出 ONNX。

**Q: 误报太多？**
A: 调高 `CONFIDENCE_THRESHOLD`（如 0.9）或用滑动窗口投票（连续 3 帧都是"跌倒"才告警）。

## 部署到边缘

```python
# 导出 ONNX
dummy_input = torch.randn(1, 30, 34)
torch.onnx.export(model, dummy_input, "model.onnx")

# 在 Jetson / Raspberry Pi 上用 ONNX Runtime 推理
```
