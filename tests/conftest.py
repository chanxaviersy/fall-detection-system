"""pytest 共享 fixtures"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 把 src/ 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_skeleton_sequence():
    """示例骨架序列 (30帧, 17关节, 2坐标) - MediaPipe 风格"""
    np.random.seed(42)
    return np.random.randn(30, 17, 2).astype(np.float32)


@pytest.fixture
def sample_skeleton_dataset():
    """示例骨架数据集 (10 samples, 30 frames, 17 joints, 2 coords)"""
    np.random.seed(42)
    X = np.random.randn(10, 30, 17, 2).astype(np.float32)
    y = np.random.randint(0, 4, size=10)  # 4 类
    return X, y


@pytest.fixture
def sample_predictions():
    """示例预测结果"""
    y_true = np.array([0, 1, 2, 3, 3, 3, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2, 3, 3, 0, 0, 2, 0])
    return y_true, y_pred
