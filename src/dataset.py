"""PyTorch 数据集封装。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """骨架序列数据集。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X shape: (N, T, J, C) → 我们把它平展为 (N, T, J*C)
        N, T, J, C = X.shape
        self.X = X.reshape(N, T, J * C).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)