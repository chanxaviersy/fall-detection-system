"""序列预处理：归一化、滑窗。"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_skeleton(sequence: np.ndarray) -> np.ndarray:
    """对单个骨架序列做归一化：以髋部中心为原点、按躯干长度缩放。"""
    seq = sequence.copy()
    # 髋部中心 = (left_hip + right_hip) / 2
    center = (seq[:, 11, :] + seq[:, 12, :]) / 2  # (seq_len, 2)
    seq = seq - center[:, None, :]

    # 躯干长度 = nose 到 hip center 的距离
    torso_len = np.linalg.norm(seq[:, 0, :] - (seq[:, 11, :] + seq[:, 12, :]) / 2, axis=-1)
    torso_len = np.mean(torso_len) + 1e-6
    seq = seq / torso_len
    return seq


def preprocess_dataset(X: np.ndarray) -> np.ndarray:
    """对整个数据集做归一化。
    输入: (N, T, J, C)
    输出: (N, T, J*C)  - 平展为特征向量
    """
    N, T, J, C = X.shape
    X_norm = np.zeros_like(X)
    for i in range(N):
        X_norm[i] = normalize_skeleton(X[i])

    # 整体再做一次标准化
    scaler = StandardScaler()
    X_flat = X_norm.reshape(N, -1)
    X_flat = scaler.fit_transform(X_flat)
    return X_flat.reshape(N, T, J, C).astype(np.float32)