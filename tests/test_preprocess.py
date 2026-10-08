"""测试 preprocess.py"""
from __future__ import annotations

import numpy as np


def test_normalize_skeleton_shape(sample_skeleton_sequence):
    """归一化后 shape 不变"""
    from preprocess import normalize_skeleton

    result = normalize_skeleton(sample_skeleton_sequence)
    assert result.shape == sample_skeleton_sequence.shape


def test_normalize_skeleton_dtype(sample_skeleton_sequence):
    """归一化后 dtype 应为 float32"""
    from preprocess import normalize_skeleton

    result = normalize_skeleton(sample_skeleton_sequence)
    assert result.dtype == np.float32


def test_preprocess_dataset_shape(sample_skeleton_dataset):
    """预处理后 shape 应保持 (N, T, J, C)，但被平展后 reshape 回来"""
    from preprocess import preprocess_dataset

    X, _ = sample_skeleton_dataset
    result = preprocess_dataset(X)
    # 输出在源码中 reshape 回去 (N, T, J*C) → (N, T, J, C)
    assert result.shape == (10, 30, 17, 2)
    assert result.dtype == np.float32


def test_preprocess_dataset_normalized(sample_skeleton_dataset):
    """预处理后整体应接近均值为 0"""
    from preprocess import preprocess_dataset

    X, _ = sample_skeleton_dataset
    result = preprocess_dataset(X)
    # StandardScaler 后的均值应接近 0
    assert abs(result.mean()) < 1.0
