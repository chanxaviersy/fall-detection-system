"""测试 dataset.py"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch 未安装，跳过 dataset 测试")


def test_skeleton_dataset_length(sample_skeleton_dataset):
    """Dataset 长度等于样本数"""
    from dataset import SkeletonDataset

    X, y = sample_skeleton_dataset
    ds = SkeletonDataset(X, y)
    assert len(ds) == len(X)


def test_skeleton_dataset_item_shape(sample_skeleton_dataset):
    """每个样本的 X shape = (T, J*C), y 是标量"""
    from dataset import SkeletonDataset

    X, y = sample_skeleton_dataset
    ds = SkeletonDataset(X, y)
    x, label = ds[0]
    assert x.shape == (30, 17 * 2)
    assert label.dim() == 0  # scalar tensor


def test_skeleton_dataset_values(sample_skeleton_dataset):
    """Dataset 取出的值应与原数据一致"""
    from dataset import SkeletonDataset
    import numpy as np

    X, y = sample_skeleton_dataset
    ds = SkeletonDataset(X, y)
    x, label = ds[5]
    expected_x = X[5].reshape(30, -1).astype(np.float32)
    assert torch.allclose(x, torch.from_numpy(expected_x))
    assert label.item() == y[5]
