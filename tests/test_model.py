"""测试 model.py"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch 未安装，跳过 model 测试")


def test_attention_pool_output_shape():
    """AttentionPool 输出 shape = (batch, hidden_dim)"""
    from model import AttentionPool

    pool = AttentionPool(hidden_dim=128)
    x = torch.randn(4, 30, 128)
    out = pool(x)
    assert out.shape == (4, 128)


def test_fall_detection_model_forward_shape():
    """FallDetectionModel 前向传播输出 shape = (batch, num_classes)"""
    from model import FallDetectionModel

    model = FallDetectionModel(
        input_dim=34,  # 17 joints * 2 coords
        hidden_dim=64,
        num_classes=4,
        num_layers=2,
        dropout=0.3,
    )
    model.eval()
    x = torch.randn(8, 30, 34)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 4)


def test_fall_detection_model_different_configs():
    """不同配置应能正常 forward"""
    from model import FallDetectionModel

    for num_layers, dropout in [(1, 0.0), (2, 0.5), (3, 0.2)]:
        model = FallDetectionModel(
            input_dim=34,
            hidden_dim=32,
            num_classes=2,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.eval()
        x = torch.randn(2, 20, 34)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)
