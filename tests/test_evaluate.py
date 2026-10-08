"""测试 evaluate.py"""
from __future__ import annotations

import numpy as np
import pytest


def test_evaluate_fall_detection_returns_dict(sample_predictions):
    """应返回包含 6 个指标的字典"""
    from evaluate import evaluate_fall_detection

    y_true, y_pred = sample_predictions
    metrics = evaluate_fall_detection(y_true, y_pred, fall_class=3)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics


def test_evaluate_fall_detection_perfect():
    """完美预测时所有指标应为 1.0 / 0.0"""
    from evaluate import evaluate_fall_detection

    y_true = np.array([0, 1, 2, 3, 3, 3, 0, 1])
    metrics = evaluate_fall_detection(y_true, y_true, fall_class=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_evaluate_fall_detection_fpr_calculation():
    """FPR 应正确计算"""
    from evaluate import evaluate_fall_detection

    # 3 个负样本中 1 个被误判为正 → FPR = 1/3
    y_true = np.array([0, 0, 0, 3])
    y_pred = np.array([3, 0, 0, 3])  # 第一个 0 被错判为 3
    metrics = evaluate_fall_detection(y_true, y_pred, fall_class=3)
    assert pytest.approx(metrics["fpr"], abs=0.01) == 1 / 3


def test_evaluate_fall_detection_fnr_calculation():
    """FNR 应正确计算"""
    from evaluate import evaluate_fall_detection

    # 2 个正样本中 1 个被漏报 → FNR = 1/2
    y_true = np.array([0, 3, 3])
    y_pred = np.array([0, 0, 3])  # 第二个 3 被漏报
    metrics = evaluate_fall_detection(y_true, y_pred, fall_class=3)
    assert pytest.approx(metrics["fnr"], abs=0.01) == 0.5


def test_evaluate_no_positive_samples():
    """没有正样本时不应崩"""
    from evaluate import evaluate_fall_detection

    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])
    metrics = evaluate_fall_detection(y_true, y_pred, fall_class=3)
    # FNR 分母为 0，应返回 0.0
    assert metrics["fnr"] == 0.0
