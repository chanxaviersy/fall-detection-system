"""误报率 / 漏报率专项评估模块。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_fall_detection(
    y_true: np.ndarray, y_pred: np.ndarray, fall_class: int = 3
) -> dict[str, float]:
    """跌倒检测专项评估：把问题简化成"跌倒 vs 非跌倒"二分类。

    返回关键指标：Accuracy、Precision、Recall、F1、FPR、FNR
    """
    binary_true = (y_true == fall_class).astype(int)
    binary_pred = (y_pred == fall_class).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_true, binary_pred, average="binary", zero_division=0
    )

    # 误报率 (False Positive Rate)
    n_neg = (binary_true == 0).sum()
    fp = ((binary_pred == 1) & (binary_true == 0)).sum()
    fpr = fp / n_neg if n_neg > 0 else 0.0

    # 漏报率 (False Negative Rate)
    n_pos = (binary_true == 1).sum()
    fn = ((binary_pred == 0) & (binary_true == 1)).sum()
    fnr = fn / n_pos if n_pos > 0 else 0.0

    accuracy = (y_true == y_pred).mean()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def print_full_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """打印完整的分类报告与混淆矩阵。"""
    print("\n========== 分类报告 ==========")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("========== 混淆矩阵 ==========")
    cm = confusion_matrix(y_true, y_pred)
    header = "         " + "  ".join([f"{n:>7s}" for n in class_names])
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>9s} " + "  ".join([f"{v:>7d}" for v in row]))