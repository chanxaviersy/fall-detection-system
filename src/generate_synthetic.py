"""生成模拟骨架数据（无需真实视频）。
每条数据是一段 30 帧的人体骨架序列，每帧包含 17 个关键点的 (x, y) 坐标。

类别：
- 0: 站立（standing）
- 1: 行走（walking）
- 2: 坐下（sitting）
- 3: 跌倒（falling）

注意：跌倒动作会在中间几帧出现质心快速下降 + 身体倾斜的特征。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# 17 个关键点（COCO 骨架格式）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def generate_standing(seq_len: int = 30, noise: float = 0.02) -> np.ndarray:
    """站立：基本不动的小幅晃动。"""
    base = np.array([
        [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.47, 0.09], [0.53, 0.09],
        [0.42, 0.25], [0.58, 0.25], [0.38, 0.45], [0.62, 0.45],
        [0.35, 0.6], [0.65, 0.6], [0.45, 0.55], [0.55, 0.55],
        [0.43, 0.75], [0.57, 0.75], [0.42, 0.95], [0.58, 0.95],
    ])
    return base + np.random.normal(0, noise, (seq_len, 17, 2))


def generate_walking(seq_len: int = 30, noise: float = 0.03) -> np.ndarray:
    """行走：下肢周期性摆动 + 整体水平位移。"""
    base = generate_standing(seq_len, noise=noise * 0.5)
    sequence = np.zeros_like(base)
    for t in range(seq_len):
        swing = 0.1 * np.sin(2 * np.pi * t / 15)
        seq = base[t].copy()
        # 下肢关键点周期摆动
        seq[13, 0] += swing  # left_knee
        seq[15, 0] += swing * 1.2  # left_ankle
        seq[14, 0] -= swing  # right_knee
        seq[16, 0] -= swing * 1.2  # right_ankle
        # 整体水平位移
        seq[:, 0] += t * 0.005
        sequence[t] = seq
    return sequence


def generate_sitting(seq_len: int = 30, noise: float = 0.02) -> np.ndarray:
    """坐下：质心下降、髋部与膝部接近同一高度。"""
    base = np.array([
        [0.5, 0.2], [0.48, 0.18], [0.52, 0.18], [0.47, 0.19], [0.53, 0.19],
        [0.42, 0.4], [0.58, 0.4], [0.38, 0.55], [0.62, 0.55],
        [0.35, 0.7], [0.65, 0.7], [0.45, 0.65], [0.55, 0.65],
        [0.43, 0.85], [0.57, 0.85], [0.42, 0.95], [0.58, 0.95],
    ])
    sequence = np.zeros((seq_len, 17, 2))
    for t in range(seq_len):
        # 渐变下降
        ratio = min(t / (seq_len - 1), 1.0)
        standing = generate_standing(1, noise=0).reshape(17, 2)
        sequence[t] = standing * (1 - ratio * 0.4) + base * ratio * 0.4
        sequence[t] += np.random.normal(0, noise, (17, 2))
    return sequence


def generate_falling(seq_len: int = 30, noise: float = 0.03) -> np.ndarray:
    """跌倒：中间帧出现身体倾斜 + 质心骤降。"""
    base = generate_standing(seq_len, noise=noise * 0.5)
    fall_start = seq_len // 3
    fall_end = 2 * seq_len // 3

    for t in range(fall_start, fall_end):
        progress = (t - fall_start) / (fall_end - fall_start)
        # 整体下移 + 旋转（通过 x 坐标倾斜模拟）
        base[t, :, 1] += progress * 0.5
        base[t, :, 0] += progress * 0.2  # 倾斜
    # 落地后保持躺倒姿势
    for t in range(fall_end, seq_len):
        base[t, :, 1] += 0.5
        base[t, :, 0] += 0.2
    return base + np.random.normal(0, noise, base.shape)


def generate_dataset(
    n_samples_per_class: int = 200,
    seq_len: int = 30,
    save_dir: str | Path = "data",
) -> None:
    """生成完整的模拟数据集并保存为 .npz。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("standing", generate_standing),
        ("walking", generate_walking),
        ("sitting", generate_sitting),
        ("falling", generate_falling),
    ]

    X_list, y_list = [], []
    for class_id, (_, gen) in enumerate(generators):
        print(f"[INFO] 生成 {n_samples_per_class} 个 class={class_id} 样本...")
        for _ in range(n_samples_per_class):
            X_list.append(gen(seq_len=seq_len))
            y_list.append(class_id)

    X = np.stack(X_list).astype(np.float32)  # (N, seq_len, 17, 2)
    y = np.array(y_list, dtype=np.int64)

    # 划分训练/测试
    n = len(X)
    idx = np.random.permutation(n)
    train_size = int(n * 0.8)
    train_idx, test_idx = idx[:train_size], idx[train_size:]

    np.savez(
        save_dir / "fall_dataset.npz",
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        keypoint_names=np.array(KEYPOINT_NAMES),
    )
    print(f"[INFO] 数据已保存到 {save_dir / 'fall_dataset.npz'}")
    print(f"[INFO] 训练集: {len(train_idx)} 条，测试集: {len(test_idx)} 条")


if __name__ == "__main__":
    generate_dataset(n_samples_per_class=200)