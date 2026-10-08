"""一键运行 Demo：生成数据 → 训练 → 评估 → 推理。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import SkeletonDataset  # noqa: E402
from evaluate import evaluate_fall_detection, print_full_report  # noqa: E402
from generate_synthetic import generate_dataset  # noqa: E402
from infer import predict_sequence  # noqa: E402
from model import FallDetectionModel  # noqa: E402
from preprocess import preprocess_dataset  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("  人类跌倒行为识别系统 - 一键 Demo")
    print("=" * 60)

    # 1. 生成数据
    data_path = Path("data/fall_dataset.npz")
    if not data_path.exists():
        generate_dataset(n_samples_per_class=200)
    data = np.load(data_path)
    X_train = preprocess_dataset(data["X_train"])
    X_test = preprocess_dataset(data["X_test"])
    y_train, y_test = data["y_train"], data["y_test"]

    # 2. 训练（精简版 20 epoch）
    device = "cpu"
    train_ds = SkeletonDataset(X_train, y_train)
    test_ds = SkeletonDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    model = FallDetectionModel(input_dim=X_train.shape[-1], hidden_dim=64, num_classes=4).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n[DEMO] 开始训练（20 epoch 精简版）...")
    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d}/20 | loss: {loss.item():.4f}")

    # 3. 评估
    print("\n[DEMO] 测试集评估...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(dim=-1).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    class_names = ["standing", "walking", "sitting", "falling"]
    print_full_report(targets, preds, class_names)

    metrics = evaluate_fall_detection(targets, preds, fall_class=3)
    print("========== 跌倒专项指标 ==========")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # 5. 推理 demo
    print("\n[DEMO] 单样本推理...")
    from generate_synthetic import generate_falling, generate_standing, generate_walking, generate_sitting

    for name, gen in [
        ("standing", generate_standing),
        ("walking", generate_walking),
        ("sitting", generate_sitting),
        ("falling", generate_falling),
    ]:
        seq = gen()
        pred_class, confidence = predict_sequence(model, seq, device)
        status = "OK" if pred_class == name else "MISS"
        print(f"  [{status}] 真实: {name:9s} → 预测: {pred_class:9s} | 置信度: {confidence:.2%}")


if __name__ == "__main__":
    main()