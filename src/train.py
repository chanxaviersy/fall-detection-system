"""训练脚本（含严格评估）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SkeletonDataset
from generate_synthetic import generate_dataset
from model import FallDetectionModel
from preprocess import preprocess_dataset


def main(args: argparse.Namespace) -> None:
    # 1. 数据
    data_path = Path("data/fall_dataset.npz")
    if not data_path.exists():
        print("[INFO] 未找到数据，自动生成模拟数据集...")
        generate_dataset(n_samples_per_class=args.n_samples)

    data = np.load(data_path)
    X_train = preprocess_dataset(data["X_train"])
    X_test = preprocess_dataset(data["X_test"])
    y_train, y_test = data["y_train"], data["y_test"]
    print(f"[INFO] 数据形状: X_train={X_train.shape}, X_test={X_test.shape}")

    train_ds = SkeletonDataset(X_train, y_train)
    test_ds = SkeletonDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. 模型
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = FallDetectionModel(
        input_dim=X_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 训练
    # 类别权重：跌倒样本更重要，避免被多数类淹没
    class_counts = np.bincount(y_train, minlength=args.num_classes)
    weights = (1.0 / (class_counts + 1e-6))
    weights = weights / weights.sum() * args.num_classes
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"[INFO] 类别权重: {weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_fpr = 1.0
    save_path = Path("models/best_fall_detector.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 评估
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).argmax(dim=-1).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(y.numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # 重点计算"跌倒 vs 其他"的二分类 FPR
        binary_pred = (all_preds == 3).astype(int)
        binary_true = (all_targets == 3).astype(int)
        # 误报率 = 实际非跌倒但被判为跌倒的比例
        fpr = binary_pred[(binary_true == 0)].mean() if (binary_true == 0).any() else 0.0
        acc = (all_preds == all_targets).mean()
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | FPR: {fpr:.4f}")

        if fpr < best_fpr:
            best_fpr = fpr
            torch.save({"model_state": model.state_dict()}, save_path)

    print(f"[INFO] 最佳 FPR: {best_fpr:.4f}，模型已保存到 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=4)
    main(parser.parse_args())