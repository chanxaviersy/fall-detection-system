"""推理脚本：对单条序列做预测。"""
from __future__ import annotations

import numpy as np
import torch

from generate_synthetic import generate_falling, generate_standing, generate_walking, generate_sitting
from model import FallDetectionModel
from preprocess import normalize_skeleton


CLASS_NAMES = ["standing", "walking", "sitting", "falling"]


def predict_sequence(model: FallDetectionModel, sequence: np.ndarray, device: str = "cpu") -> tuple[str, float]:
    """对单条序列做预测，返回 (类别名, 置信度)。"""
    model.eval()
    seq = normalize_skeleton(sequence)  # (T, J, C)
    seq = seq.reshape(1, seq.shape[0], -1)  # (1, T, J*C)
    with torch.no_grad():
        logits = model(torch.from_numpy(seq.astype(np.float32)).to(device))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_id = int(probs.argmax())
    return CLASS_NAMES[pred_id], float(probs[pred_id])


def main() -> None:
    device = "cpu"
    model = FallDetectionModel(input_dim=17 * 2, hidden_dim=128, num_classes=4).to(device)
    ckpt_path = "models/best_fall_detector.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] 已加载模型: {ckpt_path}")
    except FileNotFoundError:
        print(f"[WARN] 未找到 {ckpt_path}，使用未训练模型（结果仅供参考）")

    generators = {
        "standing": generate_standing,
        "walking": generate_walking,
        "sitting": generate_sitting,
        "falling": generate_falling,
    }

    print("\n========== 推理 Demo ==========")
    for true_class, gen in generators.items():
        seq = gen()
        pred_class, confidence = predict_sequence(model, seq, device)
        status = "✅" if pred_class == true_class else "❌"
        print(f"  {status} 真实: {true_class:9s} | 预测: {pred_class:9s} | 置信度: {confidence:.2%}")


if __name__ == "__main__":
    main()