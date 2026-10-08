"""模型定义：CNN + LSTM + Attention 混合架构。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """时序维度的注意力池化。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        scores = self.attn(x).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)
        # 加权求和
        out = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return out


class FallDetectionModel(nn.Module):
    """跌倒检测模型：Conv1D → LSTM → Attention → Classifier
    输入: (batch, seq_len, joint*coord)
    输出: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        # 1D-CNN：提取每个时间步的局部特征
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        # LSTM：时序建模
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention
        self.attention = AttentionPool(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        # Conv1D 期望输入是 (batch, channels, seq_len)，所以先转置
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.dropout_cnn(x)
        x = self.relu(self.conv2(x))
        x = self.dropout_cnn(x)
        # 转回 (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # LSTM
        out, _ = self.lstm(x)

        # Attention pool
        pooled = self.attention(out)

        # Classifier
        logits = self.classifier(pooled)
        return logits