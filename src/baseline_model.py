import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.net(x).squeeze(-1)  # [B]
