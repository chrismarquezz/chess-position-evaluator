"""Stage 4: neural network architecture.

Input:
  board  (B, 12, 8, 8)  uint8 → cast to float32 inside forward
  meta   (B, 7)          float32

Output:
  logits (B, 3)  — [White-win, Draw, Black-win], apply softmax for probabilities
"""

import torch
import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 64 channels × 8 × 8 = 4096, plus 7 meta features
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 3),
        )

    def forward(self, board: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = self.conv(board.float() / 1.0)   # already 0/1 uint8 → float32
        x = x.flatten(1)
        x = torch.cat([x, meta], dim=1)
        return self.head(x)
