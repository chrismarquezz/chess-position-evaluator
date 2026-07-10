"""Stage 4: PyTorch Dataset for encoded chess positions.

Reads the memmap/npy files written by encode.py.
Labels (y) map: 1.0 → class 0 (White win), 0.5 → class 1 (Draw), 0.0 → class 2 (Black win).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = "data"

# y value → class index
RESULT_TO_CLASS = {1.0: 0, 0.5: 1, 0.0: 2}


class ChessDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
        n = meta["n_samples"]

        self.board = np.memmap(
            os.path.join(DATA_DIR, "X_board.bin"),
            dtype="uint8", mode="r", shape=(n, 12, 8, 8),
        )
        self.meta  = np.load(os.path.join(DATA_DIR, "X_meta.npy"))
        y_raw      = np.load(os.path.join(DATA_DIR, "y.npy"))

        self.indices = indices
        # pre-map float labels to integer class indices
        self.labels = np.array(
            [RESULT_TO_CLASS[round(float(v), 1)] for v in y_raw[indices]],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pos = self.indices[idx]
        board = torch.from_numpy(self.board[pos].copy())   # (12,8,8) uint8
        meta  = torch.from_numpy(self.meta[pos].copy())    # (7,)  float32
        label = self.labels[idx]                            # int64 scalar
        return board, meta, label
