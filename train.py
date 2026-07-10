"""Stage 4: train ChessNet on encoded positions.

Splits (game-level, written by make_splits.py): 80% train / 10% val / 10% test.
Loss: CrossEntropy over 3 classes (White-win / Draw / Black-win).
Device: MPS (Apple Silicon) → CPU fallback.
Saves best checkpoint to checkpoints/best.pt.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChessDataset
from model import ChessNet

# ── hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 15
BATCH       = 2048
LR          = 1e-3
NUM_WORKERS = 0          # 0 is safest on macOS with memmap
CKPT_DIR    = "checkpoints"
# ──────────────────────────────────────────────────────────────────────────────


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_acc = 0.0
    with torch.set_grad_enabled(train):
        for board, meta, labels in loader:
            board  = board.to(device)
            meta   = meta.to(device)
            labels = labels.to(device)

            logits = model(board, meta)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)

    return total_loss / len(loader), total_acc / len(loader)


def main():
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"device: {device}")

    meta = json.load(open("data/meta.json"))
    print(f"total positions: {meta['n_samples']:,}")

    train_idx = np.load("data/train_idx_thin.npy")
    val_idx   = np.load("data/val_idx.npy")
    print(f"split  train={len(train_idx):,} (thinned, every 5th ply)  val={len(val_idx):,}  (game-level)")

    train_ds = ChessDataset(train_idx)
    val_ds   = ChessDataset(val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    model     = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}\n")

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 3

    print(f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}  {'time':>6}", flush=True)
    print("-" * 58, flush=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {elapsed:>5.0f}s", flush=True)

        scheduler.step(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": vl_loss, "val_acc": vl_acc}, f"{CKPT_DIR}/best.pt")
            print(f"         ↑ saved best checkpoint (val_loss={vl_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"         early stop: no improvement for {EARLY_STOP_PATIENCE} epochs", flush=True)
                break

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CKPT_DIR}/best.pt")

    print("Test indices are in data/test_idx.npy (written by make_splits.py)")


if __name__ == "__main__":
    main()
