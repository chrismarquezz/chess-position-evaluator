"""Stage 5: calibration evaluation on the held-out test set.

Loads checkpoints/best.pt and data/test_idx.npy.
Outputs: printed tables + calibration.png
"""

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ChessDataset
from model import ChessNet

CKPT    = "checkpoints/best.pt"
CSV_IN  = "twic/positions.csv"
N_BINS  = 10
BATCH   = 2048
FIG_OUT = "calibration.png"


def load_ply_data():
    """One pass over positions.csv: returns per-position ply and plies-from-end."""
    game_id, plies = [], []
    g = -1
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ply = int(row["ply"])
            if ply == 1:
                g += 1
            game_id.append(g)
            plies.append(ply)

    game_id = np.array(game_id, dtype=np.int32)
    plies   = np.array(plies,   dtype=np.int32)

    max_ply_per_game = np.zeros(game_id[-1] + 1, dtype=np.int32)
    np.maximum.at(max_ply_per_game, game_id, plies)

    plies_from_end = max_ply_per_game[game_id] - plies
    return plies, plies_from_end


def compute_ece(confidences, correct, n_bins=10):
    """ECE: size-weighted mean |confidence - accuracy| across equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        n = in_bin.sum()
        if n == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        mc = confidences[in_bin].mean()
        af = correct[in_bin].mean()
        ece += (n / len(confidences)) * abs(mc - af)
        rows.append((lo, hi, n, mc, af))
    return ece, rows


def ce_smoothed(probs, labels, alpha=0.1, K=3):
    """CrossEntropy with label_smoothing=alpha applied to probability arrays."""
    log_p = np.log(probs + 1e-9)
    true_log_p = log_p[np.arange(len(labels)), labels]
    mean_log_p = log_p.mean(axis=1)
    return float(np.mean(-(1 - alpha) * true_log_p - alpha * mean_log_p))


def report_slice(name, mask, probs, labels):
    n = mask.sum()
    if n == 0:
        return
    sl_p = probs[mask]; sl_l = labels[mask]
    sl_conf = sl_p.max(axis=1)
    sl_corr = (sl_p.argmax(axis=1) == sl_l).astype(float)
    acc  = sl_corr.mean()
    loss = ce_smoothed(sl_p, sl_l)
    ece, _ = compute_ece(sl_conf, sl_corr, N_BINS)
    print(f"  {name:35}  {n:>7,}  {acc:>6.3f}  {loss:>6.3f}  {ece:>6.4f}")


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    print("Loading ply data...", flush=True)
    all_plies, all_plies_from_end = load_ply_data()

    test_idx = np.load("data/test_idx.npy")
    n_test   = len(test_idx)
    print(f"test set: {n_test:,} positions")

    test_plies         = all_plies[test_idx]
    test_plies_from_end = all_plies_from_end[test_idx]

    # load model
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = ChessNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}\n")

    # inference
    print("Running inference...", flush=True)
    ds     = ChessDataset(test_idx)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for board, meta, labels_b in loader:
            board, meta, labels_b = board.to(device), meta.to(device), labels_b.to(device)
            logits = model(board, meta)
            total_loss += nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels_b).item()
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels_b.cpu().numpy())

    probs  = np.concatenate(all_probs)   # (N, 3)
    labels = np.concatenate(all_labels)  # (N,)
    test_loss = total_loss / len(loader)

    confidence = probs.max(axis=1)
    correct    = (probs.argmax(axis=1) == labels).astype(float)
    test_acc   = correct.mean()

    ece_overall, bin_rows = compute_ece(confidence, correct, N_BINS)

    # ── headline metrics ─────────────────────────────────────────────────────
    print("=" * 60)
    print("HEADLINE METRICS — TEST SET")
    print("=" * 60)
    print(f"  accuracy            : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  CE loss (smooth=0.1): {test_loss:.4f}")
    print(f"  ECE                 : {ece_overall:.4f}")
    print(f"  base-rate CE        : 1.0944  (smoothed baseline)")
    print(f"  gap vs baseline     : {test_loss - 1.0944:+.4f}")

    # ── reliability diagram table ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"RELIABILITY DIAGRAM  ({N_BINS} equal-width bins)")
    print(f"{'─'*60}")
    print(f"  {'bin range':>13}  {'count':>8}  {'mean_conf':>9}  {'actual_freq':>11}  {'gap':>6}")
    print(f"  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*6}")
    for lo, hi, n, mc, af in bin_rows:
        if n == 0:
            print(f"  [{lo:.2f}, {hi:.2f})  {'0':>8}  {'—':>9}  {'—':>11}  {'—':>6}")
        else:
            print(f"  [{lo:.2f}, {hi:.2f})  {n:>8,}  {mc:>9.3f}  {af:>11.3f}  {mc-af:>+6.3f}")

    # ── per-slice breakdown ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PER-SLICE BREAKDOWN")
    print(f"{'─'*60}")
    print(f"  {'slice':35}  {'n':>7}  {'acc':>6}  {'loss':>6}  {'ece':>6}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    report_slice("Opening    (ply ≤ 30)",         test_plies <= 30,                             probs, labels)
    report_slice("Middlegame (ply 31–70)",         (test_plies > 30) & (test_plies <= 70),       probs, labels)
    report_slice("Endgame    (ply > 70, not near)", (test_plies > 70) & (test_plies_from_end > 10), probs, labels)
    report_slice("Near-end   (last 11 plies)",     test_plies_from_end <= 10,                   probs, labels)

    # ── confidence distribution ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("CONFIDENCE DISTRIBUTION  (max predicted probability)")
    print(f"{'─'*60}")
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    vs = np.percentile(confidence, qs)
    for q, v in zip(qs, vs):
        print(f"  p{q:>3}: {v:.3f}")
    print(f"  fraction < 0.40 : {(confidence < 0.40).mean():.1%}")
    print(f"  fraction < 0.50 : {(confidence < 0.50).mean():.1%}")
    print(f"  fraction < 0.60 : {(confidence < 0.60).mean():.1%}")
    print(f"  fraction ≥ 0.80 : {(confidence >= 0.80).mean():.1%}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # reliability diagram
    valid_bins  = [(mc, af, n) for _, _, n, mc, af in bin_rows if n > 0]
    bmc, baf, bn = zip(*valid_bins)
    max_n = max(bn)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1.5, label="Perfect calibration")
    ax1.scatter(bmc, baf, s=[300 * n / max_n for n in bn],
                c=bmc, cmap="Blues", edgecolors="steelblue", linewidths=1.5, zorder=3)
    for mc, af, n in valid_bins:
        ax1.annotate(f"{n/1000:.0f}k", (mc, af), textcoords="offset points",
                     xytext=(4, 4), fontsize=7, color="gray")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Actual frequency (accuracy)")
    ax1.set_title(f"Reliability diagram — ECE = {ece_overall:.4f}")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)

    # confidence histogram
    ax2.hist(confidence, bins=40, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax2.axvline(1/3, color="red", linestyle="--", lw=1.5, label="Chance (0.333)")
    ax2.set_xlabel("Max predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {FIG_OUT}")


if __name__ == "__main__":
    main()
