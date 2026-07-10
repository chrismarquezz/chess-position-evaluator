#!/usr/bin/env python3
"""Stage 3: encode FEN positions to tensors.

Reads twic/positions.csv and encodes each row as:
  board  (12, 8, 8) uint8  — 6 piece types × 2 colors, planes 0-5 White, 6-11 Black
  meta   (7,)       float32 — side_to_move, wk/wq/bk/bq castling, ep_file, halfmove_clock
  y      float32            — game result from White's perspective (1.0 / 0.5 / 0.0)

Output in data/:
  X_board.bin   raw memmap uint8  shape (N, 12, 8, 8)
  X_meta.npy    float32           shape (N, 7)
  y.npy         float32           shape (N,)
  meta.json     {"n_samples": N, "board_shape": [12,8,8], "meta_features": 7}
"""

import csv
import json
import os
import sys

import numpy as np
import chess

CSV_IN   = "twic/positions.csv"
DATA_DIR = "data"

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_fen(fen: str):
    board = chess.Board(fen)

    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    for ci, color in enumerate((chess.WHITE, chess.BLACK)):
        for pi, pt in enumerate(PIECE_TYPES):
            ch = ci * 6 + pi
            for sq in board.pieces(pt, color):
                planes[ch, sq >> 3, sq & 7] = 1

    ep = board.ep_square
    meta = np.array([
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        float(chess.square_file(ep)) if ep is not None else -1.0,
        board.halfmove_clock / 100.0,
    ], dtype=np.float32)

    return planes, meta


def count_rows(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n - 1  # subtract header


def main():
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # optional row cap for testing

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Counting rows...", flush=True)
    n_total = count_rows(CSV_IN)
    n = n_limit if n_limit else n_total
    print(f"  {n_total:,} total, encoding {n:,}", flush=True)

    board_mm = np.memmap(
        os.path.join(DATA_DIR, "X_board.bin"),
        dtype="uint8", mode="w+", shape=(n, 12, 8, 8),
    )
    X_meta = np.empty((n, 7),  dtype=np.float32)
    y      = np.empty(n,       dtype=np.float32)

    with open(CSV_IN, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            planes, meta = encode_fen(row["fen"])
            board_mm[i] = planes
            X_meta[i]   = meta
            y[i]        = float(row["result"])

            if (i + 1) % 200_000 == 0:
                print(f"  {i+1:,} / {n:,}  ({100*(i+1)/n:.0f}%)", flush=True)

    board_mm.flush()
    np.save(os.path.join(DATA_DIR, "X_meta.npy"), X_meta)
    np.save(os.path.join(DATA_DIR, "y.npy"),      y)
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"n_samples": n, "board_shape": [12, 8, 8], "meta_features": 7}, f, indent=2)

    board_gb = n * 12 * 8 * 8 / 1e9
    meta_mb  = n * 7 * 4 / 1e6
    print(f"\nDone -> {DATA_DIR}/")
    print(f"  X_board.bin  {board_gb:.2f} GB  (uint8, {n}×12×8×8)")
    print(f"  X_meta.npy   {meta_mb:.1f} MB  (float32, {n}×7)")
    print(f"  y.npy        {n*4/1e6:.1f} MB  (float32, {n})")


if __name__ == "__main__":
    main()
