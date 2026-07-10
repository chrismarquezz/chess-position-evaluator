"""Build train/val/test splits grouped by game.

All positions from a single game stay in the same split.
Writes:
  data/train_idx.npy       — all training positions
  data/train_idx_thin.npy  — every 5th ply per training game (~19 pos/game)
  data/val_idx.npy
  data/test_idx.npy
"""

import csv
import numpy as np

CSV_IN       = "twic/positions.csv"
SEED         = 42
THIN_STEP    = 5   # keep ply where (ply - 1) % THIN_STEP == 0  →  ply 1, 6, 11, …

def main():
    print("Scanning positions.csv for game boundaries...", flush=True)

    game_id = []
    plies   = []
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
    n_pos   = len(game_id)
    n_games = game_id[-1] + 1
    print(f"  {n_pos:,} positions across {n_games:,} games")

    # shuffle at the game level
    rng = np.random.default_rng(SEED)
    game_order = rng.permutation(n_games)

    t = int(0.80 * n_games)
    v = int(0.90 * n_games)
    train_games = set(game_order[:t].tolist())
    val_games   = set(game_order[t:v].tolist())
    test_games  = set(game_order[v:].tolist())

    pos_idx   = np.arange(n_pos)
    train_idx = pos_idx[np.isin(game_id, list(train_games))]
    val_idx   = pos_idx[np.isin(game_id, list(val_games))]
    test_idx  = pos_idx[np.isin(game_id, list(test_games))]

    # thinned training set: one position every THIN_STEP plies per game
    thin_mask      = np.isin(game_id, list(train_games)) & ((plies - 1) % THIN_STEP == 0)
    train_idx_thin = pos_idx[thin_mask]

    np.save("data/train_idx.npy",      train_idx)
    np.save("data/train_idx_thin.npy", train_idx_thin)
    np.save("data/val_idx.npy",        val_idx)
    np.save("data/test_idx.npy",       test_idx)

    thin_per_game = len(train_idx_thin) / t
    print(f"  train (full): {len(train_idx):,} positions ({t:,} games)")
    print(f"  train (thin): {len(train_idx_thin):,} positions (~{thin_per_game:.0f}/game, every {THIN_STEP}th ply)")
    print(f"  val:          {len(val_idx):,} positions ({v-t:,} games)  [unchanged]")
    print(f"  test:         {len(test_idx):,} positions ({n_games-v:,} games)  [unchanged]")
    print("Saved to data/train_idx.npy, train_idx_thin.npy, val_idx.npy, test_idx.npy")

if __name__ == "__main__":
    main()
