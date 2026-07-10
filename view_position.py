import numpy as np
import json
import chess

# load the metadata so we know shapes/dtypes
meta = json.load(open("data/meta.json"))
N = meta["shapes"]["X_board"][0] if "shapes" in meta else None

# memmap the big board file (don't load all 3.4GB), plain-load the small ones
X_board = np.memmap("data/X_board.bin", dtype="uint8", mode="r",
                    shape=(4448070, 12, 8, 8))
X_meta = np.load("data/X_meta.npy")
y = np.load("data/y.npy")

# --- pick which position to look at ---
i = 0                      # change this to inspect any row
board_planes = X_board[i]  # shape (12, 8, 8)
meta_vec = X_meta[i]       # shape (7,)
label = y[i]

PIECES = ["P", "N", "B", "R", "Q", "K"]   # plane order: white 0-5, black 6-11

print(f"=== position {i} ===")
print("label (side-to-move perspective):", label)
print("meta vector:", meta_vec)
print()

# 1) show each of the 12 planes as an 8x8 grid of 0/1
for p in range(12):
    color = "white" if p < 6 else "black"
    name = PIECES[p % 6]
    plane = board_planes[p]
    if plane.sum() == 0:
        continue                      # skip empty planes to reduce clutter
    print(f"plane {p:2d}  {color} {name}  (count={int(plane.sum())})")
    for rank in plane:
        print("   " + " ".join(str(int(c)) for c in rank))
    print()

# 2) collapse all 12 planes back into one human-readable board
print("=== reconstructed board (from the planes) ===")
symbols = ["P","N","B","R","Q","K","p","n","b","r","q","k"]
grid = [["." for _ in range(8)] for _ in range(8)]
for p in range(12):
    for r in range(8):
        for c in range(8):
            if board_planes[p][r][c] == 1:
                grid[r][c] = symbols[p]
for row in grid:
    print("   " + " ".join(row))