import chess.pgn
from collections import Counter
c = Counter()
with open("twic_2400_classical.pgn", encoding="utf-8", errors="replace") as f:
    while (g := chess.pgn.read_game(f)) is not None:
        c[g.headers.get("Result")] += 1
print(c, "draw rate:", c["1/2-1/2"] / sum(c.values()))