import chess.pgn

FAST = ("blitz", "rapid", "titled tuesday", "armageddon", "bullet")

def is_classical(h):
    tc = h.get("TimeControl", "")
    base = tc.split("+")[0]
    if base.isdigit():
        return int(base) >= 1500          # 25+ min base = classical
    return not any(k in h.get("Event", "").lower() for k in FAST)

def keep(h):
    try:
        we, be = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
    except ValueError:
        return False
    return (we >= 2400 and be >= 2400
            and h.get("Result") in ("1-0", "0-1", "1/2-1/2")
            and is_classical(h))

kept = dropped = 0
with open("twic_all.pgn", encoding="utf-8", errors="replace") as fin, \
     open("twic_2400_classical.pgn", "w", encoding="utf-8") as fout:
    while (game := chess.pgn.read_game(fin)) is not None:
        if keep(game.headers):
            print(game, file=fout, end="\n\n")
            kept += 1
        else:
            dropped += 1
print("kept:", kept, "dropped:", dropped)