import chess.pgn

n = 0
with open("twic_2400_classical.pgn", encoding="utf-8") as f:
    while (g := chess.pgn.read_game(f)) is not None:
        n += 1
        if n <= 8:
            h = g.headers
            print(h.get("WhiteElo"), h.get("BlackElo"),
                  h.get("Result"), h.get("Event"))
print("total games:", n)