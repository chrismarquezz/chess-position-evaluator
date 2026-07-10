import chess.pgn
import csv

IN = "twic_2400_classical.pgn"
OUT = "positions.csv"

# game result -> label from White's perspective
LABEL = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}

games = 0
rows = 0
with open(IN, encoding="utf-8", errors="replace") as fin, \
     open(OUT, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow(["fen", "ply", "side_to_move", "result"])

    while (game := chess.pgn.read_game(fin)) is not None:
        result = LABEL.get(game.headers.get("Result"))
        if result is None:          # skip anything unexpected
            continue

        board = game.board()        # starting position
        ply = 0
        for move in game.mainline_moves():
            board.push(move)        # make the move
            ply += 1
            # 'w' if White to move in this new position, else 'b'
            side = "w" if board.turn else "b"
            writer.writerow([board.fen(), ply, side, result])
            rows += 1

        games += 1
        if games % 5000 == 0:
            print(f"{games} games -> {rows} positions")

print(f"done: {games} games, {rows} positions -> {OUT}")