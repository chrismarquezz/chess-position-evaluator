# Chess Win-Probability Model

## Goal

Neural net that predicts win/draw/loss probabilities from a board position.
Trained on elite (2400+) classical games. Pure NN on the raw board —
NO Stockfish features, NO engine at train or serve time. Labels are game
outcomes. Calibration is the headline metric. Inspired by ChessDS (which
used LightGBM + Stockfish features); differentiator is end-to-end deep
learning from position alone.

## Data pipeline (done)

- Source: TWIC issues 1600–1652, OTB classical.
- twic/twic_all.pgn — 363,146 raw games (315MB)
- twic/twic_2400_classical.pgn — 46,289 games, filtered to both players
  2400+ and classical time control
- twic/positions.csv — 4,448,070 rows, columns: fen, ply, side_to_move, result
  (result = 1.0 White win / 0.5 draw / 0.0 Black loss, White's perspective)
- Sanity checked: mean result 0.539 (correct white advantage), draws 29.8%
  at position level (fine — decisive games are longer so contribute more rows)

## Stages

1. Gather data — DONE
2. Games -> position+outcome examples — DONE
3. Encode FEN -> tensor (NEXT)
4. Build + train the network
5. Evaluate, esp. calibration (does 70% mean 70%)
6. Wrap in a small web UI with the win-probability ribbon

## Scripts (in twic/)

filter.py, build_dataset.py, sanity.py

## Notes

- venv at chess-ML/.venv, `chess` installed
- 46k games is plenty for building/debugging; scale download toward
  issue 210 later if needed (ChessDS plateaued ~100k games)
- Keep stages in separate files; prove pipeline on a small slice before scaling
