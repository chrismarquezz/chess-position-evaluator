# Chess Position Evaluator

A neural network that predicts win/draw/loss probabilities from a raw chess board position, trained end-to-end on elite human games with no engine features.

---

## Overview

Chess Position Evaluator learns to output calibrated outcome probabilities directly from board position, with no Stockfish evaluation or hand-engineered features in the loop. It is trained exclusively on classical games between 2400+ Elo players, so it models realistic human outcomes rather than objective engine assessments.

This distinguishes it from engine-based approaches: Stockfish computes the theoretically best-play evaluation; Chess Position Evaluator estimates the probability distribution over *actual* game results — a different and complementary signal, especially useful for understanding how humans handle a position.

---

## Results

| Metric | Value |
|---|---|
| Training positions | 4.45M (from 46,289 elite games) |
| Raw games filtered | 363K TWIC games → 2400+ Elo classical only |
| Parameters | 551K |
| Test accuracy | **47.9%** (majority-class baseline: 38.2%) |
| Expected Calibration Error (ECE) | **0.038** (~4% probability error) |
| Cross-entropy loss (base-rate → test) | 1.094 → **1.035** |

The model is modest in scale but well-calibrated: its probability estimates are honest to within ~4%, which is the primary design goal.

---

## Approach

- **Data pipeline** — PGN parsing and filtering via `python-chess`; positions encoded as 12-plane binary board tensors (one plane per piece type × color), castling rights, and side-to-move flag.
- **Train/val/test split at the game level** — all positions from a given game land in the same split, preventing data leakage from correlated board states.
- **Regularization** — label smoothing, dropout, and weight decay to address overfitting on repeated opening positions.
- **Calibration as the primary metric** — ECE is tracked alongside accuracy; a model that outputs confident but wrong probabilities is considered a failure.

---

## Tech Stack

Python · PyTorch · FastAPI · React · TypeScript · NumPy · python-chess

---

## Running It

**Backend** (FastAPI, port 8000):

```bash
pip install -r requirements.txt   # if not already done
uvicorn app:app --reload
```

**Frontend** (Vite/React, port 5173):

```bash
cd frontend
npm install                        # first time only
npm run dev
```

Open `http://localhost:5173` — the frontend proxies position FENs to the backend and displays the predicted win/draw/loss probabilities.
