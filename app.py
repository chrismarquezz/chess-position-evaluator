"""Stage 6 backend: serve ChessNet win-probability predictions via FastAPI."""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import ChessNet
from encode import encode_fen  # stage-3 encoder — imported verbatim, no changes

# ── load model once at startup ────────────────────────────────────────────────
CKPT   = "checkpoints/best.pt"

ckpt  = torch.load(CKPT, map_location="cpu", weights_only=True)
model = ChessNet().to("cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, device=cpu")

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class FENRequest(BaseModel):
    fen: str


@app.post("/predict")
async def predict(req: FENRequest):
    try:
        # encode_fen returns (planes: uint8 (12,8,8), meta: float32 (7,))
        # Serving-time adaptation: .unsqueeze(0) adds the batch dim that
        # DataLoader provided during training. Encoding logic is unchanged.
        planes, meta = encode_fen(req.fen)
        board_t = torch.from_numpy(planes).unsqueeze(0)  # (1,12,8,8) uint8, cpu
        meta_t  = torch.from_numpy(meta).unsqueeze(0)    # (1,7) float32, cpu

        with torch.no_grad():
            logits = model(board_t, meta_t)
            probs  = torch.softmax(logits, dim=1).numpy()[0]          # (3,)

        return {
            "white_win": float(probs[0]),
            "draw":      float(probs[1]),
            "black_win": float(probs[2]),
            "raw":       probs.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
