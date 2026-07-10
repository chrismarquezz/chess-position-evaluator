import { useState, useEffect, useCallback } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, Legend, ResponsiveContainer,
} from 'recharts'

const API       = 'http://localhost:8000'
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

async function fetchPred(fen) {
  const res = await fetch(`${API}/predict`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ fen }),
  })
  if (!res.ok) throw new Error(`Server ${res.status}`)
  return res.json()
}

function ProbBar({ pred }) {
  if (!pred) return <div className="prob-bar-empty">waiting for server…</div>
  const { white_win, draw, black_win } = pred
  const fmt = v => `${(v * 100).toFixed(1)}%`
  return (
    <div className="prob-bar">
      <div className="seg seg-white" style={{ flex: white_win }}>{white_win > 0.15 && fmt(white_win)}</div>
      <div className="seg seg-draw"  style={{ flex: draw      }}>{draw       > 0.15 && fmt(draw)}</div>
      <div className="seg seg-black" style={{ flex: black_win }}>{black_win  > 0.15 && fmt(black_win)}</div>
    </div>
  )
}

function ProbLabels({ pred }) {
  const fmt = (v) => pred ? `${(v * 100).toFixed(1)}%` : '—'
  return (
    <div className="prob-labels">
      <span><span className="dot white-dot" />White {fmt(pred?.white_win)}</span>
      <span><span className="dot draw-dot"  />Draw  {fmt(pred?.draw)}</span>
      <span><span className="dot black-dot" />Black {fmt(pred?.black_win)}</span>
    </div>
  )
}

export default function App() {
  // history: [{fen, pred}]; idx: current position
  const [history, setHistory] = useState([{ fen: START_FEN, pred: null }])
  const [idx, setIdx]         = useState(0)
  const [pgn, setPgn]         = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const cur = history[idx]

  // fetch starting position on mount
  useEffect(() => {
    fetchPred(START_FEN)
      .then(pred => setHistory([{ fen: START_FEN, pred }]))
      .catch(() => {})
  }, [])

  // arrow key navigation
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowLeft')  setIdx(i => Math.max(0, i - 1))
      if (e.key === 'ArrowRight') setIdx(i => Math.min(history.length - 1, i + 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [history.length])

  // load PGN: fetch predictions for all positions in parallel
  const loadPgn = useCallback(async () => {
    if (!pgn.trim()) return
    setLoading(true)
    setError('')
    try {
      const game = new Chess()
      game.loadPgn(pgn.trim())

      const c = new Chess()
      const fens = [c.fen()]
      for (const move of game.history()) {
        c.move(move)
        fens.push(c.fen())
      }

      const preds = await Promise.all(fens.map(fetchPred))
      setHistory(fens.map((fen, i) => ({ fen, pred: preds[i] })))
      setIdx(0)
    } catch (e) {
      setError(e.message)
    }
    setLoading(false)
  }, [pgn])

  // interactive drag-and-drop: branch from current position
  const onDrop = useCallback(async (from, to) => {
    const game = new Chess(cur.fen)
    const move = game.move({ from, to, promotion: 'q' })
    if (!move) return false

    const fen  = game.fen()
    const pred = await fetchPred(fen).catch(() => null)
    const next = [...history.slice(0, idx + 1), { fen, pred }]
    setHistory(next)
    setIdx(next.length - 1)
    return true
  }, [cur?.fen, history, idx])

  // ribbon data — stack: black (bottom) / draw / white (top)
  const ribbonData = history.map(({ pred }, i) => ({
    ply:   i,
    black: pred ? +(pred.black_win * 100).toFixed(1) : null,
    draw:  pred ? +(pred.draw      * 100).toFixed(1) : null,
    white: pred ? +(pred.white_win * 100).toFixed(1) : null,
  }))

  return (
    <div className="app">
      <header>
        <h1>Chess Win Probability</h1>
        <p className="subtitle">Neural net · 46k elite games (2400+ Elo) · end-to-end from raw board</p>
      </header>

      <div className="pgn-row">
        <textarea
          value={pgn}
          onChange={e => setPgn(e.target.value)}
          placeholder="Paste PGN to replay a game — or drag pieces to play interactively"
          rows={3}
        />
        <button onClick={loadPgn} disabled={loading || !pgn.trim()}>
          {loading ? 'Loading…' : 'Load PGN'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}

      <div className="main-layout">
        {/* ── board panel ── */}
        <div className="board-panel">
          <Chessboard
            position={cur?.fen ?? START_FEN}
            onPieceDrop={onDrop}
            boardWidth={380}
            customBoardStyle={{ borderRadius: 6, boxShadow: '0 4px 20px rgba(0,0,0,.4)' }}
          />

          <div className="nav">
            <button onClick={() => setIdx(0)}                                     disabled={idx === 0}>⏮</button>
            <button onClick={() => setIdx(i => Math.max(0, i - 1))}              disabled={idx === 0}>◀</button>
            <span className="ply-label">Ply {idx} / {history.length - 1}</span>
            <button onClick={() => setIdx(i => Math.min(history.length-1, i+1))} disabled={idx === history.length - 1}>▶</button>
            <button onClick={() => setIdx(history.length - 1)}                    disabled={idx === history.length - 1}>⏭</button>
          </div>

          <ProbBar    pred={cur?.pred} />
          <ProbLabels pred={cur?.pred} />
        </div>

        {/* ── ribbon panel ── */}
        <div className="ribbon-panel">
          <h3>Win Probability Ribbon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={ribbonData} margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis
                dataKey="ply"
                tick={{ fontSize: 11, fill: '#aaa' }}
                label={{ value: 'Ply', position: 'insideBottom', offset: -14, fill: '#aaa', fontSize: 12 }}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={v => `${Math.round(v)}%`}
                tick={{ fontSize: 11, fill: '#aaa' }}
                width={38}
              />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #444', borderRadius: 6 }}
                labelStyle={{ color: '#ccc' }}
                formatter={(v, name) => [`${Number(v).toFixed(1)}%`, name]}
                labelFormatter={p => `Ply ${p}`}
              />
              <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={idx} stroke="#f0c040" strokeWidth={2} strokeDasharray="4 3" />
              {/* stacked bottom→top: black / draw / white */}
              <Area type="monotone" dataKey="black" stackId="1" name="Black win" stroke="#333" fill="#444" />
              <Area type="monotone" dataKey="draw"  stackId="1" name="Draw"      stroke="#777" fill="#999" />
              <Area type="monotone" dataKey="white" stackId="1" name="White win" stroke="#bbb" fill="#ddd" />
            </AreaChart>
          </ResponsiveContainer>
          <p className="hint">← → arrow keys · drag pieces to branch</p>
        </div>
      </div>
    </div>
  )
}
