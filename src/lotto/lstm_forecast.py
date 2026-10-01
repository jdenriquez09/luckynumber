"""
LSTM-based draw-sum forecaster for dynamic sum filtering.

Trains a small LSTM on historical draw sums to predict the next draw's
expected sum range, replacing the static percentile-based sum filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .games import GameConfig


def _draw_sums(df: pd.DataFrame, game_cfg: GameConfig) -> np.ndarray:
    cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    clean = df.dropna(subset=cols).sort_values("draw_date").reset_index(drop=True)
    sums = []
    for _, row in clean.iterrows():
        vals = [v for v in (row[c] for c in cols) if v is not None and not pd.isna(v)]
        sums.append(sum(int(v) for v in vals))
    return np.array(sums, dtype=float)


def train_sum_lstm(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    seq_len: int = 20,
    hidden_size: int = 32,
    epochs: int = 80,
    lr: float = 0.002,
) -> dict | None:
    """
    Train a small LSTM to forecast next draw sum.

    Returns dict with model, scaler params, and seq_len, or None on failure.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    sums = _draw_sums(df, game_cfg)
    if len(sums) < seq_len + 20:
        return None

    mean, std = float(sums.mean()), float(sums.std())
    if std < 1e-6:
        return None
    normed = (sums - mean) / std

    X_seqs, y_seqs = [], []
    for i in range(len(normed) - seq_len):
        X_seqs.append(normed[i : i + seq_len])
        y_seqs.append(normed[i + seq_len])

    X_t = torch.tensor(np.array(X_seqs), dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(np.array(y_seqs), dtype=torch.float32)

    class SumLSTM(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]).squeeze(-1)

    model = SumLSTM(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "seq_len": seq_len,
        "game": game_cfg.id,
    }


def predict_sum_range(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    lstm_result: dict,
    margin_pct: float = 0.12,
) -> dict:
    """
    Predict the next draw's expected sum and a confidence range.

    Returns {"predicted_sum", "sum_lo", "sum_hi", "theoretical_mean"}.
    """
    import torch

    sums = _draw_sums(df, game_cfg)
    seq_len = lstm_result["seq_len"]
    mean, std = lstm_result["mean"], lstm_result["std"]

    recent = sums[-seq_len:]
    normed = (recent - mean) / std
    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    model = lstm_result["model"]
    model.eval()
    with torch.no_grad():
        pred_normed = model(x).item()

    predicted = pred_normed * std + mean
    theoretical_mean = (game_cfg.low_boundary + game_cfg.pool) / 2 * game_cfg.picks
    margin = theoretical_mean * margin_pct

    return {
        "predicted_sum": round(predicted, 1),
        "sum_lo": round(predicted - margin, 1),
        "sum_hi": round(predicted + margin, 1),
        "theoretical_mean": round(theoretical_mean, 1),
    }
