"""LSTM variance forecaster trained with the Gaussian likelihood (QLIKE) loss.

Walk-forward: at the start of each calendar year the network is retrained on every day
before it (the last 15% held out for early stopping) and then forecasts that year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import har_features

SEQ = 22


def features(r: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Per-day inputs known at the close of day t."""
    h = har_features(r, vix)
    return pd.DataFrame({
        "r": r,
        "log_d": np.log(h["d"] + 1e-2),
        "log_w": np.log(h["w"] + 1e-2),
        "log_m": np.log(h["m"] + 1e-2),
        "log_vix": np.log(h["vix"]),
    }, index=r.index)


class Net(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # -> log variance
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def qlike_loss(log_h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood up to constants: log h + y / h."""
    return (log_h + y * torch.exp(-log_h)).mean()


def _windows(F: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Input sequences ending at day t-1 for each target day t in idx."""
    return np.stack([F[t - SEQ:t] for t in idx])


def _train(X, y, Xv, yv, seed: int, device: str, epochs: int = 60, patience: int = 6) -> Net:
    torch.manual_seed(seed)
    net = Net(X.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    X, y = torch.tensor(X, device=device), torch.tensor(y, device=device)
    Xv, yv = torch.tensor(Xv, device=device), torch.tensor(yv, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(X), generator=gen).split(256):
            opt.zero_grad()
            loss = qlike_loss(net(X[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = qlike_loss(net(Xv), yv).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


def lstm(r: pd.Series, vix: pd.Series, start: int, seeds=(0, 1, 2), device: str | None = None) -> pd.Series:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    F_df = features(r, vix)
    first = int(F_df.notna().all(axis=1).to_numpy().argmax())  # first row with every feature
    y_all = (r ** 2).to_numpy(np.float32)
    out = pd.Series(np.nan, index=r.index)
    years = sorted(set(r.index[start:].year))
    for year in years:
        test = np.where(r.index.year == year)[0]
        test = test[test >= start]
        if len(test) == 0:
            continue
        T = test[0]
        train_idx = np.arange(first + SEQ, T)
        # standardise with training statistics only
        mu, sd = F_df.iloc[first:T].mean(), F_df.iloc[first:T].std()
        F = ((F_df - mu) / sd).to_numpy(np.float32)
        cut = int(len(train_idx) * 0.85)
        tr, va = train_idx[:cut], train_idx[cut:]
        preds = []
        for seed in seeds:
            net = _train(_windows(F, tr), y_all[tr], _windows(F, va), y_all[va], seed, device)
            with torch.no_grad():
                preds.append(torch.exp(net(torch.tensor(_windows(F, test), device=device))).cpu().numpy())
        out.iloc[test] = np.mean(preds, axis=0)
    return out
