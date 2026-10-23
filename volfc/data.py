"""Daily index closes and the VIX from FRED, cached locally.

FRED redistributes these series under their owners' terms (S&P Dow Jones Indices, Nasdaq,
Cboe), so the raw files are downloaded on first use and never committed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

INDICES = {
    "NASDAQCOM": "NASDAQ Composite",
    "SP500": "S&P 500",
}


def fred(series: str, refresh: bool = False) -> pd.Series:
    path = CACHE / f"{series}.csv"
    if refresh or not path.exists():
        resp = requests.get(FRED_URL.format(series=series), timeout=60)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
    df = pd.read_csv(path, parse_dates=["observation_date"], na_values=["."])
    return df.set_index("observation_date").iloc[:, 0].astype(float).dropna()


def load(index: str = "NASDAQCOM", start: str = "1990-01-02", refresh: bool = False) -> pd.DataFrame:
    """Daily frame with the close, the log return in percent (r) and the VIX close.

    Rows are trading days on which both the index and the VIX are published.
    """
    close = fred(index, refresh)
    r = 100 * np.log(close).diff()  # before the join, so a missing VIX day never merges two returns
    vix = fred("VIXCLS", refresh)
    df = pd.concat({"close": close, "r": r, "vix": vix}, axis=1, join="inner").dropna()
    return df[df.index >= pd.Timestamp(start)]
