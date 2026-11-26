"""Download real OHLCV data via yfinance. No orders, read-only."""
from pathlib import Path
import pandas as pd


FOREX_TICKERS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'AUDUSD': 'AUDUSD=X',
}


def fetch_yahoo(symbol: str = 'EURUSD', period: str = '2y',
                interval: str = '1h', output: str | Path = 'data/real.csv') -> pd.DataFrame:
    """Fetch forex data from Yahoo Finance. Returns OHLCV DataFrame."""
    import yfinance as yf

    ticker = FOREX_TICKERS.get(symbol.upper(), symbol)
    raw = yf.download(ticker, period=period, interval=interval, progress=False)
    if raw.empty:
        raise RuntimeError(f'No data returned for {ticker} (period={period}, interval={interval})')

    df = raw.reset_index()

    # yfinance returns MultiIndex columns when single ticker — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]

    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low in ('datetime', 'date'):
            rename_map[col] = 'time'
        elif low == 'adj close':
            continue
        elif low in ('open', 'high', 'low', 'close', 'volume'):
            rename_map[col] = low

    df = df.rename(columns=rename_map)
    if 'time' not in df.columns:
        raise RuntimeError(f'No time column found in {list(df.columns)}')

    df['time'] = pd.to_datetime(df['time'], utc=True)
    keep = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=['open', 'high', 'low', 'close']).sort_values('time').reset_index(drop=True)

    if 'volume' not in df.columns or df['volume'].isna().all():
        df['volume'] = 0

    df['volume'] = df['volume'].fillna(0).astype(int)

    if len(df) < 100:
        raise RuntimeError(f'Only {len(df)} bars retrieved — need at least 100')

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
