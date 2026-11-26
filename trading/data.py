from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ('time', 'open', 'high', 'low', 'close', 'volume')

def generate_demo(path: str | Path, n: int = 1600, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices: for plumbing tests ONLY, not evidence of a market edge."""
    if n < 100: raise ValueError('n must be >=100')
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.003, n) + 0.00012*np.sin(np.arange(n)/90)
    close = 1.1*np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0018, 0.0006, n))
    high = np.maximum(open_, close)*(1+spread)
    low = np.minimum(open_, close)*(1-spread)
    df = pd.DataFrame({'time': pd.date_range('2022-01-01', periods=n, freq='h', tz='UTC'),
       'open':open_, 'high':high, 'low':low, 'close':close,
       'volume': rng.integers(100, 5000, n)})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='raise')
    for col in REQUIRED[1:]: df[col] = pd.to_numeric(df[col],errors='raise')
    df = df.sort_values('time').reset_index(drop=True)
    if df.time.duplicated().any(): raise ValueError('Duplicate timestamps')
    if df[list(REQUIRED[1:])].isna().any().any(): raise ValueError('NaN prices')
    if (df[['open','high','low','close']]<=0).any().any():raise ValueError('Prices must be positive')
    if (df['volume']<0).any(): raise ValueError('Negative volume')
    if ((df.high < df[['open','close','low']].max(axis=1)) | (df.low > df[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC')
    if len(df)<100: raise ValueError('Need at least 100 bars')
    return df

def features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features computed using only current and past CLOSED bars."""
    p = df.copy()
    p['ret1'] = p.close.pct_change()
    p['ret5'] = p.close.pct_change(5)
    p['ema20'] = p.close.ewm(span=20,adjust=False).mean()
    p['ema50'] = p.close.ewm(span=50,adjust=False).mean()
    p['trend'] = p.ema20/p.ema50-1
    p['vol20'] = p.ret1.rolling(20).std()
    p['volume_z'] = (p.volume-p.volume.rolling(20).mean())/p.volume.rolling(20).std().replace(0,np.nan)
    return p.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
