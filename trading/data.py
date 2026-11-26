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
    c, h, l, v = p.close, p.high, p.low, p.volume

    # Returns
    p['ret1'] = c.pct_change()
    p['ret5'] = c.pct_change(5)
    p['ret10'] = c.pct_change(10)
    p['ret20'] = c.pct_change(20)

    # EMAs + trend
    p['ema20'] = c.ewm(span=20, adjust=False).mean()
    p['ema50'] = c.ewm(span=50, adjust=False).mean()
    p['ema200'] = c.ewm(span=200, adjust=False).mean()
    p['trend'] = p.ema20 / p.ema50 - 1
    p['trend_long'] = p.ema50 / p.ema200 - 1

    # Volatility
    p['vol20'] = p.ret1.rolling(20).std()
    p['vol50'] = p.ret1.rolling(50).std()
    p['vol_ratio'] = p.vol20 / p.vol50.replace(0, np.nan)

    # Volume
    p['volume_z'] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)

    # RSI (14 periods)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    p['rsi'] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    p['macd'] = ema12 - ema26
    p['macd_signal'] = p.macd.ewm(span=9, adjust=False).mean()
    p['macd_hist'] = p.macd - p.macd_signal

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    p['bb_upper'] = sma20 + 2 * std20
    p['bb_lower'] = sma20 - 2 * std20
    p['bb_pct'] = (c - p.bb_lower) / (p.bb_upper - p.bb_lower).replace(0, np.nan)

    # ATR (14 periods)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    p['atr'] = tr.rolling(14).mean()
    p['atr_pct'] = p.atr / c

    # Stochastic %K / %D
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    p['stoch_k'] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    p['stoch_d'] = p.stoch_k.rolling(3).mean()

    # Price position relative to range
    p['hl_ratio'] = (c - l) / (h - l).replace(0, np.nan)

    # Candle patterns
    body = (c - p.open).abs()
    full_range = (h - l).replace(0, np.nan)
    p['body_ratio'] = body / full_range
    p['upper_shadow'] = (h - pd.concat([c, p.open], axis=1).max(axis=1)) / full_range
    p['lower_shadow'] = (pd.concat([c, p.open], axis=1).min(axis=1) - l) / full_range

    # Macro/regime features
    from .macro_features import add_macro_features
    p = add_macro_features(p)

    return p.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
