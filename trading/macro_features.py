"""Macro/calendar features derived from price action patterns.
No external API needed — extracts regime signals from the data itself.
"""
import numpy as np
import pandas as pd


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime and macro-proxy features to price data.
    Works on any timeframe but most useful on D1.
    """
    p = df.copy()
    c = p.close

    # Volatility regime (high vol = risk-off, low vol = trend-friendly)
    ret = c.pct_change()
    vol_fast = ret.rolling(10).std()
    vol_slow = ret.rolling(60).std()
    p['vol_regime'] = vol_fast / vol_slow.replace(0, np.nan)

    # Trend regime via ADX proxy (directional movement)
    high, low = p.high, p.low
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - c.shift(1)).abs(),
        (low - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    p['adx'] = dx.rolling(14).mean()
    p['di_diff'] = plus_di - minus_di

    # Mean reversion signal (z-score of price vs 50-day SMA)
    sma50 = c.rolling(50).mean()
    std50 = c.rolling(50).std()
    p['price_zscore'] = (c - sma50) / std50.replace(0, np.nan)

    # Momentum breakout (close vs 20-day high/low)
    p['donchian_pct'] = (c - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min()).replace(0, np.nan)

    # Day-of-week (for D1: Monday effect, Friday risk-off)
    if hasattr(p, 'time'):
        time_col = pd.to_datetime(p.time)
        p['dow'] = time_col.dt.dayofweek / 4.0  # 0=Mon, 1=Fri normalized
        p['is_monday'] = (time_col.dt.dayofweek == 0).astype(float)
        p['is_friday'] = (time_col.dt.dayofweek == 4).astype(float)

    # Monthly seasonality proxy
    if hasattr(p, 'time'):
        p['month_sin'] = np.sin(2 * np.pi * time_col.dt.month / 12)
        p['month_cos'] = np.cos(2 * np.pi * time_col.dt.month / 12)

    # Rate of change (longer-term momentum)
    p['roc_50'] = c.pct_change(50)
    p['roc_100'] = c.pct_change(100)

    # Regime label: trending vs ranging (ADX > 25 = trending)
    p['is_trending'] = (p.adx > 25).astype(float)

    return p
