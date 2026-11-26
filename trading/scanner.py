"""Multi-asset scanner: find tradeable opportunities across all available symbols.
Scans forex, commodities, indices, stocks — anything available on the MT5 broker.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import features
from .ml import COLS
from .mt5_executor import _init_mt5, _require_demo

logger = logging.getLogger('atlas.scanner')

# Symbol categories to scan
FOREX_MAJORS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
]
FOREX_CROSSES = [
    'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'AUDNZD', 'EURAUD', 'GBPAUD',
    'CADJPY', 'CHFJPY', 'AUDCAD', 'NZDJPY', 'GBPCAD',
]
COMMODITIES = [
    'XAUUSD', 'XAGUSD',  # Gold, Silver
]
INDICES = [
    'US500', 'US30', 'USTEC', 'DE40', 'UK100', 'JP225', 'FR40',
]
CRYPTO = [
    'BTCUSD', 'ETHUSD',
]

ALL_WATCHLIST = FOREX_MAJORS + FOREX_CROSSES + COMMODITIES + INDICES + CRYPTO


def get_available_symbols(watchlist: list[str] | None = None) -> list[dict]:
    """Check which symbols from the watchlist are available on this broker."""
    mt5 = _init_mt5()
    try:
        _require_demo(mt5)
        symbols = watchlist or ALL_WATCHLIST
        available = []
        for sym_name in symbols:
            if mt5.symbol_select(sym_name, True):
                info = mt5.symbol_info(sym_name)
                if info is not None and info.trade_mode != 0:  # 0 = disabled
                    available.append({
                        'symbol': sym_name,
                        'bid': info.bid,
                        'ask': info.ask,
                        'spread': info.spread,
                        'digits': info.digits,
                        'volume_min': info.volume_min,
                        'contract_size': info.trade_contract_size,
                        'currency_profit': info.currency_profit,
                    })
        return available
    finally:
        mt5.shutdown()


def fetch_bars(symbol: str, count: int = 200, timeframe: str = 'H1') -> pd.DataFrame | None:
    """Fetch bars for a single symbol. Returns None if unavailable."""
    mt5 = _init_mt5()
    try:
        _require_demo(mt5)
        tf_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        if not mt5.symbol_select(symbol, True):
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf, 1, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={'tick_volume': 'volume'})
        return df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time')
    except Exception as e:
        logger.debug('Failed to fetch %s: %s', symbol, e)
        return None
    finally:
        mt5.shutdown()


def score_symbol(bars: pd.DataFrame, model_path: str = 'models/xgb.json') -> dict:
    """Score a symbol: trend strength, volatility, model confidence."""
    try:
        p = features(bars)
        if len(p) < 20:
            return {'tradeable': False, 'reason': 'insufficient_bars'}

        last = p.iloc[-1]
        trend = float(last.trend)
        vol = float(last.vol20)
        ret5 = float(last.ret5)

        # EMA signal
        ema_signal = 1.0 if last.ema20 > last.ema50 else -1.0

        # XGBoost confidence (if model exists)
        xgb_proba = 0.5
        try:
            from xgboost import XGBClassifier
            from pathlib import Path
            if Path(model_path).exists():
                model = XGBClassifier()
                model.load_model(model_path)
                xgb_proba = float(model.predict_proba(p.iloc[-1:][COLS])[:, 1][0])
        except Exception:
            pass

        # Composite score: combines trend, momentum, model confidence
        # Higher = more favorable for LONG
        score = (
            0.3 * (xgb_proba - 0.5) * 2  # Model confidence [-1, 1]
            + 0.3 * np.clip(trend * 20, -1, 1)  # Trend strength
            + 0.2 * np.clip(ret5 * 50, -1, 1)  # Momentum
            + 0.2 * ema_signal * 0.5  # EMA direction
        )

        return {
            'tradeable': True,
            'score': float(score),
            'direction': 'BUY' if score > 0 else 'SELL' if score < -0.1 else 'FLAT',
            'confidence': float(abs(score)),
            'xgb_proba': xgb_proba,
            'trend': trend,
            'volatility': vol,
            'momentum_5bar': ret5,
            'ema_signal': 'UP' if ema_signal > 0 else 'DOWN',
        }

    except Exception as e:
        return {'tradeable': False, 'reason': str(e)}


def scan_all(timeframe: str = 'H1', min_confidence: float = 0.15,
             watchlist: list[str] | None = None,
             model_path: str = 'models/xgb.json') -> dict:
    """Scan all available symbols and rank by opportunity score."""
    available = get_available_symbols(watchlist)
    logger.info('Scanning %d available symbols...', len(available))

    opportunities = []
    skipped = []

    for sym_info in available:
        symbol = sym_info['symbol']
        bars = fetch_bars(symbol, count=200, timeframe=timeframe)
        if bars is None or len(bars) < 100:
            skipped.append({'symbol': symbol, 'reason': 'insufficient_data'})
            continue

        score_result = score_symbol(bars, model_path)
        if not score_result.get('tradeable'):
            skipped.append({'symbol': symbol, 'reason': score_result.get('reason', 'not_tradeable')})
            continue

        entry = {**sym_info, **score_result}

        if score_result['confidence'] >= min_confidence:
            opportunities.append(entry)
        else:
            skipped.append({'symbol': symbol, 'reason': f'low_confidence ({score_result["confidence"]:.3f})'})

    # Sort by absolute score (best opportunities first)
    opportunities.sort(key=lambda x: abs(x['score']), reverse=True)

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'opportunities': opportunities,
        'skipped': skipped,
        'total_scanned': len(available),
        'total_opportunities': len(opportunities),
    }
