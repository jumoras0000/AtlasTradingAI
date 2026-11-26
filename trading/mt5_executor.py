"""MT5 order executor — DEMO accounts ONLY.
Sends market orders, manages positions, enforces risk limits.
Will refuse to operate on live/real accounts.
"""
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger('atlas.executor')

TIMEFRAMES = {
    'M1': 'TIMEFRAME_M1', 'M5': 'TIMEFRAME_M5', 'M15': 'TIMEFRAME_M15',
    'H1': 'TIMEFRAME_H1', 'H4': 'TIMEFRAME_H4', 'D1': 'TIMEFRAME_D1',
}


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    volume: float
    price: float
    comment: str
    timestamp: str


def _init_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError('pip install -r requirements-mt5.txt') from e

    path = os.getenv('MT5_TERMINAL_PATH') or None
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        raise RuntimeError(f'MT5 init failed: {mt5.last_error()}')
    return mt5


def _require_demo(mt5):
    info = mt5.account_info()
    if info is None:
        raise RuntimeError('No MT5 account connected')
    if info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown()
        raise PermissionError(
            'SAFETY: Only DEMO accounts allowed. '
            f'Current account {info.login} is NOT demo (trade_mode={info.trade_mode}). '
            'Refusing to trade.'
        )
    return info


def get_account_info() -> dict:
    mt5 = _init_mt5()
    try:
        info = _require_demo(mt5)
        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin_free': info.margin_free,
            'currency': info.currency,
            'trade_mode': 'DEMO',
        }
    finally:
        mt5.shutdown()


def get_symbol_info(symbol: str) -> dict:
    mt5 = _init_mt5()
    try:
        _require_demo(mt5)
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f'Symbol unavailable: {symbol}')
        info = mt5.symbol_info(symbol)
        return {
            'symbol': symbol,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'point': info.point,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'digits': info.digits,
        }
    finally:
        mt5.shutdown()


def get_positions(symbol: str | None = None) -> list[dict]:
    mt5 = _init_mt5()
    try:
        _require_demo(mt5)
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == 0 else 'SELL',
                'volume': p.volume,
                'price_open': p.price_open,
                'price_current': p.price_current,
                'profit': p.profit,
                'sl': p.sl,
                'tp': p.tp,
                'time': datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
            }
            for p in positions
        ]
    finally:
        mt5.shutdown()


def _round_volume(volume: float, vol_min: float, vol_max: float, vol_step: float) -> float:
    if volume < vol_min:
        return 0.0
    volume = min(volume, vol_max)
    steps = round(volume / vol_step)
    return round(steps * vol_step, 8)


def open_position(symbol: str, direction: str, volume: float,
                  sl_points: int = 0, tp_points: int = 0,
                  comment: str = 'AtlasAI_DEMO') -> OrderResult:
    """Open a market position. direction: 'BUY' or 'SELL'."""
    mt5 = _init_mt5()
    try:
        info = _require_demo(mt5)

        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f'Symbol unavailable: {symbol}')

        sym = mt5.symbol_info(symbol)
        volume = _round_volume(volume, sym.volume_min, sym.volume_max, sym.volume_step)
        if volume <= 0:
            return OrderResult(False, None, 0, 0, 'Volume too small after rounding', _now())

        if direction.upper() == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = sym.ask
            sl = price - sl_points * sym.point if sl_points > 0 else 0.0
            tp = price + tp_points * sym.point if tp_points > 0 else 0.0
        elif direction.upper() == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
            price = sym.bid
            sl = price + sl_points * sym.point if sl_points > 0 else 0.0
            tp = price - tp_points * sym.point if tp_points > 0 else 0.0
        else:
            raise ValueError(f'Invalid direction: {direction}')

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'sl': sl,
            'tp': tp,
            'deviation': 20,
            'magic': 123456,
            'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, volume, price, f'order_send returned None: {mt5.last_error()}', _now())

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, volume, price, f'Rejected: {result.retcode} {result.comment}', _now())

        logger.info('OPEN %s %s %.5f @ %.5f ticket=%d', direction, symbol, volume, result.price, result.order)
        return OrderResult(True, result.order, volume, result.price, 'OK', _now())

    finally:
        mt5.shutdown()


def close_position(ticket: int, comment: str = 'AtlasAI_DEMO_close') -> OrderResult:
    """Close an existing position by ticket."""
    mt5 = _init_mt5()
    try:
        _require_demo(mt5)
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(False, ticket, 0, 0, f'Position {ticket} not found', _now())

        pos = position[0]
        sym = mt5.symbol_info(pos.symbol)

        if pos.type == 0:  # BUY -> close by SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = sym.bid
        else:  # SELL -> close by BUY
            order_type = mt5.ORDER_TYPE_BUY
            price = sym.ask

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': pos.symbol,
            'volume': pos.volume,
            'type': order_type,
            'position': ticket,
            'price': price,
            'deviation': 20,
            'magic': 123456,
            'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, ticket, pos.volume, price, f'close failed: {mt5.last_error()}', _now())

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, ticket, pos.volume, price, f'Rejected: {result.retcode} {result.comment}', _now())

        logger.info('CLOSE ticket=%d %s %.5f @ %.5f', ticket, pos.symbol, pos.volume, result.price)
        return OrderResult(True, ticket, pos.volume, result.price, 'OK', _now())

    finally:
        mt5.shutdown()


def close_all(symbol: str | None = None, comment: str = 'AtlasAI_DEMO_closeall') -> list[OrderResult]:
    """Close all open positions (optionally filtered by symbol)."""
    positions = get_positions(symbol)
    results = []
    for p in positions:
        results.append(close_position(p['ticket'], comment=comment))
    return results


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
