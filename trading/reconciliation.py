"""Position reconciliation: compare expected vs actual MT5 positions.
Detects drift, orphan positions, and sync issues.
"""
import logging
from datetime import datetime, timezone
from .mt5_executor import get_positions, get_symbol_info, get_account_info

logger = logging.getLogger('atlas.reconcile')


def reconcile(symbol: str, expected_exposure: float, tolerance: float = 0.02) -> dict:
    """Compare expected exposure fraction vs actual MT5 positions.

    Returns dict with status ('ok', 'drift', 'orphan') and details.
    """
    account = get_account_info()
    equity = account['equity']
    positions = get_positions(symbol)
    sym = get_symbol_info(symbol)

    buy_volume = sum(p['volume'] for p in positions if p['type'] == 'BUY')
    sell_volume = sum(p['volume'] for p in positions if p['type'] == 'SELL')

    if equity > 0:
        buy_value = buy_volume * sym['trade_contract_size'] * sym['bid']
        actual_exposure = buy_value / equity
    else:
        actual_exposure = 0.0

    drift = abs(actual_exposure - expected_exposure)

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': symbol,
        'expected_exposure': expected_exposure,
        'actual_exposure': actual_exposure,
        'drift': drift,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'num_positions': len(positions),
        'equity': equity,
    }

    if sell_volume > 0:
        result['status'] = 'orphan_sell'
        result['message'] = f'Unexpected SELL positions: {sell_volume} lots'
        logger.warning('RECONCILE: orphan SELL positions detected')
    elif drift > tolerance:
        result['status'] = 'drift'
        result['message'] = f'Exposure drift {drift:.4f} exceeds tolerance {tolerance}'
        logger.warning('RECONCILE: drift %.4f (expected=%.3f actual=%.3f)',
                       drift, expected_exposure, actual_exposure)
    else:
        result['status'] = 'ok'
        result['message'] = 'Positions match expected exposure'

    return result


def check_session_hours(symbol: str = 'EURUSD') -> dict:
    """Check if the market is currently open for trading.
    Forex: Sunday 22:00 UTC to Friday 22:00 UTC.
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour

    # Forex closed: Friday after 22:00 UTC through Sunday 22:00 UTC
    if weekday == 4 and hour >= 22:
        open_now = False
    elif weekday == 5:
        open_now = False
    elif weekday == 6 and hour < 22:
        open_now = False
    else:
        open_now = True

    return {
        'timestamp': now.isoformat(),
        'weekday': now.strftime('%A'),
        'hour_utc': hour,
        'market_open': open_now,
        'symbol': symbol,
        'note': 'Forex hours approximation. Actual hours depend on broker.',
    }


def full_health_check(symbol: str, expected_exposure: float = 0.0) -> dict:
    """Complete health check: session + account + reconciliation."""
    session = check_session_hours(symbol)
    checks = {'session': session}

    if not session['market_open']:
        checks['status'] = 'market_closed'
        checks['tradeable'] = False
        return checks

    try:
        account = get_account_info()
        checks['account'] = account
        checks['account_ok'] = True
    except Exception as e:
        checks['account_error'] = str(e)
        checks['account_ok'] = False
        checks['status'] = 'account_error'
        checks['tradeable'] = False
        return checks

    try:
        recon = reconcile(symbol, expected_exposure)
        checks['reconciliation'] = recon
    except Exception as e:
        checks['reconciliation_error'] = str(e)

    checks['status'] = 'ok'
    checks['tradeable'] = True
    return checks
