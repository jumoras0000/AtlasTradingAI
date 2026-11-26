"""Multi-asset autonomous trader with auto-learning.
Scans all markets, picks best opportunities, trades them, learns continuously.

Usage:
    python -m trading.multi_trader --timeframe H1 --max-positions 5

DEMO ONLY.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config, make_risk, make_settings
from .risk import clamp_target
from .scanner import scan_all, fetch_bars, get_available_symbols
from .data import features
from .ml import COLS
from .mt5_executor import (
    get_account_info, get_positions, get_symbol_info,
    open_position, close_position, close_all, _init_mt5, _require_demo,
)
from .auto_learn import AutoLearner, PerformanceTracker

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'multi_trader.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('atlas.multi')


class MultiTrader:
    """Autonomous multi-asset trader with scanning and auto-learning."""

    def __init__(self, timeframe: str = 'H1', max_positions: int = 5,
                 max_exposure_per_symbol: float = 0.10,
                 max_total_exposure: float = 0.50,
                 min_score: float = 0.15,
                 config_path: str | None = None,
                 model_path: str = 'models/xgb.json'):
        self.timeframe = timeframe
        self.max_positions = max_positions
        self.max_per_symbol = max_exposure_per_symbol
        self.max_total = max_total_exposure
        self.min_score = min_score
        self.model_path = model_path

        cfg = load_config(config_path)
        self.risk = make_risk(cfg)
        self.settings = make_settings(cfg)

        self.learner = AutoLearner(
            symbol='MULTI', timeframe=timeframe,
            retrain_interval_hours=24, eval_window=50,
        )

        self.peak_equity = 0.0
        self.day_start_equity = 0.0
        self.current_day = None
        self.trade_log_file = LOG_DIR / 'multi_trades.jsonl'

    def _log_trade(self, action: str, details: dict):
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            **details,
        }
        with open(self.trade_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        logger.info('TRADE %s %s', action, details.get('symbol', ''))

    def _get_current_positions(self) -> dict[str, list[dict]]:
        """Group open positions by symbol."""
        positions = get_positions()
        by_symbol: dict[str, list[dict]] = {}
        for p in positions:
            by_symbol.setdefault(p['symbol'], []).append(p)
        return by_symbol

    def _total_exposure(self, account: dict) -> float:
        """Total exposure across all symbols."""
        equity = account['equity']
        if equity <= 0:
            return 0.0
        positions = get_positions()
        total = 0.0
        for p in positions:
            try:
                sym = get_symbol_info(p['symbol'])
                total += p['volume'] * sym['trade_contract_size'] * p['price_current']
            except Exception:
                pass
        return total / equity

    def _symbol_exposure(self, symbol: str, equity: float) -> float:
        positions = get_positions(symbol)
        if not positions or equity <= 0:
            return 0.0
        try:
            sym = get_symbol_info(symbol)
            total = sum(p['volume'] * sym['trade_contract_size'] * p['price_current']
                       for p in positions if p['type'] == 'BUY')
            return total / equity
        except Exception:
            return 0.0

    def run_once(self) -> dict:
        """Single scan + trade cycle."""
        # 1. Account check
        account = get_account_info()
        equity = account['equity']
        logger.info('Account: equity=%.2f balance=%.2f', equity, account['balance'])

        if self.peak_equity == 0:
            self.peak_equity = equity
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        today = datetime.now(timezone.utc).date()
        if self.current_day != today:
            self.current_day = today
            self.day_start_equity = equity

        # Global risk check
        dd = 1 - equity / self.peak_equity if self.peak_equity > 0 else 0
        daily_loss = 1 - equity / self.day_start_equity if self.day_start_equity > 0 else 0

        if dd >= self.risk.max_drawdown_fraction:
            logger.critical('GLOBAL KILL SWITCH: drawdown %.2f%%', dd * 100)
            results = close_all()
            self._log_trade('GLOBAL_KILL_DRAWDOWN', {'dd': dd, 'closed': len(results)})
            return {'action': 'kill_switch', 'reason': 'drawdown', 'dd': dd}

        if daily_loss >= self.risk.max_daily_loss_fraction:
            logger.critical('DAILY LOSS KILL: %.2f%%', daily_loss * 100)
            results = close_all()
            self._log_trade('GLOBAL_KILL_DAILY', {'loss': daily_loss, 'closed': len(results)})
            return {'action': 'kill_switch', 'reason': 'daily_loss', 'loss': daily_loss}

        # 2. Scan markets
        scan = scan_all(
            timeframe=self.timeframe,
            min_confidence=self.min_score,
            model_path=self.model_path,
        )
        opportunities = scan['opportunities']
        logger.info('Scan: %d opportunities from %d symbols',
                     len(opportunities), scan['total_scanned'])

        # 3. Defensive mode from auto-learner
        exposure_scale = self.learner.get_exposure_scale()
        if exposure_scale < 1.0:
            logger.info('Defensive mode: scale=%.2f', exposure_scale)

        # 4. Manage positions
        current = self._get_current_positions()
        total_exposure = self._total_exposure(account)
        active_symbols = set(current.keys())

        # Close positions for symbols no longer favorable
        favorable_symbols = {o['symbol'] for o in opportunities}
        for sym in list(active_symbols):
            if sym not in favorable_symbols:
                logger.info('Closing %s — no longer favorable', sym)
                for pos in current[sym]:
                    result = close_position(pos['ticket'])
                    self._log_trade('CLOSE_UNFAVORABLE', {
                        'symbol': sym, 'ticket': pos['ticket'],
                        'result': result.__dict__,
                    })
                total_exposure = self._total_exposure(account)

        # 5. Open new positions for best opportunities
        num_positions = len(get_positions())
        for opp in opportunities:
            if num_positions >= self.max_positions:
                break
            if total_exposure >= self.max_total * exposure_scale:
                break

            symbol = opp['symbol']
            if symbol in active_symbols:
                continue
            direction = opp['direction']
            if direction == 'FLAT':
                continue

            target_exposure = min(
                self.max_per_symbol * exposure_scale,
                self.max_total * exposure_scale - total_exposure,
            )
            if target_exposure < 0.01:
                continue

            try:
                sym_info = get_symbol_info(symbol)
                value = target_exposure * equity
                price = sym_info['ask'] if direction == 'BUY' else sym_info['bid']
                volume = value / (sym_info['trade_contract_size'] * price)

                result = open_position(symbol, direction, volume,
                                       comment=f'Atlas_score={opp["score"]:.3f}')
                self._log_trade('OPEN_OPPORTUNITY', {
                    'symbol': symbol,
                    'score': opp['score'],
                    'confidence': opp['confidence'],
                    'volume': volume,
                    'result': result.__dict__,
                })

                if result.success:
                    num_positions += 1
                    total_exposure += target_exposure
                    active_symbols.add(symbol)

            except Exception as e:
                logger.error('Failed to open %s: %s', symbol, e)

        # 6. Auto-learning check
        retrain_result = self.learner.maybe_retrain()
        if retrain_result:
            self._log_trade('RETRAIN', retrain_result)

        # 7. Status
        final_positions = get_positions()
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': equity,
            'total_exposure': self._total_exposure(account),
            'num_positions': len(final_positions),
            'symbols': list({p['symbol'] for p in final_positions}),
            'opportunities_found': len(opportunities),
            'defensive_scale': exposure_scale,
            'autolearn': self.learner.status(),
        }

    def run_loop(self, interval_seconds: int = 3600, max_iterations: int = 0):
        """Main trading loop."""
        logger.info('=== MULTI-TRADER START ===')
        logger.info('TF=%s max_pos=%d max_total_exp=%.0f%% min_score=%.2f',
                     self.timeframe, self.max_positions,
                     self.max_total * 100, self.min_score)

        # Verify DEMO
        mt5 = _init_mt5()
        try:
            info = _require_demo(mt5)
            logger.info('DEMO verified: %d @ %s', info.login, info.server)
        finally:
            mt5.shutdown()

        iteration = 0
        while True:
            iteration += 1
            try:
                result = self.run_once()
                logger.info(
                    'Iter %d: equity=%.2f positions=%d exposure=%.1f%% symbols=%s',
                    iteration, result['equity'], result['num_positions'],
                    result['total_exposure'] * 100,
                    ','.join(result['symbols']) or 'none',
                )
            except PermissionError as e:
                logger.critical('NOT DEMO: %s — STOPPING', e)
                break
            except Exception as e:
                logger.error('Iter %d error: %s', iteration, e, exc_info=True)

            if 0 < max_iterations <= iteration:
                break

            logger.info('Sleeping %ds...', interval_seconds)
            time.sleep(interval_seconds)

        logger.info('=== MULTI-TRADER END === %d iterations', iteration)


def main():
    ap = argparse.ArgumentParser(description='AtlasTradingAI multi-asset trader (DEMO ONLY)')
    ap.add_argument('--timeframe', default='H1', choices=['M5', 'M15', 'H1', 'H4', 'D1'])
    ap.add_argument('--max-positions', type=int, default=5)
    ap.add_argument('--max-exposure', type=float, default=0.50)
    ap.add_argument('--min-score', type=float, default=0.15)
    ap.add_argument('--model', default='models/xgb.json')
    ap.add_argument('--interval', type=int, default=0)
    ap.add_argument('--max-iter', type=int, default=0)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    interval = args.interval or {'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400}[args.timeframe]

    trader = MultiTrader(
        timeframe=args.timeframe,
        max_positions=args.max_positions,
        max_total_exposure=args.max_exposure,
        min_score=args.min_score,
        model_path=args.model,
    )

    if args.once:
        result = trader.run_once()
        print(json.dumps(result, indent=2))
    else:
        trader.run_loop(interval_seconds=interval, max_iterations=args.max_iter)


if __name__ == '__main__':
    main()
