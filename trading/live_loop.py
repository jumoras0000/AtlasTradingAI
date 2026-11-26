"""Live trading loop for MT5 DEMO accounts.
Reads latest bars -> computes signal -> executes orders -> logs everything.

Usage:
    python -m trading.live_loop --symbol EURUSD --timeframe H1 --strategy xgb

DEMO ONLY. Will refuse real accounts.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, make_risk, make_settings
from .data import features
from .backtest import ema_signals
from .risk import RiskConfig, clamp_target
from .mt5_bridge import read_demo_bars
from .mt5_executor import (
    get_account_info, get_positions, get_symbol_info,
    open_position, close_position, close_all, _require_demo, _init_mt5,
)

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'live.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('atlas.live')


class LiveTrader:
    def __init__(self, symbol: str, timeframe: str, strategy: str,
                 config_path: str | None = None, model_path: str = 'models/xgb.json'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.model_path = model_path

        cfg = load_config(config_path)
        self.risk = make_risk(cfg)
        self.settings = make_settings(cfg)

        self.equity_start = 0.0
        self.peak_equity = 0.0
        self.day_start_equity = 0.0
        self.current_day = None

        self.trade_log: list[dict] = []

    def _log_trade(self, action: str, details: dict):
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            **details,
        }
        self.trade_log.append(entry)
        log_file = LOG_DIR / 'trades.jsonl'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        logger.info('TRADE %s %s', action, json.dumps(details))

    def _get_signal(self, bars: pd.DataFrame) -> float:
        """Compute target exposure fraction from strategy."""
        p = features(bars)
        if len(p) < 2:
            return 0.0

        if self.strategy == 'ema':
            signals = ema_signals(p)
            return float(signals[-1])

        elif self.strategy == 'xgb':
            from .ml import COLS
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(self.model_path)
            row = p.iloc[-1:][COLS]
            proba = model.predict_proba(row)[:, 1][0]
            return 0.20 if proba >= 0.55 else 0.0

        elif self.strategy == 'ppo':
            from .ml import COLS
            from stable_baselines3 import PPO
            model = PPO.load(self.model_path.replace('xgb.json', 'ppo.zip'))
            row = p.iloc[-1]
            positions = get_positions(self.symbol)
            current_pos = sum(po['volume'] for po in positions if po['type'] == 'BUY')
            obs = np.nan_to_num(
                np.r_[row[COLS].to_numpy(dtype=float), current_pos, 0.0],
                nan=0, posinf=0, neginf=0
            ).astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            return (0., 0.1, 0.2)[int(action)]

        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def _current_exposure(self, account_equity: float) -> float:
        """Current long exposure as fraction of equity."""
        positions = get_positions(self.symbol)
        if not positions or account_equity <= 0:
            return 0.0
        sym = get_symbol_info(self.symbol)
        total_value = sum(
            p['volume'] * sym['trade_contract_size'] * p['price_current']
            for p in positions if p['type'] == 'BUY'
        )
        return total_value / account_equity

    def _execute_target(self, target: float, account: dict):
        """Adjust position to match target exposure."""
        equity = account['equity']
        current = self._current_exposure(equity)

        # Apply risk manager
        today = datetime.now(timezone.utc).date()
        if self.current_day != today:
            self.current_day = today
            self.day_start_equity = equity

        self.peak_equity = max(self.peak_equity, equity)

        clamped, reason = clamp_target(
            target, current, equity, self.peak_equity, self.day_start_equity, self.risk
        )

        if reason != 'approved':
            logger.warning('Risk clamp: %s (target=%.3f -> %.3f)', reason, target, clamped)
            if reason in ('drawdown_kill_switch', 'daily_loss_kill_switch', 'equity_zero'):
                logger.critical('KILL SWITCH: %s — closing all positions', reason)
                results = close_all(self.symbol)
                self._log_trade('KILL_SWITCH', {'reason': reason, 'results': [r.__dict__ for r in results]})
                return

        diff = clamped - current
        sym = get_symbol_info(self.symbol)

        if abs(diff) < 0.005:
            logger.debug('Exposure diff %.4f too small, skip', diff)
            return

        if diff > 0:
            # Need to buy more
            value = diff * equity
            volume = value / (sym['trade_contract_size'] * sym['ask'])
            result = open_position(self.symbol, 'BUY', volume)
            self._log_trade('BUY', {
                'target': clamped, 'current': current, 'volume': volume,
                'reason': reason, 'result': result.__dict__,
            })

        elif diff < 0:
            # Need to reduce — close positions partially or fully
            positions = get_positions(self.symbol)
            buy_positions = [p for p in positions if p['type'] == 'BUY']

            if clamped <= 0.005:
                # Close all
                results = close_all(self.symbol)
                self._log_trade('CLOSE_ALL', {
                    'target': clamped, 'reason': reason,
                    'results': [r.__dict__ for r in results],
                })
            else:
                # Close excess — start with smallest positions
                buy_positions.sort(key=lambda p: p['volume'])
                excess_value = abs(diff) * equity
                closed_value = 0
                for pos in buy_positions:
                    if closed_value >= excess_value:
                        break
                    result = close_position(pos['ticket'])
                    pos_value = pos['volume'] * sym['trade_contract_size'] * pos['price_current']
                    closed_value += pos_value
                    self._log_trade('PARTIAL_CLOSE', {
                        'ticket': pos['ticket'], 'volume': pos['volume'],
                        'reason': reason, 'result': result.__dict__,
                    })

    def run_once(self) -> dict:
        """Single iteration: read bars, compute signal, execute."""
        account = get_account_info()
        logger.info('Account: balance=%.2f equity=%.2f', account['balance'], account['equity'])

        if self.equity_start == 0:
            self.equity_start = account['equity']
            self.peak_equity = account['equity']
            self.day_start_equity = account['equity']

        bars = read_demo_bars(self.symbol, count=200, timeframe=self.timeframe)
        signal = self._get_signal(bars)
        logger.info('Signal: strategy=%s target_exposure=%.3f', self.strategy, signal)

        self._execute_target(signal, account)

        positions = get_positions(self.symbol)
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': account['equity'],
            'signal': signal,
            'positions': len(positions),
            'exposure': self._current_exposure(account['equity']),
        }

    def run_loop(self, interval_seconds: int = 60, max_iterations: int = 0):
        """Main trading loop. Runs every interval_seconds.
        max_iterations=0 means infinite.
        """
        logger.info('=== STARTING LIVE LOOP ===')
        logger.info('Symbol=%s TF=%s Strategy=%s Interval=%ds',
                     self.symbol, self.timeframe, self.strategy, interval_seconds)

        # Verify DEMO at startup
        mt5 = _init_mt5()
        try:
            info = _require_demo(mt5)
            logger.info('Account %d on %s (DEMO verified)', info.login, info.server)
        finally:
            mt5.shutdown()

        iteration = 0
        while True:
            iteration += 1
            try:
                result = self.run_once()
                logger.info('Iteration %d: equity=%.2f exposure=%.3f positions=%d',
                           iteration, result['equity'], result['exposure'], result['positions'])
            except PermissionError as e:
                logger.critical('PERMISSION ERROR: %s — STOPPING', e)
                break
            except Exception as e:
                logger.error('Iteration %d error: %s', iteration, e, exc_info=True)

            if 0 < max_iterations <= iteration:
                logger.info('Max iterations reached (%d)', max_iterations)
                break

            logger.info('Sleeping %ds...', interval_seconds)
            time.sleep(interval_seconds)

        logger.info('=== LOOP ENDED === iterations=%d', iteration)


def _interval_for_timeframe(tf: str) -> int:
    """Default poll interval in seconds based on timeframe."""
    return {
        'M1': 60, 'M5': 300, 'M15': 900,
        'H1': 3600, 'H4': 14400, 'D1': 86400,
    }.get(tf, 3600)


def main():
    ap = argparse.ArgumentParser(description='AtlasTradingAI live trading loop (DEMO ONLY)')
    ap.add_argument('--symbol', default='EURUSD')
    ap.add_argument('--timeframe', default='H1', choices=['M1', 'M5', 'M15', 'H1', 'H4', 'D1'])
    ap.add_argument('--strategy', default='ema', choices=['ema', 'xgb', 'ppo'])
    ap.add_argument('--model', default='models/xgb.json')
    ap.add_argument('--interval', type=int, default=0, help='Poll interval in seconds (0=auto from timeframe)')
    ap.add_argument('--max-iter', type=int, default=0, help='Max iterations (0=infinite)')
    ap.add_argument('--once', action='store_true', help='Run once and exit')
    args = ap.parse_args()

    interval = args.interval or _interval_for_timeframe(args.timeframe)

    trader = LiveTrader(
        symbol=args.symbol, timeframe=args.timeframe,
        strategy=args.strategy, model_path=args.model,
    )

    if args.once:
        result = trader.run_once()
        print(json.dumps(result, indent=2))
    else:
        trader.run_loop(interval_seconds=interval, max_iterations=args.max_iter)


if __name__ == '__main__':
    main()
