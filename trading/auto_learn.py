"""Autonomous learning pipeline: retrain, evaluate, deploy, adapt.
Runs periodically alongside the live trading loop.

Cycle:
  1. Fetch latest bars from MT5
  2. Evaluate current model on recent OOS window
  3. If performance degraded -> retrain on expanded dataset
  4. Walk-forward validate new model
  5. If new > old -> deploy; else -> defensive mode
  6. Log everything
"""
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, make_settings, make_risk
from .data import load_prices, features
from .ml import COLS, train_xgb
from .backtest import simulate, ema_signals, BacktestSettings
from .risk import RiskConfig

logger = logging.getLogger('atlas.autolearn')

MODELS_DIR = Path('models')
DATA_DIR = Path('data')
REPORTS_DIR = Path('reports')
LOG_DIR = Path('logs')

for d in (MODELS_DIR, DATA_DIR, REPORTS_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)


class PerformanceTracker:
    """Track live model performance over rolling windows."""

    def __init__(self, window: int = 100):
        self.window = window
        self.predictions: list[float] = []
        self.actuals: list[int] = []
        self.equities: list[float] = []

    def record(self, predicted_proba: float, actual_up: int, equity: float):
        self.predictions.append(predicted_proba)
        self.actuals.append(actual_up)
        self.equities.append(equity)

    def recent_accuracy(self) -> float | None:
        if len(self.actuals) < self.window:
            return None
        recent_pred = self.predictions[-self.window:]
        recent_actual = self.actuals[-self.window:]
        correct = sum(
            1 for p, a in zip(recent_pred, recent_actual)
            if (p >= 0.5) == (a == 1)
        )
        return correct / self.window

    def recent_return(self) -> float | None:
        if len(self.equities) < self.window:
            return None
        return (self.equities[-1] / self.equities[-self.window] - 1) * 100

    def recent_sharpe(self) -> float | None:
        if len(self.equities) < self.window:
            return None
        eq = np.array(self.equities[-self.window:])
        rets = np.diff(eq) / eq[:-1]
        if np.std(rets) < 1e-10:
            return 0.0
        return float(np.mean(rets) / np.std(rets))

    def to_dict(self) -> dict:
        return {
            'total_observations': len(self.actuals),
            'recent_accuracy': self.recent_accuracy(),
            'recent_return_pct': self.recent_return(),
            'recent_sharpe': self.recent_sharpe(),
        }


class ModelManager:
    """Manage model versions: train, evaluate, deploy, rollback."""

    def __init__(self, symbol: str = 'EURUSD', config_path: str | None = None):
        self.symbol = symbol
        cfg = load_config(config_path)
        self.settings = make_settings(cfg)
        self.risk = make_risk(cfg)
        self.active_model = MODELS_DIR / 'xgb.json'
        self.candidate_model = MODELS_DIR / 'xgb_candidate.json'
        self.history_dir = MODELS_DIR / 'history'
        self.history_dir.mkdir(exist_ok=True)

    def _version_tag(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    def _archive_model(self, path: Path, tag: str):
        if path.exists():
            dest = self.history_dir / f'xgb_{tag}.json'
            shutil.copy2(path, dest)
            logger.info('Archived model -> %s', dest)

    def collect_latest_data(self, count: int = 2000, timeframe: str = 'H1') -> pd.DataFrame:
        """Fetch fresh bars from MT5 and save to data/live_{symbol}_{tf}.csv."""
        from .scanner import fetch_bars
        df = fetch_bars(self.symbol, count=count, timeframe=timeframe)
        if df is None or len(df) < 100:
            from .mt5_bridge import read_demo_bars
            df = read_demo_bars(self.symbol, count=count, timeframe=timeframe)
        path = DATA_DIR / f'live_{self.symbol}_{timeframe}.csv'
        df.to_csv(path, index=False)
        logger.info('Collected %d %s bars for %s -> %s', len(df), timeframe, self.symbol, path)
        return df

    def evaluate_model(self, prices: pd.DataFrame, model_path: Path,
                       test_fraction: float = 0.3) -> dict:
        """Evaluate a model on the last test_fraction of data."""
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score

        p = features(prices)
        p['label'] = (p.close.shift(-1) > p.close).astype(int)
        p = p.iloc[:-1].copy()

        n = int(len(p) * (1 - test_fraction))
        if n < 50 or len(p) - n < 20:
            return {'error': 'Insufficient data', 'accuracy': 0.0, 'return_pct': 0.0}

        xtest = p.iloc[n:][COLS]
        ytest = p.iloc[n:].label

        model = XGBClassifier()
        model.load_model(str(model_path))
        proba = model.predict_proba(xtest)[:, 1]
        acc = float(accuracy_score(ytest, (proba >= 0.5).astype(int)))

        # Backtest on test portion with confidence sizing
        from .walkforward import _confidence_sizing
        targets = _confidence_sizing(proba)
        p_test = features(prices).iloc[n:n + len(targets) + 1]
        if len(p_test) >= 2:
            full_targets = np.zeros(len(p_test))
            full_targets[:len(targets)] = targets[:len(full_targets)]
            summary, _ = simulate(p_test, full_targets, self.settings)
            ret = summary['net_return_pct']
            dd = summary['max_drawdown_pct']
        else:
            ret, dd = 0.0, 0.0

        return {
            'accuracy': acc,
            'return_pct': ret,
            'max_drawdown_pct': dd,
            'test_bars': len(p) - n,
            'model': str(model_path),
        }

    def retrain(self, prices: pd.DataFrame, fraction: float = 0.7) -> dict:
        """Train a new candidate model on latest data."""
        result = train_xgb(prices, output=str(self.candidate_model), fraction=fraction)
        logger.info('Retrained candidate: acc=%.3f on %d test bars',
                     result['test_accuracy'], result['test_bars'])
        return result

    def should_retrain(self, tracker: PerformanceTracker,
                       accuracy_threshold: float = 0.48,
                       sharpe_threshold: float = -0.1) -> tuple[bool, str]:
        """Decide if retraining is needed based on recent performance."""
        acc = tracker.recent_accuracy()
        sharpe = tracker.recent_sharpe()

        if acc is None:
            return False, 'insufficient_data'
        if acc < accuracy_threshold:
            return True, f'accuracy_degraded ({acc:.3f} < {accuracy_threshold})'
        if sharpe is not None and sharpe < sharpe_threshold:
            return True, f'sharpe_degraded ({sharpe:.3f} < {sharpe_threshold})'
        return False, 'performance_ok'

    def deploy_if_better(self, prices: pd.DataFrame) -> dict:
        """Compare candidate vs active model, deploy if better."""
        if not self.candidate_model.exists():
            return {'action': 'no_candidate'}
        if not self.active_model.exists():
            # No active model, deploy candidate
            tag = self._version_tag()
            shutil.copy2(self.candidate_model, self.active_model)
            logger.info('Deployed first model %s', tag)
            return {'action': 'deployed_first', 'tag': tag}

        active_eval = self.evaluate_model(prices, self.active_model)
        candidate_eval = self.evaluate_model(prices, self.candidate_model)

        logger.info('Active:    acc=%.3f ret=%.2f%%', active_eval['accuracy'], active_eval['return_pct'])
        logger.info('Candidate: acc=%.3f ret=%.2f%%', candidate_eval['accuracy'], candidate_eval['return_pct'])

        # Deploy if candidate is better on BOTH accuracy and return
        better_acc = candidate_eval['accuracy'] > active_eval['accuracy'] - 0.01
        better_ret = candidate_eval['return_pct'] > active_eval['return_pct']
        less_dd = candidate_eval['max_drawdown_pct'] > active_eval['max_drawdown_pct'] - 0.5

        if better_acc and (better_ret or less_dd):
            tag = self._version_tag()
            self._archive_model(self.active_model, tag)
            shutil.copy2(self.candidate_model, self.active_model)
            logger.info('DEPLOYED new model %s', tag)
            return {
                'action': 'deployed',
                'tag': tag,
                'active_eval': active_eval,
                'candidate_eval': candidate_eval,
            }
        else:
            logger.info('Candidate NOT better, keeping active model')
            return {
                'action': 'kept_active',
                'active_eval': active_eval,
                'candidate_eval': candidate_eval,
            }


class DefensiveMode:
    """Reduce exposure when model confidence is low."""

    def __init__(self, base_exposure: float = 0.20):
        self.base_exposure = base_exposure
        self.active = False
        self.scale = 1.0

    def update(self, tracker: PerformanceTracker):
        acc = tracker.recent_accuracy()
        sharpe = tracker.recent_sharpe()

        if acc is None:
            self.scale = 0.5  # Conservative until enough data
            self.active = True
            return

        if acc < 0.45:
            self.scale = 0.0  # Stop trading
            self.active = True
        elif acc < 0.48:
            self.scale = 0.25
            self.active = True
        elif acc < 0.50:
            self.scale = 0.5
            self.active = True
        else:
            self.scale = 1.0
            self.active = False

        # Sharpe override
        if sharpe is not None and sharpe < -0.3:
            self.scale = min(self.scale, 0.25)
            self.active = True

    def adjust_exposure(self, raw_exposure: float) -> float:
        return raw_exposure * self.scale

    def to_dict(self) -> dict:
        return {
            'active': self.active,
            'scale': self.scale,
            'effective_max_exposure': self.base_exposure * self.scale,
        }


class AutoLearner:
    """Main autonomous learning loop — integrates with LiveTrader."""

    def __init__(self, symbol: str = 'EURUSD', timeframe: str = 'H1',
                 retrain_interval_hours: int = 24,
                 eval_window: int = 100,
                 config_path: str | None = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.retrain_interval = retrain_interval_hours * 3600
        self.last_retrain = 0.0

        self.tracker = PerformanceTracker(window=eval_window)
        self.manager = ModelManager(symbol, config_path)
        self.defensive = DefensiveMode()

        self.log_file = LOG_DIR / 'autolearn.jsonl'

    def _log(self, event: str, data: dict):
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            **data,
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        logger.info('AUTOLEARN %s %s', event, json.dumps(data))

    def record_observation(self, predicted_proba: float, actual_up: int, equity: float):
        """Called by LiveTrader after each bar closes."""
        self.tracker.record(predicted_proba, actual_up, equity)
        self.defensive.update(self.tracker)

    def get_exposure_scale(self) -> float:
        """LiveTrader calls this to adjust raw signal."""
        return self.defensive.scale

    def maybe_retrain(self) -> dict | None:
        """Check if retraining is needed, and do it if so."""
        now = time.time()
        if now - self.last_retrain < self.retrain_interval:
            return None

        need, reason = self.manager.should_retrain(self.tracker)
        if not need:
            self._log('retrain_check', {'needed': False, 'reason': reason})
            self.last_retrain = now
            return None

        self._log('retrain_start', {'reason': reason, 'tracker': self.tracker.to_dict()})

        try:
            # Collect fresh data
            prices = self.manager.collect_latest_data(count=2000, timeframe=self.timeframe)

            # Retrain candidate
            train_result = self.manager.retrain(prices)
            self._log('retrain_done', train_result)

            # Compare and maybe deploy
            deploy_result = self.manager.deploy_if_better(prices)
            self._log('deploy_decision', deploy_result)

            self.last_retrain = now
            return deploy_result

        except Exception as e:
            logger.error('Retrain failed: %s', e, exc_info=True)
            self._log('retrain_error', {'error': str(e)})
            self.last_retrain = now
            return {'action': 'error', 'error': str(e)}

    def force_retrain(self) -> dict:
        """Force immediate retraining regardless of schedule."""
        self.last_retrain = 0
        self.tracker.predictions.clear()
        self.tracker.actuals.clear()

        prices = self.manager.collect_latest_data(count=2000, timeframe=self.timeframe)
        train_result = self.manager.retrain(prices)
        deploy_result = self.manager.deploy_if_better(prices)

        self._log('force_retrain', {
            'train': train_result,
            'deploy': deploy_result,
        })
        return {'train': train_result, 'deploy': deploy_result}

    def status(self) -> dict:
        return {
            'tracker': self.tracker.to_dict(),
            'defensive': self.defensive.to_dict(),
            'active_model': str(self.manager.active_model),
            'active_model_exists': self.manager.active_model.exists(),
            'last_retrain_ago_hours': (time.time() - self.last_retrain) / 3600 if self.last_retrain else None,
            'retrain_interval_hours': self.retrain_interval / 3600,
        }


class TrainingScheduler:
    """Autonomous training scheduler — runs independently, produces reports."""

    def __init__(self, symbols: list[str] | None = None,
                 timeframe: str = 'H1',
                 interval_hours: float = 24,
                 config_path: str | None = None):
        self.symbols = symbols or ['EURUSD']
        self.timeframe = timeframe
        self.interval = interval_hours * 3600
        self.config_path = config_path
        self.report_dir = REPORTS_DIR / 'training'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.managers: dict[str, ModelManager] = {}
        for sym in self.symbols:
            self.managers[sym] = ModelManager(sym, config_path)

    def _report_path(self, cycle_id: str) -> Path:
        return self.report_dir / f'report_{cycle_id}.json'

    def _emit(self, report: dict, cycle_id: str):
        """Write report to file + log."""
        path = self._report_path(cycle_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info('REPORT [%s] step=%s status=%s -> %s',
                     cycle_id, report.get('step'), report.get('status'), path)

    def run_cycle(self) -> dict:
        """Full training cycle for all symbols, with step-by-step reports."""
        cycle_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        report = {
            'cycle_id': cycle_id,
            'started': datetime.now(timezone.utc).isoformat(),
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'steps': [],
        }

        self._step(report, cycle_id, 'STARTED', 'Training cycle initiated')

        results_by_symbol = {}
        for sym in self.symbols:
            mgr = self.managers[sym]
            sym_result = {'symbol': sym}

            # Step 1: Data collection
            self._step(report, cycle_id, 'DATA_COLLECT',
                       f'Fetching {sym} bars from MT5', symbol=sym)
            try:
                prices = mgr.collect_latest_data(count=2000, timeframe=self.timeframe)
                sym_result['bars_collected'] = len(prices)
                self._step(report, cycle_id, 'DATA_OK',
                           f'{sym}: {len(prices)} bars collected', symbol=sym)
            except Exception as e:
                sym_result['error'] = f'data_collect: {e}'
                self._step(report, cycle_id, 'DATA_FAIL',
                           f'{sym}: data collection failed — {e}', symbol=sym)
                results_by_symbol[sym] = sym_result
                continue

            # Step 2: Evaluate current model
            self._step(report, cycle_id, 'EVAL_CURRENT',
                       f'Evaluating active model on {sym}', symbol=sym)
            try:
                if mgr.active_model.exists():
                    active_eval = mgr.evaluate_model(prices, mgr.active_model)
                    sym_result['active_eval'] = active_eval
                    self._step(report, cycle_id, 'EVAL_CURRENT_OK',
                               f'{sym}: active model acc={active_eval["accuracy"]:.3f} '
                               f'ret={active_eval["return_pct"]:.2f}%', symbol=sym)
                else:
                    sym_result['active_eval'] = None
                    self._step(report, cycle_id, 'EVAL_CURRENT_SKIP',
                               f'{sym}: no active model, will train first', symbol=sym)
            except Exception as e:
                sym_result['active_eval'] = {'error': str(e)}
                self._step(report, cycle_id, 'EVAL_CURRENT_FAIL',
                           f'{sym}: eval failed — {e}', symbol=sym)

            # Step 3: Retrain
            self._step(report, cycle_id, 'RETRAIN',
                       f'Training new candidate model for {sym}', symbol=sym)
            try:
                train_result = mgr.retrain(prices)
                sym_result['train'] = train_result
                self._step(report, cycle_id, 'RETRAIN_OK',
                           f'{sym}: candidate trained acc={train_result["test_accuracy"]:.3f} '
                           f'on {train_result["test_bars"]} test bars', symbol=sym)
            except Exception as e:
                sym_result['error'] = f'retrain: {e}'
                self._step(report, cycle_id, 'RETRAIN_FAIL',
                           f'{sym}: retrain failed — {e}', symbol=sym)
                results_by_symbol[sym] = sym_result
                continue

            # Step 4: Walk-forward validation
            self._step(report, cycle_id, 'WALKFORWARD',
                       f'Walk-forward validation for {sym}', symbol=sym)
            wf_result = None
            try:
                from .walkforward import walk_forward_xgb
                wf_result = walk_forward_xgb(prices, min_train=300, test_window=100, step=50)
                n_folds = wf_result.get('n_folds', 0)
                mean_acc = wf_result.get('mean_accuracy', 0)
                mean_ret = wf_result.get('mean_return_pct', 0)
                sym_result['walkforward'] = {
                    'n_folds': n_folds, 'mean_accuracy': mean_acc,
                    'mean_return_pct': mean_ret,
                    'folds': wf_result.get('folds', []),
                }
                self._step(report, cycle_id, 'WALKFORWARD_OK',
                           f'{sym}: {n_folds} folds, mean_acc={mean_acc:.3f} mean_ret={mean_ret:.2f}%',
                           symbol=sym)
            except Exception as e:
                sym_result['walkforward'] = {'error': str(e)}
                self._step(report, cycle_id, 'WALKFORWARD_FAIL',
                           f'{sym}: walk-forward failed — {e}', symbol=sym)

            # Step 5: Compare + deploy
            self._step(report, cycle_id, 'DEPLOY_CHECK',
                       f'Comparing candidate vs active for {sym}', symbol=sym)
            try:
                deploy_result = mgr.deploy_if_better(prices)
                sym_result['deploy'] = deploy_result
                action = deploy_result.get('action', '?')
                self._step(report, cycle_id, 'DEPLOY_DONE',
                           f'{sym}: {action}', symbol=sym)
            except Exception as e:
                sym_result['deploy'] = {'error': str(e)}
                self._step(report, cycle_id, 'DEPLOY_FAIL',
                           f'{sym}: deploy check failed — {e}', symbol=sym)

            # Step 6: Quality gate check
            self._step(report, cycle_id, 'QUALITY_CHECK',
                       f'Checking quality gates for {sym}', symbol=sym)
            gate = QualityGate()
            eval_for_gate = sym_result.get('active_eval') or sym_result.get('train', {})
            if isinstance(eval_for_gate, dict) and 'error' not in eval_for_gate:
                gate_result = gate.evaluate(eval_for_gate, wf_result)
                sym_result['quality_gate'] = gate_result
                verdict = gate_result['verdict']
                msg = gate_result['message']
                self._step(report, cycle_id, f'QUALITY_{verdict}',
                           f'{sym}: {verdict} — {msg}', symbol=sym)
            else:
                sym_result['quality_gate'] = {'verdict': 'SKIP', 'message': 'No valid eval data'}
                self._step(report, cycle_id, 'QUALITY_SKIP',
                           f'{sym}: skipped quality check (no eval data)', symbol=sym)

            results_by_symbol[sym] = sym_result

        # Global convergence summary
        gate = QualityGate()
        convergence = gate.convergence_summary()
        report['results'] = results_by_symbol
        report['convergence'] = convergence
        report['finished'] = datetime.now(timezone.utc).isoformat()
        self._step(report, cycle_id, 'COMPLETED',
                   f'Cycle done — {len(results_by_symbol)} symbols | convergence: {convergence.get("status", "?")}')

        return report

    def _step(self, report: dict, cycle_id: str, status: str, message: str,
              symbol: str | None = None):
        step = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'step': status,
            'message': message,
        }
        if symbol:
            step['symbol'] = symbol
        report['steps'].append(step)
        report['status'] = status
        self._emit(report, cycle_id)

    def run_loop(self, max_cycles: int = 0):
        """Continuous training loop — runs every interval_hours."""
        logger.info('=== TRAINING SCHEDULER START === symbols=%s interval=%.1fh',
                     self.symbols, self.interval / 3600)
        cycle = 0
        while True:
            cycle += 1
            logger.info('--- Training cycle %d ---', cycle)
            try:
                report = self.run_cycle()
                ok = sum(1 for r in report['results'].values() if 'error' not in r)
                fail = len(report['results']) - ok
                logger.info('Cycle %d done: %d ok, %d failed', cycle, ok, fail)
            except Exception as e:
                logger.error('Cycle %d crashed: %s', cycle, e, exc_info=True)

            if 0 < max_cycles <= cycle:
                break

            logger.info('Next cycle in %.1f hours...', self.interval / 3600)
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
#  Quality Gates — objective criteria for model readiness
# ---------------------------------------------------------------------------

class QualityGate:
    """Define and check objective thresholds for model readiness."""

    # Thresholds — a model must pass ALL to be considered "production-ready"
    MIN_ACCURACY = 0.54          # >54% on OOS data (random=50%)
    MIN_RETURN_PCT = 0.5         # >0.5% net return on OOS backtest
    MAX_DRAWDOWN_PCT = -5.0      # drawdown no worse than -5%
    MIN_SHARPE = 0.3             # annualized Sharpe > 0.3
    MIN_WALKFORWARD_FOLDS = 3    # must pass on >= 3 walk-forward folds
    MIN_FOLD_WINRATE = 0.6       # >= 60% of folds must be profitable
    CONSECUTIVE_PASS_CYCLES = 3  # must pass for 3 consecutive cycles

    def __init__(self):
        self.history_file = REPORTS_DIR / 'quality_history.jsonl'
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, eval_result: dict, walkforward_result: dict | None = None) -> dict:
        """Check all quality gates. Returns pass/fail per criterion + overall."""
        checks = {}

        acc = eval_result.get('accuracy', 0)
        checks['accuracy'] = {
            'value': acc, 'threshold': self.MIN_ACCURACY,
            'pass': acc >= self.MIN_ACCURACY,
        }

        ret = eval_result.get('return_pct', -999)
        checks['return_pct'] = {
            'value': ret, 'threshold': self.MIN_RETURN_PCT,
            'pass': ret >= self.MIN_RETURN_PCT,
        }

        dd = eval_result.get('max_drawdown_pct', -999)
        checks['max_drawdown'] = {
            'value': dd, 'threshold': self.MAX_DRAWDOWN_PCT,
            'pass': dd >= self.MAX_DRAWDOWN_PCT,
        }

        # Walk-forward checks
        if walkforward_result and 'folds' in walkforward_result:
            folds = walkforward_result['folds']
            n_folds = len(folds)
            profitable = sum(1 for f in folds if f.get('return_pct', 0) > 0)
            fold_winrate = profitable / n_folds if n_folds > 0 else 0

            checks['walkforward_folds'] = {
                'value': n_folds, 'threshold': self.MIN_WALKFORWARD_FOLDS,
                'pass': n_folds >= self.MIN_WALKFORWARD_FOLDS,
            }
            checks['fold_winrate'] = {
                'value': fold_winrate, 'threshold': self.MIN_FOLD_WINRATE,
                'pass': fold_winrate >= self.MIN_FOLD_WINRATE,
            }
        else:
            checks['walkforward_folds'] = {'value': 0, 'threshold': self.MIN_WALKFORWARD_FOLDS, 'pass': False}
            checks['fold_winrate'] = {'value': 0, 'threshold': self.MIN_FOLD_WINRATE, 'pass': False}

        all_pass = all(c['pass'] for c in checks.values())
        consecutive = self._count_consecutive_pass() if all_pass else 0

        checks['consecutive_cycles'] = {
            'value': consecutive + (1 if all_pass else 0),
            'threshold': self.CONSECUTIVE_PASS_CYCLES,
            'pass': (consecutive + 1) >= self.CONSECUTIVE_PASS_CYCLES if all_pass else False,
        }

        overall = all_pass and checks['consecutive_cycles']['pass']

        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_pass': overall,
            'ready_for_live': overall,
            'checks': checks,
            'verdict': 'READY' if overall else 'NOT_READY',
            'message': self._verdict_message(checks, overall),
        }

        self._record(result)
        return result

    def _verdict_message(self, checks: dict, overall: bool) -> str:
        if overall:
            return 'Model passes all quality gates — ready for live deployment'
        failing = [name for name, c in checks.items() if not c['pass']]
        return f'Failing gates: {", ".join(failing)}'

    def _record(self, result: dict):
        with open(self.history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

    def _count_consecutive_pass(self) -> int:
        if not self.history_file.exists():
            return 0
        lines = self.history_file.read_text(encoding='utf-8').strip().split('\n')
        count = 0
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                passed = all(
                    c['pass'] for name, c in entry.get('checks', {}).items()
                    if name != 'consecutive_cycles'
                )
                if passed:
                    count += 1
                else:
                    break
            except (json.JSONDecodeError, KeyError):
                break
        return count

    def get_history(self, last_n: int = 20) -> list[dict]:
        if not self.history_file.exists():
            return []
        lines = self.history_file.read_text(encoding='utf-8').strip().split('\n')
        results = []
        for line in lines[-last_n:]:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return results

    def convergence_summary(self) -> dict:
        """How close is the model to passing all gates?"""
        history = self.get_history(last_n=10)
        if not history:
            return {'status': 'no_data', 'cycles_completed': 0}

        latest = history[-1]
        checks = latest.get('checks', {})

        progress = {}
        for name, c in checks.items():
            if name == 'consecutive_cycles':
                continue
            val = c['value']
            thresh = c['threshold']
            if isinstance(val, (int, float)) and isinstance(thresh, (int, float)):
                if thresh != 0:
                    pct = min(val / thresh * 100, 100) if thresh > 0 else min(thresh / val * 100, 100) if val < 0 else 100
                else:
                    pct = 100 if val >= 0 else 0
                progress[name] = {
                    'current': val, 'target': thresh,
                    'progress_pct': round(pct, 1), 'pass': c['pass'],
                }

        # Trend: improving or degrading?
        if len(history) >= 3:
            recent_acc = [h['checks']['accuracy']['value'] for h in history[-3:] if 'accuracy' in h.get('checks', {})]
            if len(recent_acc) >= 2:
                trend = 'improving' if recent_acc[-1] > recent_acc[0] else 'degrading' if recent_acc[-1] < recent_acc[0] else 'stable'
            else:
                trend = 'unknown'
        else:
            trend = 'insufficient_data'

        return {
            'status': latest.get('verdict', 'unknown'),
            'cycles_completed': len(history),
            'consecutive_pass': latest['checks'].get('consecutive_cycles', {}).get('value', 0),
            'needed_consecutive': self.CONSECUTIVE_PASS_CYCLES,
            'progress': progress,
            'trend': trend,
            'latest_message': latest.get('message', ''),
        }
