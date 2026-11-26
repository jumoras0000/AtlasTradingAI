"""Walk-forward evaluation: sliding window train/test for XGBoost + backtest.
Prevents overfitting by retraining on expanding windows and testing on unseen data.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import features
from .backtest import simulate, BacktestSettings
from .ml import COLS
from .risk import RiskConfig


def _confidence_sizing(proba: np.ndarray, max_exposure: float = 0.20,
                       adaptive: bool = True) -> np.ndarray:
    """Scale exposure by signal confidence with hysteresis to reduce turnover.
    adaptive=True uses quantile thresholds when proba range is compressed."""
    if adaptive and proba.std() < 0.03:
        hi = np.quantile(proba, 0.80)
        mid = np.quantile(proba, 0.60)
        lo = np.quantile(proba, 0.20)
    else:
        hi, mid, lo = 0.58, 0.53, 0.42

    targets = np.zeros(len(proba))
    current = 0.0
    for i, p in enumerate(proba):
        if p >= hi:
            desired = max_exposure
        elif p >= mid:
            desired = max_exposure * 0.5
        elif p <= lo:
            desired = 0.0
        else:
            desired = current
        if abs(desired - current) >= 0.05:
            current = desired
        targets[i] = current
    return targets


def walk_forward_xgb(prices: pd.DataFrame, min_train: int = 300,
                     test_window: int = 100, step: int = 50,
                     settings: BacktestSettings | None = None,
                     xgb_params: dict | None = None,
                     horizon: int = 1) -> dict:
    """Expanding-window walk-forward: train on [0..i], test on [i..i+test_window].
    Returns aggregated metrics across all folds.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    p = features(prices)
    p['label'] = (p.close.shift(-horizon) > p.close).astype(int)
    p = p.iloc[:-horizon].copy()

    if len(p) < min_train + test_window:
        raise ValueError(f'Need {min_train + test_window} bars, got {len(p)}')

    s = settings or BacktestSettings()
    params = xgb_params or dict(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.5,
        min_child_weight=3, gamma=0.01, random_state=42, n_jobs=2,
    )

    folds = []
    cursor = min_train

    while cursor + test_window <= len(p):
        train_slice = p.iloc[:cursor]
        test_slice = p.iloc[cursor:cursor + test_window]

        xtrain, ytrain = train_slice[COLS], train_slice['label']
        xtest, ytest = test_slice[COLS], test_slice['label']

        if ytrain.nunique() < 2:
            cursor += step
            continue

        model = XGBClassifier(**params)
        model.fit(xtrain, ytrain)
        proba = model.predict_proba(xtest)[:, 1]
        preds = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(ytest, preds))

        targets = _confidence_sizing(proba)
        # Build full-length target array for simulate (needs features-length)
        p_test = features(prices).iloc[cursor:cursor + test_window + 1]
        if len(p_test) >= 2:
            full_targets = np.zeros(len(p_test))
            full_targets[:len(targets)] = targets[:len(full_targets)]
            summary, detail = simulate(p_test, full_targets, s)
            fold_return = summary['net_return_pct']
            fold_trades = int(summary.get('n_trades', (detail.turnover > 0).sum()))
            fold_dd = summary.get('max_drawdown_pct', 0.0)
        else:
            fold_return, fold_trades, fold_dd = 0.0, 0, 0.0

        folds.append({
            'fold_start': cursor,
            'fold_end': cursor + test_window,
            'train_bars': cursor,
            'test_bars': test_window,
            'accuracy': acc,
            'return_pct': fold_return,
            'n_trades': fold_trades,
            'max_drawdown_pct': fold_dd,
        })
        cursor += step

    if not folds:
        raise ValueError('No valid folds produced')

    accs = [f['accuracy'] for f in folds]
    rets = [f['return_pct'] for f in folds]

    return {
        'n_folds': len(folds),
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_return_pct': float(np.mean(rets)),
        'std_return_pct': float(np.std(rets)),
        'cumulative_return_pct': float(np.prod([1 + r / 100 for r in rets]) * 100 - 100),
        'folds': folds,
        'note': 'Walk-forward OOS evaluation. NOT proof of profitability.',
    }


def write_walkforward_report(result: dict, directory: str = 'reports',
                             prefix: str = 'walkforward') -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{prefix}.json').write_text(
        json.dumps(result, indent=2), encoding='utf8'
    )
    folds_df = pd.DataFrame(result['folds'])
    folds_df.to_csv(d / f'{prefix}_folds.csv', index=False)
