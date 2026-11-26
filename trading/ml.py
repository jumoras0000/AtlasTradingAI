"""Walk-forward-safe supervised baseline. Train only on earlier timestamp data."""
from pathlib import Path
import numpy as np
from .data import features

COLS_BASIC = ['ret1', 'ret5', 'trend', 'vol20', 'volume_z']

COLS = [
    'ret1', 'ret5', 'ret10', 'ret20',
    'trend', 'trend_long',
    'vol20', 'vol50', 'vol_ratio',
    'volume_z',
    'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_pct',
    'atr_pct',
    'stoch_k', 'stoch_d',
    'hl_ratio',
    'body_ratio', 'upper_shadow', 'lower_shadow',
    # Macro/regime features
    'vol_regime', 'adx', 'di_diff', 'price_zscore',
    'donchian_pct', 'roc_50', 'roc_100', 'is_trending',
]
COLS_WITH_NEWS = COLS + ['news_inflation', 'news_rates', 'news_employment', 'news_total']

def train_xgb(prices, output='models/xgb.json', fraction=0.7, horizon=1):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    p = features(prices)
    p['label'] = (p.close.shift(-horizon) > p.close).astype(int)
    p = p.iloc[:-horizon].copy()
    n = int(len(p) * fraction)
    if n < 50 or len(p) - n < 20:
        raise ValueError('Insufficient train/test split')
    xtrain, ytrain = p.iloc[:n][COLS], p.iloc[:n].label
    xtest, ytest = p.iloc[n:][COLS], p.iloc[n:].label
    if ytrain.nunique() < 2:
        raise ValueError('Only one class in training labels')
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.5,
        min_child_weight=3, gamma=0.01, random_state=42, n_jobs=2,
        early_stopping_rounds=30, eval_metric='logloss',
    )
    model.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], verbose=False)
    scores = model.predict_proba(xtest)[:, 1]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return {
        'train_bars': n, 'test_bars': len(p) - n, 'horizon': horizon,
        'test_accuracy': float(accuracy_score(ytest, scores >= 0.5)),
        'note': 'Classification metric, NOT trading profitability. Test data was never fitted.',
    }

def xgb_targets(prices, model_path='models/xgb.json'):
    from xgboost import XGBClassifier
    p = features(prices)
    model = XGBClassifier()
    model.load_model(model_path)
    proba = model.predict_proba(p[COLS])[:, 1]
    targets = np.zeros(len(proba))
    targets[proba >= 0.60] = 0.20
    targets[(proba >= 0.55) & (proba < 0.60)] = 0.10
    return p, targets


def train_xgb_with_news(prices, news_path='data/news.json',
                        output='models/xgb_news.json', fraction=0.7):
    """XGBoost with news features. Same temporal split, additional news columns."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from .news_features import merge_news_features

    p = features(prices)
    p = merge_news_features(p, news_path)
    p['label'] = (p.close.shift(-1) > p.close).astype(int)
    p = p.iloc[:-1].copy()

    cols = [c for c in COLS_WITH_NEWS if c in p.columns]
    n = int(len(p) * fraction)
    if n < 50 or len(p) - n < 20:
        raise ValueError('Insufficient train/test split')

    xtrain, ytrain = p.iloc[:n][cols], p.iloc[:n].label
    xtest, ytest = p.iloc[n:][cols], p.iloc[n:].label

    if ytrain.nunique() < 2:
        raise ValueError('Only one class in training labels')

    model = XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=2
    )
    model.fit(xtrain, ytrain)
    scores = model.predict_proba(xtest)[:, 1]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    return {
        'train_bars': n, 'test_bars': len(p) - n,
        'test_accuracy': float(accuracy_score(ytest, scores >= 0.5)),
        'features_used': cols,
        'note': 'XGBoost with news features. NOT proof of trading profitability.',
    }
