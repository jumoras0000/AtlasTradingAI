from trading.data import generate_demo
from trading.walkforward import walk_forward_xgb


def test_walkforward_basic(tmp_path):
    prices = generate_demo(tmp_path / 'wf.csv', n=800)
    result = walk_forward_xgb(prices, min_train=300, test_window=100, step=100)
    assert result['n_folds'] >= 2
    assert 0 <= result['mean_accuracy'] <= 1
    assert 'folds' in result


def test_walkforward_all_folds_have_metrics(tmp_path):
    prices = generate_demo(tmp_path / 'wf.csv', n=800)
    result = walk_forward_xgb(prices, min_train=300, test_window=100, step=100)
    for fold in result['folds']:
        assert 'accuracy' in fold
        assert 'return_pct' in fold
        assert fold['train_bars'] >= 300
