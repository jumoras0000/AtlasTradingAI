import json
from trading.data import generate_demo, features
from trading.news_features import load_news, merge_news_features, CATEGORIES


def test_load_news_missing_file(tmp_path):
    df = load_news(tmp_path / 'nope.json')
    assert len(df) == 0


def test_merge_news_no_news(tmp_path):
    prices = generate_demo(tmp_path / 'p.csv', n=200)
    p = features(prices)
    merged = merge_news_features(p, tmp_path / 'nope.json')
    assert 'news_total' in merged.columns
    assert (merged['news_total'] == 0).all()


def test_merge_news_with_data(tmp_path):
    prices = generate_demo(tmp_path / 'p.csv', n=200)
    p = features(prices)
    news = {
        'items': [
            {'title': 'ECB inflation report', 'published': str(p.iloc[50].time)},
            {'title': 'Interest rate decision', 'published': str(p.iloc[60].time)},
        ],
        'errors': {},
    }
    news_path = tmp_path / 'news.json'
    news_path.write_text(json.dumps(news), encoding='utf-8')
    merged = merge_news_features(p, news_path)
    assert merged['news_total'].sum() > 0
