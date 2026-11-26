"""Align news headlines to price bars as ML features.
No lookahead: news at time T only available to bars with time >= T.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .news import keyword_tag


CATEGORIES = ['inflation', 'rates', 'employment']


def load_news(path: str | Path = 'data/news.json') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=['time', 'title'] + CATEGORIES)
    raw = json.loads(p.read_text(encoding='utf-8'))
    items = raw.get('items', [])
    if not items:
        return pd.DataFrame(columns=['time', 'title'] + CATEGORIES)

    rows = []
    for item in items:
        pub = item.get('published') or item.get('collected_at_utc')
        if not pub:
            continue
        try:
            t = pd.to_datetime(pub, utc=True)
        except (ValueError, TypeError):
            continue
        tags = keyword_tag(item.get('title', ''))
        row = {'time': t, 'title': item.get('title', '')}
        for cat in CATEGORIES:
            row[cat] = 1 if cat in tags else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['time', 'title'] + CATEGORIES)
    return pd.DataFrame(rows).sort_values('time').reset_index(drop=True)


def merge_news_features(prices: pd.DataFrame, news_path: str | Path = 'data/news.json') -> pd.DataFrame:
    """Merge news counts into price DataFrame using asof join (no lookahead).
    For each price bar, count news in the preceding 24h window per category.
    """
    df = prices.copy()
    news = load_news(news_path)

    for cat in CATEGORIES:
        df[f'news_{cat}'] = 0

    if news.empty or 'time' not in df.columns:
        df['news_total'] = 0
        return df

    df['time'] = pd.to_datetime(df['time'], utc=True)
    news['time'] = pd.to_datetime(news['time'], utc=True)

    for i, row in df.iterrows():
        bar_time = row['time']
        window_start = bar_time - pd.Timedelta('24h')
        mask = (news['time'] > window_start) & (news['time'] <= bar_time)
        window = news.loc[mask]
        for cat in CATEGORIES:
            df.at[i, f'news_{cat}'] = int(window[cat].sum()) if cat in window.columns else 0

    df['news_total'] = sum(df[f'news_{cat}'] for cat in CATEGORIES)
    return df
