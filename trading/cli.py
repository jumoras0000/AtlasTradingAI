import argparse, json
from pathlib import Path
import pandas as pd
from .data import generate_demo, load_prices, features
from .backtest import simulate, ema_signals, write_report, BacktestSettings
from .risk import RiskConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description='Trading AI offline research tools (NO LIVE ORDERS)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('generate-demo')
    g.add_argument('--output', default='data/demo.csv')
    g.add_argument('--rows', type=int, default=1600)

    b = sub.add_parser('backtest')
    b.add_argument('--csv', default='data/demo.csv')
    b.add_argument('--prefix', default='baseline')

    x = sub.add_parser('train-xgb')
    x.add_argument('--csv', default='data/demo.csv')

    p = sub.add_parser('train-ppo')
    p.add_argument('--csv', default='data/demo.csv')
    p.add_argument('--steps', type=int, default=15000)

    n = sub.add_parser('collect-news')

    e = sub.add_parser('export-mt5')
    e.add_argument('--symbol', default='EURUSD')
    e.add_argument('--count', type=int, default=2000)
    e.add_argument('--output', default='data/mt5_demo.csv')

    f = sub.add_parser('fetch-real')
    f.add_argument('--symbol', default='EURUSD')
    f.add_argument('--period', default='2y')
    f.add_argument('--interval', default='1h')
    f.add_argument('--output', default='data/real.csv')

    w = sub.add_parser('walkforward')
    w.add_argument('--csv', default='data/real.csv')
    w.add_argument('--min-train', type=int, default=300)
    w.add_argument('--test-window', type=int, default=100)
    w.add_argument('--step', type=int, default=50)
    w.add_argument('--prefix', default='walkforward')

    ep = sub.add_parser('eval-ppo')
    ep.add_argument('--csv', default='data/demo.csv')
    ep.add_argument('--model', default='models/ppo.zip')
    ep.add_argument('--prefix', default='ppo_oos')

    xn = sub.add_parser('train-xgb-news')
    xn.add_argument('--csv', default='data/demo.csv')
    xn.add_argument('--news', default='data/news.json')

    cf = sub.add_parser('compare-fees')
    cf.add_argument('--csv', default='data/demo.csv')

    args = ap.parse_args(argv)

    if args.cmd == 'generate-demo':
        df = generate_demo(args.output, args.rows)
        print(f'Generated {len(df)} SYNTHETIC candles -> {args.output}')

    elif args.cmd == 'backtest':
        data = features(load_prices(args.csv))
        s, d = simulate(data, ema_signals(data), BacktestSettings())
        write_report(s, d, prefix=args.prefix)
        print(json.dumps(s, indent=2))

    elif args.cmd == 'train-xgb':
        from .ml import train_xgb
        print(json.dumps(train_xgb(load_prices(args.csv)), indent=2))

    elif args.cmd == 'train-ppo':
        from .rl import train_ppo
        print(json.dumps(train_ppo(load_prices(args.csv), timesteps=args.steps), indent=2))

    elif args.cmd == 'collect-news':
        from .news import collect
        print(json.dumps(collect(), indent=2))

    elif args.cmd == 'export-mt5':
        from .mt5_bridge import read_demo_bars
        df = read_demo_bars(args.symbol, args.count)
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f'Exported {len(df)} bars from DEMO -> {dest}')

    elif args.cmd == 'fetch-real':
        from .fetch import fetch_yahoo
        df = fetch_yahoo(args.symbol, args.period, args.interval, args.output)
        print(f'Fetched {len(df)} real bars ({args.symbol}) -> {args.output}')

    elif args.cmd == 'walkforward':
        from .walkforward import walk_forward_xgb, write_walkforward_report
        prices = load_prices(args.csv)
        result = walk_forward_xgb(
            prices, min_train=args.min_train,
            test_window=args.test_window, step=args.step
        )
        write_walkforward_report(result, prefix=args.prefix)
        summary = {k: v for k, v in result.items() if k != 'folds'}
        print(json.dumps(summary, indent=2))

    elif args.cmd == 'eval-ppo':
        from .eval_ppo import evaluate_ppo, write_eval_report
        prices = load_prices(args.csv)
        result = evaluate_ppo(prices, model_path=args.model)
        write_eval_report(result, prefix=args.prefix)
        print(json.dumps(result, indent=2))

    elif args.cmd == 'train-xgb-news':
        from .ml import train_xgb_with_news
        result = train_xgb_with_news(load_prices(args.csv), news_path=args.news)
        print(json.dumps(result, indent=2))

    elif args.cmd == 'compare-fees':
        from .config import load_config, make_settings
        cfg = load_config()
        profiles = cfg.get('profiles', {})
        prices = load_prices(args.csv)
        data = features(prices)
        targets = ema_signals(data)
        results = {}
        for name, overrides in profiles.items():
            merged = {**cfg, **overrides}
            s = make_settings(merged)
            summary, detail = simulate(data, targets, s)
            write_report(summary, detail, prefix=f'fees_{name}')
            results[name] = {
                'fee_bps': overrides.get('fee_bps', cfg['fee_bps']),
                'slippage_bps': overrides.get('slippage_bps', cfg['slippage_bps']),
                'net_return_pct': summary['net_return_pct'],
                'max_drawdown_pct': summary['max_drawdown_pct'],
                'total_cost': summary['total_cost'],
            }
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

