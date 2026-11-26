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

    lv = sub.add_parser('live')
    lv.add_argument('--symbol', default='EURUSD')
    lv.add_argument('--timeframe', default='H1', choices=['M1', 'M5', 'M15', 'H1', 'H4', 'D1'])
    lv.add_argument('--strategy', default='ema', choices=['ema', 'xgb', 'ppo'])
    lv.add_argument('--model', default='models/xgb.json')
    lv.add_argument('--interval', type=int, default=0)
    lv.add_argument('--max-iter', type=int, default=0)
    lv.add_argument('--once', action='store_true')

    hc = sub.add_parser('health-check')
    hc.add_argument('--symbol', default='EURUSD')

    rc = sub.add_parser('reconcile')
    rc.add_argument('--symbol', default='EURUSD')
    rc.add_argument('--expected', type=float, default=0.0)

    ac = sub.add_parser('account')

    cl = sub.add_parser('close-all')
    cl.add_argument('--symbol', default='EURUSD')

    sc = sub.add_parser('scan')
    sc.add_argument('--timeframe', default='H1')
    sc.add_argument('--min-score', type=float, default=0.15)

    mt = sub.add_parser('multi')
    mt.add_argument('--timeframe', default='H1', choices=['M5', 'M15', 'H1', 'H4', 'D1'])
    mt.add_argument('--max-positions', type=int, default=5)
    mt.add_argument('--max-exposure', type=float, default=0.50)
    mt.add_argument('--min-score', type=float, default=0.15)
    mt.add_argument('--model', default='models/xgb.json')
    mt.add_argument('--interval', type=int, default=0)
    mt.add_argument('--max-iter', type=int, default=0)
    mt.add_argument('--once', action='store_true')

    rt = sub.add_parser('retrain')

    qc = sub.add_parser('quality-check')
    qc.add_argument('--symbol', default='EURUSD')

    cv = sub.add_parser('convergence')

    at = sub.add_parser('auto-train')
    at.add_argument('--symbols', nargs='+', default=['EURUSD'])
    at.add_argument('--timeframe', default='H1', choices=['M5', 'M15', 'H1', 'H4', 'D1'])
    at.add_argument('--interval', type=float, default=24, help='Hours between cycles')
    at.add_argument('--max-cycles', type=int, default=0, help='0=infinite')
    at.add_argument('--once', action='store_true', help='Single cycle then exit')
    at.add_argument('--all-symbols', action='store_true', help='Train on all scanned symbols')

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

    elif args.cmd == 'live':
        from .live_loop import LiveTrader
        interval = args.interval or {'M1': 60, 'M5': 300, 'H1': 3600, 'H4': 14400, 'D1': 86400}.get(args.timeframe, 3600)
        trader = LiveTrader(args.symbol, args.timeframe, args.strategy, model_path=args.model)
        if args.once:
            print(json.dumps(trader.run_once(), indent=2))
        else:
            trader.run_loop(interval_seconds=interval, max_iterations=args.max_iter)

    elif args.cmd == 'health-check':
        from .reconciliation import full_health_check
        print(json.dumps(full_health_check(args.symbol), indent=2))

    elif args.cmd == 'reconcile':
        from .reconciliation import reconcile
        print(json.dumps(reconcile(args.symbol, args.expected), indent=2))

    elif args.cmd == 'account':
        from .mt5_executor import get_account_info
        print(json.dumps(get_account_info(), indent=2))

    elif args.cmd == 'close-all':
        from .mt5_executor import close_all
        results = close_all(args.symbol)
        print(json.dumps([r.__dict__ for r in results], indent=2))

    elif args.cmd == 'scan':
        from .scanner import scan_all
        result = scan_all(timeframe=args.timeframe, min_confidence=args.min_score)
        # Print summary
        print(f"\n{'Symbol':<12} {'Direction':<8} {'Score':>8} {'Confidence':>10} {'Trend':>8} {'XGB':>6}")
        print('-' * 60)
        for o in result['opportunities']:
            print(f"{o['symbol']:<12} {o['direction']:<8} {o['score']:>8.3f} {o['confidence']:>10.3f} {o['trend']:>8.4f} {o['xgb_proba']:>6.3f}")
        print(f"\n{result['total_opportunities']} opportunities / {result['total_scanned']} scanned")

    elif args.cmd == 'multi':
        from .multi_trader import MultiTrader
        interval = args.interval or {'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400}[args.timeframe]
        trader = MultiTrader(
            timeframe=args.timeframe, max_positions=args.max_positions,
            max_total_exposure=args.max_exposure, min_score=args.min_score, model_path=args.model,
        )
        if args.once:
            print(json.dumps(trader.run_once(), indent=2))
        else:
            trader.run_loop(interval_seconds=interval, max_iterations=args.max_iter)

    elif args.cmd == 'quality-check':
        from .auto_learn import QualityGate, ModelManager
        from .walkforward import walk_forward_xgb
        mgr = ModelManager(args.symbol)
        prices = mgr.collect_latest_data(count=2000)
        eval_result = mgr.evaluate_model(prices, mgr.active_model)
        wf = walk_forward_xgb(prices, min_train=300, test_window=100, step=50)
        gate = QualityGate()
        result = gate.evaluate(eval_result, wf)
        print(f"\n{'Gate':<25} {'Value':>10} {'Threshold':>10} {'Status':>8}")
        print('-' * 58)
        for name, c in result['checks'].items():
            v = c['value']
            t = c['threshold']
            s = 'PASS' if c['pass'] else 'FAIL'
            print(f"{name:<25} {v:>10.3f} {t:>10.3f} {s:>8}")
        print(f"\nVerdict: {result['verdict']}")
        print(result['message'])

    elif args.cmd == 'convergence':
        from .auto_learn import QualityGate
        gate = QualityGate()
        summary = gate.convergence_summary()
        if summary['status'] == 'no_data':
            print('No training history yet. Run: python -m trading.cli auto-train --once')
        else:
            print(f"\nStatus: {summary['status']}")
            print(f"Cycles: {summary['cycles_completed']}")
            print(f"Consecutive pass: {summary['consecutive_pass']} / {summary['needed_consecutive']}")
            print(f"Trend: {summary['trend']}")
            print(f"\n{'Metric':<20} {'Current':>10} {'Target':>10} {'Progress':>10} {'Status':>8}")
            print('-' * 63)
            for name, p in summary.get('progress', {}).items():
                print(f"{name:<20} {p['current']:>10.3f} {p['target']:>10.3f} {p['progress_pct']:>9.1f}% {'PASS' if p['pass'] else 'FAIL':>8}")
            print(f"\n{summary['latest_message']}")

    elif args.cmd == 'retrain':
        from .auto_learn import AutoLearner
        learner = AutoLearner()
        result = learner.force_retrain()
        print(json.dumps(result, indent=2))

    elif args.cmd == 'auto-train':
        from .auto_learn import TrainingScheduler
        symbols = args.symbols
        if args.all_symbols:
            from .scanner import get_available_symbols, ALL_WATCHLIST
            available = get_available_symbols()
            symbols = [s['symbol'] for s in available]
            print(f'Training on {len(symbols)} available symbols')
        scheduler = TrainingScheduler(
            symbols=symbols, timeframe=args.timeframe,
            interval_hours=args.interval,
        )
        if args.once:
            report = scheduler.run_cycle()
            print(json.dumps(report, indent=2))
        else:
            scheduler.run_loop(max_cycles=args.max_cycles)


if __name__ == '__main__':
    main()

