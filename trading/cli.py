import argparse,json
from pathlib import Path
import pandas as pd
from .data import generate_demo,load_prices,features
from .backtest import simulate,ema_signals,write_report,BacktestSettings
from .risk import RiskConfig

def main(argv=None):
    ap=argparse.ArgumentParser(description='Trading AI offline research tools (NO LIVE ORDERS)')
    sub=ap.add_subparsers(dest='cmd',required=True)
    g=sub.add_parser('generate-demo');g.add_argument('--output',default='data/demo.csv');g.add_argument('--rows',type=int,default=1600)
    b=sub.add_parser('backtest');b.add_argument('--csv',default='data/demo.csv');b.add_argument('--prefix',default='baseline')
    x=sub.add_parser('train-xgb');x.add_argument('--csv',default='data/demo.csv')
    p=sub.add_parser('train-ppo');p.add_argument('--csv',default='data/demo.csv');p.add_argument('--steps',type=int,default=15000)
    n=sub.add_parser('collect-news')
    e=sub.add_parser('export-mt5');e.add_argument('--symbol',default='EURUSD');e.add_argument('--count',type=int,default=2000);e.add_argument('--output',default='data/mt5_demo.csv')
    args=ap.parse_args(argv)
    if args.cmd=='generate-demo':
        df=generate_demo(args.output,args.rows);print(f'Generated {len(df)} SYNTHETIC candles -> {args.output}')
    elif args.cmd=='backtest':
        data=features(load_prices(args.csv))
        s,d=simulate(data,ema_signals(data),BacktestSettings())
        write_report(s,d,prefix=args.prefix);print(json.dumps(s,indent=2))
    elif args.cmd=='train-xgb':
        from .ml import train_xgb
        print(json.dumps(train_xgb(load_prices(args.csv)),indent=2))
    elif args.cmd=='train-ppo':
        from .rl import train_ppo
        print(json.dumps(train_ppo(load_prices(args.csv),timesteps=args.steps),indent=2))
    elif args.cmd=='collect-news':
        from .news import collect
        print(json.dumps(collect(),indent=2))
    elif args.cmd=='export-mt5':
        from .mt5_bridge import read_demo_bars
        df=read_demo_bars(args.symbol,args.count)
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(dest,index=False)
        print(f'Exported {len(df)} bars from DEMO -> {dest}')
if __name__=='__main__':main()
