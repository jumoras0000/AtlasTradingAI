"""DEMO-only read bridge. No order_send, no live trading functionality.
Run on Windows computer with installed MT5 terminal and MetaTrader5 package.
"""
import os
import pandas as pd

def read_demo_bars(symbol='EURUSD',count=2000,timeframe='H1'):
    try:import MetaTrader5 as mt5
    except ImportError as e:raise RuntimeError('Install requirements-mt5.txt on Windows') from e
    path=os.getenv('MT5_TERMINAL_PATH') or None
    initialized = mt5.initialize(path=path) if path else mt5.initialize()
    if not initialized:
        raise RuntimeError(f'MT5 terminal initialization failed: {mt5.last_error()}')
    try:
        info=mt5.account_info()
        if info is None:raise RuntimeError('No MT5 account connected')
        if info.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise PermissionError('Only DEMO accounts permitted for data export in this package')
        tf={'M1':mt5.TIMEFRAME_M1,'M5':mt5.TIMEFRAME_M5,'H1':mt5.TIMEFRAME_H1,
            'H4':mt5.TIMEFRAME_H4,'D1':mt5.TIMEFRAME_D1}[timeframe]
        if not mt5.symbol_select(symbol,True):raise RuntimeError(f'Symbol unavailable: {symbol}')
        rates=mt5.copy_rates_from_pos(symbol,tf,1,count) # position 1 avoids current open bar
        if rates is None or len(rates)==0:raise RuntimeError(f'No bars: {mt5.last_error()}')
        df=pd.DataFrame(rates)
        df['time']=pd.to_datetime(df.time,unit='s',utc=True)
        df=df.rename(columns={'tick_volume':'volume'})
        return df[['time','open','high','low','close','volume']].sort_values('time')
    finally:mt5.shutdown()
