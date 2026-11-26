import numpy as np
import pandas as pd
import pytest
from trading.data import generate_demo,load_prices,features
from trading.backtest import simulate,ema_signals,BacktestSettings
from trading.risk import clamp_target,RiskConfig
from trading.news import keyword_tag

def test_demo_valid(tmp_path):
    p=tmp_path/'prices.csv'
    generate_demo(p,n=180)
    df=load_prices(p)
    assert len(df)==180
    assert df.time.dt.tz is not None
    assert len(features(df))>100

def test_reject_bad_ohlc(tmp_path):
    p=tmp_path/'bad.csv';df=generate_demo(p,n=120)
    df.loc[0,'high']=0.01;df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Inconsistent'):load_prices(p)

def test_risk_clamps_and_kill_switch():
    r=RiskConfig()
    assert clamp_target(.9,0,100,100,100,r)==(.1,'approved')
    assert clamp_target(.2,.2,94,100,100,r)[1]=='drawdown_kill_switch'
    assert clamp_target(.2,.2,98,98,100,r)[1]=='daily_loss_kill_switch'

def test_simulation_exposure_and_cash(tmp_path):
    p=features(generate_demo(tmp_path/'x.csv',n=220))
    summary,detail=simulate(p,ema_signals(p))
    assert summary['execution_mode']=='offline_simulation_only'
    assert (detail.cash>=-1e-8).all()
    assert (detail.units>=-1e-8).all()
    assert detail.equity.gt(0).all()
    assert detail.target.le(.20+1e-8).all()

def test_no_lookahead_future_signal(tmp_path):
    p=features(generate_demo(tmp_path/'x.csv',n=220))
    t=np.zeros(len(p));t[-1]=1.
    _,detail=simulate(p,t)
    assert (detail.turnover==0).all(), 'Last close signal must not trade before future open'

def test_fees_lower_pnl(tmp_path):
    p=features(generate_demo(tmp_path/'x.csv',n=220))
    targets=np.full(len(p),.2)
    a,_=simulate(p,targets,BacktestSettings(fee_bps=0,slippage_bps=0))
    b,_=simulate(p,targets,BacktestSettings(fee_bps=50,slippage_bps=50))
    assert b['final_equity']<a['final_equity']

def test_news_keywords():
    assert 'inflation' in keyword_tag('Consumer Prices and Inflation')
