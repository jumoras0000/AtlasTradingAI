"""Closed-bar signal applied at NEXT bar open. Long-only no leverage.
Research simulator, not a broker fill-engine. Fees and slippage modeled per turnover.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .risk import RiskConfig,clamp_target

@dataclass
class BacktestSettings:
    starting_cash:float=10000.
    fee_bps:float=8.
    slippage_bps:float=5.
    risk:RiskConfig|None=None

    def __post_init__(self):
        if self.starting_cash<=0 or self.fee_bps<0 or self.slippage_bps<0:raise ValueError('Invalid backtest settings')
        if self.risk is None:self.risk=RiskConfig()

def ema_signals(p:pd.DataFrame)->np.ndarray:
    return (p.ema20>p.ema50).astype(float).to_numpy()*0.20

def simulate(p:pd.DataFrame, targets, settings:BacktestSettings|None=None):
    """Targets at bar t CLOSE applied at bar t+1 OPEN, then mark at t+1 CLOSE.
    Rebalances at opens; no fills at future inaccessible timestamps.
    """
    s = settings or BacktestSettings()
    targets=np.asarray(targets,dtype=float)
    if len(targets)!=len(p):raise ValueError('Target length mismatch')
    if len(p)<2:raise ValueError('Need >=2 bars')
    cash=float(s.starting_cash); units=0.0; peak=cash; day_start=cash
    current_day=p.iloc[0].time.date()
    rows=[]
    for i in range(1,len(p)):
        bar=p.iloc[i]; open_price=float(bar.open); close_price=float(bar.close)
        equity_open=cash+units*open_price
        if bar.time.date()!=current_day:
            current_day=bar.time.date();day_start=equity_open
        peak=max(peak,equity_open)
        current_fraction=units*open_price/equity_open if equity_open>0 else 0
        target,reason=clamp_target(float(targets[i-1]),current_fraction,equity_open,peak,day_start,s.risk)
        # Sell first if decreasing; for buys account for transaction costs.
        requested_value=(target-current_fraction)*equity_open
        cost_rate=(s.fee_bps+s.slippage_bps)/10000
        if requested_value>0:
            requested_value=min(requested_value,cash/(1+cost_rate))
        elif requested_value<0:
            requested_value=max(requested_value,-units*open_price)
        qty=requested_value/open_price
        turnover=abs(requested_value)
        cost=turnover*cost_rate
        cash-=requested_value+cost
        units+=qty
        equity=cash+units*close_price
        peak=max(peak,equity)
        rows.append({'time':str(bar.time),'equity':equity,'cash':cash,'units':units,
                     'target':target,'reason':reason,'turnover':turnover,'cost':cost})
    result=pd.DataFrame(rows)
    equity=result.equity
    drawdown=(equity/equity.cummax()-1).min()
    summary={'initial_equity':s.starting_cash,'final_equity':float(equity.iloc[-1]),
             'net_return_pct':float((equity.iloc[-1]/s.starting_cash-1)*100),
             'max_drawdown_pct':float(drawdown*100),
             'total_cost':float(result.cost.sum()),
             'rebalance_events':int((result.turnover>1e-7).sum()),
             'execution_mode':'offline_simulation_only'}
    return summary,result

def write_report(summary:dict,detail:pd.DataFrame,directory='reports',prefix='baseline'):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    (d/f'{prefix}.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    detail.to_csv(d/f'{prefix}.csv',index=False)
