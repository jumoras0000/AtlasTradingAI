from dataclasses import dataclass

@dataclass
class RiskConfig:
    max_exposure: float = 0.20
    max_trade_fraction: float = 0.10
    max_daily_loss_fraction: float = 0.01
    max_drawdown_fraction: float = 0.05

    def __post_init__(self):
        for v in vars(self).values():
            if not 0 < v <= 1: raise ValueError('Risk fractions must be in (0,1]')

def clamp_target(target: float, current: float, equity: float, peak: float, day_start: float,
                 config: RiskConfig) -> tuple[float, str]:
    """Target is long-only portfolio fraction; positive exposure changes capped."""
    if equity<=0: return 0.0,'equity_zero'
    if equity < peak*(1-config.max_drawdown_fraction): return 0.0,'drawdown_kill_switch'
    if equity < day_start*(1-config.max_daily_loss_fraction): return 0.0,'daily_loss_kill_switch'
    if target != target: return current,'invalid_target'
    target = max(0.,min(float(target),config.max_exposure))
    # Allow reductions immediately; cap increases to a single-trade increment.
    if target>current: target=min(target,current+config.max_trade_fraction)
    return target,'approved'
