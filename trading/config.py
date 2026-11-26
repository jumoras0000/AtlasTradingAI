"""Load settings from YAML config. Single source of truth for all modules."""
from pathlib import Path
import yaml
from .backtest import BacktestSettings
from .risk import RiskConfig

DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'config' / 'default.yaml'


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    with open(p, encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_risk(cfg: dict) -> RiskConfig:
    return RiskConfig(
        max_exposure=cfg.get('max_exposure', 0.20),
        max_trade_fraction=cfg.get('max_trade_fraction', 0.10),
        max_daily_loss_fraction=cfg.get('max_daily_loss_fraction', 0.01),
        max_drawdown_fraction=cfg.get('max_drawdown_fraction', 0.05),
    )


def make_settings(cfg: dict) -> BacktestSettings:
    return BacktestSettings(
        starting_cash=cfg.get('starting_cash', 10000.0),
        fee_bps=cfg.get('fee_bps', 8.0),
        slippage_bps=cfg.get('slippage_bps', 5.0),
        risk=make_risk(cfg),
    )
