"""Out-of-sample evaluation for trained PPO agent."""
import json
from pathlib import Path
import numpy as np
from .data import features
from .rl import make_env
from .risk import RiskConfig


def evaluate_ppo(prices, model_path='models/ppo.zip', train_fraction=0.7,
                 fee_bps=8, slippage_bps=5, risk: RiskConfig | None = None) -> dict:
    """Run trained PPO on the OOS portion (after train_fraction) and collect metrics."""
    from stable_baselines3 import PPO

    n = int(len(prices) * train_fraction)
    oos = prices.iloc[n:].copy().reset_index(drop=True)

    if len(oos) < 100:
        raise ValueError(f'Only {len(oos)} OOS bars — need at least 100')

    env = make_env(oos, fee_bps=fee_bps, slippage_bps=slippage_bps, risk=risk)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    equities = [1.0]
    actions_taken = []
    reasons = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        equities.append(info['equity'])
        actions_taken.append(int(action))
        reasons.append(info.get('risk_reason', 'unknown'))
        done = terminated or truncated

    equities = np.array(equities)
    peak = np.maximum.accumulate(equities)
    drawdowns = equities / peak - 1

    action_counts = {
        'flat': actions_taken.count(0),
        'exposure_10pct': actions_taken.count(1),
        'exposure_20pct': actions_taken.count(2),
    }
    risk_events = {r: reasons.count(r) for r in set(reasons) if r != 'approved'}

    return {
        'oos_bars': len(oos),
        'steps_played': len(actions_taken),
        'final_equity': float(equities[-1]),
        'oos_return_pct': float((equities[-1] - 1) * 100),
        'max_drawdown_pct': float(drawdowns.min() * 100),
        'sharpe_approx': float(
            np.mean(np.diff(equities) / equities[:-1]) /
            max(np.std(np.diff(equities) / equities[:-1]), 1e-10)
        ),
        'action_distribution': action_counts,
        'risk_events': risk_events,
        'note': 'OOS evaluation only. NOT proof of profitability.',
    }


def write_eval_report(result: dict, directory='reports', prefix='ppo_oos') -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{prefix}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
