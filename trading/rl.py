"""Optional PPO research environment. Requires requirements-rl.txt.
Observation at t is closed bar t; action establishes exposure at t+1 OPEN.
"""
from pathlib import Path
import numpy as np
from .data import features
from .ml import COLS


def make_env(prices, fee_bps=8, slippage_bps=5):
    import gymnasium as gym
    from gymnasium import spaces
    class TradingEnv(gym.Env):
        metadata={'render_modes':[]}
        def __init__(self):
            self.p=features(prices).reset_index(drop=True)
            self.observation_space=spaces.Box(-1e8,1e8,shape=(len(COLS)+1,),dtype=np.float32)
            self.action_space=spaces.Discrete(3) # flat, 10%, 20%; LONG ONLY
            self.reset()
        def _obs(self):
            return np.nan_to_num(np.r_[self.p.loc[self.i,COLS].to_numpy(dtype=float),self.position],
                                nan=0,posinf=0,neginf=0).astype(np.float32)
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            self.i=0;self.position=0.;self.equity=1.;self.peak=1.
            return self._obs(),{}
        def step(self,action):
            if self.i>=len(self.p)-2: raise RuntimeError('Episode finished, call reset')
            target=(0.,.1,.2)[int(action)]
            fee=(fee_bps+slippage_bps)/10000*abs(target-self.position)
            self.position=target
            nxt=self.p.iloc[self.i+1]
            # Exposure established at NEXT open, earns next open->close, not past close->open.
            r=float(nxt.close/nxt.open-1)
            pnl=self.position*r-fee
            self.equity*=max(1e-8,1+pnl)
            self.peak=max(self.peak,self.equity)
            drawdown=1-self.equity/self.peak
            reward=float(pnl-0.02*drawdown)
            self.i+=1
            terminated=self.i>=len(self.p)-2 or drawdown>=0.05
            return self._obs(),reward,terminated,False,{'equity':self.equity,'drawdown':drawdown}
    return TradingEnv()

def train_ppo(prices,output='models/ppo',timesteps=15000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    p=prices.iloc[:int(len(prices)*.7)].copy()
    env=make_env(p)
    model=PPO('MlpPolicy',env,verbose=0,seed=42,n_steps=256,batch_size=64,
              learning_rate=0.0003,device='cpu')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=CheckpointCallback(save_freq=5000,save_path=str(path.parent),name_prefix='ppo_ckpt')
    model.learn(total_timesteps=timesteps,callback=checkpoint)
    model.save(str(path));env.close()
    return {'training_bars':len(p),'timesteps':timesteps,'model_path':str(path)+'.zip',
            'note':'Unvalidated research model; never use without out-of-sample tests'}
