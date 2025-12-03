import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
import os
import numpy as np
from matplotlib import pyplot as plt
import datetime
import shutil
from utils import *
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import skvideo.io
from arm12 import Arm12 

seed = 2025

set_random_seed(seed=seed, using_cuda=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
current_file = os.path.basename(__file__)
nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

save_dir = os.path.join(current_dir, f"output/PPO-Arm12-Time-{nowtime}-seed-{seed}")
model_dir = os.path.join(save_dir, "models")
loggir_dir = os.path.join(save_dir, "logs") 

os.makedirs(model_dir, exist_ok=True)
os.makedirs(loggir_dir, exist_ok=True) 

shutil.copyfile(os.path.join(current_dir, current_file), os.path.join(save_dir, current_file))
shutil.copyfile(os.path.join(current_dir, 'arm12.py'), os.path.join(save_dir, 'arm12.py'))

env = Arm12(render_mode=False)
env.reset()

net_shape = [32, 32]
policy_kwargs = dict(net_arch=dict(pi=net_shape, vf=net_shape))
model = PPO("MlpPolicy", 
            env, verbose=1, 
            policy_kwargs=policy_kwargs,
            tensorboard_log=loggir_dir, 
            n_steps=16384, 
            batch_size=512, 
            n_epochs=10)

TimeSteps = 1_000_000

checkpoint_callback = CheckpointCallback(
    save_freq=TimeSteps,
    save_path=model_dir,
    save_vecnormalize=True,
    verbose=1,
)

tensorboard_callback = TensorboardCallback("arm", info_keywords=ENV_INFO["arm"])

callback_list = [checkpoint_callback,  
                 tensorboard_callback
                 ]

model.learn(total_timesteps = 10 * TimeSteps + 1, callback = callback_list)
