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
from arm12 import Arm12, save_video
plt.rcParams['font.family'] = 'Times New Roman'
env = Arm12(render_mode=False)
state = env.reset()
model_dir = r'output/PPO-Arm12-Time-2025-12-03-12-54-seed-2025/models/rl_model_10000000_steps.zip' 

step_number = model_dir.split('_')[-2]   
pi = PPO.load(model_dir)
for t in range(5):  
    state, _ = env.reset() 
    target = env.target 

    for i in range(300):
        action = pi.predict(state, deterministic=True)[0]
        state, reward, done, _, info = env.step(action)  
        time = env.mj_data.time
        qpos = env.mj_data.qpos.copy()
        env.render()

        if i == 0:
            qpos_list = []
            target_list = []
            time_list = []
            action_list = []
        action_list.append(action)
        qpos_list.append(qpos)
        target_list.append(target) 
        time_list.append(time)

video_path = f"test_video.mp4"  
save_video(env.frames, video_path)

print('done')