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

env = Arm12(render_mode=False)
state = env.reset()
model_dir = r'output/PPO-Arm12-Time-2025-12-03-11-42-seed-2025/models/rl_model_4000000_steps.zip'
pi = PPO.load(model_dir)

step_number = model_dir.split('_')[-2]
test_dir = os.path.join('test', step_number)
os.makedirs(test_dir, exist_ok=True)

for t in range(10):
    state, _ = env.reset() 
    target = env.target
    env.frames = []  

    for i in range(500):
        action = pi.predict(state)[0]
        state, reward, done, _, info = env.step(action)  
        qpos = env.mj_data.qpos.copy()
        env.render()

        if i == 0:
            qpos_list = []
            target_list = []
        qpos_list.append(qpos)
        target_list.append(target) 
    video_path = os.path.join(test_dir, f"episode_{t + 1}_video.mp4")
    save_video(env.frames, video_path)
 
    qpos_array = np.array(qpos_list)
    target_array = np.array(target_list)

    plot_path = os.path.join(test_dir, f"episode_{t + 1}_qpos_target.png")
    plt.figure()
    plt.plot(qpos_array, label='qpos')
    plt.plot(target_array, label='target', linestyle='--')
    plt.legend()
    plt.title(f'Episode {t + 1}')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.savefig(plot_path)
    plt.close()

print('done')