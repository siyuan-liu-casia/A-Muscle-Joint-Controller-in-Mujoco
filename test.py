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
# model_dir = r'output/PPO-Arm12-Time-2025-12-03-11-42-seed-2025/models/rl_model_10000000_steps.zip'


step_number = model_dir.split('_')[-2]
# test_dir = os.path.join('test', f'alpha_0_{step_number}')
test_dir = os.path.join('test', f'alpha_0.5_{step_number}')
os.makedirs(test_dir, exist_ok=True)
pi = PPO.load(model_dir)
for t in range(10):
    state, _ = env.reset() 
    target = env.target
    env.frames = []  

    for i in range(500):
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

    video_path = os.path.join(test_dir, f"episode_{t + 1}_video.mp4")
    save_video(env.frames, video_path)
 
    qpos_array = np.array(qpos_list)
    target_array = np.array(target_list)
    time_array = np.array(time_list) - time_list[0]  # Shift time to start from 0
    action_array = np.array(action_list)

    plot_path = os.path.join(test_dir, f"episode_{t + 1}_qpos_target.jpg")
    plt.figure(figsize=(4, 3))
    plt.plot(time_array, qpos_array, label='qpos')
    plt.plot(time_array, target_array, label='target', linestyle='--')
    plt.legend()
    plt.title(f'Episode {t + 1}', fontsize=14)
    plt.xlabel('Time / s', fontsize=12)
    plt.ylabel('Joint Angle / rad', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plot_path = os.path.join(test_dir, f"episode_{t + 1}_action.jpg")
    plt.figure(figsize=(4, 3))
    plt.plot(time_array, action_array[:, 0], label='EF')
    plt.plot(time_array, action_array[:, 1], label='EE')
    plt.legend()
    plt.ylim([-0.1, 1.1])
    plt.title(f'Episode {t + 1}', fontsize=14)
    plt.xlabel('Time / s', fontsize=12)
    plt.ylabel('Action', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

print('done')