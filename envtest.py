import mujoco
import time
from mujoco import viewer
import os
import gym
import numpy as np
import skvideo.io
from arm12 import Arm12, save_video

if __name__ == "__main__":
  env = Arm12(render_mode=True)
  env.reset()
  state = []
  for i in range(1000): 
      action = env.action_space.sample()
      state, reward, done, _ = env.step(action) 
      env.render()
  if not env.render_mode:
    save_video(env.frames) 
