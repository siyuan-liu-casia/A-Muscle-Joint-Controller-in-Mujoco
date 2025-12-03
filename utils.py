import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque as dq
import os
import skvideo.io

ARM_INFO_KEYS = {
    "qpos_err" 
}

ENV_INFO={
    "arm": ARM_INFO_KEYS,
}

class TensorboardCallback(BaseCallback):
    def __init__(self, env_name, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.env_name = env_name
        self.info_keywords = info_keywords
        self.rollout_info = {}
        
    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}
        
    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True
    
    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record(f"{self.env_name}/" + key, np.mean(self.rollout_info[key]))

 