import mujoco
import time
from mujoco import viewer
import os 
import gymnasium as gym
import numpy as np
import skvideo.io

FRAMESKIP = 2
FRAMERATE = 100

# offline without sphere drawing so far
OFFLINE_RENDER = False
curr_dir = os.path.dirname(os.path.realpath(__file__))

def save_video(frames, filename):
    save_path = os.path.join(curr_dir, f"{filename}")
    print(f"Saving video to: {save_path}")
    skvideo.io.vwrite(
        save_path,
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p"},
        inputdict={"-r": str(FRAMERATE)},
    ) 

# class Arm12:
class Arm12(gym.Env): 
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FRAMERATE}
    def __init__(self, render_mode=False, max_steps=500):
        curr_dir = os.path.dirname(__file__)
        xml_path = os.path.join(curr_dir, "./assets/arm12.xml")

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.init_qpos = self.mj_data.qpos.copy()
        nactions = self.mj_model.nu
        self.action_space = gym.spaces.Box(np.zeros((nactions,)), np.ones((nactions,)))
        self.observation_space = gym.spaces.Box(-np.ones((6,)), np.ones((6,)))    
        self.init = False
        self.init_visuals = False
        self.timestep = 0
        self.render_mode = render_mode
        self.max_steps = max_steps

    def generate_target(self):
        self.target = np.random.uniform(0, 2.09, size=(1,))
 
    def draw_sphere(self):
        if hasattr(self, "viewer"):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                return
            if not self.init_visuals:
                self.viewer.user_scn.ngeom += 1   
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[self.timestep],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.02, 0, 0], dtype=np.float64),
                    pos=np.array([self.target[0], 0, 0], dtype=np.float64),
                    mat=np.eye(3, dtype=np.float64).flatten(),
                    rgba=np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
                )
                self.init_visuals = True
            else:
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1].pos[:] = ( 
                    np.array([self.target[0], 0.0, 0.0], dtype=np.float64)
                )

            self.viewer.sync()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) 
        self.timestep = 0
        self.draw_sphere()
        self.generate_target()
        self.mj_data.qpos[:] = np.array([0.0])
        self.mj_data.ctrl[:] = np.array([0.0])
        self.mj_data.act[:] = np.random.uniform(size=self.mj_model.nu)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.init = True
        return self.compute_obs(), {}

    def compute_obs(self):
        return np.concatenate(
            [
                self.mj_data.qpos.copy(),
                self.mj_data.qvel.copy(), 
                self.target,
                self.target-self.mj_data.qpos.copy(),
                self.muscle_activity(), 
            ],
            dtype=np.float32, 
        ).copy()
    
    def reward(self):
        qpos = self.mj_data.qpos.copy()
        qpos_err = self.target - qpos
        return np.exp(-10 * np.linalg.norm(qpos_err))

    def step(self, action):
        self.timestep += 1
        if not self.init:
            raise Exception("Reset has to be called once before step")
        self.mj_data.ctrl[:] = action
        for i in range(FRAMESKIP):
            mujoco.mj_step(self.mj_model, self.mj_data)
        if self.timestep>=self.max_steps:
            done = True
        else:
            done = False
        state = self.compute_obs()
        reward = self.reward()
        info = {}
        info["qpos_err"] = np.linalg.norm(self.target-self.mj_data.qpos.copy())
        return state, reward, done, False, info 
    

    def render(self):
        if self.render_mode:
            if not hasattr(self, "viewer"):
                self.viewer = viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                self.viewer.sync()
            with self.viewer.lock():
                self.draw_sphere()
            self.viewer.sync()
            time.sleep(0.01)

        else:
            if not hasattr(self, "renderer"):
                self.renderer = mujoco.Renderer(
                    self.mj_model,
                    height=1080,
                    width=1080,
                )
                self.frames = []
            self.renderer.update_scene(self.mj_data, camera="right")
            frame = self.renderer.render()
            self.frames.append(frame)

    def muscle_length(self):
        return self.mj_data.actuator_length.copy()

    def muscle_velocity(self):
        return self.mj_data.actuator_velocity.copy()

    def muscle_force(self):
        return self.mj_data.actuator_force.copy()

    def muscle_activity(self):
        return self.mj_data.act.copy()


if __name__ == "__main__":
    env = Arm12(render_mode=True)
 
    for ep in range(10):
        env.reset()
        print(f"Episode {ep+1}")
        for i in range(100): 
            action = env.action_space.sample()
            env.step(action)
            env.render()

    if OFFLINE_RENDER:
        save_video(env.frames, "rendered_video")
