import gymnasium
import numpy as np
from dm_control import suite
from dm_env import specs
import random

from config import DMC_CFGS

class DMC(gymnasium.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, domain_name:str, task_name:str, seed:int, render_size:int=112, camera_id:int=0,
                 frame_skip:int=1, render_mode: str|None=None):
        self.domain_name = domain_name
        self.task_name = task_name
        self.render_size = render_size
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self._env = suite.load(domain_name=self.domain_name, task_name=self.task_name,  task_kwargs={'random': seed})
        self._observation_spec = self._env.observation_spec()
        self._action_spec = self._env.action_spec()
        
        flat_dim = int(sum(np.prod(spec.shape) for spec in self._observation_spec.values()))
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

        self.action_space = self._spec_to_box(self._action_spec)
        self._np_random = None
        self.seed(seed)
        self.current_time_step = None

    def _spec_to_box(self, spec):
        return gymnasium.spaces.Box(spec.minimum, spec.maximum, dtype=spec.dtype)

    def seed(self, seed=None):
        self._np_random, seed = gymnasium.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def _flatten_obs(self, time_step):
        obs = time_step.observation
        return np.concatenate([np.array(v).ravel() for v in obs.values()]).astype(np.float32)

    def _get_obs(self):
        
        return self._flatten_obs(self.current_time_step)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_time_step = self._env.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = 0.0
        for _ in range(self.frame_skip):
            self.current_time_step = self._env.step(action)
            reward += self.current_time_step.reward or 0.0
            if self.current_time_step.last():
                break

        obs = self._get_obs()
        terminated = self.current_time_step.last()
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(height=self.render_size, width=self.render_size, camera_id=self.camera_id)
        elif self.render_mode == "human":
            raise NotImplementedError("Human rendering is not supported.")
        return None

    def close(self):
        pass

def setup_dmc_env(task_name:str, seed:int, render_mode:str="rgb_array"):

    cfg = DMC_CFGS[task_name]

    env = DMC(cfg["domain_name"], cfg["task_name"], seed, render_mode=render_mode, camera_id=cfg["camera"], frame_skip=2)
    
    return env
