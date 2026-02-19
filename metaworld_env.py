from gymnasium.core import Env
import numpy as np
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from config import METAWORLD_CFGS
from motion_detector import get_motion_detecor

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.65, 0.0]),
    }

DEFAULT_SIZE=112

class VisualWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, motion_detector, seed):
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, default_cam_config=DEFAULT_CAMERA_CONFIG, width=DEFAULT_SIZE, height=DEFAULT_SIZE)

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seeded_rand_vec = True
        
        self.seed = self.unwrapped.seed(seed)

        self.motion_detector = motion_detector

    def reset(self, **kwargs):
        current_obs, info = super().reset(**kwargs)
        
        current_obs = current_obs.tolist()
        #current_obs = current_obs[:18] + current_obs[-3:]
        self.previous_obs = current_obs
        info["is_grasped"] = False
        info["motion"] = ""

        vector_obs = current_obs
        return vector_obs, info

    def step(self, action):
        current_obs, reward, done, truncate, info = self.env.step(action)
        
        current_obs = current_obs.tolist()

        try:
            is_grasped = self.unwrapped.touching_main_object
        except:
            is_grasped = False

        motion = ""
        if self.motion_detector is not None:
            motion = self.motion_detector.get_motion_label(self.previous_obs, current_obs, is_grasped)
            
        info["motion"] = motion
        info["is_grasped"] = is_grasped

        vector_obs = current_obs
        self.previous_obs = current_obs

        return vector_obs, reward, done, truncate, info
    
def setup_metaworld_env(task_name:str, seed:int, render_mode:str="rgb_array"):
    cfgs = METAWORLD_CFGS[task_name]
    env_cls = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[cfgs["env_name"]]
    
    motion_detector = get_motion_detecor(task_name)
    env = VisualWrapper(env_cls(render_mode=render_mode), motion_detector, seed)
    
    return env