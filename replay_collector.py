import os
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{
        "file_format_version" : "1.0.0",
        "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
        }
    }
    """)
        
os.environ["MUJOCO_GL"] = "egl"

import argparse


import gymnasium
import numpy as np
import torch
import yaml
import json
from uuid import uuid4
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import trange

from config import METAWORLD_CFGS, DMC_CFGS
from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer

from model.encoder import EncoderNet
from model.actor import DDPGActorNet
from model.critic import CriticNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--visual', action='store_true', help='visual observation or not')
    #parse.add_argument('--weight_epoch', type=int, default=0, help='weight trained after epoch')
   
    return parse.parse_args()
        
class Collector:
    def __init__(self, task_name:str, visual:bool=False):
        
        if task_name in METAWORLD_CFGS:
            config_path = "configs/ddpg_metaworld.yaml"
            self.task_motions = motions[task_name]
            self.num_motions = len(self.task_motions) - 1
        elif task_name in DMC_CFGS:
            config_path = "configs/ddpg_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(0)
        
        self.visual = visual
        self.seed = 0
        self.task_name = task_name
        self.epochs = config["epochs"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.vector.SyncVectorEnv([lambda seed=i : self.setup_env(task_name, seed) for i in range(config["num_envs"])])
        
        frame_dim = self.envs.single_observation_space["pixels"].shape  
        state_dim = self.envs.single_observation_space["state"].shape[-1:]
        action_dim = self.envs.single_action_space.shape

        if visual:
            obs_dim = frame_dim
        else:
            obs_dim = state_dim

        self.num_envs = config["num_envs"]
        
        
        self.encoder = EncoderNet(np.prod(state_dim), config["encoder_layers"]).to(self.device)
        self.actor = DDPGActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.critic = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)

        self.replay_buffer = ReplayBuffer(config["num_envs"], config["max_buffer_size"], obs_dim, action_dim)

        self.std = 1

        if visual:
            self.path = f"replays/{task_name}/visual"
        else:
            self.path = f"replays/{task_name}/state"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def load_weight(self):
        weight_file = f"weights/ddpg/{self.task_name}/actor_best_200.pt"
        encoder_weight, actor_weight, critic_weight = torch.load(weight_file, weights_only=True)
        self.encoder.load_state_dict(encoder_weight)
        self.actor.load_state_dict(actor_weight)
        self.critic.load_state_dict(critic_weight)
        
    def setup_env(self, task_name:str, seed:int) -> gymnasium.Env:
        if task_name in METAWORLD_CFGS:
            env = setup_metaworld_env(task_name, seed, "rgb_array")
            self.task_env = "metaworld"
            self.task_motions = motions[task_name]
            self.num_motions = len(self.task_motions) - 1
            self.eval_steps = 500
        elif task_name in DMC_CFGS:
            env = setup_dmc_env(task_name, seed, "rgb_array")
            self.task_env = "dmc"
            self.eval_steps = 500
        env = gymnasium.wrappers.AddRenderObservation(env, render_only=False)
        env = gymnasium.wrappers.FrameStackObservation(env, 2)

        return env
    
    @torch.no_grad()
    def get_action(self, obs_batch, deterministic, random):
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        obs_batch = self.encoder(obs_batch)
        dist = self.actor(obs_batch, self.std)
        if deterministic:
            action = dist.mean
        else:    
            action = dist.sample(clip=None)

            if random:
                action.uniform_(-1, 1)
        
        action = action.cpu().numpy()
        
        return action.tolist()
    
    def get_motion_reward(self, motions:list[str]) -> list[float]:
        rewards = []
        for motion in motions:
            idx = self.task_motions.index(motion)
            reward = idx / self.num_motions
            rewards.append(reward)

        return rewards
    
    def rollout(self, deterministic:bool, random:bool):
        obs, info = self.envs.reset()    
        for step in range(100):
            action = self.get_action(obs["state"][:, -1].tolist(), deterministic, random)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])

            if self.visual:
                self.replay_buffer.add_steps(obs["pixels"], action, reward, done, next_obs["pixels"])
            else:
                self.replay_buffer.add_steps(obs["state"][:, -1], action, reward, done, next_obs["state"][:, -1])

            obs = next_obs

    def collect(self):

        for epoch in trange(self.epochs, desc="Collect Epochs"):
            collector.rollout(True, False)

        self.replay_buffer.save(self.path)
            
if __name__ == '__main__':
    args = get_train_args()

    collector = Collector(args.task, args.visual)

    collector.load_weight()

    collector.collect()
    
    collector.envs.close()