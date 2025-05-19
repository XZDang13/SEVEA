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

from model.encoder import EncoderNet
from model.actor import DDPGActorNet, PPOActorNet, IQLActorNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
    parse.add_argument('--alg', type=str, default="ddpg", help='RL algorithm name')
   
    return parse.parse_args()

class PairDataCollector:
    @staticmethod
    def save(obs, path):
        # Ensure the directories exist
        json_dir = os.path.join(path, "json")
        img_dir = os.path.join(path, "img")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        def process_entry(args):
            state, frame = args
            step_id = str(uuid4())

            # Save JSON data
            data = {
                "step_id": step_id,
                "state": state.tolist(),
            }
            with open(f"{json_dir}/{step_id}.json", "w") as f:
                json.dump(data, f)

            # Save images
            for i in range(2):
                Image.fromarray(frame[i]).save(f"{img_dir}/{step_id}_{i}.png")

        # Use ThreadPool for parallel processing
        with ThreadPool() as pool:
            pool.map(process_entry, zip(obs["state"], obs["pixels"]))
        
class Collector:
    def __init__(self, task_name:str, seed:int, alg:str):
        
        if task_name in METAWORLD_CFGS:
            if alg == "ddpg":
                config_path = "configs/ddpg_metaworld.yaml"
            elif alg == "ppo":
                config_path = "configs/ppo_metaworld.yaml"
            elif alg == "iql":
                config_path = "configs/iql_metaworld.yaml"

            self.task_motions = motions[task_name]
            self.num_motions = len(self.task_motions) - 1
        elif task_name in DMC_CFGS:
            if alg == "ddpg":
                config_path = "configs/ddpg_dmc.yaml"
            elif alg == "ppo":
                config_path = "configs/ppo_dmc.yaml"
            elif alg == "iql":
                config_path = "configs/iql_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(0)
        
        self.seed = seed
        self.task_name = task_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.vector.SyncVectorEnv([lambda seed=i : self.setup_env(task_name, self.seed+seed) for i in range(config["num_envs"])])
        
        frame_dim = self.envs.single_observation_space["pixels"].shape  
        state_dim = self.envs.single_observation_space["state"].shape[-1:]
        action_dim = self.envs.single_action_space.shape

        self.num_envs = config["num_envs"]
        
        self.alg = alg

        if alg == "ddpg":
            self.encoder = EncoderNet(np.prod(state_dim), config["encoder_layers"]).to(self.device)
            self.actor = DDPGActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        elif alg == "ppo":
            self.encoder = EncoderNet(np.prod(state_dim), config["encoder_layers"]).to(self.device)
            self.actor = PPOActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        elif alg == "iql":
            self.encoder = EncoderNet(np.prod(state_dim), config["encoder_layers"]).to(self.device)
            self.actor = IQLActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)


        self.std = 1

        self.path = f"pair_data/{task_name}"

        if not os.path.exists(self.path):
            os.makedirs(f"{self.path}/json")
            os.makedirs(f"{self.path}/img")

    def load_weight(self):
        weight_file = f"weights/{self.alg}/{self.task_name}/actor_best_{self.seed}.pt"
        encoder_weight, actor_weight, critic_weight = torch.load(weight_file, weights_only=True)
        self.encoder.load_state_dict(encoder_weight)
        self.actor.load_state_dict(actor_weight)
        
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
    def get_action_ddpg(self, obs_batch, deterministic, random):
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
    
    @torch.no_grad()
    def get_action_ppo_iql(self, obs_batch, determine):
        obs_batch = torch.as_tensor(obs_batch).float().to(self.device)
        obs_batch = self.encoder(obs_batch)
        pi, action, log_prob = self.actor(obs_batch)
        
        if determine:
            action = pi.mean

        return action.tolist()
    
    def rollout(self, deterministic:bool):
        obs, info = self.envs.reset()    
        for step in range(100):
            PairDataCollector.save(obs, self.path)
            if self.alg == "ddpg":
                action = self.get_action_ddpg(obs["state"][:, -1].tolist(), deterministic, False)
            elif self.alg == "ppo" or self.alg == "iql":
                action = self.get_action_ppo_iql(obs["state"][:, -1].tolist(), deterministic)
            obs, reward, done, timeout, info = self.envs.step(action)

    def collect(self):
        set_deterministic = False
        total_epoch = 50
        noise_epoch = int(total_epoch * 0.7)
        for epoch in trange(total_epoch):
            self.rollout(set_deterministic)

            if epoch >= noise_epoch:
                set_deterministic = True  
            
if __name__ == '__main__':
    args = get_train_args()

    collector = Collector(args.task, args.seed, args.alg)

    collector.load_weight()

    collector.collect()
    
    collector.envs.close()