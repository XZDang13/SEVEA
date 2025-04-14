import argparse
import os

import gymnasium
import numpy as np
import torch
import yaml
import json
from uuid import uuid4
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

from metaworld_env import setup_metaworld_env
from motion_detector import motions
from RLAlg.utils import set_seed_everywhere

from model.encoder import EncoderNet
from model.actor import DDPGActorNet
from model.critic import CriticNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--weight_epoch', type=int, default=0, help='weight trained after epoch')
   
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
    def __init__(self, task_name:str):
        
        with open("configs/ddpg.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(0)
        
        self.seed = 0
        self.task_name = task_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.vector.SyncVectorEnv([lambda seed=i : self.setup_env(task_name, seed) for i in range(config["num_envs"])])
        
        frame_dim = self.envs.single_observation_space["pixels"].shape  
        state_dim = self.envs.single_observation_space["state"].shape[-1:]
        action_dim = self.envs.single_action_space.shape

        self.num_envs = config["num_envs"]
        self.task_motions = motions[task_name]
        self.num_motions = len(self.task_motions) - 1
        
        
        self.encoder = EncoderNet(np.prod(state_dim), config["encoder_layers"]).to(self.device)
        self.actor = DDPGActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.critic = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)

        self.std = 1

        self.path = f"pair_data/{task_name}"

        if not os.path.exists(self.path):
            os.makedirs(f"{self.path}/json")
            os.makedirs(f"{self.path}/img")

    def load_weight(self):
        weight_file = f"weights/ddpg/{self.task_name}/actor_0_100.pt"
        encoder_weight, actor_weight, critic_weight = torch.load(weight_file, weights_only=True)
        self.encoder.load_state_dict(encoder_weight)
        self.actor.load_state_dict(actor_weight)
        self.critic.load_state_dict(critic_weight)
        
    def setup_env(self, task_name:str, seed:int) -> gymnasium.Env:
        env = setup_metaworld_env(task_name, False, seed, "rgb_array")
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
    
    def rollout(self, deterministic:bool, random:bool):
        obs, info = self.envs.reset()    
        for step in range(250):
            PairDataCollector.save(obs, self.path)
            action = self.get_action(obs["state"][:, -1].tolist(), deterministic, random)
            obs, reward, done, timeout, info = self.envs.step(action)       
            
if __name__ == '__main__':
    args = get_train_args()

    collector = Collector(args.task)

    collector.load_weight()

    #for _ in range(5):
    #    collector.rollout(False, True)

    for _ in range(30):
        collector.rollout(False, False)

    for _ in range(20):
        collector.rollout(True, False)
    
    collector.envs.close()