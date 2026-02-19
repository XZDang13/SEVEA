import os
'''
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
'''

import argparse

import gymnasium
import numpy as np
import torch
import yaml
from tqdm import trange

from config import METAWORLD_CFGS, DMC_CFGS
from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.layers import NormPosition
from RLAlg.nn.steps import DeterministicContinuousPolicyStep

from model.encoder import StateObservationEncoderNet
from model.actor import DDPGActor
from model.critic import DDPGCritic


def get_collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='draweropen', help='the task name')
    parser.add_argument('--seed', type=int, default=0, help='environment seed')
    parser.add_argument('--weight-seed', type=int, default=200, help='seed id used in actor_best_<seed>.pt')
    parser.add_argument('--visual', action='store_true', help='collect visual observation replay')

    return parser.parse_args()
        
class Collector:
    def __init__(self, task_name:str, seed:int, weight_seed:int, visual:bool=False):
        
        if task_name in METAWORLD_CFGS:
            config_path = "configs/ddpg_metaworld.yaml"
            self.task_motions = motions[task_name]
            self.num_motions = len(self.task_motions) - 1
        elif task_name in DMC_CFGS:
            config_path = "configs/ddpg_dmc.yaml"
        else:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"MetaWorld tasks: {sorted(METAWORLD_CFGS.keys())}; DMC tasks: {sorted(DMC_CFGS.keys())}"
            )
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.visual = visual
        self.seed = seed
        self.weight_seed = weight_seed
        self.task_name = task_name
        self.epochs = 1
        self.max_steps = config["max_steps"]
        self.std = config["std"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.vector.SyncVectorEnv([lambda offset=i: self.setup_env(task_name, self.seed + offset) for i in range(config["num_envs"])])
        
        frame_dim = self.envs.single_observation_space["pixels"].shape  
        state_dim = self.envs.single_observation_space["state"].shape[-1:]
        action_dim = self.envs.single_action_space.shape

        if visual:
            obs_dim = frame_dim
        else:
            obs_dim = state_dim

        self.num_envs = config["num_envs"]
        
        self.encoder = StateObservationEncoderNet(
            np.prod(state_dim),
            config["encoder_layers"],
            norm_position=NormPosition.POST,
            dropout_prob=config["encoder_dropout_prob"],
        ).to(self.device)
        self.actor = DDPGActor(
            self.encoder.feature_dim,
            np.prod(action_dim),
            config["actor_layers"],
            norm_position=NormPosition.POST,
        ).to(self.device)
        self.critic = DDPGCritic(
            self.encoder.feature_dim,
            np.prod(action_dim),
            config["critic_layers"],
            norm_position=NormPosition.POST,
        ).to(self.device)

        self.replay_buffer = ReplayBuffer(config["num_envs"], 10, device=torch.device("cpu"))
        obs_dtype = torch.uint8 if visual else torch.float32
        self.replay_buffer.create_storage_space("observations", obs_dim, obs_dtype)
        self.replay_buffer.create_storage_space("next_observations", obs_dim, obs_dtype)
        self.replay_buffer.create_storage_space("actions", action_dim, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)

        replay_type = "visual" if visual else "state"
        self.path = f"replays/{task_name}/{replay_type}"
        self.replay_path = os.path.join(self.path, "replays.pt")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def load_weight(self):
        weight_file = f"weights/ddpg/{self.task_name}/actor_best_{self.weight_seed}.pt"
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(f"Weight file not found: {weight_file}")

        encoder_weight, actor_weight, critic_weight = torch.load(weight_file, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(encoder_weight)
        self.actor.load_state_dict(actor_weight)
        self.critic.load_state_dict(critic_weight)
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        
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
        actor_step: DeterministicContinuousPolicyStep = self.actor(obs_batch, self.std)
        if deterministic:
            action = actor_step.mean
        else:
            action = actor_step.pi.rsample()

        if random:
            action = torch.empty_like(actor_step.mean).uniform_(-1, 1)
        
        action = action.cpu().numpy()
        
        return action
    
    def get_motion_reward(self, motions:list[str]) -> list[float]:
        rewards = []
        for motion in motions:
            idx = self.task_motions.index(motion)
            reward = idx / self.num_motions
            rewards.append(reward)

        return rewards
    
    def rollout(self, deterministic:bool, random:bool):
        obs, info = self.envs.reset()    
        for _ in range(self.max_steps):
            action = self.get_action(obs["state"][:, -1], deterministic, random)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])

            if self.visual:
                observation = obs["pixels"]
                next_observation = next_obs["pixels"]
            else:
                observation = obs["state"][:, -1]
                next_observation = next_obs["state"][:, -1]

            record = {
                "observations": observation,
                "next_observations": next_observation,
                "actions": action,
                "rewards": reward,
                "dones": done,
            }

            self.replay_buffer.add_records(record)

            obs = next_obs

    def collect(self):

        for _ in trange(self.epochs, desc="Collect Epochs"):
            self.rollout(True, False)

        self.replay_buffer.save(self.replay_path)
        print(f"[Collector] Replay saved to '{self.replay_path}'")
            
if __name__ == '__main__':
    args = get_collect_args()

    collector = Collector(args.task, args.seed, args.weight_seed, args.visual)

    try:
        collector.load_weight()
        collector.collect()
    finally:
        collector.envs.close()
