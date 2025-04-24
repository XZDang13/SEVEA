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

from datetime import datetime
import gymnasium
import numpy as np
import torch
import torch.optim as optim
import yaml
from torchvision.transforms import v2
from einops import rearrange

from config import METAWORLD_CFGS, DMC_CFGS
from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer

from model.encoder import FrameObservationEncoderNet
from model.actor import DDPGActorNet
from model.critic import CriticNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int):
        
        if task_name in METAWORLD_CFGS:
            config_path = "configs/ddpg_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/ddpg_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.seed = seed
        self.task_name = task_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset, render_mode="rgb_array") for i in range(config["num_envs"])])
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000, render_mode="rgb_array") for i in range(config["num_envs"])])
        

        self.transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406), (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
        ])
        
        obs_dim = (2, 112, 112, 3)
        action_dim = self.train_envs.single_action_space.shape
        
        self.encoder = FrameObservationEncoderNet(6, 256).to(self.device)
        self.actor = DDPGActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.critic = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["learning_rate"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        
        self.replay_buffer = ReplayBuffer(config["num_envs"], config["max_buffer_size"], obs_dim, action_dim, state_dtype=torch.uint8)
        self.max_steps = config["max_steps"]
        self.epochs = config["epochs"]
        self.update_iteration = config["update_iteration"]
        self.batch_size = config["batch_size"]
        self.eval_frequence = config["eval_frequence"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.regularization_weight = config["regularization_weight"]
        self.std = config["std"]
    
    def setup_env(self, task_name:str, seed:int, render_mode:str|None = None) -> gymnasium.Env:
        if task_name in METAWORLD_CFGS:
            env = setup_metaworld_env(task_name, seed, render_mode)
            self.task_env = "metaworld"
            self.task_motions = motions[task_name]
            self.num_motions = len(self.task_motions) - 1
            self.eval_steps = 500
        elif task_name in DMC_CFGS:
            env = setup_dmc_env(task_name, seed, render_mode)
            self.task_env = "dmc"
            self.eval_steps = 500
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = gymnasium.wrappers.AddRenderObservation(env, render_only=True)
        env = gymnasium.wrappers.FrameStackObservation(env, 2)

        return env
    
    def get_motion_reward(self, motions:list[str]) -> list[float]:
        rewards = []
        for motion in motions:
            idx = self.task_motions.index(motion)
            reward = idx / self.num_motions
            rewards.append(reward)

        return rewards
    
    def preprpcess(self, obs_batch):
        obs_batch = rearrange(obs_batch, "b l h w c -> b (l c) h w")
        obs_batch = self.transform(obs_batch)
        return obs_batch
    
    @torch.no_grad()
    def get_action(self, obs_batch, deterministic, random):
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        obs_batch = self.preprpcess(obs_batch)
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
    
    def rollout(self):
        obs, info = self.train_envs.reset()
        for i in range(self.max_steps):
            action = self.get_action(obs, False, False)
            next_obs, reward, done, timeout, info = self.train_envs.step(action)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])
            self.replay_buffer.add_steps(obs, action, reward, done, next_obs)
            obs = next_obs
        
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        episode_success = 0
        for i in range(self.eval_steps):
            action = self.get_action(obs, True, False)
            next_obs, reward, done, timeout, info = self.eval_envs.step(action)
            if "success" in info:
                episode_success += info["success"]
            obs = next_obs
            
        log_data["avg_episode_reward"] = np.mean(info['episode']['r']).item()

        if self.task_env == "metaworld":
            log_data["avg_success_rate"] = np.mean(episode_success / 500).item()
        
        return log_data
        
        
    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample(batch_size)
            obs_batch = batch["states"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            next_obs_batch = batch["next_states"].to(self.device)

            obs_batch = self.preprpcess(obs_batch)
            next_obs_batch = self.preprpcess(next_obs_batch)

            feature_batch = self.encoder(obs_batch, True)
            with torch.no_grad():
                next_feature_batch = self.encoder(next_obs_batch, True)

            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = DDPGDoubleQ.compute_critic_loss(
                self.actor, self.critic, self.critic_target, feature_batch, action_batch, reward_batch, next_feature_batch, done_batch, self.std, self.gamma
            )
            critic_loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = DDPGDoubleQ.compute_actor_loss(self.actor, self.critic, feature_batch.detach(), self.std, self.regularization_weight)
            actor_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DDPGDoubleQ.update_target_param(self.critic, self.critic_target, self.tau)
    
    def log(self, step, log_data):
        time = datetime.now()
        avg_reward = log_data.get("avg_episode_reward", 0.0)
        avg_success = log_data.get("avg_success_rate", None)

        print(f"[{time}] Step {step}")
        if avg_success is not None:
            print(f"Avg: eval success rate is: {avg_success:.3f}, eval reward is {avg_reward:.3f}")
        else:
            print(f"Avg: eval reward is {avg_reward:.3f}")
    
    def train(self):
        best_record = 0
        print("-------------------------------")
        print(f"task: {self.task_name}, seed: {self.seed}")
        time = datetime.now()
        print(f"[{time}] start")

        for epoch in range(self.epochs):
            self.rollout()
            self.update(self.update_iteration, self.batch_size)
            
            mix = np.clip(epoch/self.epochs, 0, 1)
            self.std = (1-mix) * 1 + mix * 0.1
            
            if (epoch + 1) % self.eval_frequence == 0:
                log_data = self.eval()
                self.log(epoch+1, log_data)
                torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/drqv2/{self.task_name}/actor_{self.seed}_{epoch+1}.pt")

                episode_success_rate = log_data.get("avg_success_rate", None)
                episode_reward = log_data.get("avg_episode_reward", None)
                metric = episode_success_rate if episode_success_rate is not None else episode_reward
                
                if metric >= best_record:
                    best_record = metric
                    torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/drqv2/{self.task_name}/actor_best_{self.seed}.pt")

        time = datetime.now()
        print(f"[{time}] end")

        print("-------------------------------")

if __name__ == '__main__':
    
    args = get_train_args()
    weight_folder = f"weights/drqv2/{args.task}"
        
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    trainer = Trainer(args.task, args.seed)
    trainer.train()   