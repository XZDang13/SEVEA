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
import tqdm
from tqdm import trange

from config import METAWORLD_CFGS, DMC_CFGS

from RLAlg.alg.iql import IQL
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.offline_buffer import OfflineReplayBuffer

from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from model.encoder import EncoderNet
from model.actor import IQLActorNet
from model.critic import ValueNet, CriticNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int):
        
        if task_name in METAWORLD_CFGS:
            config_path = "configs/iql_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/iql_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.task_name = task_name
        self.seed = seed
        
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(config["num_envs"])])
        
        obs_dim = self.eval_envs.single_observation_space.shape
        action_dim = self.eval_envs.single_action_space.shape
        
        self.encoder = EncoderNet(np.prod(obs_dim), config["encoder_layers"]).to(self.device)
        self.actor = IQLActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.value = ValueNet(self.encoder.dim, config["value_layers"]).to(self.device)
        self.critic = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["learning_rate"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        
        self.replays_path = config["replay_path"]
        self.replay_buffer = OfflineReplayBuffer(f"{self.replays_path}/{self.task_name}/state/replays.pt")
        self.epochs = config["epochs"]
        self.update_iteration = config["update_iteration"]
        self.batch_size = config["batch_size"]
        self.eval_frequence = config["eval_frequence"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.expectile = config["expectile"]
        self.beta = config["beta"]
    
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

        return env
    
    def get_motion_reward(self, motions:list[str]) -> list[float]:
        rewards = []
        for motion in motions:
            idx = self.task_motions.index(motion)
            reward = idx / self.num_motions
            rewards.append(reward)

        return rewards
    
    @torch.no_grad()
    def get_action(self, obs:list[list[float]]):
        obs = torch.as_tensor(obs).float().to(self.device)
        obs = self.encoder(obs)
        pi, _, _ = self.actor(obs)
        
        action = pi.mean
        
        return action.cpu().tolist()
    
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        episode_success = 0
        for i in range(self.eval_steps):
            action = self.get_action(obs)
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
            
            feature_batch = self.encoder(obs_batch, True)
            with torch.no_grad():
                next_feature_batch = self.encoder(next_obs_batch, True)
            
            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss = IQL.compute_value_loss(self.value, self.critic_target, feature_batch, action_batch, self.expectile)
            value_loss.backward()
            self.value_optimizer.step()
            self.encoder_optimizer.step()

            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = IQL.compute_critic_loss(self.value, self.critic, feature_batch.detach(), action_batch, reward_batch, done_batch, next_feature_batch, self.gamma)
            critic_loss.backward()
            self.critic_optimizer.step()    
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = IQL.compute_actor_loss(self.actor, self.value, self.critic_target, feature_batch.detach(), action_batch, self.beta)
            actor_loss.backward()
            self.actor_optimizer.step()
            
            IQL.update_target_param(self.critic, self.critic_target, self.tau)
                
    def log(self, step, log_data):
        time = datetime.now()
        avg_reward = log_data.get("avg_episode_reward", 0.0)
        avg_success = log_data.get("avg_success_rate", None)

        tqdm.tqdm.write(f"[{time}] Step {step}")
        if avg_success is not None:
            tqdm.tqdm.write(f"Avg: eval success rate is: {avg_success:.3f}, eval reward is {avg_reward:.3f}")
        else:
            tqdm.tqdm.write(f"Avg: eval reward is {avg_reward:.3f}")
    
    def train(self):
        best_record = 0
        print("-------------------------------")
        print(f"task: {self.task_name}, seed: {self.seed}")
        time = datetime.now()
        print(f"[{time}] start")
        for epoch in trange(self.epochs, desc="Training Epochs"):
            self.update(self.update_iteration, self.batch_size)
            
            if (epoch + 1) % self.eval_frequence == 0:
                log_data = self.eval()
                self.log(epoch+1, log_data)
                torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/iql/{self.task_name}/actor_{self.seed}_{epoch+1}.pt")

                episode_success_rate = log_data.get("avg_success_rate", None)
                episode_reward = log_data.get("avg_episode_reward", None)
                metric = episode_success_rate if episode_success_rate is not None else episode_reward
                
                if metric >= best_record:
                    best_record = metric
                    torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/iql/{self.task_name}/actor_best_{self.seed}.pt")

        time = datetime.now()
        print(f"[{time}] end")

        print("-------------------------------")
        
if __name__ == '__main__':
    args = get_train_args()
    weight_folder = f"weights/iql/{args.task}"
        
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    trainer = Trainer(args.task, args.seed)
    trainer.train()