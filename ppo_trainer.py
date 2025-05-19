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

from RLAlg.alg.ppo import PPO
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.rollout_buffer import RolloutBuffer

from config import METAWORLD_CFGS, DMC_CFGS
from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from model.encoder import EncoderNet
from model.actor import PPOActorNet
from model.critic import ValueNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int):

        if task_name in METAWORLD_CFGS:
            config_path = "configs/ppo_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/ppo_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.task_name = task_name
        self.seed = seed
        
        self.train_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset) for i in range(config["num_envs"])])
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(config["num_envs"])])
        
        obs_dim = self.train_envs.single_observation_space.shape
        action_dim = self.train_envs.single_action_space.shape
        
        self.encoder = EncoderNet(np.prod(obs_dim), config["encoder_layers"]).to(self.device)
        self.actor = PPOActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.critic = ValueNet(self.encoder.dim, config["value_layers"]).to(self.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=config["learning_rate"]
        )
        
        self.rollout_buffer = RolloutBuffer(config["num_envs"],  config["max_steps"], obs_dim, action_dim)
        self.max_steps =  config["max_steps"]
        self.epochs = config["epochs"]
        self.update_iteration = config["update_iteration"]
        self.batch_size = config["batch_size"]
        self.eval_frequence = config["eval_frequence"]
        self.gamma = config["gamma"]
        self.lambda_ = config["lambda_"]
        self.value_loss_weight = config["value_loss_weight"]
        self.entropy_weight = config["entropy_weight"]
        self.max_grad_norm = config["max_grad_norm"]
        self.clip_ratio = config["clip_ratio"]
        self.regularization_weight = config["regularization_weight"]
    
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
    def get_action(self, obs_batch:list[list[float]], determine:bool=False):
        obs_batch = torch.as_tensor(obs_batch).float().to(self.device)
        obs_batch = self.encoder(obs_batch)
        pi, action, log_prob = self.actor(obs_batch)
        
        if determine:
            action = pi.mean
        
        value = self.critic(obs_batch)

        return action.cpu().tolist(), log_prob.cpu().tolist(), value.cpu().tolist()
    
    def rollout(self):
        obs, info = self.train_envs.reset()
        for i in range(self.max_steps):
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, done, timeout, info = self.train_envs.step(action)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])
            self.rollout_buffer.add_steps(i, obs, action, log_prob, reward, done, value)
            obs = next_obs

        _, _, value = self.get_action(obs)
        self.rollout_buffer.compute_gae(value, gamma=self.gamma, lambda_=self.lambda_)
        
        
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        episode_success = 0
        for i in range(self.eval_steps):
            action, log_prob, value = self.get_action(obs, True)
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
            for batch in self.rollout_buffer.batch_sample(batch_size):
                obs_batch = batch["states"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                feature_batch = self.encoder(obs_batch, True)
                policy_loss, entropy = PPO.compute_policy_loss(self.actor, log_prob_batch, feature_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)

                value_loss = PPO.compute_clipped_value_loss(self.critic, feature_batch, value_batch, return_batch, self.clip_ratio)
                #value_loss = PPO.compute_value_loss(self.critic, feature_batch, return_batch)

                loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
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
            self.rollout()
            self.update(self.update_iteration, self.batch_size)
            
            if (epoch + 1) % self.eval_frequence == 0:
                log_data = self.eval()
                self.log(epoch+1, log_data)
                torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ppo/{self.task_name}/actor_{self.seed}_{epoch+1}.pt")

                episode_success_rate = log_data.get("avg_success_rate", None)
                episode_reward = log_data.get("avg_episode_reward", None)
                metric = episode_success_rate if episode_success_rate is not None else episode_reward
                
                if metric >= best_record:
                    best_record = metric
                    torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ppo/{self.task_name}/actor_best_{self.seed}.pt")

        time = datetime.now()
        print(f"[{time}] end")

        print("-------------------------------")

if __name__ == '__main__':
    
    args = get_train_args()
    weight_folder = f"weights/ppo/{args.task}"
        
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    trainer = Trainer(args.task, args.seed)
    trainer.train()