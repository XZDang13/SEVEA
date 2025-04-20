import argparse
import os

from datetime import datetime
import gymnasium
import numpy as np
import torch
import torch.optim as optim
import yaml

from metaworld_env import setup_metaworld_env
from motion_detector import motions
from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer

from model.encoder import EncoderNet
from model.actor import DDPGActorNet
from model.critic import CriticNet

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int):
        
        with open("configs/ddpg.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.seed = seed
        self.task_name = task_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset) for i in range(config["num_envs"])])
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(config["num_envs"])])
        
        self.task_motions = motions[task_name]
        self.num_motions = len(self.task_motions) - 1
        
        obs_dim = self.train_envs.single_observation_space.shape
        action_dim = self.train_envs.single_action_space.shape
        
        self.encoder = EncoderNet(np.prod(obs_dim), config["num_blocks"], config["encoder_layers"]).to(self.device)
        self.actor = DDPGActorNet(self.encoder.dim, np.prod(action_dim), config["actor_layers"]).to(self.device)
        self.critic = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target = CriticNet(self.encoder.dim, np.prod(action_dim), config["critic_layers"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["learning_rate"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        
        self.replay_buffer = ReplayBuffer(config["num_envs"], config["max_buffer_size"], obs_dim, action_dim)
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
        env = setup_metaworld_env(task_name, seed, render_mode)
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
    
    def rollout(self):
        obs, info = self.train_envs.reset()
        for i in range(self.max_steps):
            action = self.get_action(obs, False, False)
            next_obs, reward, done, timeout, info = self.train_envs.step(action)
            reward = self.get_motion_reward(info["motion"])
            self.replay_buffer.add_steps(obs, action, reward, done, next_obs)
            obs = next_obs
        
    def eval(self):
        obs, info = self.eval_envs.reset()
        episode_success = 0
        for i in range(500):
            action = self.get_action(obs, True, False)
            next_obs, reward, done, timeout, info = self.eval_envs.step(action)
            episode_success += info["success"]
            obs = next_obs
            
        avg_episode_reward = np.mean(info['episode']['r']).item()
        avg_success_rate = np.mean(episode_success / 500).item()
        
        return avg_episode_reward, avg_success_rate
        
        
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
    
    def log(self, step, avg_reward, avg_success):
        time = datetime.now()
        print(f"[{time}] {step}")
        print(f"Avg: eval success rate is: {avg_success:.3f}, eval reward is {avg_reward:.3f}")
    
    def train(self):
        best_success = 0
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
                episode_reward, episode_success_rate = self.eval()
                self.log(epoch+1, episode_reward, episode_success_rate)
                torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ddpg/{self.task_name}/actor_{self.seed}_{epoch+1}.pt")

                if episode_success_rate > best_success:
                    best_success = episode_success_rate
                    torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ddpg/{self.task_name}/actor_best_{self.seed}.pt")

        time = datetime.now()
        print(f"[{time}] end")

        print("-------------------------------")

if __name__ == '__main__':
    args = get_train_args()
    weight_folder = f"weights/ddpg/{args.task}"
        
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    trainer = Trainer(args.task, args.seed)
    trainer.train()   