import os
import json
import math
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

from datetime import datetime
import gymnasium
import numpy as np
import torch
import torch.optim as optim
import yaml
import tqdm
from torchvision.transforms import v2
from tqdm import trange

from config import METAWORLD_CFGS, DMC_CFGS
from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions

from RLAlg.alg.ppo import PPO
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.layers import NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep

from model.encoder import StateObservationEncoderNet, VisualObservationEncoderNet
from model.actor import PPOActor
from model.critic import PPOCritic

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
    parse.add_argument('--visual', action='store_true', help='use visual observations (rendered frame stack)')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int, visual:bool=False):

        if task_name in METAWORLD_CFGS:
            config_path = "configs/ppo_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/ppo_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_transform = None
        if visual:
            self.pixel_transform = v2.Compose([
                v2.Lambda(
                    lambda x: x.permute(0, 1, 4, 2, 3).reshape(
                        x.shape[0], -1, x.shape[2], x.shape[3]
                    )
                )
            ])
        
        self.task_name = task_name
        self.seed = seed
        self.visual = visual
        self.num_envs = config["num_envs"]
        self.eval_log_path = f"weights/ppo/{self.task_name}/eval_{self.seed}.jsonl"
        os.makedirs(os.path.dirname(self.eval_log_path), exist_ok=True)
        
        self.train_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset) for i in range(self.num_envs)])
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(self.num_envs)])
        
        obs_space = self.train_envs.single_observation_space
        obs_dim = obs_space["pixels"].shape if self.visual else obs_space.shape
        action_dim = self.train_envs.single_action_space.shape
        
        if self.visual:
            encoder_input = torch.zeros((1, *obs_dim), dtype=torch.uint8)
            in_channel = self.pixel_transform(encoder_input).shape[1]
            self.encoder = VisualObservationEncoderNet(in_channel).to(self.device)
        else:
            self.encoder = StateObservationEncoderNet(np.prod(obs_dim), config["encoder_layers"], norm_position=NormPosition.POST, dropout_prob=config["encoder_dropout_prob"]).to(self.device)
        self.actor = PPOActor(self.encoder.feature_dim, np.prod(action_dim), config["actor_layers"], norm_position=NormPosition.POST).to(self.device)
        self.critic = PPOCritic(self.encoder.feature_dim, config["critic_layers"], norm_position=NormPosition.POST).to(self.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=config["learning_rate"]
        )
        
        total_buffer_size = int(config.get("max_buffer_size", config["max_steps"] * self.num_envs))
        self.max_steps = max(1, math.ceil(total_buffer_size / self.num_envs))
        print(self.max_steps)
        self.replay_buffer = ReplayBuffer(self.num_envs, self.max_steps, device=self.device)
        obs_dtype = torch.uint8 if self.visual else torch.float32
        self.replay_buffer.create_storage_space("observations", obs_dim, obs_dtype)
        self.replay_buffer.create_storage_space("actions", action_dim, torch.float32)
        self.replay_buffer.create_storage_space("log_probs", (), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)
        
        self.batch_keys = ["observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]
        
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
        render_mode = "rgb_array" if self.visual else render_mode
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
        if self.visual:
            env = gymnasium.wrappers.AddRenderObservation(env, render_only=False)
            env = gymnasium.wrappers.FrameStackObservation(env, 2)

        env = gymnasium.wrappers.RecordEpisodeStatistics(env)

        return env

    def _select_observation(self, obs):
        if self.visual:
            return obs["pixels"]
        return obs

    def _observation_to_tensor(self, obs) -> torch.Tensor:
        if self.visual:
            obs = torch.as_tensor(obs, dtype=torch.uint8)
            obs = self.pixel_transform(obs)
            obs = obs.to(self.device, dtype=torch.float32) / 255.0
            return obs
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return obs.reshape(obs.shape[0], -1)
    
    def get_motion_reward(self, motions:list[str]) -> list[float]:
        rewards = []
        for motion in motions:
            idx = self.task_motions.index(motion)
            reward = idx / self.num_motions
            rewards.append(reward)

        return rewards
    
    @torch.no_grad()
    def get_action(self, obs_batch:list[list[float]], determine:bool=False):
        obs_batch = self._observation_to_tensor(obs_batch)
        obs_batch = self.encoder(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        value_step:ValueStep = self.critic(obs_batch)
        
        if determine:
            action = actor_step.mean
        else:
            action = actor_step.action
        log_prob = actor_step.log_prob

        value = value_step.value

        return action, log_prob, value
    
    def rollout(self):
        obs, info = self.train_envs.reset()
        obs = self._select_observation(obs)
        for i in range(self.max_steps):
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, done, timeout, info = self.train_envs.step(action.cpu().numpy())
            next_obs = self._select_observation(next_obs)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])
            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "dones": done
            }
            
            self.replay_buffer.add_records(record)
            
            obs = next_obs
            
        _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.replay_buffer.data["rewards"],
            self.replay_buffer.data["values"],
            self.replay_buffer.data["dones"],
            value,
            self.gamma,
            self.lambda_
            )
        
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)
        
        
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        obs = self._select_observation(obs)
        episode_success = 0
        for i in range(self.eval_steps):
            action, log_prob, value = self.get_action(obs, True)
            next_obs, reward, done, timeout, info = self.eval_envs.step(action.cpu().numpy())
            next_obs = self._select_observation(next_obs)
            if "success" in info:
                episode_success += info["success"]
            obs = next_obs
            
        log_data["avg_episode_reward"] = np.mean(info['episode']['r']).item()

        if self.task_env == "metaworld":
            log_data["avg_success_rate"] = np.mean(episode_success / 500).item()
        
        return log_data
        
        
    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = self._observation_to_tensor(batch["observations"])
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                aug = True
                if self.visual:
                    aug = False
                    
                feature_batch = self.encoder(obs_batch, aug)
                policy_loss_dict = PPO.compute_policy_loss(self.actor, log_prob_batch, feature_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)

                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]
                
                value_loss_dict = PPO.compute_clipped_value_loss(self.critic, feature_batch, value_batch, return_batch, self.clip_ratio)
                value_loss = value_loss_dict["loss"]
                
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
        self.save_eval_log(step, log_data)

    def save_eval_log(self, step, log_data):
        record = {
            "time": datetime.now().isoformat(),
            "step": int(step),
            "avg_episode_reward": float(log_data.get("avg_episode_reward", 0.0))
        }
        avg_success = log_data.get("avg_success_rate", None)
        if avg_success is not None:
            record["avg_success_rate"] = float(avg_success)

        with open(self.eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
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

    trainer = Trainer(args.task, args.seed, args.visual)
    trainer.train()
