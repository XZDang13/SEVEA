import os
import glob
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

from RLAlg.alg.iql import IQL
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.layers import NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from dmc import setup_dmc_env
from metaworld_env import setup_metaworld_env
from motion_detector import motions
from model.encoder import StateObservationEncoderNet, VisualObservationEncoderNet
from model.actor import IQLActor
from model.critic import IQLCritic, IQLValue

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
    parse.add_argument('--replay-file', type=str, default=None, help='optional replay file path override')
    parse.add_argument('--visual', action='store_true', help='use visual observations (rendered frame stack)')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int, replay_file:str|None=None, visual:bool=False):
        
        if task_name in METAWORLD_CFGS:
            config_path = "configs/iql_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/iql_dmc.yaml"
        
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
        self.eval_log_path = f"weights/iql/{self.task_name}/eval_{self.seed}.jsonl"
        os.makedirs(os.path.dirname(self.eval_log_path), exist_ok=True)
        
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(config["num_envs"])])
        
        obs_space = self.eval_envs.single_observation_space
        obs_dim = obs_space["pixels"].shape if self.visual else obs_space.shape
        action_space = self.eval_envs.single_action_space
        action_dim = action_space.shape
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)
        if not np.all(np.isfinite(action_low)) or not np.all(np.isfinite(action_high)):
            raise ValueError("IQL requires finite action bounds.")
        if not np.allclose(action_low, -action_high, atol=1e-6):
            raise ValueError(
                f"IQL squashed policy expects symmetric action bounds, got low={action_low}, high={action_high}"
            )
        self.max_action = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        
        if self.visual:
            encoder_input = torch.zeros((1, *obs_dim), dtype=torch.uint8)
            in_channel = self.pixel_transform(encoder_input).shape[1]
            self.encoder = VisualObservationEncoderNet(in_channel).to(self.device)
        else:
            self.encoder = StateObservationEncoderNet(
                np.prod(obs_dim),
                config["encoder_layers"],
                norm_position=NormPosition.POST,
                dropout_prob=config["encoder_dropout_prob"],
            ).to(self.device)
        self.actor = IQLActor(
            self.encoder.feature_dim,
            np.prod(action_dim),
            config["actor_layers"],
            max_action=self.max_action,
            norm_position=NormPosition.POST,
        ).to(self.device)
        self.value = IQLValue(
            self.encoder.feature_dim,
            config["value_layers"],
            norm_position=NormPosition.POST,
        ).to(self.device)
        self.critic = IQLCritic(
            self.encoder.feature_dim,
            np.prod(action_dim),
            config["critic_layers"],
            norm_position=NormPosition.POST,
        ).to(self.device)
        self.critic_target = IQLCritic(
            self.encoder.feature_dim,
            np.prod(action_dim),
            config["critic_layers"],
            norm_position=NormPosition.POST,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["learning_rate"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        
        self.replays_path = config["replay_path"]
        self.max_buffer_size = int(config.get("max_buffer_size", 0))
        replay_path = replay_file or self._resolve_replay_path()
        self.replay_buffer = ReplayBuffer(1, 1, device=self.device)
        self.replay_buffer.load(replay_path, device=self.device)
        self._limit_replay_buffer_size()
        print(f"[IQL] replay file: {replay_path}")
        
        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        self.epochs = config["epochs"]
        self.update_iteration = config["update_iteration"]
        self.batch_size = config["batch_size"]
        self.eval_frequence = config["eval_frequence"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.expectile = config["expectile"]
        self.beta = config["beta"]

    def _limit_replay_buffer_size(self) -> None:
        if self.max_buffer_size <= 0:
            return
        if not getattr(self.replay_buffer, "data", None):
            return

        sample_tensor = next(iter(self.replay_buffer.data.values()))
        num_envs = int(getattr(self.replay_buffer, "num_envs", 1))

        # Vectorized replay format: [steps, num_envs, ...]
        if sample_tensor.ndim >= 2 and sample_tensor.shape[1] == num_envs:
            before_steps = int(getattr(self.replay_buffer, "current_size", sample_tensor.shape[0]))
            before_samples = before_steps * num_envs
            keep_steps = max(1, math.ceil(self.max_buffer_size / num_envs))
            if keep_steps >= sample_tensor.shape[0]:
                return
            for key, value in self.replay_buffer.data.items():
                self.replay_buffer.data[key] = value[:keep_steps].contiguous()
            if hasattr(self.replay_buffer, "steps"):
                self.replay_buffer.steps = keep_steps
            if hasattr(self.replay_buffer, "current_size"):
                self.replay_buffer.current_size = min(before_steps, keep_steps)
            after_samples = int(getattr(self.replay_buffer, "current_size", keep_steps)) * num_envs
            print(f"[IQL] replay clipped: {before_samples} -> {after_samples} samples (max {self.max_buffer_size})")
            return

        # Flat replay format: [samples, ...]
        before_samples = sample_tensor.shape[0]
        keep_samples = max(1, self.max_buffer_size)
        if keep_samples >= before_samples:
            return
        for key, value in self.replay_buffer.data.items():
            self.replay_buffer.data[key] = value[:keep_samples].contiguous()
        if hasattr(self.replay_buffer, "steps"):
            self.replay_buffer.steps = keep_samples
        if hasattr(self.replay_buffer, "current_size"):
            self.replay_buffer.current_size = min(int(getattr(self.replay_buffer, "current_size", keep_samples)), keep_samples)
        print(f"[IQL] replay clipped: {before_samples} -> {keep_samples} samples (max {self.max_buffer_size})")

    def _resolve_replay_path(self) -> str:
        task_dir = os.path.join(self.replays_path, self.task_name)
        candidates = []
        if self.visual:
            candidates.append(os.path.join(task_dir, "visual", "replays.pt"))
        candidates.extend([
            os.path.join(task_dir, f"buffer_{self.seed}.pt"),
            os.path.join(task_dir, "buffer.pt"),
            os.path.join(task_dir, "state", "replays.pt"),
        ])
        for path in candidates:
            if os.path.isfile(path):
                return path

        pt_files = glob.glob(os.path.join(task_dir, "*.pt"))
        if pt_files:
            pt_files.sort(key=os.path.getmtime, reverse=True)
            return pt_files[0]

        raise FileNotFoundError(
            f"No replay file found for task '{self.task_name}' in '{task_dir}'. "
            "Expected one of: visual/replays.pt, buffer_<seed>.pt, buffer.pt, state/replays.pt, or any .pt file in task directory."
        )
    
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
    def get_action(self, obs:list[list[float]]):
        obs = self._observation_to_tensor(obs)
        obs = self.encoder(obs)
        actor_step: StochasticContinuousPolicyStep = self.actor(obs)
        action = actor_step.mean
        
        return action
    
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        obs = self._select_observation(obs)
        episode_success = 0
        for i in range(self.eval_steps):
            action = self.get_action(obs)
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
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = self._observation_to_tensor(batch["observations"])
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            next_obs_batch = self._observation_to_tensor(batch["next_observations"])
            
            aug = True
            if self.visual:
                aug = False
                
            feature_batch = self.encoder(obs_batch, aug)
            with torch.no_grad():
                next_feature_batch = self.encoder(next_obs_batch, aug)
            
            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss_dict = IQL.compute_value_loss(self.value, self.critic_target, feature_batch, action_batch, self.expectile)
            value_loss = value_loss_dict["loss"]
            value_loss.backward()
            self.value_optimizer.step()
            self.encoder_optimizer.step()

            
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_dict = IQL.compute_critic_loss(
                self.value,
                self.critic,
                feature_batch.detach(),
                action_batch,
                reward_batch,
                done_batch,
                next_feature_batch,
                self.gamma,
            )
            critic_loss = critic_loss_dict["loss"]
            critic_loss.backward()
            self.critic_optimizer.step()    
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss_dict = IQL.compute_policy_loss(
                self.actor,
                self.value,
                self.critic_target,
                feature_batch.detach(),
                action_batch,
                self.beta,
            )
            actor_loss = actor_loss_dict["loss"]
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

    trainer = Trainer(args.task, args.seed, args.replay_file, args.visual)
    trainer.train()
