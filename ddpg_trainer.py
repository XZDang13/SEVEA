import os
import json
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
from RLAlg.alg.ddpg_double_q import DDPGDoubleQ
from RLAlg.utils import set_seed_everywhere
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.layers import NormPosition
from RLAlg.nn.steps import DeterministicContinuousPolicyStep


from model.encoder import StateObservationEncoderNet, VisualObservationEncoderNet
from model.actor import DDPGActor
from model.critic import DDPGCritic

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
    parse.add_argument('--save-buffer', action='store_true', help='If set, save the buffer to disk after training')
    parse.add_argument('--visual', action='store_true', help='use visual observations (rendered frame stack)')
   
    return parse.parse_args()

class Trainer:
    def __init__(self, task_name:str, seed:int, visual:bool=False):

        if task_name in METAWORLD_CFGS:
            config_path = "configs/ddpg_metaworld.yaml"
        elif task_name in DMC_CFGS:
            config_path = "configs/ddpg_dmc.yaml"
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        set_seed_everywhere(seed)
        
        self.seed = seed
        self.task_name = task_name
        self.visual = visual
        self.eval_log_path = f"weights/ddpg/{self.task_name}/eval_{self.seed}.jsonl"
        os.makedirs(os.path.dirname(self.eval_log_path), exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_transform = None
        if self.visual:
            self.pixel_transform = v2.Compose([
                v2.Lambda(
                    lambda x: x.permute(0, 1, 4, 2, 3).reshape(
                        x.shape[0], -1, x.shape[2], x.shape[3]
                    )
                )
            ])
        
        self.train_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset) for i in range(config["num_envs"])])
        self.eval_envs = gymnasium.vector.SyncVectorEnv([lambda offset=i : self.setup_env(task_name, seed=self.seed+offset+1000) for i in range(config["num_envs"])])
        
        obs_space = self.train_envs.single_observation_space
        obs_dim = obs_space["pixels"].shape if self.visual else obs_space.shape
        action_dim = self.train_envs.single_action_space.shape
        
        if self.visual:
            encoder_input = torch.zeros((1, *obs_dim), dtype=torch.uint8)
            in_channel = self.pixel_transform(encoder_input).shape[1]
            self.encoder = VisualObservationEncoderNet(in_channel).to(self.device)
        else:
            self.encoder = StateObservationEncoderNet(np.prod(obs_dim), config["encoder_layers"], norm_position=NormPosition.POST, dropout_prob=config["encoder_dropout_prob"]).to(self.device)
        self.actor = DDPGActor(self.encoder.feature_dim, np.prod(action_dim), config["actor_layers"], norm_position=NormPosition.POST).to(self.device)
        self.critic = DDPGCritic(self.encoder.feature_dim, np.prod(action_dim), config["critic_layers"], norm_position=NormPosition.POST).to(self.device)
        self.critic_target = DDPGCritic(self.encoder.feature_dim, np.prod(action_dim), config["critic_layers"], norm_position=NormPosition.POST).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["learning_rate"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        
        self.replay_buffer = ReplayBuffer(config["num_envs"], config["max_buffer_size"], device=self.device)
        obs_dtype = torch.uint8 if self.visual else torch.float32
        self.replay_buffer.create_storage_space("observations", obs_dim, obs_dtype)
        self.replay_buffer.create_storage_space("next_observations", obs_dim, obs_dtype)
        self.replay_buffer.create_storage_space("actions", action_dim, torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)
        
        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        
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
    def get_action(self, obs_batch, deterministic, random):
        obs_batch = self._observation_to_tensor(obs_batch)
        obs_batch = self.encoder(obs_batch)
        actor_step:DeterministicContinuousPolicyStep = self.actor(obs_batch, self.std)
        if deterministic:
            action = actor_step.mean
        else:    
            action = actor_step.pi.rsample()

        if random:
            action = actor_step.mean.uniform_(-1, 1) * self.max_action
        
        return action
    
    def rollout(self):
        obs, info = self.train_envs.reset()
        obs = self._select_observation(obs)
        for i in range(self.max_steps):
            action = self.get_action(obs, False, False)
            next_obs, reward, done, timeout, info = self.train_envs.step(action.cpu().numpy())
            next_obs = self._select_observation(next_obs)
            if "motion" in info:
                reward = self.get_motion_reward(info["motion"])
                
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": reward,
                "dones": done
            }
            
            self.replay_buffer.add_records(record)
            obs = next_obs
        
    def eval(self):
        log_data = {}
        obs, info = self.eval_envs.reset()
        obs = self._select_observation(obs)
        episode_success = 0
        for i in range(self.eval_steps):
            action = self.get_action(obs, True, False)
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
            next_obs_batch = self._observation_to_tensor(batch["next_observations"])
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)

            aug = True
            
            feature_batch = self.encoder(obs_batch, aug)
            with torch.no_grad():
                next_feature_batch = self.encoder(next_obs_batch, aug)

            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss_dict = DDPGDoubleQ.compute_critic_loss(
                self.actor, self.critic, self.critic_target, feature_batch, action_batch, reward_batch, next_feature_batch, done_batch, self.std, self.gamma
            )
            
            critic_loss = critic_loss_dict["loss"]
            q1 = critic_loss_dict["q1"]
            q2 = critic_loss_dict["q2"]
            q_target = critic_loss_dict["q_target"]
            
            critic_loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss_dict = DDPGDoubleQ.compute_policy_loss(self.actor, self.critic, feature_batch.detach(), self.std, self.regularization_weight)
            policy_loss = policy_loss_dict["loss"]
            policy_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            DDPGDoubleQ.update_target_param(self.critic, self.critic_target, self.tau)

    
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
            
            mix = np.clip(epoch/self.epochs, 0, 1)
            self.std = (1-mix) * 1 + mix * 0.1
            
            if (epoch + 1) % self.eval_frequence == 0:
                log_data = self.eval()
                self.log(epoch+1, log_data)
                torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ddpg/{self.task_name}/actor_{self.seed}_{epoch+1}.pt")

                episode_success_rate = log_data.get("avg_success_rate", None)
                episode_reward = log_data.get("avg_episode_reward", None)
                metric = episode_success_rate if episode_success_rate is not None else episode_reward
                
                if metric >= best_record:
                    best_record = metric
                    torch.save([self.encoder.state_dict(), self.actor.state_dict(), self.critic.state_dict()], f"weights/ddpg/{self.task_name}/actor_best_{self.seed}.pt")

        time = datetime.now()
        print(f"[{time}] end")

        print("-------------------------------")

if __name__ == '__main__':
    
    args = get_train_args()
    weight_folder = f"weights/ddpg/{args.task}"
        
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    trainer = Trainer(args.task, args.seed, args.visual)
    trainer.train()

    if args.save_buffer:
        replay_dir = f"replays/{args.task}"
        os.makedirs(replay_dir, exist_ok=True)
        replay_path = os.path.join(replay_dir, f"buffer_{args.seed}.pt")
        trainer.replay_buffer.save(replay_path)
