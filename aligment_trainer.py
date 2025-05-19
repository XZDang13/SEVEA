import os
import argparse

import yaml
from tqdm import trange
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from RLAlg.utils import set_seed_everywhere

from config import METAWORLD_CFGS, DMC_CFGS
from model.encoder import FrameObservationEncoderNet, EncoderNet
from state_frame_dataset import get_dataloader

def get_train_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default='buttonpress', help='the task name')
    parse.add_argument('--seed', type=int, default=0, help='the random seed to reproduce results')
    parse.add_argument('--alg', type=str, default="ddpg", help='RL algorithm name')
    parse.add_argument('--feature_layer', type=int, default=-1, help='the layer of the encoder to be used')

    return parse.parse_args()

def cosine_loss(x, y):
    return 1. - F.cosine_similarity(x, y, dim=-1).mean()

def mse_loss(x, y):
    return F.mse_loss(x, y, reduction="mean")

class Alignment(nn.Module):
    def __init__(self, state_encoder, frame_encoder, state_feature_layer=-1):
        super().__init__()

        self.state_feature_layer = state_feature_layer

        self.state_encoder = state_encoder

        self.frame_encoder = frame_encoder

        self.state_encoder.eval()

        for param in self.state_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_states(self, vectors):
        state_features = self.state_encoder.get_features(vectors)

        return state_features[self.state_feature_layer]

    def encode_frames(self, frames):
        frame_features = self.frame_encoder(frames, False, False)

        return frame_features
    
    def forward(self, frames, states):
        frame_features = self.encode_frames(frames)
        state_features = self.encode_states(states)

        return frame_features, state_features
    
class Trainer:
    def __init__(self, task_name:str, seed:int, alg:str, feature_layer:int):
        if task_name in METAWORLD_CFGS:
            config_path = "configs/ddpg_metaworld.yaml"
            state_dim = METAWORLD_CFGS[task_name]["state_dim"]
        elif task_name in DMC_CFGS:
            config_path = "configs/ddpg_dmc.yaml"
            state_dim = DMC_CFGS[task_name]["state_dim"]

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)


        self.epochs = 20

        self.task_name = task_name
        self.seed = seed
        self.alg = alg
        self.feature_layer = feature_layer

        weight_folder = f"weights/{alg}/{task_name}"
                
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

        set_seed_everywhere(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_weight, _, _ = torch.load(f"{weight_folder}/actor_best_{seed}.pt", weights_only=True)

        state_encoder = EncoderNet(state_dim, config["encoder_layers"]).to(self.device)
        frame_encoder = FrameObservationEncoderNet(6, state_encoder.dim).to(self.device)

        state_encoder.load_state_dict(encoder_weight)

        self.model = Alignment(state_encoder, frame_encoder, feature_layer).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.epochs)
        self.dataloader = get_dataloader(task_name)
        self.size = len(self.dataloader.dataset)

    def train(self):
        for _ in trange(self.epochs, desc="Epochs"):
            running_loss = 0.0
            for _, (vectors, frames) in enumerate(self.dataloader):
                vectors = vectors.to(self.device)
                frames = frames.to(self.device)
                
                frame_features, vector_features = self.model(frames, vectors)
                
                loss = 0.5 * mse_loss(frame_features, vector_features) + 0.5 * cosine_loss(frame_features, vector_features)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * vectors.size(0)
            
            self.scheduler.step()
            train_loss = running_loss / self.size
            tqdm.tqdm.write(f"Train Loss: {train_loss:.4f}")

        torch.save(self.model.frame_encoder.state_dict(), f"weights/{self.alg}/{self.task_name}/frame_encoder_{self.seed}_{self.feature_layer}.pt")

if __name__ == '__main__':
    
    args = get_train_args()

    trainer = Trainer(args.task, args.seed, args.alg, args.feature_layer)
    trainer.train()