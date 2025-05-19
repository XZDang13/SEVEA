import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import GuassianHead, DeterminicHead, SquashedGaussianHead, make_mlp_layers, MLPLayer
from RLAlg.distribution import TruncatedNormal

class PPOActorNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)

        self.policy = GuassianHead(dim, action_dim, max_action)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None=None) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        pi, action, log_prob = self.policy(x, action)

        return pi, action, log_prob
    
class DDPGActorNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)
        self.max_action = max_action
        self.policy = DeterminicHead(dim, action_dim, max_action)

    def forward(self, x:torch.Tensor, std:float) -> TruncatedNormal:
        x = self.layers(x)
        
        action = self.policy(x)
        std = torch.ones_like(action) * std
        dist = TruncatedNormal(action, std, -self.max_action, self.max_action)

        return dist
    
class IQLActorNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)

        self.policy = GuassianHead(dim, action_dim, max_action)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None=None) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        pi, action, log_prob = self.policy(x, action)

        return pi, action, log_prob