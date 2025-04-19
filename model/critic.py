import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import CriticHead, make_mlp_layers, MLPLayer

class ValueNet(nn.Module):
    def __init__(self, feature_dim:int, hidden_dims:list[int]):
        super().__init__()
        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)

        self.value = CriticHead(dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        value = self.value(x)

        return value
    
class QNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int]):
        super().__init__()
        self.layers, in_dim = make_mlp_layers(feature_dim+action_dim, hidden_dims, F.silu, True)
        self.critic_layer = CriticHead(in_dim)

    def forward(self, feature:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([feature, action], 1)
        x = self.layers(x)
       
        q = self.critic_layer(x)

        return q
    
class CriticNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int]):
        super().__init__()
        self.qnet_1 = QNet(feature_dim, action_dim, hidden_dims)
        self.qnet_2 = QNet(feature_dim, action_dim, hidden_dims)

    def forward(self, feature:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.qnet_1(feature, action)
        q2 = self.qnet_2(feature, action)

        return q1, q2