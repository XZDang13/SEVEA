from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, GaussianHead, DeterministicHead, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DeterministicContinuousPolicyStep

class PPOActor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = GaussianHead(feature_dim, action_dim, max_action=max_action, state_dependent_std=False)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x, action)

        return step
    
class DDPGActor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = DeterministicHead(feature_dim, action_dim, max_action=max_action)

    def forward(self, x:torch.Tensor, std:float) -> DeterministicContinuousPolicyStep:
        x = self.layers(x)

        step:DeterministicContinuousPolicyStep = self.head(x, std)

        return step
    
class IQLActor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = GaussianHead(feature_dim, action_dim, max_action=max_action, state_dependent_std=False)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x, action)

        return step