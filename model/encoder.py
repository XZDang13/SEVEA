from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, MLPLayer, NormPosition

class StateObservationEncoderNet(nn.Module):
    def __init__(self, obs_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST, dropout_prob:float=0.2):
        super().__init__()

        self.layers, self.feature_dim = make_mlp_layers(obs_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)
        
        self.dropout_prob = dropout_prob

    def forward(self, obs:torch.Tensor, aug:bool=False) -> torch.Tensor:
        x = self.layers(obs)
        if aug:
            x = F.dropout(x, p=self.dropout_prob)
            
        return x