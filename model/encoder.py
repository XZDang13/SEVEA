from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, MLPLayer, NormPosition

class StateObservationEncoderNet(nn.Module):
    def __init__(self, obs_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST, dropout_prob:float=0.2):
        super().__init__()

        if len(hidden_dims) > 1:
            self.layers, feature_dim = make_mlp_layers(obs_dim, hidden_dims[:-1], activate_function=nn.SiLU(), norm_position=norm_position)
        else:
            self.layers = nn.Identity()
            feature_dim = obs_dim
            
        self.feature_layer = MLPLayer(feature_dim, hidden_dims[-1], activate_func=nn.Identity(), norm_position=NormPosition.NONE) 
        self.feature_dim = hidden_dims[-1]
        
        self.dropout_prob = dropout_prob

    def forward(self, obs:torch.Tensor, aug:bool=False) -> torch.Tensor:
        x = self.layers(obs)
        x = self.feature_layer(x)
        if aug:
            x = F.dropout(x, p=self.dropout_prob)
            
        return x