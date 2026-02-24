from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, Conv2DLayer, MLPLayer, NormPosition

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
    
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
    
class VisualObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int):
        super().__init__()
        
        self.aug = RandomShiftsAug(pad=4)
        self.feature_dim = 256
        
        self.layers = nn.Sequential(
            Conv2DLayer(in_channel=in_channel, out_channel=16, kernel_size=3, stride=2, padding=1, activate_func=nn.SiLU(), norm=True),
            Conv2DLayer(in_channel=16, out_channel=32, kernel_size=3, stride=2, padding=1, activate_func=nn.SiLU(), norm=True),
            Conv2DLayer(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1, activate_func=nn.SiLU(), norm=True),
            Conv2DLayer(in_channel=64, out_channel=128, kernel_size=3, stride=2, padding=1, activate_func=nn.SiLU(), norm=True)
        )
        
        self.projection = MLPLayer(6272, self.feature_dim, activate_func=nn.SiLU(), norm_position=NormPosition.POST)
        
    def forward(self, obs:torch.Tensor, aug:bool=False) -> torch.Tensor:
        if aug:
            obs = self.aug(obs)
            
        x = self.layers(obs)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        
        return x
