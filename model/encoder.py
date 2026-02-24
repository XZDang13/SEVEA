from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from r3m import load_r3m
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
    
class R3MObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int=6):
        super().__init__()
        
        self.feature_dim = 512
        
        self.r3m = load_r3m('resnet18')
        self._expand_input_conv(in_channel)
        self._expand_input_norm(in_channel)

    def _expand_input_conv(self, in_channel:int) -> None:
        backbone = self.r3m.module if hasattr(self.r3m, "module") else self.r3m
        conv1: nn.Conv2d = backbone.convnet.conv1
        old_in_channel = conv1.in_channels

        if in_channel == old_in_channel:
            return
        if in_channel < old_in_channel:
            raise ValueError(
                f"R3M input channels must be >= {old_in_channel}, got {in_channel}"
            )
        if in_channel % old_in_channel != 0:
            raise ValueError(
                f"CCE requires in_channel to be {old_in_channel}m, got {in_channel}"
            )

        expansion_ratio = in_channel // old_in_channel

        new_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        ).to(device=conv1.weight.device, dtype=conv1.weight.dtype)

        with torch.no_grad():
            expanded_weight = conv1.weight.repeat(1, expansion_ratio, 1, 1)
            expanded_weight.mul_(1.0 / expansion_ratio)
            new_conv.weight.copy_(expanded_weight)
            if conv1.bias is not None:
                new_conv.bias.copy_(conv1.bias)

        backbone.convnet.conv1 = new_conv

    def _expand_input_norm(self, in_channel:int) -> None:
        backbone = self.r3m.module if hasattr(self.r3m, "module") else self.r3m
        normlayer = backbone.normlayer
        old_norm_channels = len(normlayer.mean)

        if in_channel == old_norm_channels:
            return
        if in_channel % old_norm_channels != 0:
            raise ValueError(
                f"CCE norm expansion requires in_channel to be {old_norm_channels}m, got {in_channel}"
            )

        expansion_ratio = in_channel // old_norm_channels
        expanded_mean = list(normlayer.mean) * expansion_ratio
        expanded_std = list(normlayer.std) * expansion_ratio
        backbone.normlayer = type(normlayer)(mean=expanded_mean, std=expanded_std)
        
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.r3m(obs)
            
        return x
