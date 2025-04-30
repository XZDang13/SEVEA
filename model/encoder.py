import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, Conv2DLayer, MLPLayer
    
class EncoderNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dims:list[int]):
        super().__init__()
        
        self.layers = nn.ModuleList(self.init_layers(state_dim, hidden_dims))

    def init_layers(self, in_dim:int, hidden_dims:list[int]):
        layers = []
        dim = in_dim
        
        for hidden_dim in hidden_dims:
            mlp = MLPLayer(dim, hidden_dim, None, True)
            dim = hidden_dim

            layers.append(mlp)

        self.dim = dim
        return layers
    
    def get_features(self, x:torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = F.silu(x)

        return features

    def forward(self, x:torch.Tensor, aug:bool=False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = F.silu(x)
            x = F.dropout(x, p=0.1, training=aug)

        return x
    
class FrameObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int, feature_dim:int):
        super().__init__()
        
        self.aug = RandomShiftsAug(4)

        self.dim = feature_dim

        self.cnn_layers = nn.Sequential(
            Conv2DLayer(in_channel, 64, 3, 2, 1, F.silu, True),
            Conv2DLayer(64, 128, 3, 2, 1, F.silu, True),
            Conv2DLayer(128, 256, 3, 2, 1, F.silu, True),
            Conv2DLayer(256, 512, 3, 2, 1, F.silu, True)
        )
        
        self.mlp_layer = nn.Sequential(
            MLPLayer(512*7*7, feature_dim, None, True),
            #MLPLayer(1024, feature_dim, F.silu, True),
        )
        
    def forward(self, x:torch.Tensor, with_act_func:bool=True, aug:bool=False) -> torch.Tensor:
        if aug:
            x = self.aug(x)
        x = self.cnn_layers(x)
        #x = F.avg_pool2d(x, 7)
        x = x.flatten(1)
        x = self.mlp_layer(x)
        if with_act_func:
            x = F.silu(x)
        return x
    
class MobileFrameObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int, feature_dim:int):
        super().__init__()
        
        self.dim = feature_dim
        
        self.cnn_layers = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
        self.cnn_layers.features[0][0].weight.data = self.cnn_layers.features[0][0].weight.data.repeat(1, 2, 1, 1) / 2
        self.cnn_layers.classifier = nn.Identity()
        
        self.mlp_layer = nn.Sequential(
            MLPLayer(960, feature_dim, None, True),
            #MLPLayer(1024, feature_dim, F.silu, True),
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        #x = F.avg_pool2d(x, 7)
        x = x.flatten(1)
        x = self.mlp_layer(x)
        
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