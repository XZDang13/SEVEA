import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, Conv2DLayer, MLPLayer

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, dims: list[int]):
        super().__init__()

        self.layers, out_dim = make_mlp_layers(in_dim, dims, F.silu, True)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layers(x)
        return out + identity
    
class EncoderNet(nn.Module):
    def __init__(self, state_dim:int, num_blocks:int, hidden_dims:list[int]):
        super().__init__()
        
        self.embedding = MLPLayer(state_dim, hidden_dims[0], None, False)

        self.layers = nn.ModuleList(self.init_layers(hidden_dims[0], num_blocks, hidden_dims))

    def init_layers(self, in_dim:int, num_blocks:int, hidden_dims:list[int]):
        layers = []
        dim = in_dim
        for _ in range(num_blocks):
            layer = ResidualBlock(dim, hidden_dims)
            dim = layer.out_dim
            layers.append(layer)

        self.dim = dim
        
        return layers

    def forward(self, x:torch.Tensor, aug:bool=False) -> torch.Tensor:
        x = self.embedding(x)
        x = F.silu(x)
        for block in self.layers:
            x = block(x)
            x = F.silu(x)

        x = F.dropout(x, p=0.25, training=aug) 
        #x = self.final_layer(x)
        #x = F.dropout(x, p=0.1, training=aug)

        return x
    
class FrameObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int, feature_dim:int):
        super().__init__()
        
        self.dim = feature_dim

        self.cnn_layers = nn.Sequential(
            Conv2DLayer(in_channel, 64, 3, 2, 1, F.silu, True),
            Conv2DLayer(64, 128, 3, 2, 1, F.silu, True),
            Conv2DLayer(128, 256, 3, 2, 1, F.silu, True),
            Conv2DLayer(256, 512, 3, 2, 1, F.silu, True)
        )
        
        self.mlp_layer = nn.Sequential(
            MLPLayer(512*7*7, feature_dim, F.tanh, True),
            #MLPLayer(1024, feature_dim, F.silu, True),
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        #x = F.avg_pool2d(x, 7)
        x = x.flatten(1)
        x = self.mlp_layer(x)
        
        return x
    
class MobileFrameObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int, feature_dim:int):
        super().__init__()
        
        self.dim = feature_dim
        
        self.cnn_layers = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
        self.cnn_layers.features[0][0].weight.data = self.cnn_layers.features[0][0].weight.data.repeat(1, 2, 1, 1) / 2
        self.cnn_layers.classifier = nn.Identity()
        
        self.mlp_layer = nn.Sequential(
            MLPLayer(960, feature_dim, F.tanh, True),
            #MLPLayer(1024, feature_dim, F.silu, True),
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        #x = F.avg_pool2d(x, 7)
        x = x.flatten(1)
        x = self.mlp_layer(x)
        
        return x