import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, Conv2DLayer, MLPLayer
    
class FrameObservationEncoderNet(nn.Module):
    def __init__(self, in_channel:int, feature_dim:int):
        super().__init__()
        
        self.dim = feature_dim
        
        #self.cnn_layers = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
        #self.cnn_layers.features[0][0].weight.data = self.cnn_layers.features[0][0].weight.data.repeat(1, 2, 1, 1) / 2
        #self.cnn_layers.classifier = nn.Identity()

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
    
class EncoderNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dims:list[int]):
        super().__init__()
        
        in_dim = state_dim
        layers = []

        for dim in hidden_dims:
            mlp = MLPLayer(in_dim, dim, F.tanh, True)
            in_dim = dim

            layers.append(mlp)

        self.layers = nn.ModuleList(layers)
        self.dim = in_dim
        
    def forward(self, x:torch.Tensor, drop_out:bool=False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = F.dropout(x, 0.25, training=drop_out)
        
        return x