from typing import List
import torch
import torch.nn as nn
from typing import List


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels)
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = torch.cat((x, skip_connection), dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List = [64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        all_features = [in_channels, *features] # 3, 64, 128, 256, 512
        # List containing input and output feature tuples for all blocks of an encoder
        feature_tuples = zip(all_features, all_features[1:])
        # Module list containing all encoder blocks
        self.down_convs = nn.ModuleList([DoubleConv(*fts) for fts in feature_tuples])
        # Module list containing all decoder blocks
        self.up_convs = nn.ModuleList([DecoderBlock(2 * fts, fts) for fts in features[::-1]])
        # Bottleneck conv
        self.bottleneck = DoubleConv(features[-1], 2*features[-1])
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)
        




if __name__== '__main__':
    model = UNet()