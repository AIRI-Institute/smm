import torch
from torch import nn

from .resblock import ResnetBlock


class Encoder(nn.Module):
    """
    CNN encoder for set prediction task, which reduces the spatial resolution 4 times
    """
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_size, 5, padding=(2, 2)), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(2, 2)), nn.ReLU()
        ])

    def forward(self, inputs):
        return self.layers(inputs)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0
        )

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class EfficientEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample_steps=1, dropout=0.1):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (downsample_steps + 1)
        assert len(hidden_channels) == (downsample_steps + 1)
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                hidden_channels[0], 
                kernel_size=5, 
                stride=1, 
                padding=2
            )
        ])
        for i in range(downsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(hidden_channels[i], hidden_channels[i + 1], dropout),
                    Downsample(hidden_channels[i+1])
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
