import torch
from torch import nn

from .resblock import ResnetBlock


def nonlinearity(x):
    return x*torch.sigmoid(x)


class Decoder(nn.Module):
    """
    Decoder for autoencoder model
    """
    def __init__(self, num_channels=64):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
        ])
        self.final_module = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(2, 2), output_padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels, 4, 3, padding=(1,1), output_padding=0, stride=1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)[:, :, :-1, :-1]
        return self.final_module(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class EfficientDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, upsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        hidden_channels = [hidden_channels] + [64] * upsample_steps
        for i in range(upsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(hidden_channels[i], hidden_channels[i+1], dropout),
                    Upsample(hidden_channels[i+1])
                )
            )
        self.norm = Normalize(hidden_channels[-1])
        self.conv_out = nn.Conv2d(
            hidden_channels[-1],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return torch.tanh(x)
