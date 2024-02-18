import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, channels, height, width
        residue = x
        n, c, h, w = x.shape

        # batch, channels, height, width -> batch, channels, height * width
        x = x.view(n, c, -1) 

        # batch, channels, height * width -> batch, height * width, channels
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # batch, height * width, channels -> batch, channels, height * width
        x = x.transpose(-1, -2)

        # batch, channels, height * width -> batch, channels, height, width
        x = x.view(n, c, h, w)

        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = batch, in_channels, height, width
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x += self.skip(residual)
        return x

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 8, width / 8 -> batch, 512, height / 8, width / 8
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 8, width / 8 -> batch, 512, height / 4, width / 4
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 4, width / 4 -> batch, 512, height / 2, width / 2
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # batch, 256, height / 2, width / 2 -> batch, 256, height, width
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            # batch, 128, height, width -> batch, 3, height, width
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, 4, height / 8, width / 8

        x /= 0.81215

        for module in self:
            x = module(x)
        
        return x