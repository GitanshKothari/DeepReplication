import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self) -> None:
        super.__init__(
            # batch, channel, height, width -> batch, 128, height, width
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # batch, 128, height, width -> batch, 128, height, width
            VAE_ResidualBlock(128, 128),

            # batch, 128, height, width -> batch, 128, height, width
            VAE_ResidualBlock(128, 128),

            #  batch, 128, height, width -> batch, 128, height/2, width/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # batch, 128, height / 2, width / 2 -> batch, 256, height / 2, width / 2
            VAE_ResidualBlock(128, 256),

            # batch, 256, height / 2, width / 2 -> batch, 256, height / 2, width / 2
            VAE_ResidualBlock(256, 256),

            #  batch, 256, height / 2, width / 2 -> batch, 256, height / 4, width / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # batch, 256, height height / 4, width / 4 -> batch, 512, height height / 4, width / 4
            VAE_ResidualBlock(256, 512),

            # batch, 512, height height / 4, width / 4 -> batch, 512, height height / 4, width / 4
            VAE_ResidualBlock(512, 512),

            #  batch, 512, height / 4, width / 4 -> batch, 512, height/8, width/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # batch, 512, height / 8, width / 8-> batch, 512, height / 8, width / 8
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 8, width / 8-> batch, 512, height / 8, width / 8
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 8, width / 8-> batch, 512, height / 8, width / 8
            VAE_ResidualBlock(512, 512),

            # batch, 512, height / 8, width / 8-> batch, 512, height / 8, width / 8
            VAE_AttentionBlock(512),

            # batch, 512, height / 8, width / 8-> batch, 512, height / 8, width / 8
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # batch, 512, height / 8, width / 8-> batch, 8, height / 8, width / 8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # batch, 8, height / 8, width / 8-> batch, 8, height / 8, width / 8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),

        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: batch, channel, height, width
        # noise: batch, out_channels, height / 8, width / 8

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # apply padding on right and bottom 
                x = F.pad(x, (0, 1, 0, 1)) # sequence of (left, right, top, bottom)
            
            x = module(x)
        
        # batch, 8, height / 8, width / 8 -> two tensors of shape (batch, 4, height / 8, width / 8)
        mean, log_variance = x.chunk(2, dim=1)

        log_variance = log_variance.clamp(-30, 20)
        variance = log_variance.exp()
        std_dev = variance.sqrt()

        # Noise = N(0, 1) --> X = N(mean, variance)
        x = mean + std_dev * noise

        # Scale by constant
        x *= 0.18215

        return x




