import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 1, 320

        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        # x: 1, 1280
        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, UNetAttentionBlock):
                x = module(x, context)
            elif isinstance(module, UNetResidualBlock):
                x = module(x, time)
            else:
                x = module(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, channels, height, width

        # batch_size, channels, height, width -> batch_size, channels, height * 2, width * 2
        x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
        x = self.conv(x)
        return x


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280) -> None:
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.linear = nn.Linear(n_time, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: batch_size, in_channels, height, width
        # time: 1, 1280
        residue = x
        # batch_size, in_channels, height, width -> batch_size, out_channels, height, width
        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time = F.silu(time)
        time  = self.linear(time).unsqueeze(-1).unsqueeze(-1)

        merged = x + time
        # batch_size, out_channels, height, width -> batch_size, out_channels, height, width
        x = self.group_norm2(merged)
        x = F.silu(merged)
        x = self.conv2(merged)

        # batch_size, in_channels, height, width -> batch_size, out_channels, height, width
        return merged + self.skip(residue)

class UNetAttentionBlock(nn.Module):
    def __init__(self, num_head: int, n_embed: int, d_context = 768) -> None:
        super().__init__()
        channels = num_head * n_embed

        self.group_norm1 = nn.GroupNorm(32, channels, eps = 1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 1)
        self.layer_norm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(num_head, channels, in_proj_bias=False)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(num_head, channels, d_context, in_proj_bias=False)
        self.layer_norm3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels * 4, channels * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 1)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: batch_size, channels, height, width
        # context: batch_size, seq_len, dim

        residue_long = x

        # batch_size, channels, height, width -> batch_size, channels, height, width
        x = self.group_norm1(x)
        x = self.conv1(x)

        n, c, h, w = x.shape

        # batch_size, channels, height, width -> batch_size, channels, height * width -> batch_size, height * width, channels
        x = x.view(n, c, -1).transpose(-1, -2)

        residue_short = x

        # batch_size, channels, height, width -> batch_size, channels, height, width
        x = self.layer_norm1(x)
        x = self.attention1(x)
        x += residue_short

        residue_short = x
        # batch_size, channels, height, width -> batch_size, channels, height, width
        x = self.layer_norm2(x)
        x = self.attention2(x, context)
        x += residue_short

        # batch_size, channels, height, width -> batch_size, channels, height, width
        x = self.layer_norm3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        # batch_size, height * width, channels -> batch_size, channels, height * width -> batch_size, channels, height, width
        x = x.transpose(-1, -2).view(n, c, h, w)

        return self.conv2(x) + residue_long

class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.ModuleList([

            # batch_size, 4, height / 8, width / 8 -> batch_size, 320, height / 8, width / 8
            SwitchSequential(
                nn.Conv2d(4, 320, kernel_size = 3, padding = 1),
            ),

            SwitchSequential(
                UNetResidualBlock(320, 320),
                UNetAttentionBlock(8, 40),
            ),

            SwitchSequential(
                UNetResidualBlock(320, 320),
                UNetAttentionBlock(8, 40),
            ),

            # batch_size, 320, height / 8, width / 8 -> batch_size, 320, height / 16, width / 16
            SwitchSequential(
                nn.Conv2d(320, 320, kernel_size = 3, stride = 2, padding = 1),
            ),

            SwitchSequential(
                UNetResidualBlock(320, 640),
                UNetAttentionBlock(8, 80),
            ),

            SwitchSequential(
                UNetResidualBlock(640, 640),
                UNetAttentionBlock(8, 80),
            ),

            # batch_size, 640, height / 16, width / 16 -> batch_size, 640, height / 32, width / 32
            SwitchSequential(
                nn.Conv2d(640, 640, kernel_size = 3, stride = 2, padding = 1),
            ),

            SwitchSequential(
                UNetResidualBlock(640, 1280),
                UNetAttentionBlock(8, 160),
            ),

            SwitchSequential(
                UNetResidualBlock(1280, 1280),
                UNetAttentionBlock(8, 160),
            ),

            # batch_size, 1280, height / 32, width / 32 -> batch_size, 1280, height / 64, width / 64
            SwitchSequential(
                nn.Conv2d(1280, 1280, kernel_size = 3, stride = 2, padding = 1),
            ),

            SwitchSequential(
                UNetResidualBlock(1280, 1280),
            ),

            SwitchSequential(
                UNetResidualBlock(1280, 1280),
            ),
        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),

            UNetAttentionBlock(8, 160),

            UNetResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([

            SwitchSequential(
                # batch_size, 2560, height / 64, width / 64 -> batch_size, 1280, height / 64, width / 64
                UNetResidualBlock(2560, 1280),
            ),

            SwitchSequential(
                # batch_size, 2560, height / 64, width / 64 -> batch_size, 1280, height / 64, width / 64
                UNetResidualBlock(2560, 1280),
            ),

            SwitchSequential(
                # batch_size, 2560, height / 64, width / 64 -> batch_size, 1280, height / 64, width / 64
                UNetResidualBlock(2560, 1280),
                UpSample(1280)
            ),

            SwitchSequential(
                UNetResidualBlock(2560, 1280),
                UNetAttentionBlock(8, 160),
            ),

            SwitchSequential(
                UNetResidualBlock(2560, 1280),
                UNetAttentionBlock(8, 160),
            ),

            SwitchSequential(
                UNetResidualBlock(1920, 1280),
                UNetAttentionBlock(8, 160),
                UpSample(1280)
            ),

            SwitchSequential(
                UNetResidualBlock(1920, 640),
                UNetAttentionBlock(8, 80),
            ),

            SwitchSequential(
                UNetResidualBlock(1280, 640),
                UNetAttentionBlock(8, 160),
            ),

            SwitchSequential(
                UNetResidualBlock(960, 640),
                UNetAttentionBlock(8, 80),
                UpSample(640)
            ),

            SwitchSequential(
                UNetResidualBlock(960, 320),
                UNetAttentionBlock(8, 40),
            ),

            SwitchSequential(
                UNetResidualBlock(640, 320),
                UNetAttentionBlock(8, 80),
            ),

            SwitchSequential(
                UNetResidualBlock(640, 320),
                UNetAttentionBlock(8, 40),
            ),

        ])


class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, in_channels, height / 8, width / 8

        x = self.group_norm(x)
        x = F.silu(x)
        # batch_size, in_channels, height / 8, width / 8 -> batch_size, out_channels, height / 8, width / 8
        x = self.conv(x)
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, seq_len: int) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final_layer = UNet_OutputLayer(320, 4)    

    def forward(self, latent: torch.Tensor ,context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:

        # latent: batch_size, 4, height / 8, width / 8
        # context: batch_size, seq_len, dim
        # time: 1, 320

        # (1, 320) -> (1, 320, 1280)
        time = self.time_embedding(time)

        # batch_size, 4, height / 8, width / 8 -> batch_size, 320, height / 8, width / 8
        output = self.unet(latent, context, time)

        # batch_size, 320, height / 8, width / 8 -> batch_size, 4, height / 8, width / 8
        output = self.final_layer(output)

        return output
