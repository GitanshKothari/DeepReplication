import torch
import torch.nn as nn

def get_time_embed(timesteps, embed_dim):
    denom = 10000 ** (torch.arange(0, embed_dim//2, device = timesteps.device) / (embed_dim//2))
    embeddings = timesteps[:, None].repeat(1, embed_dim//2) / denom
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], -1)
    return embeddings

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, downsample, num_heads):
        super().__init__()
        self.downsample = downsample
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

        self.t_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )   

        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ) 

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True) 
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.downsample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if downsample else nn.Identity()

    def forward(self, x, t_embed):
        
        out = self.resnet_conv1(x)
        out = out + self.t_embedding_layer(t_embed)[:, :, None, None]
        out = self.resnet_conv2(out)
        out = out + self.residual_conv(x)

        batch_size, channels, height, width = out.shape
        attn_input = out.reshape(batch_size, channels, -1)
        attn_input = self.attention_norm(attn_input)
        attn_input = attn_input.transpose(1, 2)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + attn_output

        out = self.downsample_conv(out)
        return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_heads):
        super().__init__()
        self.resnet_conv1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            ),
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )
        ])

        self.t_embedding_layer = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, out_channels),
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, out_channels),
            )
        ])

        self.resnet_conv2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            ),
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )
        ])
        
        self.attention_norm = nn.GroupNorm(8, in_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True) 

        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 1),
        ])

    def forward(self, x, t_embed):
    
        out = self.resnet_conv1[0](x)
        out = out + self.t_embedding_layer[0](t_embed)[:, :, None, None]
        out = self.resnet_conv2[0](out)
        out = out + self.residual_conv[0](x)

        batch_size, channels, height, width = out.shape
        attn_input = out.reshape(batch_size, channels, -1)
        attn_input = self.attention_norm(attn_input)
        attn_input = attn_input.transpose(1, 2)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + attn_output

        out = self.resnet_conv1[1](out)
        out = out + self.t_embedding_layer[1](t_embed)[:, :, None, None]
        out = self.resnet_conv2[1](out)
        out = out + self.residual_conv[1](x)

        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, upsample, num_heads):
        super().__init__()
        self.upsample = upsample
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

        self.t_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )   

        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ) 

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True) 
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample_conv = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1) if upsample else nn.Identity()

    def forward(self, x, downblock_output, t_embed):
        
        x = self.upsample_conv(x)
        x = torch.cat([x, downblock_output], 1)

        out = self.resnet_conv1(x)
        out = out + self.t_embedding_layer(t_embed)[:, :, None, None]
        out = self.resnet_conv2(out)
        out = out + self.residual_conv(x)

        batch_size, channels, height, width = out.shape
        attn_input = out.reshape(batch_size, channels, -1)
        attn_input = self.attention_norm(attn_input)
        attn_input = attn_input.transpose(1, 2)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + attn_output

        return out

class UNet(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.downchannels = [32, 64, 128, 256]
        self.midchannels = [256, 256, 128]
        self.t_embed_dim = 128
        self.downsample = [True, True, False]

        self.t_projection = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
            nn.SiLU(),
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
        )

        self.upsample = list(reversed(self.downsample))
        self.conv_in = nn.Conv2d(in_channels, self.downchannels[0], 3, 1)

        self.down = nn.ModuleList([])
        for i in range(len(self.downchannels)):
            self.down.append(DownBlock(self.downchannels[i], self.downchannels[i+1], self.t_embed_dim, self.downsample[i], 4))

        self.mid = nn.ModuleList([])
        for i in range(len(self.midchannels)):
            self.mid.append(MidBlock(self.midchannels[i], self.midchannels[i+1], self.t_embed_dim, 4))
        
        self.up = nn.ModuleList([])
        for i in reversed(range(len(self.downchannels))):
            self.up.append(UpBlock(self.downchannels[i] * 2, self.downchannels[i - 1] if i != 0 else 16, self.t_embed_dim, self.upsample[i], 4))

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(16, in_channels, 3, 1),
        )
            

    def forward(self, x, timesteps):
        out = self.conv_in(x)
        t_embed = get_time_embed(timesteps, self.t_embed_dim)
        t_embed = self.t_projection(t_embed)

        downblock_outputs = []
        for down in self.down:
            downblock_outputs.append(out)
            out = down(out, t_embed)
        
        for mid in self.mid:
            out = mid(out, t_embed)
        
        for up in self.up:
            out = up(out, downblock_outputs.pop(), t_embed)
        
        out = self.norm_out(out)
        out = self.conv_out(out)

        return out



