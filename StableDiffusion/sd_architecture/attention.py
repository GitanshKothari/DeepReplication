import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True) -> None:
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.num_heads = num_heads
        self.dim_head = d_embed // num_heads
    
    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: batch, seq_len, d_embed
        batch_size, seq_len, d_embed = x.shape

        # batch, seq_len, d_embed -> batch, seq_len, d_embed * 3 -> 3 tensors of shape (batch, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # batch, seq_len, d_embed -> batch, seq_len, num_heads, dim_head -> batch, num_heads, seq_len, dim_head
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)

        weight = q @ k.transpose(-2, -1) / math.sqrt(self.dim_head)
        if causal_mask:
            mask = torch.triu(torch.ones_like(seq_len), diagonal = 1)
            weight = weight.masked_fill(mask, float('-inf'))
        
        weight = F.softmax(weight, dim = -1)

        # (batch_size, num_heads, seq_len, seq_len) @ batch, num_heads, seq_len, dim_head -> batch_size, num_heads, seq_len, dim_head
        output = weight @ v

        output = output.transpose(1, 2).view(batch_size, seq_len, d_embed)
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True) -> None:
        super().__init__()

        self.q = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k = nn.Linear(d_cross, d_embed, bias = out_proj_bias)
        self.v = nn.Linear(d_cross, d_embed, bias = out_proj_bias)
        self.out = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.num_heads = num_heads
        self.dim_head = d_embed // num_heads

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        # x: batch, seq_lenQ, dimQ
        # cross: batch, seq_lenKV, d_crossKV = Batchsize, 77, 768
        batch_size, seq_len, d_embed = x.shape

        # batch, seq_len, d_embed -> batch, seq_len, d_embed * 3 -> 3 tensors of shape (batch, seq_len, d_embed)
        q = self.q(x)
        k = self.k(cross)
        v = self.v(cross)

        # batch, seq_len, d_embed -> batch, seq_len, num_heads, dim_head -> batch, num_heads, seq_len, dim_head
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.dim_head)
        weight = F.softmax(weight, dim = -1)

        # (batch_size, num_heads, seq_len, seq_len) @ batch, num_heads, seq_len, dim_head -> batch_size, num_heads, seq_len, dim_head
        output = weight @ v

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_embed)
        output = self.out(output)

        return output
