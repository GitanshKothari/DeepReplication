import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, dim))
    
    def forward(self, tokens):
        # tokens: batch, seq_len

        # batch, seq_len -> batch, seq_len, dim
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, num_head: int, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(num_head, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, seq_len, dim

        residue = x
        x = self.norm1(x)
        x = self.attention(x, causal_mask = True)
        x += residue

        residue = x
        x = self.norm2(x)
        x = self.linear1(x)
        x *= torch.sigmoid(1.702 * x) # quick gelu activation
        x = self.linear2(x)
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.norm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: batch, seq_len
        tokens = tokens.type(torch.long)

        # batch, seq_len -> batch, seq_len, dim
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x)
        
        # batch, seq_len, dim
        x = self.norm(x)
        return x