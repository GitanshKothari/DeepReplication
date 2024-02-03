import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096                     # embedding dimension
    n_layers: int = 32                  # Number of repetition
    n_heads: int = 32                   # for queries
    n_kv_heads: Optional[int] = None    # for keys and values
    vocab_size: int = -1 
    multiple_of: int = 256
    ff_dim_multiplier: Optional[int] = None
    norm_epsilon: float = 1e-5          # stability purposes for RMSNorm

    # For KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def compute_thetha_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float) -> torch.Tensor:

    assert head_dim % 2 == 0, "Head dimension must be even"

    # creates an array [0, 2, 4, 6, 8, ...] with shape (head_dim // 2)
    freqs = torch.arange(0, head_dim, 2.0, device = device).float()
    
    # (head_dim // 2) -> (1, head_dim // 2): [0, 2, 4, 6, 8] -> [[theta_1, theta_2, ... theta_d/2]]
    inv_freqs = 1 / (theta ** (freqs / head_dim)).to(device)

    # creates an array [0, 1, 2, 3, 4, ... seq_len - 1] with shape (seq_len) for the positions
    positions = torch.arange(0, seq_len, device = device)

    # creates a matrix (seq_len, head_dim // 2) with every element of positions multiplied with every element of inv_freqs
    freqs = torch.outer(positions, inv_freqs).float()

    # converts each element of the above matrix to its corresponding complex number: element m*theta_n -> (cos(m*theta_n), sin(m*theta_n))
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_pos_encodings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:

    # (B, seq_len, Head, head_dim) -> (B, seq_len, Head, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, seq_len, Head, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, Head, head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # (B, seq_len, Head, head_dim / 2) -> (B, seq_len, Head, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (torch.sqrt((x ** 2).mean(-1, keepdim = True) + self.eps)) * self.scale + self.bias
    

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        # Number of heads for keys and values
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        
        # Number of heads for queries
        self.n_heads_q = args.n_heads
        
        # Number of repetitions of keys and values heads to match the number of queries heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        # dimension of the embeddings for each head
        self.head_dim = args.dim // self.n_heads_q 


        # Linear Layer for Queries
        self.to_q = nn.Linear(args.dim, args.n_heads * args.head_dim, bias = False)

        # Linear Layer for Keys
        self.to_k = nn.Linear(args.dim, args.n_heads * args.head_dim, bias = False)

        # Linear Layer for Values
        self.to_v = nn.Linear(args.dim, args.n_heads * args.head_dim, bias = False)

        # Linear Layer for Output
        self.to_out = nn.Linear(args.n_heads * args.head_dim, args.dim, bias = False)

        self.cache_keys = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))
        self.cache_values = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, dim = x.shape #(B, 1, dim)

        # (B, 1, dim) -> (B, 1, n_heads_q, head_dim)
        q = self.to_q(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, dim) -> (B, 1, n_kv_heads, head_dim) -> (B, 1, n_rep, n_kv_heads, head_dim)
        k = self.to_k(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)#.repeat(1, 1, self.n_rep, 1).view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, dim) -> (B, 1, n_kv_heads, head_dim) -> (B, 1, n_rep, n_kv_heads, head_dim)
        v = self.to_v(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)#.repeat(1, 1, self.n_rep, 1).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        
        # Apply the rotary position encodings on queries and keys
        q = apply_rotary_pos_encodings(q, freqs_complex, device = x.device)
        k = apply_rotary_pos_encodings(k, freqs_complex, device = x.device)

        # add the keys and values to the cache
        self.cache_keys[:batch_size, start_pos:start_pos+seq_len] = k
        self.cache_values[:batch_size, start_pos:start_pos+seq_len] = v

        # retrieve the keys and values from the cache only upto the current position
        # (B, seq_len_KV, n_kv_heads, head_dim)
        k = self.cache_keys[:batch_size, :start_pos+seq_len]
        v = self.cache_values[:batch_size, :start_pos+seq_len]

        # Repeate K and V to match the number of heads for queries using n_rep calculated above
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # (B, 1, h_q, head_dim) -> (B, h_q, 1, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (B, h_q, 1, head_dim) @ (B, h_q, head_dim, seq_len_kv) -> (B, h_q, 1, seq_len_kv)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn.float(), dim = -1).type_as(q)

        # (B, h_q, 1, seq_len_kv) @ (B, h_q, seq_len_kv, head_dim) -> (B, h_q, 1, head_dim)
        out = attn @ v

        # (B, h_q, 1, head_dim) -> (B, 1, h_q, head_dim) -> (B, 1, n_heads_q * head_dim)
        out = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
    
        # (B, 1, n_heads_q * head_dim = dim) -> (B, 1, dim)
        return self.to_out(out)


    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        else:
            return x.repeat(1, 1, self.n_rep, 1).view(x.shape[0], x.shape[1], self.n_heads_q, self.head_dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ff_dim_multiplier is not None:
            hidden_dim = args.dim * args.ff_dim_multiplier

        # round it to the nearest multiple of *multiple_of*
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # Example hidden = 7, multiple_of = 5
        # ((7 + 5) - 1) = 11
        # 11 // 5 = 2
        # 2 * 5 = 10

        self.fc1 = nn.Linear(args.dim, hidden_dim)
        self.fc2 = nn.Linear(args.dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, args.dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.fc1(x))
        x_V = self.fc2(x)

        x = swish * x_V
        return self.fc3(x)


class EncoderBlock(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.ff = FeedForward(args)

        self.norm1 = RMSNorm(self.dim, eps = args.norm_epsilon)
        self.norm2 = RMSNorm(self.dim, eps = args.norm_epsilon)

    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:    

        # Apply Self Attention on the normalized input (embeddings) and perform skip connection
        # (B, seq_len, dim) + (B, seq_len, dim)  -> (B, seq_len, dim)
        h = x + self.attention.forward(self.norm1(x), start_pos, freqs_complex)
 
        # Apply Feed Forward on the normalized attention output and perform skip connection
        out = h + self.ff(self.norm2((h)))
        return out



class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Set the Vocabulary Size"

        self.args = args
        self.tok_embeddings = nn.Embedding(self.args.vocab_size, self.args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(self.args))
        
        self.norm = RMSNorm(self.args.dim, eps = self.args.norm_epsilon)
        self.ff = nn.Linear(self.args.dim, self.args.vocab_size, bias = False)

        self.freqs_complex = compute_thetha_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)
    

    def forward(self, tokens: torch.Tensor, start_pos: int):
        
        # (B, seq_len)
        batch_size, seq_len = tokens.shape

        assert seq_len == 1, "For next token prediction, we process only 1 sequence at a time"

        # (B, seq_len) -> (B, seq_len, embedding_dim)
        h = self.tok_embeddings(tokens)

        # Retrieveing (m, thetha) corresponding to the positions [start_pos : start_pos + seq_len]
        freq_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)

        # Apply RMSNorm
        h = self.norm(h)

        # Apply linear layer
        h = self.ff(h).float()
        return h



