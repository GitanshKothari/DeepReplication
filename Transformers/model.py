import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        
        super().__init__()
        self.d_model = d_model # dimensions of the model
        self.vocab_size = vocab_size # number of words in the vocab

        # initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len # max length of one seq
        self.dropout = nn.Dropout(p=dropout)

        # creating a matrix of shape (seq_len, d_model)
        self.p_encoding = torch.zeros(self.seq_len, self.d_model)
        numerator = torch.arange(0, self.seq_len, dtype = torch.float).unsqueeze(1)
        inv_divide_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # Apply sin on all even terms, cos on all odd terms on d_model dimension
        self.p_encoding[:, 0::2] = torch.sin(numerator * inv_divide_term)
        self.p_encoding[:, 1::2] = torch.cos(numerator * inv_divide_term)

        self.p_encoding = self.p_encoding.unsqueeze(0) # Add a new dimension for batch input (seq_len, d_model) -> (batch, seq_len, d_model)

        self.register_buffer('p_encoding', self.p_encoding)
    
    def forward(self, x):
        # Adds positional encoding to the input
        x = x + (self.p_encoding[:, :x.shape[1], :]).requires_grad(False) # want to go only uptil the length of the current input 
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) #multiplicative parameter
        self.beta = nn.Parameter(torch.zeros(1)) #additive parameter

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim = -1, keepdim=True)

        return self.alpha * ((x - mean) / (std + self.epsilon)) + self.beta

        
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model) #W2 and b2
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear2(x)

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads 

        assert d_model % h == 0, "d_model is not dividible by h" #need to make sure d_model is divisible by number of heads

        self.d_k = d_model // h
        self.Wq = nn.Linear(d_model, d_model) # creates a matrix of size (d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model) # creates a matrix of size (d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model) # creates a matrix of size (d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model) # creates a matrix of size ((dv = dk) * h = d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> # (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)    # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)   # applies Dropout
        
        return (attention_scores @ value), attention_scores 


    def forward(self, query, key, value, mask):
        q_prime = self.Wq(query) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model) 
        k_prime = self.Wk(key) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        v_prime = self.Wv(value) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        q_prime = q_prime.view(q_prime.shape[0], q_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        k_prime = k_prime.view(k_prime.shape[0], k_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        v_prime = v_prime.view(v_prime.shape[0], v_prime.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(q_prime, k_prime, v_prime, mask, self.dropout)
        
        #(batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model) 
        return self.Wo(x)

class SkipConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, attention_block: MultiheadAttentionBlock, feedforward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(p=dropout)
        self.skip_connections = nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self, src_mask):
        x = self.skip_connections[0](x = x, sublayer = lambda x: self.attention_block(x, x, x, src_mask))
        x = self.skip_connections[1](x, sublayer = self.feedforward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock,
                  feedforward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(p=dropout)

        self.skipconnections = nn.ModuleList([SkipConnection(dropout) for _ in range (3)])
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.skipconnections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.skipconnections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.skipconnections[2](x, self.feedforward_block)
                                    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

class Projectionlayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)

class TransformerBlock(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 target_embed: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding,
                 proj_layer: Projectionlayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.proj_layer(x)
    
def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, num_blocks: int = 6, h: int = 8, dropout: float = 0.1, d_ff = 2048) -> TransformerBlock:
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(num_blocks):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feedforward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(num_blocks):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feedforward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    proj_layer = Projectionlayer(d_model, target_vocab_size)

    transformer = TransformerBlock(encoder, decoder, src_embed, target_embed, src_pos, target_pos, proj_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    
    return transformer
    