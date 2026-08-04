import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.p_attn = MultiHeadAttention.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Encoderblock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x  

class Encoder(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNorm()

    def forward(self, x, src_mask=None):
        for block in self.encoder_blocks:
            x = block(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_blocks: nn.ModuleList):
        super().__init__()
        self.decoder_blocks = decoder_blocks
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        for block in self.decoder_blocks:
            x = block(x, encoder_output, tgt_mask, src_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embedding: InputEmbedding,target_embedding: InputEmbedding, src_positional_encoding: PositionalEncoding, tgt_positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.target_embedding = target_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask=None):
        src_embedded = self.input_embedding(src)
        src_encoded = self.src_positional_encoding(src_embedded)
        return self.encoder(src_encoded, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        tgt_embedded = self.target_embedding(tgt)
        tgt_decoded = self.decoder(self.tgt_positional_encoding(tgt_embedded), encoder_output, tgt_mask, src_mask)
        return self.decoder(tgt_decoded, encoder_output, tgt_mask, src_mask)

    def project(self, decoder_output):
        return self.projection_layer(decoder_output)

def build_transformer(d_model: int = 512, src_vocab_size: int = 30000, tgt_vocab_size: int = 30000, src_seq_len: int = 128, tgt_seq_len: int = 128, N: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
    input_embedding = InputEmbedding(d_model, src_vocab_size)
    target_embedding = InputEmbedding(d_model, tgt_vocab_size)
    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = nn.ModuleList([
        Encoderblock(
            MultiHeadAttention(d_model, h, dropout),
            FeedForward(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])
    encoder = Encoder(encoder_blocks)

    decoder_blocks = nn.ModuleList([
        DecoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            MultiHeadAttention(d_model, h, dropout),
            FeedForward(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, input_embedding,target_embedding ,src_positional_encoding,tgt_positional_encoding ,projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer