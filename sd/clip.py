import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class CLIPEmbedding(nn.Module):
  def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
    super().__init__()
    self.token_embedding = nn.EmbeddingBag(n_vocab, n_embed)
    self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

  def forward(self, tokens):
    # (Batch_size, Sequence_Length) -> (Batch_size, Sequence_lenght, Dimension)
    x = self.token_embedding(tokens)
    x += self.position_embedding
    return x

class CLIPLayer(nn.Module):
  def __init__(self, n_head: int, n_embed: int):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(n_embed)
    self.attention = SelfAttention(n_head, n_embed)
    self.layer_norm_2 = nn.LayerNorm(n_embed)
    self.linear_1 = nn.Linear(n_embed, n_embed * 4)
    self.linear_2 = nn.Linear(n_embed * 4, n_embed)

  def forward(self, x: torch.Tensor) ->  torch.Tensor:
    # (Batch_Size, Sequence_lenght, Dimension)
    residue = x

    ## Self Attention
    x = self.layernorm_1(x)
    x = self.attention(x, causal_mask=True)

    x += residue

    ## Feed forward layer
    residue = x

    x = self.layernorm_2(x)
    x = self.linear_1(x)

    # quickGELU activation function
    # work better in practice, no other reason
    x = x * torch.sigmoid(1.702 * x)

    x = self.linear_2(x)
    x += residue

    return x

class CLIP(nn.Module):
  def __init__(self) -> None:
    self.embedding = CLIPEmbedding(49480, 768, 77)
    self.layers = nn.Moules([
      CLIPLayer(12, 768) for _ in range(12)
    ])

    self.layer_norm = nn.LayerNorm(768)

  def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    tokens = tokens.type(torch.long)

    # (Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, Dimension)
    state = self.embedding(tokens)

    for layer in self.layers:
      state = layer(state)

    # (Batch_Size, Sequence_Length, Dimension)
    output = self.layer_norm(state)

    return output
