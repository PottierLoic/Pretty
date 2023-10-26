import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
  def __init__(self, nb_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
    super().__init__()

    self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
    self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.n_heads = nb_heads
    self.d_head = d_embed // nb_heads

  def forward(self, x: torch.Tensor, causal_mask=False):
    # x: (Batch_Size, Seq_Lenght, Dim)

    input_shape = x.shape
    batch_size, sequence_length, d_embed = input_shape

    intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

    # (Batch_Size, Sequence_Length, Dim) To (Batch_Size, Sequence_Length, Dim * 3) -> 3 tensors of shape (Batch_Size, Sequence_Length, Dim)
    q, k, v = self.in_proj(x).chunk(3, dim=-1)

    # (Batch_Size, Sequence_Length, Dim) To (Batch_Size, Sequence_Length, Heads, Dim / Heads) To (Batch_Size, Heads, Sequence_Length, Dim / Heads)
    q = q.view(intermim_shape).transpose(1, 2)
    k = k.view(intermim_shape).transpose(1, 2)
    v = v.view(intermim_shape).transpose(1, 2)

    # (Batch_Size, Heads, Sequence_Length, Sequence_lenght)
    weight = q @ k.transpose(-1, -2)

    if causal_mask:
      mask =torch.ones_like(weight, dtype=torch.bool).triu(1)
      weight.masked_fill_(mask, -torch.inf)
      weight /= math.sqrt(self.d_head)
      weight = F.softmax(weight, dim=-1)

      # (Batch_Size, Sequence_Length, Sequence_Length) To (Batch_Size, Heads, Sequence_Length, Dimension / Heads) To (Batch_Size, Heads, Sequence_Length, Dimension / Heads)
      output = weight @ v

      # (Batch_Size, Heads, Sequence_Length, Dimension / Heads) To (Batch_Size, Sequence_Length, Heads, Dimension / Heads)
      output = output.transpose(1, 2)

      output = output.reshape(input_shape)

      output = self.out_proj(output)

      # (Batch_Size, Sequence_Length, Dimension)
      return output
