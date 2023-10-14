import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

  def __init__(self):
    super().__init__(
      # From (Batch_Size, Channel, Height, Width) To (Batch_Size, 128, Height, Width)
      # So just increase the number of Channels without changing size.
      nn.Conv2d(3, 128, kernel_size=3, padding=1),

      # (Batch_Size, 128, Height, Width) To (Batch_Size, 128, Height, Width)
      # Remains the same.
      VAE_ResidualBlock(128, 128),

      # (Batch_Size, 128, Height, Width) To (Batch_Size, 128, Height, Width)
      # Remains the same.
      VAE_ResidualBlock(128, 128),

      # (Batch_Size, 128, Height, Width) To (Batch_Size, 128, Height/2, Width/2)
      # Reduce the size of the image and keep the same amount of Channels.
      nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

      # (Batch_Size, 128, Height/2, Width/2) To (Batch_Size, 256, Height/2, Width/2)
      # Double amount of Channels.
      VAE_ResidualBlock(128, 256),

      # (Batch_Size, 256, Height/2, Width/2) To (Batch_Size, 256, Height/2, Width/2)
      # Remains the same.
      VAE_ResidualBlock(256, 256),

      # (Batch_Size, 256, Height/2, Width/2) To (Batch_Size, 256, Height/4, Width/4)
      # Reduce the size of the image and keep the same amount of Channels.
      nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

      # (Batch_Size, 256, Height/4, Width/4) To (Batch_Size, 512, Height/4, Width/4)
      # Double amount of Channels.
      VAE_ResidualBlock(256, 512),

      # (Batch_Size, 512, Height/4, Width/4) To (Batch_Size, 512, Height/4, Width/4)
      # Remains the same.
      VAE_ResidualBlock(512, 512),

      # (Batch_Size, 512, Height/4, Width/4) To (Batch_Size, 512, Height8, Width/8)
      # Reduce the size of the image and keep the same amount of Channels.
      nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 512, Height8, Width/8)
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 512, Height8, Width/8)
      VAE_AttentionBlock(512),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 512, Height8, Width/8)
      VAE_ResidualBlock(512, 512),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 512, Height8, Width/8)
      nn.GroupNorm(32, 512),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 512, Height8, Width/8)
      nn.SiLU(),

      # (Batch_Size, 512, Height/8, Width/8) To (Batch_Size, 8, Height8, Width/8)
      nn.Conv2d(512, 8, kernel_size=3, padding=1),

      # (Batch_Size, 8, Height/8, Width/8) To (Batch_Size, 8, Height8, Width/8)
      nn.Conv2d(8, 8, kernel_size=1, padding=0)
    )

  def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # x: (Batch_Size, Channel, Height, Width)
    # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
    for module in self:
      if getattr(module, 'stride', None) == (2, 2):
        # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
        x = F.pad(x, (0, 1, 0, 1))
      x = module(x)

    # (Batch_Size, 8, Height/8, Width/8) To two tensors of shape (Batch_Size, 4, Height/8, Width/8)
    mean, log_variance = torch.chunk(x, 2, dim=1)

    # (Batch_Size, 4, Height/8, Width/8) To (Batch_Size, 4, Height/8, Width/8)
    log_variance = torch.clamp(log_variance, -30, 20)

    # (Batch_Size, 4, Height/8, Width/8) To (Batch_Size, 4, Height/8, Width/8)
    variance = log_variance.exp()

    # (Batch_Size, 4, Height/8, Width/8) To (Batch_Size, 4, Height/8, Width/8)
    stdev = variance.sqrt()

    # Z = N(0, 1) To N(mean, variance) = X ?
    # X = mean + stdev * Z
    x = mean + stdev * noise

    # Scale the output by a constant
    # It is not clear why this is done. (but it is used on all stable diffusion models)
    x *= 0.18215

    return x