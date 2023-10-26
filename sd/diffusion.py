import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
  def __init__(self, n_embed: int):
    super().__init__()
    self.linear_1 = nn.Linear(n_embed, n_embed * 4)
    self.linear_2 = nn.Linear(n_embed * 4, n_embed)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (1, 320)
    x = self.linear_1(x)
    x = F.silu(x)
    x = self.linear_2(x)

    # (1, 1280)
    return x

class UNET(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoders = nn.Module([
      # (Batch_Size, 4, Height / 8, Width / 8)
      SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
      SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
      SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

      # (Batch_Size, 320, Height / 8, Width / 8) To (Batch_Size, 320, Height / 16, Width / 16)
      SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
      SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
      SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

      # (Batch_Size, 640, Height / 16, Width / 16) To (Batch_Size, 640, Height / 32, Width / 32)
      SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
      SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
      SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

      # (Batch_Size, 1280, Height / 32, Width / 32) To (Batch_Size, 1280, Height / 64, Width / 64)
      SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
      SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

      # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
      SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
    ])

    self.bottle_neck = SwitchSequential(
      UNET_ResidualBlock(1280, 1280),
      UNET_AttentionBlock(8, 160),
      UNET_ResidualBlock(1280, 1280),
    )

    self.decoders = nn.Module([
      # decoders takes the output of the encoder and the output of the previous decoder.
      # That's the reason why we expect double the size.

      # (Batch_Size, 2560, Height / 64, Width / 64) To (Batch_Size, 1280, Height / 32, Width / 32)
      SwitchSequential(UNET_ResidualBlock(2560, 1280)),
      SwitchSequential(UNET_ResidualBlock(2560, 1280)),
      SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
      SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
      SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
      SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
      SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
      SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
      SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
      SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
      SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
      SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
    ])

class UNET_ResidualBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int,  n_time: int = 1280):
    super().__init__()
    self.group_norm_feature = nn.GroupNorm(32, in_channels)
    self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.linear_time = nn.Linear(n_time, out_channels)

    self.group_norm_merged = nn.GroupNorm(32, out_channels)
    self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

  def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    # feature: (Batch_Size, In_Channels, Height, Width)
    # time: (1, 1280)
    residue = feature
    feature = self.group_norm_feature(feature)
    feature = F.silu(feature)
    feature = self.conv_feature(feature)

    time =  F.silu(time)
    time = self.linear_time(time)

    merged = feature + time.unsqueeze(-1).unsqueeze(-1)
    merged = self.group_norm_merged(merged)
    merged = F.silu(merged)
    merged = self.conv_merged(merged)

    return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
  def __init__(self, n_head:  int, n_embed: int, d_context: int = 768):
    super().__init__()
    channels = n_head *  n_embed

    self.group_norm = nn.GroupNorm(32, channels)
    self.conv_imput = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    self.layer_norm_1 = nn.LayerNorm(channels)
    self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
    self.layer_norm_2 = nn.LayerNorm(channels)
    self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
    self.layer_norm_3 = nn.LayerNorm(channels)
    self.linear_geglu_1 = nn.Linear(channels, channels * 4 * 2)
    self.linear_geglu_2 = nn.Linear(channels * 4, channels)

    self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

  def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    # x: (Batch_Size, Features, Height, Width)
    # context: (Batch_Size, Sequence_Length, Dimension)

    residue_long = x

    x = self.group_norm(x)
    x = self.conv_imput(x)

    n, c, h, w = x.shape

    # (Batch_Size, Features, Height, Width) To (Batch_Size, Features, Height * Width)
    x = x.view((n, c, h * w))

    # (Batch_Size, Features, Height * Width) To (Batch_Size, Height * Width, Features)
    x  = x.transpose(-1, -2)

    # Normalization + SelfAttention with skip connection.
    residue_short = x

    x = self.layer_norm_1(x)
    x = self.attention_1(x)
    x =+ residue_short

    residue_short = x

    # Normalization + CrossAttention with skip connection.
    x = self.layer_norm_2(x)
    x = self.attention_2(x, context) # CrossAttention

    x =+ residue_short

    residue_short = x

    # Normalization + FeedForward Layer with geglu and skip connection.
    x = self.layer_norm_3(x)

    x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
    x *= F.gelu(gate)
    x = self.linear_geglu_2(x)

    x += residue_short

    # (Batch_Size, Height * Width, Features) To (Batch_Size, Features, Height * Width)
    x = x.transpose(-1, -2)

    # (Batch_Size, Features, Height * Width) To (Batch_Size, Features, Height, Width)
    x = x.view((n, c, h, w))

    return self.conv_output(x) + residue_long



class UpSample(nn.Module):
  def ___init__(self, channels: int):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height * 2, Width * 2)
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    return self.conv(x)

class SwitchSequential(nn.Sequential):
  def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    for layer in self:
      if isinstance(layer, UNET_AttentionBlock):
        x = layer(x, context)
      elif isinstance(layer, UNET_ResidualBlock):
        x = layer(x, time)
      else :
        x = layer(x)
    return x

class UNET_OutputLayer(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.group_norm = nn.GroupNorm(32, in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (Batch_Size, 320, Height / 8, Width / 8)
    x = self.group_norm(x)
    x = F.silu(x)
    x = self.conv(x)

    # (Batch_Size, 4, Height / 8, Width / 8)
    return x

class Diffusion(nn.Module):
  def __init__(self):
    self.time_embedding = TimeEmbedding(320)
    self.u_net = UNET()
    self.final = UNET_OutputLayer(320, 4)

  def forward(self,  latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
    # Sizes:
    # latent (Batch_Size, 4, Height / 8, Width / 8)
    # Context (Batch_Size, Sequence_Length, Dimension)
    # Time (1, 320)

    # (1, 320) To (1, 1280)
    time = self.time_embedding(time)

    # (Batch_Size, 4, Height / 8, Width / 8) To (Batch_Size, 320, Height / 8, Width / 8)
    output = self.unet(latent, context, time)

    # (Batch_Size, 320, Height / 8, Width / 8) To (Batch_Size, 4, Height / 8, Width / 8)
    output = self.final(output)

    # (Batch_Size, 4, Height / 8, Width / 8)
    return output
