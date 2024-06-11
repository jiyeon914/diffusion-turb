from functools import partial

from torch import nn
import torch.nn.functional as F

# from .modules import default, Residual, PreNorm, Downsample, ResnetBlock, Attention, LinearAttention, make_attn
from .modules import default, Downsample, ResnetBlock



class Encoder(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
                 z_channels=None, resnet_block_groups=32):
        super().__init__()

        # determine dimensions
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(
            in_channels=input_channels, out_channels=init_dim, kernel_size=3, padding=1, padding_mode='circular')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_blocks = len(in_out)

        # layers
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        self.downs = nn.ModuleList([])
        for idx, (in_channels, out_channels) in enumerate(in_out):
            is_last = idx >= (num_blocks - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                    block_klass(in_channels, in_channels),
                    Downsample(in_channels, out_channels) if not is_last
                    else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
                    ]
                )
            )

        mid_channels = dims[-1]
        self.z_channels = z_channels if z_channels is not None else mid_channels
        self.mid_block = block_klass(mid_channels, mid_channels)
        self.conv_out = nn.Conv3d(mid_channels, self.z_channels, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x):
        h = self.init_conv(x)
        for block, downsample in self.downs:
            h = block(h)
            h = downsample(h)

        h = self.mid_block(h)
        h = self.conv_out(h)
        return h
    
# class Encoder(nn.Module):
#     def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
#                  z_channels=None, resnet_block_groups=32, attn_type="vanilla"):
#         super().__init__()

#         # determine dimensions
#         input_channels = channels
#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv3d(
#             in_channels=input_channels, out_channels=init_dim, kernel_size=3, padding=1, padding_mode='circular')

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#         num_blocks = len(in_out)

#         # layers
#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)
#         self.downs = nn.ModuleList([])
#         for idx, (in_channels, out_channels) in enumerate(in_out):
#             is_last = idx >= (num_blocks - 1)
#             self.downs.append(
#                 nn.ModuleList([
#                     block_klass(in_channels, in_channels),
#                     block_klass(in_channels, in_channels),
#                     Residual(PreNorm(in_channels, make_attn(in_channels, attn_type=attn_type))),
#                     Downsample(in_channels, out_channels) if not is_last
#                     else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),])
#             )

#         mid_channels = dims[-1]
#         self.z_channels = z_channels if z_channels is not None else mid_channels
#         self.mid_block1 = block_klass(mid_channels, mid_channels)
#         self.mid_attn = Residual(PreNorm(mid_channels, Attention(mid_channels)))
#         self.mid_block2 = block_klass(mid_channels, self.z_channels)

#     def forward(self, x):
#         h = self.init_conv(x)
#         for block1, block2, attn, downsample in self.downs:
#             h = block1(h)
#             h = block2(h)
#             h = attn(h)
#             h = downsample(h)

#         h = self.mid_block1(h)
#         h = self.mid_attn(h)
#         h = self.mid_block2(h)
#         return h
    
# class Encoder(nn.Module):
#     def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
#                  z_channels=None, resnet_block_groups=32, use_linear_attn=False, attn_type="vanilla"):
#         super().__init__()
#         if use_linear_attn: attn_type = "linear"

#         # determine dimensions
#         input_channels = channels
#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv3d(
#             in_channels=input_channels, out_channels=init_dim, kernel_size=3, padding=1, padding_mode='circular')

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#         num_blocks = len(in_out)

#         # layers
#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)
#         self.downs = nn.ModuleList([])
#         for idx, (in_channels, out_channels) in enumerate(in_out):
#             is_last = idx >= (num_blocks - 1)
#             self.downs.append(
#                 nn.ModuleList([
#                     block_klass(in_channels, in_channels),
#                     block_klass(in_channels, in_channels),
#                     Residual(PreNorm(in_channels, make_attn(in_channels, attn_type=attn_type))),
#                     Downsample(in_channels, out_channels) if not is_last
#                     else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),])
#             )

#         mid_channels = dims[-1]
#         self.z_channels = z_channels if z_channels is not None else mid_channels
#         self.mid_block1 = block_klass(mid_channels, mid_channels)
#         self.mid_attn = Residual(PreNorm(mid_channels, Attention(mid_channels)))
#         self.mid_block2 = block_klass(mid_channels, self.z_channels)

#     def forward(self, x):
#         h = self.init_conv(x)
#         for block1, block2, attn, downsample in self.downs:
#             h = block1(h)
#             h = block2(h)
#             h = attn(h)
#             h = downsample(h)

#         h = self.mid_block1(h)
#         h = self.mid_attn(h)
#         h = self.mid_block2(h)
#         return h
